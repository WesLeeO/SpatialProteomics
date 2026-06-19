"""
Cell-level AUPRC under MIPHEI's EXACT protocol, for ANY prediction source.

Scores per-cell marker-positivity predictions the same way MIPHEI scores their paper
table (ORION), so numbers are directly comparable to it (MIPHEI-ViT = 0.517).

Prediction sources (--source):
  ours    : our token model's per-cell features  cell_token_features_<slide>.parquet
            (mean_<marker>  ->  <marker>_pred), joined onto their cells by nucleus label.
  parquet : any MIPHEI results parquet that already has <marker>_pred columns
            (e.g. checkpoints/MIPHEI-vit/orion_cell_dataframe_logreg.parquet,
             MIPHEI-convnext/..., or any baseline they released). --pred_parquet PATH.

Held identical to MIPHEI regardless of source (only the _pred values differ):
  * GT _pos + split  -> from --gt_parquet (model-independent; their re-gated labels,
    NOT csv_nuclei_pos). train = CRC19+CRC30, test = CRC11+CRC02.
  * head             -> OneVsRest LogisticRegression(class_weight="balanced",
    random_state=42) + StandardScaler, fit on TRAIN cells.
  * metric           -> average_precision_score per marker, mean over 15 markers.
  * bootstrap        -> resample TEST tiles, np.random.seed(42) -> 1000 child seeds,
    report mean + 2.5/97.5 percentile CI. x/y recovered from csv_nuclei_pos.

Examples:
  python eval_cell_auprc.py                                   # our model (default)
  python eval_cell_auprc.py --source parquet \
      --pred_parquet checkpoints/MIPHEI-convnext/orion_cell_dataframe_logreg.parquet
"""
import argparse, re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score
from sklearn.utils import resample
from joblib import Parallel, delayed

TILE = Path("/mnt/ssd/virtual_proteomics/data/ORIONCRC_dataset_tile_20x")
FEAT = Path("/home/wesley/spatial_proteomics/cell_cls/orion_cell_cls/")
DEFAULT_GT = "/home/wesley/spatial_proteomics/checkpoints/MIPHEI-vit/orion_cell_dataframe_logreg.parquet"

MARKERS = ["CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a", "CD45RO", "CD20",
           "PD-L1", "CD3e", "CD163", "E-cadherin", "Ki67", "Pan-CK", "SMA"]
SLIDE_KEY = {"CRC19": "_C19_", "CRC30": "_C30_", "CRC11": "_C11_", "CRC02": "18459"}
TEST_KEYS = ("_C11_", "18459")
POS = [f"{m}_pos" for m in MARKERS]
PRED = [f"{m}_pred" for m in MARKERS]

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def slide_meta() -> pd.DataFrame:
    """orion_slide_id -> full in_slide_name + nuclei_csv_path."""
    return pd.read_csv(TILE / "slide_dataframe.csv")


def load_gt(gt_parquet: str) -> pd.DataFrame:
    """Model-independent GT: cell_id, slide_name, split, *_pos."""
    gt = pd.read_parquet(gt_parquet, columns=["cell_id", "slide_name", "split"] + POS)
    gt["slide_name"] = gt["slide_name"].astype(str)
    gt["cell_id"] = gt["cell_id"].astype(np.int64)
    return gt


def load_preds(source: str, pred_parquet: str, feat_dir: Path,
               meta: pd.DataFrame) -> pd.DataFrame:
    """Return cell_id, slide_name, *_pred for the requested source."""
    if source == "parquet":
        d = pd.read_parquet(pred_parquet, columns=["cell_id", "slide_name"] + PRED)
        d["slide_name"] = d["slide_name"].astype(str)
        d["cell_id"] = d["cell_id"].astype(np.int64)
        return d
    # source == "ours": per-slide map our mean_<marker> -> <marker>_pred, attach full slide_name
    name = dict(zip(meta["orion_slide_id"], meta["in_slide_name"]))
    parts = []
    for sid in SLIDE_KEY:
        u = pd.read_parquet(feat_dir / f"cell_token_features_{sid}.parquet")
        umean = {norm(c[len("mean_"):]): c for c in u.columns if c.startswith("mean_")}
        ren = {umean[norm(m)]: f"{m}_pred" for m in MARKERS}
        u = u[["label"] + list(ren)].rename(columns=ren).rename(columns={"label": "cell_id"})
        u["cell_id"] = u["cell_id"].astype(np.int64)
        u["slide_name"] = str(name[sid])
        parts.append(u)
    return pd.concat(parts, ignore_index=True)


def recover_xy(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Attach level-0 centroid x,y from csv_nuclei_pos (by slide + cell_id). Test slides only."""
    csv_of = dict(zip(meta["in_slide_name"].astype(str), meta["nuclei_csv_path"]))
    out = []
    for key in TEST_KEYS:
        sl = df["slide_name"].str.contains(key)
        if not sl.any():
            continue
        full = df.loc[sl, "slide_name"].iloc[0]
        nu = pd.read_csv(TILE / csv_of[full], usecols=["label", "x", "y"])
        nu["label"] = nu["label"].astype(np.int64)
        m = df[sl].merge(nu, left_on="cell_id", right_on="label", how="left").drop(columns="label")
        out.append(m)
    return pd.concat(out, ignore_index=True)


def assign_test_tiles(df_test: pd.DataFrame) -> np.ndarray:
    """Map each TEST cell to the MIPHEI test tile (512x512, overlapping) containing it."""
    t = pd.read_csv(TILE / "test_dataframe.csv")
    def tinfo(p):
        s = p.split("_")[-5:]; s[-1] = s[-1].split(".")[0]; return list(map(int, s))
    info = np.array([tinfo(p) for p in t["image_path"]])
    t = t.assign(tx=info[:, 0], ty=info[:, 1], ts=info[:, 3])
    tile_id = np.full(len(df_test), -1, np.int64)
    for key in TEST_KEYS:
        ts = t[t["in_slide_name"].str.contains(key)]
        cm = df_test["slide_name"].str.contains(key).values
        if not cm.any() or len(ts) == 0:
            continue
        centers = np.c_[ts["tx"].values + ts["ts"].values / 2.0,
                        ts["ty"].values + ts["ts"].values / 2.0]
        pts = df_test.loc[cm, ["x", "y"]].values.astype(float)
        tree = cKDTree(centers)
        gidx = np.where(cm)[0]
        _, nn = tree.query(pts, k=1)
        tx0, ty0, tsz = ts["tx"].values[nn], ts["ty"].values[nn], ts["ts"].values[nn]
        inside = (pts[:, 0] >= tx0) & (pts[:, 0] < tx0 + tsz) & \
                 (pts[:, 1] >= ty0) & (pts[:, 1] < ty0 + tsz)
        tile_id[gidx[inside]] = ts.index.values[nn][inside]
        for gi, p in zip(gidx[~inside], pts[~inside]):
            for c in tree.query_ball_point(p, r=float(ts["ts"].max()), p=np.inf):
                if ts["tx"].values[c] <= p[0] < ts["tx"].values[c] + ts["ts"].values[c] and \
                   ts["ty"].values[c] <= p[1] < ts["ty"].values[c] + ts["ts"].values[c]:
                    tile_id[gi] = ts.index.values[c]; break
    return tile_id


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["ours", "parquet"], default="ours")
    ap.add_argument("--pred_parquet", default=None,
                    help="results parquet with <marker>_pred (required for --source parquet)")
    ap.add_argument("--feat_dir", default=str(FEAT), help="dir with cell_token_features_<slide>.parquet")
    ap.add_argument("--gt_parquet", default=DEFAULT_GT, help="source of GT _pos + split")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--tag", default=None, help="label for output csv (default: source/model name)")
    args = ap.parse_args()
    if args.source == "parquet" and not args.pred_parquet:
        ap.error("--source parquet requires --pred_parquet")
    tag = args.tag or (args.source if args.source == "ours"
                       else Path(args.pred_parquet).parent.name)

    meta = slide_meta()
    print(f"== source={args.source}  tag={tag}  gt={args.gt_parquet} ==")
    gt = load_gt(args.gt_parquet)
    preds = load_preds(args.source, args.pred_parquet, Path(args.feat_dir), meta)
    df = gt.merge(preds, on=["cell_id", "slide_name"], how="inner").dropna(subset=PRED + POS)
    for sid, key in SLIDE_KEY.items():
        n_gt = (gt["slide_name"].str.contains(key)).sum()
        n = (df["slide_name"].str.contains(key)).sum()
        print(f"  [{sid}] their cells={n_gt}  scored={n}  dropped={n_gt-n} ({(n_gt-n)/n_gt:.2%})")

    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"].reset_index(drop=True)
    te = recover_xy(te, meta)
    print(f"  train={len(tr)}  test={len(te)}")

    scaler = StandardScaler().fit(tr[PRED].values)
    clf = OneVsRestClassifier(LogisticRegression(class_weight="balanced", random_state=42))
    clf.fit(scaler.transform(tr[PRED].values), tr[POS].values.astype(int))
    proba = clf.predict_proba(scaler.transform(te[PRED].values))
    y = te[POS].values.astype(int)

    ap_point = np.array([average_precision_score(y[:, i], proba[:, i]) for i in range(len(MARKERS))])

    # F1 at MIPHEI's protocol: fixed 0.5 decision threshold on the calibrated probability
    # (== sklearn clf.predict; == their sigmoid(logreg) > 0.5). No per-marker tuning.
    f1_point = np.array([f1_score(y[:, i], proba[:, i] > 0.5, zero_division=0)
                         for i in range(len(MARKERS))])

    print("\n== per-marker AUPRC / F1 (point estimate) ==")
    for m, a, f in zip(MARKERS, ap_point, f1_point):
        print(f"  {m:12}  AUPRC={a:.3f}  F1={f:.3f}")
    print(f"  {'MEAN15':12}  AUPRC={ap_point.mean():.3f}  F1={f1_point.mean():.3f}")

    pred_df = te[["cell_id", "slide_name", "x", "y"]].copy()
    for i, m in enumerate(MARKERS):
        pred_df[f"{m}_prob"] = proba[:, i]
        pred_df[f"{m}_binary"] = (proba[:, i] > 0.5).astype(np.int8)
        pred_df[f"{m}_pos"] = y[:, i].astype(np.int8)
    pred_path = Path(args.feat_dir) / f"cell_predictions_{tag}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\nsaved per-cell predictions -> {pred_path}")
    tile_id = assign_test_tiles(te)
    keep = tile_id >= 0
    proba, y, tile_id = proba[keep], y[keep], tile_id[keep]
    order = np.argsort(tile_id, kind="stable")
    tile_id, proba, y = tile_id[order], proba[order], y[order]
    uniq, starts = np.unique(tile_id, return_index=True)
    """
    Concrete example — 6 cells across 3 tiles:
    tile_id (sorted) = [10, 10, 10, 20, 20, 30]
    uniq             = [10, 20, 30]
    starts           = [0,  3,  5]      # first row of each tile
    starts[1:]       = [3, 5]           # cut points
    np.split([0,1,2,3,4,5], [3,5])      
        -> [array([0,1,2]), array([3,4]), array([5])]
    So tile 10 owns rows 0–2, tile 20 owns rows 3–4, tile 30 owns row 5.
    """
    groups = np.split(np.arange(len(tile_id)), starts[1:])
    print(f"\n  test cells with a tile: {keep.sum()}/{len(keep)}  | tiles: {len(uniq)}")

    np.random.seed(42)
    seeds = np.random.randint(0, 10000, size=args.n_boot)

    def one(seed):
        samp = resample(np.arange(len(uniq)), replace=True, n_samples=len(uniq), random_state=seed)
        # retrive all cells beloning to tiles sampled with boostrapping 
        idx = np.concatenate([groups[g] for g in samp])
        auprc = [average_precision_score(y[idx][:, i], proba[idx][:, i]) for i in range(len(MARKERS))]
        f1 = [f1_score(y[idx][:, i], proba[idx][:, i] > 0.5, zero_division=0)
              for i in range(len(MARKERS))]
        return auprc + f1  # length 2*N_MARKERS

    print(f"\n== bootstrap {args.n_boot}x over {len(uniq)} tiles (seed 42) ==")
    raw = np.array(Parallel(n_jobs=-1, verbose=1)(delayed(one)(s) for s in seeds)) # repeat n.bootis time
    boot_auprc = raw[:, :len(MARKERS)]   # (n_boot, N)
    boot_f1    = raw[:, len(MARKERS):]   # (n_boot, N)

    mean15_auprc = boot_auprc.mean(axis=1)
    mean15_f1    = boot_f1.mean(axis=1)
    lo_a, hi_a = np.percentile(mean15_auprc, [2.5, 97.5])
    lo_f, hi_f = np.percentile(mean15_f1,    [2.5, 97.5])
    print(f"\n== HEADLINE mean-15 ({tag}) ==")
    print(f"  AUPRC  point={ap_point.mean():.3f}  boot={mean15_auprc.mean():.3f}  95%CI=[{lo_a:.3f},{hi_a:.3f}]")
    print(f"  F1     point={f1_point.mean():.3f}  boot={mean15_f1.mean():.3f}  95%CI=[{lo_f:.3f},{hi_f:.3f}]")

    out_path = FEAT / f"eval_cell_auprc_{tag}.csv"
    out_df = pd.DataFrame({
        "marker":          MARKERS,
        "auprc_point":     ap_point,
        "auprc_boot_mean": boot_auprc.mean(axis=0),
        "auprc_ci_lo":     np.percentile(boot_auprc, 2.5, axis=0),
        "auprc_ci_hi":     np.percentile(boot_auprc, 97.5, axis=0),
        "f1_point":        f1_point,
        "f1_boot_mean":    boot_f1.mean(axis=0),
        "f1_ci_lo":        np.percentile(boot_f1, 2.5, axis=0),
        "f1_ci_hi":        np.percentile(boot_f1, 97.5, axis=0),
    })
    # headline MEAN15 row: CI from the JOINT bootstrap distribution of the mean
    # (NOT the mean of per-marker CIs — that would be too wide)
    headline = pd.DataFrame([{
        "marker":          "MEAN15",
        "auprc_point":     ap_point.mean(),
        "auprc_boot_mean": mean15_auprc.mean(),
        "auprc_ci_lo":     lo_a, "auprc_ci_hi": hi_a,
        "f1_point":        f1_point.mean(),
        "f1_boot_mean":    mean15_f1.mean(),
        "f1_ci_lo":        lo_f, "f1_ci_hi": hi_f,
    }])
    out_df = pd.concat([out_df, headline], ignore_index=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()