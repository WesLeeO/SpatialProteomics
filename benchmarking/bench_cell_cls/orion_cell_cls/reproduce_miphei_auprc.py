"""
Score OUR token model's per-cell predictions with MIPHEI's EXACT cell-level protocol,
so the AUPRC is directly comparable to their paper table (ORION col, ViT=0.517).

What is kept identical to MIPHEI (so only OUR predictions differ):
  * GT positivity `_pos`  -> from MIPHEI's results parquet (the val_test_nuclei_dataframe
    gating they scored against), NOT the csv_nuclei_pos we downloaded.
  * cell set + split      -> their `cell_id` + `split` columns (train=CRC19+CRC30, test=CRC11+CRC02).
  * calibration head       -> OneVsRest LogisticRegression(class_weight="balanced",
    random_state=42) + StandardScaler, fit on TRAIN cells (src/metrics.train_logistic_regression).
  * metric                 -> average_precision_score per marker, mean over the 15 markers.
  * bootstrap              -> resample TEST *tiles* with replacement, np.random.seed(42) ->
    1000 child seeds; report mean + 2.5/97.5 percentile CI.

Only swapped in: OUR `mean_<marker>` (area-weighted token mean per nucleus) as `<marker>_pred`,
joined onto their cells by nucleus label == cell_id.
"""
import argparse, re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.utils import resample
from joblib import Parallel, delayed

TILE = Path("/mnt/ssd/virtual_proteomics/data/ORIONCRC_dataset_tile_20x")
FEAT = Path("cell_cls/orion_cell_cls")

# MIPHEI marker order (their _pos columns, Hoechst excluded)
MARKERS = ["CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a", "CD45RO", "CD20",
           "PD-L1", "CD3e", "CD163", "E-cadherin", "Ki67", "Pan-CK", "SMA"]
# user slide id -> substring in MIPHEI full slide_name
SLIDE_KEY = {"CRC19": "_C19_", "CRC30": "_C30_", "CRC11": "_C11_", "CRC02": "18459"}

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def build_cell_dataframe(gt_parquet: str) -> pd.DataFrame:
    """Join OUR mean_<marker> (as _pred) onto THEIR cells/_pos/split by nucleus label."""
    gt = pd.read_parquet(gt_parquet,
                         columns=["cell_id", "slide_name", "split"] + [f"{m}_pos" for m in MARKERS])
    gt["slide_name"] = gt["slide_name"].astype(str)
    gt["cell_id"] = gt["cell_id"].astype(np.int64)

    parts = []
    for sid, key in SLIDE_KEY.items():
        g = gt[gt["slide_name"].str.contains(key)].copy()
        u = pd.read_parquet(FEAT / f"cell_token_features_{sid}.parquet")
        u["label"] = u["label"].astype(np.int64)
        # map our mean_marker -> marker_pred via normalized marker name
        umean = {norm(c[len("mean_"):]): c for c in u.columns if c.startswith("mean_")}
        ren = {umean[norm(m)]: f"{m}_pred" for m in MARKERS}
        u = u[["label", "x", "y"] + list(ren)].rename(columns=ren)
        j = g.merge(u, left_on="cell_id", right_on="label", how="inner").drop(columns="label")
        parts.append(j)
        print(f"  [{sid}] their cells={len(g)}  ours-joined={len(j)}  "
              f"dropped(no pred)={len(g)-len(j)} ({(len(g)-len(j))/len(g):.2%})")
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=[f"{m}_pred" for m in MARKERS] + [f"{m}_pos" for m in MARKERS])
    return df


def assign_test_tiles(df_test: pd.DataFrame) -> np.ndarray:
    """Assign each TEST cell to a MIPHEI test tile (512x512, overlapping) by point-in-box."""
    t = pd.read_csv(TILE / "test_dataframe.csv")
    def tinfo(p):
        s = p.split("_")[-5:]; s[-1] = s[-1].split(".")[0]; return list(map(int, s))
    info = np.array([tinfo(p) for p in t["image_path"]])         # x,y,level,tsx,tsy
    t = t.assign(tx=info[:, 0], ty=info[:, 1], ts=info[:, 3])
    tile_id = np.full(len(df_test), -1, np.int64)
    # match per slide on the SAME coordinate space the centroids live in (level-0)
    for key in ("_C11_", "18459"):
        ts = t[t["in_slide_name"].str.contains(key)]
        cm = df_test["slide_name"].str.contains(key).values
        if not cm.any() or len(ts) == 0:
            continue
        half = ts["ts"].values[:, None] / 2.0
        centers = np.c_[ts["tx"].values + ts["ts"].values / 2.0,
                        ts["ty"].values + ts["ts"].values / 2.0]
        pts = df_test.loc[cm, ["x", "y"]].values.astype(float)
        tree = cKDTree(centers)
        # nearest tile center (Chebyshev via querying a few neighbours, check containment)
        gidx = np.where(cm)[0]
        d, nn = tree.query(pts, k=1)
        tx0 = ts["tx"].values[nn]; ty0 = ts["ty"].values[nn]; tsz = ts["ts"].values[nn]
        inside = (pts[:, 0] >= tx0) & (pts[:, 0] < tx0 + tsz) & \
                 (pts[:, 1] >= ty0) & (pts[:, 1] < ty0 + tsz)
        # global tile index = row position in t for this slide
        slide_tile_global = ts.index.values[nn]
        tile_id[gidx[inside]] = slide_tile_global[inside]
        # fallback for the few not in nearest center: ball query
        miss = gidx[~inside]
        if len(miss):
            for gi, p in zip(miss, pts[~inside]):
                cand = tree.query_ball_point(p, r=float(ts["ts"].max()), p=np.inf)
                for c in cand:
                    if ts["tx"].values[c] <= p[0] < ts["tx"].values[c]+ts["ts"].values[c] and \
                       ts["ty"].values[c] <= p[1] < ts["ty"].values[c]+ts["ts"].values[c]:
                        tile_id[gi] = ts.index.values[c]; break
    return tile_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_parquet",
                    default="checkpoints/MIPHEI-vit/orion_cell_dataframe_logreg.parquet",
                    help="source of MIPHEI GT _pos + split (their gating)")
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()

    pred_cols = [f"{m}_pred" for m in MARKERS]
    pos_cols = [f"{m}_pos" for m in MARKERS]

    print("== join our preds onto their cells/_pos ==")
    df = build_cell_dataframe(args.gt_parquet)
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"].reset_index(drop=True)
    print(f"  train cells={len(tr)}  test cells={len(te)}")

    # --- MIPHEI head: OneVsRest LR balanced + StandardScaler, fit on TRAIN ---
    scaler = StandardScaler().fit(tr[pred_cols].values)
    clf = OneVsRestClassifier(LogisticRegression(class_weight="balanced", random_state=42))
    clf.fit(scaler.transform(tr[pred_cols].values), tr[pos_cols].values.astype(int))
    proba = clf.predict_proba(scaler.transform(te[pred_cols].values))   # (Ntest, 15)
    y = te[pos_cols].values.astype(int)

    # --- point-estimate AUPRC per marker ---
    ap_point = np.array([average_precision_score(y[:, i], proba[:, i]) for i in range(len(MARKERS))])
    print("\n== per-marker AUPRC (point estimate) ==")
    for m, a in zip(MARKERS, ap_point):
        print(f"  {m:12} {a:.3f}")
    print(f"  {'MEAN15':12} {ap_point.mean():.3f}   (MIPHEI-ViT paper=0.517, our repro of theirs=0.510)")

    # --- tile bootstrap (seed 42), AUPRC ---
    tile_id = assign_test_tiles(te)
    keep = tile_id >= 0
    print(f"\n  test cells with a tile: {keep.sum()}/{len(te)} (dropped {len(te)-keep.sum()})")
    proba, y, tile_id = proba[keep], y[keep], tile_id[keep]
    order = np.argsort(tile_id, kind="stable")
    tile_id, proba, y = tile_id[order], proba[order], y[order]
    uniq, starts = np.unique(tile_id, return_index=True)
    groups = np.split(np.arange(len(tile_id)), starts[1:])   # cell indices per tile

    np.random.seed(42)
    seeds = np.random.randint(0, 10000, size=args.n_boot)

    def one(seed):
        samp = resample(np.arange(len(uniq)), replace=True, n_samples=len(uniq), random_state=seed)
        idx = np.concatenate([groups[g] for g in samp])
        yy, pp = y[idx], proba[idx]
        return [average_precision_score(yy[:, i], pp[:, i]) for i in range(len(MARKERS))]

    print(f"\n== bootstrapping {args.n_boot}x over {len(uniq)} test tiles (seed 42) ==")
    boot = np.array(Parallel(n_jobs=-1, verbose=1)(delayed(one)(s) for s in seeds))  # (n_boot,15)
    mean15 = boot.mean(axis=1)                                # per-resample mean over markers
    lo, hi = np.percentile(mean15, [2.5, 97.5])
    print(f"\n== HEADLINE: mean-over-15 AUPRC ==")
    print(f"  point estimate     : {ap_point.mean():.3f}")
    print(f"  bootstrap mean     : {mean15.mean():.3f}")
    print(f"  95% CI             : [{lo:.3f}, {hi:.3f}]")
    print(f"  MIPHEI-ViT (paper) : 0.517")

    out = pd.DataFrame({"marker": MARKERS, "auprc_point": ap_point,
                        "auprc_boot_mean": boot.mean(axis=0),
                        "ci_lo": np.percentile(boot, 2.5, axis=0),
                        "ci_hi": np.percentile(boot, 97.5, axis=0)})
    out.to_csv(FEAT / "ours_miphei_protocol_auprc.csv", index=False)
    print(f"\nsaved per-marker -> {FEAT/'ours_miphei_protocol_auprc.csv'}")


if __name__ == "__main__":
    main()