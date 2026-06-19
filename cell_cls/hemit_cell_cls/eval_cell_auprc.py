"""
Cell-level AUPRC + F1 for HEMIT (2 markers: Pan-CK, CD3), directly comparable to
MIPHEI — same cells, same GT, same 20/80 split, only the _pred differs.

We VERIFIED our cells == MIPHEI's: cell_id == our csv `label`, and Pan-CK_pos/CD3_pos
agree 100% per tile. MIPHEI treats HEMIT as an external dataset: the logreg is trained
on a stratified random 20% of ALL cells and evaluated on the other 80%
(dataset_evaluator.py HEMITBaseEvaluator). The exact per-cell assignment lives in the
`split` column of their hemit parquet, so we JOIN it (don't re-run train_test_split).

Held identical to MIPHEI regardless of source (only _pred differs):
  * GT _pos + split  -> from --gt_parquet (MIPHEI-vit hemit parquet), by (cell_id, slide_name).
  * head             -> OneVsRest LogisticRegression(class_weight="balanced",
                        random_state=42) + StandardScaler, fit on the 20% train cells.
  * F1               -> fixed 0.5 decision threshold.
  * metric           -> average_precision_score per marker.
  * CI               -> bootstrap over tiles (slide_name) within the 80% test, seed 42.

Sources (--source):
  ours    : our cell_token_features_{train,val,test}.parquet (mean_<m> -> <m>_pred),
            joined to MIPHEI cells by (label==cell_id, image_name==slide_name stem).
  parquet : any MIPHEI hemit parquet with <m>_pred (--pred_parquet); reproduces their
            number. Note their cols are Pan-CK_pred + CD3e_pred (CD3e -> CD3).

Examples:
  python cell_cls/hemit_cell_cls/eval_cell_auprc.py                        # our model
  python cell_cls/hemit_cell_cls/eval_cell_auprc.py --source parquet \
      --pred_parquet checkpoints/MIPHEI-vit/hemit_cell_dataframe_logreg.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score
from sklearn.utils import resample
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_GT = str(REPO / "checkpoints/MIPHEI-vit/hemit_cell_dataframe_logreg.parquet")
MARKERS = ["Pan-CK", "CD3"]                  # GT targets (have _pos)
POS = [f"{m}_pos" for m in MARKERS]
# logreg INPUT features: 3 predicted channels incl the nuclei/DAPI channel,
# matching MIPHEI (marker_names = [Hoechst, CD3e, Pan-CK] -> 2 targets).
PRED = ["Pan-CK_pred", "CD3_pred", "DAPI_pred"]
# how each feature is named in a MIPHEI hemit parquet (_pred) vs our features (mean_)
PARQUET_FEAT = {"Pan-CK_pred": "Pan-CK_pred", "CD3_pred": "CD3e_pred", "DAPI_pred": "Hoechst_pred"}
# our mean_<marker> column, with fallbacks: an ORION-trained model (zero-shot) names them
# CD3e/Hoechst; a HEMIT-native model would name them CD3/Dapi. First column present wins.
OURS_FEAT    = {"Pan-CK_pred": ["mean_Pan-CK"],
                "CD3_pred":    ["mean_CD3e", "mean_CD3"],
                "DAPI_pred":   ["mean_Hoechst", "mean_Dapi"]}


def stem(s):
    return s.str.replace(".tif", "", regex=False)


def load_gt(gt_parquet: str) -> pd.DataFrame:
    gt = pd.read_parquet(gt_parquet, columns=["cell_id", "slide_name", "split"] + POS)
    gt["slide_name"] = stem(gt["slide_name"].astype(str))
    gt["cell_id"] = gt["cell_id"].astype(np.int64)
    for c in POS:
        gt[c] = gt[c].astype(int)
    return gt


def load_preds(args) -> pd.DataFrame:
    if args.source == "parquet":
        cols = ["cell_id", "slide_name"] + [PARQUET_FEAT[f] for f in PRED]
        d = pd.read_parquet(args.pred_parquet, columns=cols)
        d["slide_name"] = stem(d["slide_name"].astype(str))
        d["cell_id"] = d["cell_id"].astype(np.int64)
        return d.rename(columns={PARQUET_FEAT[f]: f for f in PRED})
    # source == ours: pool our per-split feature parquets
    parts = []
    for split in args.pools:
        f = Path(args.feat_dir) / f"cell_token_features_{split}.parquet"
        if not f.exists():
            raise SystemExit(f"{f} not found — run build_cell_token_features.py --split {split}")
        parts.append(pd.read_parquet(f))
    d = pd.concat(parts, ignore_index=True)
    # resolve each feature to whichever candidate column is present (ORION vs HEMIT names)
    ren = {}
    for f in PRED:
        col = next((c for c in OURS_FEAT[f] if c in d.columns), None)
        if col is None:
            raise SystemExit(f"none of {OURS_FEAT[f]} found for {f} in {args.feat_dir} features")
        ren[col] = f
    d = d.rename(columns=ren).rename(columns={"label": "cell_id", "image_name": "slide_name"})
    d["cell_id"] = d["cell_id"].astype(np.int64)
    d["slide_name"] = d["slide_name"].astype(str)
    return d[["cell_id", "slide_name"] + PRED]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["ours", "parquet"], default="ours")
    ap.add_argument("--pred_parquet", default=None, help="MIPHEI hemit parquet (for --source parquet)")
    ap.add_argument("--feat_dir", default=str(HERE))
    ap.add_argument("--pools", nargs="*", default=["train", "val", "test"],
                    help="our feature splits to pool (the cell pool is ALL tiles)")
    ap.add_argument("--gt_parquet", default=DEFAULT_GT)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()
    if args.source == "parquet" and not args.pred_parquet:
        ap.error("--source parquet requires --pred_parquet")
    tag = args.tag or (args.source if args.source == "ours" else Path(args.pred_parquet).parent.name)

    gt = load_gt(args.gt_parquet)
    preds = load_preds(args)
    df = gt.merge(preds, on=["cell_id", "slide_name"], how="inner").dropna(subset=PRED + POS)
    print(f"== source={args.source} tag={tag} ==")
    print(f"  MIPHEI cells={len(gt)}  scored={len(df)}  dropped={len(gt)-len(df)} "
          f"({(len(gt)-len(df))/len(gt):.2%})")

    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"].reset_index(drop=True)
    print(f"  train(20%)={len(tr)}  test(80%)={len(te)}  tiles(test)={te['slide_name'].nunique()}")

    scaler = StandardScaler().fit(tr[PRED].values)
    clf = OneVsRestClassifier(LogisticRegression(class_weight="balanced", random_state=42))
    clf.fit(scaler.transform(tr[PRED].values), tr[POS].values.astype(int))
    proba = clf.predict_proba(scaler.transform(te[PRED].values))
    y = te[POS].values.astype(int)

    ap_pt = np.array([average_precision_score(y[:, i], proba[:, i]) for i in range(len(MARKERS))])
    f1_pt = np.array([f1_score(y[:, i], proba[:, i] > 0.5, zero_division=0) for i in range(len(MARKERS))])

    tiles = te["slide_name"].values
    uniq = np.unique(tiles)
    groups = {t: np.where(tiles == t)[0] for t in uniq}
    np.random.seed(42)
    seeds = np.random.randint(0, 10000, size=args.n_boot)

    def one(seed):
        samp = resample(uniq, replace=True, n_samples=len(uniq), random_state=seed)
        idx = np.concatenate([groups[t] for t in samp])
        yy, pp = y[idx], proba[idx]
        au = [average_precision_score(yy[:, i], pp[:, i]) for i in range(len(MARKERS))]
        f1 = [f1_score(yy[:, i], pp[:, i] > 0.5, zero_division=0) for i in range(len(MARKERS))]
        return au + f1

    boot = np.array(Parallel(n_jobs=-1)(delayed(one)(s) for s in seeds))
    b_au, b_f1 = boot[:, :len(MARKERS)], boot[:, len(MARKERS):]

    rows = []
    print(f"\n== TEST (80%, {tag}) ==")
    for i, m in enumerate(MARKERS):
        print(f"  {m:8} AUPRC={ap_pt[i]:.3f} [{np.percentile(b_au[:,i],2.5):.3f},"
              f"{np.percentile(b_au[:,i],97.5):.3f}]  F1={f1_pt[i]:.3f}")
        rows.append({"marker": m, "auprc_point": ap_pt[i], "auprc_boot_mean": b_au[:, i].mean(),
                     "auprc_ci_lo": np.percentile(b_au[:, i], 2.5), "auprc_ci_hi": np.percentile(b_au[:, i], 97.5),
                     "f1_point": f1_pt[i], "f1_boot_mean": b_f1[:, i].mean(),
                     "f1_ci_lo": np.percentile(b_f1[:, i], 2.5), "f1_ci_hi": np.percentile(b_f1[:, i], 97.5)})
    m_au, m_f1 = b_au.mean(axis=1), b_f1.mean(axis=1)
    print(f"  {'MEAN':8} AUPRC={ap_pt.mean():.3f} [{np.percentile(m_au,2.5):.3f},"
          f"{np.percentile(m_au,97.5):.3f}]  F1={f1_pt.mean():.3f}")
    rows.append({"marker": "MEAN", "auprc_point": ap_pt.mean(), "auprc_boot_mean": m_au.mean(),
                 "auprc_ci_lo": np.percentile(m_au, 2.5), "auprc_ci_hi": np.percentile(m_au, 97.5),
                 "f1_point": f1_pt.mean(), "f1_boot_mean": m_f1.mean(),
                 "f1_ci_lo": np.percentile(m_f1, 2.5), "f1_ci_hi": np.percentile(m_f1, 97.5)})

    out = Path(args.feat_dir) / f"eval_cell_auprc_{tag}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()