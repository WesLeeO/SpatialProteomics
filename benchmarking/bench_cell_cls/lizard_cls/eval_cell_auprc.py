"""
Cell-level AUPRC for Lizard cell-type classification under MIPHEI's EXACT protocol
(the Lizard analog of cell_cls/pathocell_cls/eval_cell_auprc.py).

A logreg maps per-cell predicted marker expression to the 5 Lizard nuclei classes,
on MIPHEI's fixed 20%/80% split. Comparable to checkpoints/<model>/lizard_logreg.csv.

Held identical to MIPHEI regardless of source (only <marker>_pred differs):
  * GT one-hot + split -> from --gt_parquet (model-independent; their stratified
    train_test_split(test_size=0.8, random_state=42), baked into `split`).
  * head               -> StandardScaler + OneVsRestClassifier(LogisticRegression(
    class_weight="balanced", random_state=42)), fit on TRAIN cells.
  * metric             -> average_precision_score per class, mean over 5 (MEAN5).
  * bootstrap          -> resample TEST images (slide_name) with replacement, seed 42.

Sources (--source):
  ours    : our cell_token_features_<tag>.parquet (mean_<m> -> <m>_pred).
  parquet : a MIPHEI lizard parquet with <marker>_pred (--pred_parquet) — reproduces
            their lizard_logreg.csv as a harness check.

Examples:
  python cell_cls/lizard_cls/eval_cell_auprc.py --tag bg0.2
  python cell_cls/lizard_cls/eval_cell_auprc.py --source parquet \
      --pred_parquet checkpoints/MIPHEI-vit/lizard_cell_dataframe_logreg.parquet
"""
import argparse, re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score
from sklearn.utils import resample
from joblib import Parallel, delayed

from utils import FEAT_DIR, DEFAULT_GT_PARQUET, NUCLEI_CLASSES

NCLS = len(NUCLEI_CLASSES)   # 5 -> MEAN5 headline


def load_gt(gt_parquet: str) -> pd.DataFrame:
    gt = pd.read_parquet(gt_parquet, columns=["cell_id", "slide_name", "split"] + NUCLEI_CLASSES)
    gt["slide_name"] = gt["slide_name"].astype(str)
    gt["cell_id"] = gt["cell_id"].astype(np.int64)
    gt[NUCLEI_CLASSES] = gt[NUCLEI_CLASSES].astype(np.int8)
    return gt


def load_preds(source: str, features: str, pred_parquet: str):
    if source == "ours":
        u = pd.read_parquet(features)
        if "cell_id" not in u.columns:
            u = u.rename(columns={"label": "cell_id"})
        ren = {c: f"{c[len('mean_'):]}_pred" for c in u.columns if c.startswith("mean_")}
        d = u.rename(columns=ren)
        pred_cols = list(ren.values())
        d = d[["cell_id", "slide_name"] + pred_cols]
    else:
        d = pd.read_parquet(pred_parquet)
        pred_cols = [c for c in d.columns if c.endswith("_pred")]
        d = d[["cell_id", "slide_name"] + pred_cols]
    d["slide_name"] = d["slide_name"].astype(str)
    d["cell_id"] = d["cell_id"].astype(np.int64)
    return d, pred_cols


def score_block(y: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = np.full((NCLS, 2), np.nan)
    for k in range(NCLS):
        yk = y[:, k]
        if yk.sum() == 0 or yk.sum() == len(yk):
            continue
        out[k, 0] = average_precision_score(yk, proba[:, k])
        out[k, 1] = f1_score(yk, pred[:, k], zero_division=0)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["ours", "parquet"], default="ours")
    ap.add_argument("--features", default=None)
    ap.add_argument("--pred_parquet", default=None)
    ap.add_argument("--gt_parquet", default=str(DEFAULT_GT_PARQUET))
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.source == "parquet" and not args.pred_parquet:
        ap.error("--source parquet requires --pred_parquet")
    tag = args.tag or (Path(args.pred_parquet).parent.name if args.source == "parquet" else "ours")
    features = args.features or str(FEAT_DIR / f"cell_token_features_{tag}.parquet")

    print(f"== source={args.source}  tag={tag} ==")
    gt = load_gt(args.gt_parquet)
    preds, pred_cols = load_preds(args.source, features, args.pred_parquet)
    print(f"  {len(pred_cols)} pred features: {pred_cols}")
    df = gt.merge(preds, on=["cell_id", "slide_name"], how="inner").dropna(subset=pred_cols)
    print(f"  their cells={len(gt)}  scored={len(df)}  dropped={len(gt)-len(df)} "
          f"({(len(gt)-len(df))/len(gt):.2%})")

    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"].reset_index(drop=True)
    print(f"  train={len(tr)}  test={len(te)}  ({te['slide_name'].nunique()} test images)")

    scaler = StandardScaler().fit(tr[pred_cols].values)
    clf = OneVsRestClassifier(LogisticRegression(class_weight="balanced", random_state=42))
    clf.fit(scaler.transform(tr[pred_cols].values), tr[NUCLEI_CLASSES].values)
    proba = clf.predict_proba(scaler.transform(te[pred_cols].values))
    binr  = (proba > 0.5).astype(int)
    y     = te[NUCLEI_CLASSES].values.astype(int)

    point = score_block(y, proba, binr)
    ap_point, f1_point = point[:, 0], point[:, 1]
    print("\n== per class AUPRC / F1 (point) ==")
    for ct, a, f in zip(NUCLEI_CLASSES, ap_point, f1_point):
        print(f"  {ct:30}  AUPRC={a:.3f}  F1={f:.3f}")
    print(f"  {'MEAN'+str(NCLS):30}  AUPRC={np.nanmean(ap_point):.3f}  F1={np.nanmean(f1_point):.3f}")

    # bootstrap over test images
    grp = te["slide_name"].values
    uniq = np.unique(grp)
    groups = {g: np.where(grp == g)[0] for g in uniq}
    np.random.seed(42)
    seeds = np.random.randint(0, 10000, size=args.n_boot)

    def one(seed):
        samp = resample(uniq, replace=True, n_samples=len(uniq), random_state=seed)
        idx = np.concatenate([groups[g] for g in samp])
        return score_block(y[idx], proba[idx], binr[idx])

    print(f"\n== bootstrap {args.n_boot}x over {len(uniq)} test images (seed 42) ==")
    raw = np.array(Parallel(n_jobs=-1, verbose=1)(delayed(one)(s) for s in seeds))  # (B,NCLS,2)
    boot_auprc, boot_f1 = raw[:, :, 0], raw[:, :, 1]
    mean_auprc = np.nanmean(boot_auprc, axis=1)
    mean_f1    = np.nanmean(boot_f1, axis=1)
    lo_a, hi_a = np.nanpercentile(mean_auprc, [2.5, 97.5])
    lo_f, hi_f = np.nanpercentile(mean_f1, [2.5, 97.5])
    print(f"\n== HEADLINE MEAN{NCLS} ({tag}) ==")
    print(f"  AUPRC point={np.nanmean(ap_point):.3f} boot={np.nanmean(mean_auprc):.3f} "
          f"95%CI=[{lo_a:.3f},{hi_a:.3f}]")
    print(f"  F1    point={np.nanmean(f1_point):.3f} boot={np.nanmean(mean_f1):.3f} "
          f"95%CI=[{lo_f:.3f},{hi_f:.3f}]")

    out_df = pd.DataFrame({
        "marker": NUCLEI_CLASSES,
        "auprc_point": ap_point, "auprc_boot_mean": np.nanmean(boot_auprc, axis=0),
        "auprc_ci_lo": np.nanpercentile(boot_auprc, 2.5, axis=0),
        "auprc_ci_hi": np.nanpercentile(boot_auprc, 97.5, axis=0),
        "f1_point": f1_point, "f1_boot_mean": np.nanmean(boot_f1, axis=0),
        "f1_ci_lo": np.nanpercentile(boot_f1, 2.5, axis=0),
        "f1_ci_hi": np.nanpercentile(boot_f1, 97.5, axis=0),
    })
    headline = pd.DataFrame([{
        "marker": f"MEAN{NCLS}",
        "auprc_point": np.nanmean(ap_point), "auprc_boot_mean": np.nanmean(mean_auprc),
        "auprc_ci_lo": lo_a, "auprc_ci_hi": hi_a,
        "f1_point": np.nanmean(f1_point), "f1_boot_mean": np.nanmean(mean_f1),
        "f1_ci_lo": lo_f, "f1_ci_hi": hi_f,
    }])
    out_df = pd.concat([out_df, headline], ignore_index=True)
    out_path = FEAT_DIR / f"eval_cell_auprc_{tag}.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
