"""
Reproduce MIPHEI's released PathoCell cell-type numbers with OUR harness.

For every MIPHEI checkpoint that ships both a cell parquet and a logreg CSV, we
run OUR exact logreg protocol on THEIR predictions and check we recover THEIR
published per-type AUPRC / F1 (checkpoints/<model>/pathocell_logreg.csv). If we
reproduce their numbers to the digit, the harness is unbiased — so when our model
scores higher under the same harness, the edge is real, not a scoring artifact.

For each model:
  load <model>/pathocell_cell_dataframe_logreg.parquet  (their <marker>_pred,
       the 15 celltype one-hots, and the fixed train/test split),
  fit StandardScaler + OneVsRest LogisticRegression(class_weight="balanced",
       random_state=42) on TRAIN, score TEST,
  compare our per-type AUPRC/F1 to <model>/pathocell_logreg.csv.

Example:
  python cell_cls/pathocell_cls/reproduce_miphei.py
  python cell_cls/pathocell_cls/reproduce_miphei.py --tol 0.01 --ours_tag bg0.2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score

from utils import REPO_ROOT, FEAT_DIR, NUCLEI_CLASSES

CKPT_DIR = REPO_ROOT / "checkpoints"


def eval_parquet(pq_path: Path) -> pd.DataFrame:
    """Run our logreg on a MIPHEI cell parquet -> per-type AUPRC/F1 (point estimate)."""
    df = pd.read_parquet(pq_path)
    pred_cols = [c for c in df.columns if c.endswith("_pred")]
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    scaler = StandardScaler().fit(tr[pred_cols].values)
    clf = OneVsRestClassifier(LogisticRegression(class_weight="balanced", random_state=42))
    clf.fit(scaler.transform(tr[pred_cols].values), tr[NUCLEI_CLASSES].values.astype(int))
    proba = clf.predict_proba(scaler.transform(te[pred_cols].values))
    y = te[NUCLEI_CLASSES].values.astype(int)
    rows = []
    for k, ct in enumerate(NUCLEI_CLASSES):
        rows.append((ct,
                     average_precision_score(y[:, k], proba[:, k]),
                     f1_score(y[:, k], proba[:, k] > 0.5, zero_division=0)))
    return pd.DataFrame(rows, columns=["CellType", "AUPRC", "F1"]).set_index("CellType")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol", type=float, default=0.01,
                    help="max |ours - theirs| per type to count as reproduced")
    ap.add_argument("--ours_tag", default="bg0.2",
                    help="show our MEAN AUPRC alongside (from eval_cell_auprc_<tag>.csv)")
    args = ap.parse_args()

    models = sorted(d.name for d in CKPT_DIR.iterdir()
                    if (d / "pathocell_cell_dataframe_logreg.parquet").exists()
                    and (d / "pathocell_logreg.csv").exists())
    print(f"reproducing {len(models)} MIPHEI models | tol={args.tol}\n")

    summary = []
    for model in models:
        mine   = eval_parquet(CKPT_DIR / model / "pathocell_cell_dataframe_logreg.parquet")
        theirs = pd.read_csv(CKPT_DIR / model / "pathocell_logreg.csv").set_index("Marker")
        common = [c for c in NUCLEI_CLASSES if c in theirs.index]
        d_auprc = (mine.loc[common, "AUPRC"] - theirs.loc[common, "AUPRC"]).abs()
        d_f1    = (mine.loc[common, "F1"]    - theirs.loc[common, "F1 Score"]).abs()
        ok = (d_auprc.max() < args.tol) and (d_f1.max() < args.tol)
        summary.append({
            "model": model,
            "their_mean_AUPRC": theirs.loc[common, "AUPRC"].mean(),
            "our_mean_AUPRC":   mine.loc[common, "AUPRC"].mean(),
            "max|dAUPRC|":      d_auprc.max(),
            "max|dF1|":         d_f1.max(),
            "reproduced":       "✓" if ok else "✗",
        })
        flag = "✓ reproduced" if ok else "✗ MISMATCH"
        print(f"  {model:16}  their={theirs.loc[common,'AUPRC'].mean():.4f}  "
              f"ours={mine.loc[common,'AUPRC'].mean():.4f}  "
              f"max|dAUPRC|={d_auprc.max():.4f}  max|dF1|={d_f1.max():.4f}  {flag}")

    summ = pd.DataFrame(summary)
    pd.set_option("display.width", 200)
    print("\n== reproduction summary ==")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    all_ok = (summ["reproduced"] == "✓").all()
    print(f"\n{'ALL MODELS REPRODUCED' if all_ok else 'SOME MISMATCHES — investigate'} "
          f"(harness {'is unbiased' if all_ok else 'needs a look'}).")

    # our model alongside, if its eval CSV exists
    ours_csv = FEAT_DIR / f"eval_cell_auprc_{args.ours_tag}.csv"
    if ours_csv.exists():
        row = pd.read_csv(ours_csv).query("marker.str.startswith('MEAN')", engine="python").iloc[0]
        best = summ["their_mean_AUPRC"].max()
        best_m = summ.loc[summ["their_mean_AUPRC"].idxmax(), "model"]
        print(f"\nOURS ({args.ours_tag}) mean AUPRC = {row['auprc_boot_mean']:.4f} "
              f"[{row['auprc_ci_lo']:.4f}, {row['auprc_ci_hi']:.4f}]  vs  "
              f"best MIPHEI ({best_m}) = {best:.4f}")
    else:
        print(f"\n(run eval_cell_auprc.py --tag {args.ours_tag} to show ours alongside)")


if __name__ == "__main__":
    main()