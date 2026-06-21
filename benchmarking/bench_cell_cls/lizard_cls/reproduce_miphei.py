"""
Reproduce MIPHEI's released Lizard cell-type numbers with OUR harness.

For every MIPHEI checkpoint shipping a lizard cell parquet + logreg CSV, run OUR
exact logreg on THEIR predictions and check we recover THEIR published per-class
AUPRC / F1 (checkpoints/<model>/lizard_logreg.csv). If we reproduce their numbers,
the harness is unbiased — so a higher score for our model is a real edge.

Example:
  python cell_cls/lizard_cls/reproduce_miphei.py
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
    df = pd.read_parquet(pq_path)
    pred_cols = [c for c in df.columns if c.endswith("_pred")]
    tr = df[df["split"] == "train"]; te = df[df["split"] == "test"]
    scaler = StandardScaler().fit(tr[pred_cols].values)
    clf = OneVsRestClassifier(LogisticRegression(class_weight="balanced", random_state=42))
    clf.fit(scaler.transform(tr[pred_cols].values), tr[NUCLEI_CLASSES].values.astype(int))
    proba = clf.predict_proba(scaler.transform(te[pred_cols].values))
    y = te[NUCLEI_CLASSES].values.astype(int)
    rows = [(ct, average_precision_score(y[:, k], proba[:, k]),
             f1_score(y[:, k], proba[:, k] > 0.5, zero_division=0))
            for k, ct in enumerate(NUCLEI_CLASSES)]
    return pd.DataFrame(rows, columns=["CellType", "AUPRC", "F1"]).set_index("CellType")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--ours_tag", default="bg0.2")
    args = ap.parse_args()

    models = sorted(d.name for d in CKPT_DIR.iterdir()
                    if (d / "lizard_cell_dataframe_logreg.parquet").exists()
                    and (d / "lizard_logreg.csv").exists())
    print(f"reproducing {len(models)} MIPHEI models | tol={args.tol}\n")

    summ = []
    for model in models:
        mine = eval_parquet(CKPT_DIR / model / "lizard_cell_dataframe_logreg.parquet")
        theirs = pd.read_csv(CKPT_DIR / model / "lizard_logreg.csv").set_index("Marker")
        common = [c for c in NUCLEI_CLASSES if c in theirs.index]
        d_au = (mine.loc[common, "AUPRC"] - theirs.loc[common, "AUPRC"]).abs()
        d_f1 = (mine.loc[common, "F1"] - theirs.loc[common, "F1 Score"]).abs()
        ok = (d_au.max() < args.tol) and (d_f1.max() < args.tol)
        summ.append({"model": model, "their_mean_AUPRC": theirs.loc[common, "AUPRC"].mean(),
                     "our_mean_AUPRC": mine.loc[common, "AUPRC"].mean(),
                     "max|dAUPRC|": d_au.max(), "max|dF1|": d_f1.max(),
                     "reproduced": "✓" if ok else "✗"})
        print(f"  {model:16}  their={theirs.loc[common,'AUPRC'].mean():.4f}  "
              f"ours={mine.loc[common,'AUPRC'].mean():.4f}  max|dAUPRC|={d_au.max():.4f}  "
              f"{'✓ reproduced' if ok else '✗ MISMATCH'}")

    summ = pd.DataFrame(summ)
    print("\n== reproduction summary ==")
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    all_ok = (summ["reproduced"] == "✓").all()
    print(f"\n{'ALL REPRODUCED — harness unbiased' if all_ok else 'SOME MISMATCHES'}.")

    ours_csv = FEAT_DIR / f"eval_cell_auprc_{args.ours_tag}.csv"
    if ours_csv.exists():
        row = pd.read_csv(ours_csv).query("marker.str.startswith('MEAN')", engine="python").iloc[0]
        best = summ["their_mean_AUPRC"].max(); best_m = summ.loc[summ["their_mean_AUPRC"].idxmax(), "model"]
        print(f"\nOURS ({args.ours_tag}) mean AUPRC = {row['auprc_boot_mean']:.4f} "
              f"[{row['auprc_ci_lo']:.4f}, {row['auprc_ci_hi']:.4f}]  vs  "
              f"best MIPHEI ({best_m}) = {best:.4f}")


if __name__ == "__main__":
    main()
