"""
WSI-level cell-prediction visualiser: GT-positive vs OUR predicted-positive cells
for one marker on one slide, plus their spatial agreement (TP / FP / FN).

Loads pre-computed per-cell predictions from the CSV saved by eval_cell_auprc.py
(cell_predictions_<tag>.csv), so no retraining is needed here.

Run from repo root, e.g.:
  python cell_cls/orion_cell_cls/visualize_cell_predictions_wsi.py --slide CRC11 --marker CD8a
  python cell_cls/orion_cell_cls/visualize_cell_predictions_wsi.py --slide CRC02 --marker FOXP3 --tag ours
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, f1_score

from eval_cell_auprc import MARKERS, SLIDE_KEY, FEAT, norm


def best_f1_threshold(y, p):
    grid = np.quantile(p, np.linspace(0.5, 0.999, 200))
    f1s = [f1_score(y, p >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))]), float(np.max(f1s))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slide", default="CRC11", choices=list(SLIDE_KEY))
    ap.add_argument("--marker", default="CD8a")
    ap.add_argument("--tag", default="ours",
                    help="tag used when running eval_cell_auprc.py (determines which CSV to load)")
    ap.add_argument("--pred_csv", default=None,
                    help="explicit path to cell_predictions CSV (overrides --tag)")
    ap.add_argument("--feat_dir", default=str(FEAT))
    ap.add_argument("--threshold", type=float, default=None,
                    help="fixed prob threshold (default: F1-optimal on this slide)")
    ap.add_argument("--match_prevalence", action="store_true",
                    help="predict top-K cells positive, K = #GT positives")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    mk = next(m for m in MARKERS if norm(m) == norm(args.marker))

    csv_path = Path(args.pred_csv) if args.pred_csv else Path(args.feat_dir) / f"cell_predictions_{args.tag}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found — run eval_cell_auprc.py --tag {args.tag} first"
        )

    df = pd.read_csv(csv_path)
    sel = df[df["slide_name"].str.contains(SLIDE_KEY[args.slide])].copy()
    if sel.empty:
        raise ValueError(f"no cells found for slide {args.slide} in {csv_path}")

    prob = sel[f"{mk}_prob"].values
    y    = sel[f"{mk}_pos"].values.astype(int)
    x    = sel["x"].values
    yy   = sel["y"].values

    auprc = average_precision_score(y, prob)
    if args.match_prevalence:
        k = int(y.sum()); thr = np.partition(prob, -k)[-k] if k else 1.0; how = f"prev-matched K={k}"
    elif args.threshold is not None:
        thr = args.threshold; how = f"thr={thr:.3f}"
    else:
        thr, f1 = best_f1_threshold(y, prob); how = f"F1-opt thr={thr:.3f} (F1={f1:.3f})"
    pred = prob >= thr

    tp = y.astype(bool) & pred
    fp = (~y.astype(bool)) & pred
    fn = y.astype(bool) & (~pred)
    prec = tp.sum() / max(pred.sum(), 1); rec = tp.sum() / max(y.sum(), 1)
    f1v = 2 * prec * rec / max(prec + rec, 1e-9)

    fig, axes = plt.subplots(1, 3, figsize=(27, 9))
    for ax in axes:
        ax.scatter(x, yy, s=0.4, c="0.85", linewidths=0, rasterized=True)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([])
    axes[0].scatter(x[y == 1], yy[y == 1], s=0.8, c="#d62728", linewidths=0, rasterized=True)
    axes[0].set_title(f"GT {mk}+  (n={int(y.sum())}, {y.mean():.1%})")
    axes[1].scatter(x[pred], yy[pred], s=0.8, c="#1f77b4", linewidths=0, rasterized=True)
    axes[1].set_title(f"Pred {mk}+  (n={int(pred.sum())})  [{how}]")
    axes[2].scatter(x[tp], yy[tp], s=0.8, c="#2ca02c", linewidths=0, rasterized=True, label=f"TP {int(tp.sum())}")
    axes[2].scatter(x[fp], yy[fp], s=0.8, c="#d62728", linewidths=0, rasterized=True, label=f"FP {int(fp.sum())}")
    axes[2].scatter(x[fn], yy[fn], s=0.8, c="#1f77b4", linewidths=0, rasterized=True, label=f"FN {int(fn.sum())}")
    axes[2].set_title(f"agreement  P={prec:.2f} R={rec:.2f} F1={f1v:.2f}")
    axes[2].legend(markerscale=8, loc="upper right", framealpha=0.9)
    fig.suptitle(f"{args.slide}  {mk}  |  tag={args.tag}  |  AUPRC={auprc:.3f}  |  "
                 f"{len(sel)} cells", fontsize=15)
    fig.tight_layout()
    out = args.out or str(Path(args.feat_dir) / f"wsi_cells_{args.slide}_{mk}_{args.tag}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"{args.slide} {mk}: AUPRC={auprc:.3f}  {how}  P={prec:.2f} R={rec:.2f} F1={f1v:.2f}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()