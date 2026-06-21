"""Build token-agg cell features + predictions for EVERY model with cached token preds.

This automates, for each model, the exact MIPHEI-vit-tokenagg flow:
  1. build_cell_token_features.py  (cached <slide>_preds.npy -> per-cell mean_<marker>)
  2. eval_cell_auprc.py --source ours  (logreg + 1000x tile bootstrap -> AUPRC/F1 + CIs)

so every model is scored through the SAME token-resolution + nucleus-area aggregation and
the SAME downstream protocol (matched/fair comparison — see the fairness finding where
running MIPHEI through this pipeline raised it 0.517 -> 0.554).

Source preds: benchmarking/preds_cache/<key>/<slide>_preds.npy (full-N, benchmark coords),
the token preds the benchmark already cached for each baseline.

Outputs per model <key>:
  cell_cls/orion_cell_cls/fair_<key>_tokenagg/cell_token_features_<slide>.parquet
  cell_cls/orion_cell_cls/fair_<key>_tokenagg/cell_predictions_<tag>.csv
  cell_cls/orion_cell_cls/eval_cell_auprc_<tag>.csv         (where tag = NICE[key]+"-tokenagg")

Idempotent: a slide's parquet / a model's eval csv that already exists is skipped unless
--refresh (so re-running won't redo MIPHEI-vit). GPU-free; mask I/O dominates (~3 min/slide).

  python cell_cls/orion_cell_cls/build_all_model_cell_features.py            # all models
  python cell_cls/orion_cell_cls/build_all_model_cell_features.py --models hemit pix2pix
  python cell_cls/orion_cell_cls/build_all_model_cell_features.py --dry_run
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent             # cell_cls/orion_cell_cls
REPO = HERE.parents[2]                              # repo root
PRED_CACHE = REPO / "benchmarking" / "preds_cache"
BUILD = HERE / "build_cell_token_features.py"
EVAL  = HERE / "eval_cell_auprc.py"
SLIDES = ["CRC19", "CRC30", "CRC11", "CRC02"]       # VAL=19/30 (logreg train), TEST=11/02
# preds_cache key -> display name (tag = name + "-tokenagg"); unknown keys use the key itself.
NICE = {"miphei-vit": "MIPHEI-vit", "miphei-convnext": "MIPHEI-convnext",
        "hemit": "HEMIT", "pix2pix": "Pix2Pix"}


def models_with_preds(slides):
    """preds_cache subdirs that have a <slide>_preds.npy for every requested slide."""
    return [d.name for d in sorted(PRED_CACHE.iterdir())
            if d.is_dir() and all((d / f"{s}_preds.npy").exists() for s in slides)]


def run(cmd, dry):
    print("  →", " ".join(str(c) for c in cmd), flush=True)
    if not dry:
        subprocess.run([str(c) for c in cmd], check=True, cwd=REPO)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None,
                    help="preds_cache keys (default: all with full slide coverage)")
    ap.add_argument("--slides", nargs="*", default=SLIDES)
    ap.add_argument("--refresh", action="store_true",
                    help="rebuild even if the parquet / eval csv already exists")
    ap.add_argument("--dry_run", action="store_true", help="print commands, run nothing")
    args = ap.parse_args()

    keys = args.models or models_with_preds(args.slides)
    print(f"models: {keys}\nslides: {args.slides}\n")

    for key in keys:
        pred_dir = PRED_CACHE / key
        if not all((pred_dir / f"{s}_preds.npy").exists() for s in args.slides):
            print(f"[skip] {key}: missing some slide preds in {pred_dir}\n")
            continue
        tag     = f"{NICE.get(key, key)}-tokenagg"
        featdir = HERE / f"fair_{key}_tokenagg"
        print(f"=== {key}  (tag={tag}) ===")

        for s in args.slides:
            out = featdir / f"cell_token_features_{s}.parquet"
            if out.exists() and not args.refresh:
                print(f"  [have] {out.relative_to(REPO)}")
                continue
            run([sys.executable, BUILD, "--sample", s,
                 "--pred_dir", pred_dir, "--out", out], args.dry_run)

        evalcsv = HERE / f"eval_cell_auprc_{tag}.csv"
        if evalcsv.exists() and not args.refresh:
            print(f"  [have] {evalcsv.relative_to(REPO)}  (eval done)\n")
            continue
        run([sys.executable, EVAL, "--source", "ours",
             "--feat_dir", featdir, "--tag", tag], args.dry_run)
        print()

    print("done.")


if __name__ == "__main__":
    main()