"""Filter model token-predictions onto the artifact-CLEAN patch subset.

The crcv2 dataset is the artifact-filtered (lineage+shape) subset of the benchmark
patches AND carries cleaned targets. Each crcv2 h5 stores `benchmark_index`, the row
indices into the FULL benchmark h5 it kept (verified: benchmark.coords[benchmark_index]
== crcv2.coords).

We only need to subset each model's cached PREDICTIONS by those indices. Because
`preds_full[benchmark_index][i]` is the prediction for crcv2 patch i, the filtered
preds end up in crcv2 ROW ORDER — already aligned with crcv2's own `targets[i]`. So at
benchmark time the GT comes straight from orion_crcv2_patch_dataset (the cleaned
targets) and no runtime re-indexing is needed.

Output:
  benchmarking/preds_cache_clean/<key>/<slide>_preds.npy   (filtered, crcv2 order)

Then run the benchmark scoring against the cleaned crcv2 targets:
  python benchmark_all_slides.py --slides CRC19,CRC30,CRC11,CRC02 \
      --models miphei-vit,miphei-convnext,hemit,pix2pix \
      --h5_dir     orion_crcv2_patch_dataset \
      --pred_cache preds_cache_clean \
      --results_dir results_clean \
      --your_preds preds_cache_clean/ours --your_label "ours (clean)"
"""
import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np

REPO        = Path(__file__).resolve().parent.parent.parent
BENCH_DIR   = REPO / "datasets/orion_crc_patch_dataset_benchmark"
CRCV2_DIR   = REPO / "datasets/orion_crcv2_patch_dataset"
PRED_CACHE  = REPO / "benchmarking" / "preds_cache"
CLEAN_CACHE = REPO / "benchmarking" / "preds_cache_clean"
OURS_DIR    = REPO / "training_outputs/outputs_orion_token_UNI2_baseline_lora8x16mlp_2loss_lbg8_fg0"  # our model preds source


def clean_index(slide: str):
    """benchmark_index (rows into the full benchmark h5) + full benchmark patch count."""
    with h5py.File(CRCV2_DIR / f"{slide}_patch_dataset.h5", "r") as f:
        bi = f["benchmark_index"][:]
        cv = f["coords"][:]
    with h5py.File(BENCH_DIR / f"{slide}_patch_dataset.h5", "r") as f:
        bc = f["coords"][:]
    assert np.array_equal(bc[bi], cv), f"{slide}: benchmark_index mismatch"
    return bi, len(bc)


def subset_preds(slide: str, bi: np.ndarray, n_full: int,
                 src_dir: Path, dst_dir: Path):
    p = src_dir / f"{slide}_preds.npy"
    if not p.exists():
        print(f"  [skip] {src_dir.name}: no {slide}_preds.npy")
        return
    arr = np.load(p)
    if len(arr) != n_full:
        print(f"  [WARN] {src_dir.name} {slide}: preds N={len(arr)} != benchmark "
              f"N={n_full} — STALE, skipping (re-cache it first)")
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    np.save(dst_dir / f"{slide}_preds.npy", arr[bi])   # -> crcv2 row order
    t = src_dir / f"{slide}_time.txt"                   # carry timing sidecar if present
    if t.exists():
        shutil.copy(t, dst_dir / f"{slide}_time.txt")
    print(f"  preds -> {dst_dir.parent.name}/{dst_dir.name}/{slide}  "
          f"({len(arr)}->{len(bi)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", default="CRC19,CRC30,CRC11,CRC02")
    ap.add_argument("--ours_dir", default=str(OURS_DIR),
                    help="dir with our model's {SLIDE}_preds.npy to filter")
    args = ap.parse_args()
    slides = [s.strip() for s in args.slides.split(",") if s.strip()]
    ours_dir = Path(args.ours_dir)

    model_dirs = sorted(d for d in PRED_CACHE.iterdir() if d.is_dir())
    print(f"models in preds_cache: {[d.name for d in model_dirs]}  + ours({ours_dir.name})")

    for slide in slides:
        print(f"\n=== {slide} ===")
        bi, n_full = clean_index(slide)
        for d in model_dirs:
            subset_preds(slide, bi, n_full, d, CLEAN_CACHE / d.name)
        # our model: clean subdir keeps the SOURCE folder name (not "ours")
        subset_preds(slide, bi, n_full, ours_dir, CLEAN_CACHE / ours_dir.name)

    print(f"\nDone. clean preds -> {CLEAN_CACHE}")
    print("GT at benchmark time comes from orion_crcv2_patch_dataset (cleaned targets).")


if __name__ == "__main__":
    main()