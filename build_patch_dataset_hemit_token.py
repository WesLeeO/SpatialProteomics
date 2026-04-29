"""
HEMIT token-grid patch dataset builder  (v2 — replaces flat-mean targets)

Geometry
--------
Source tiles: 1024×1024 px at 40x (0.25 µm/px).
Resize step: 1024×1024 → 896×896 (full FOV, slight downscale factor 896/1024).
Target resolution: 20x → crop_size = 448 px in 896-space → 224 px model input.
Tiling: 2×2 non-overlapping 448×448 crops from 896×896 → 4 patches/source.
Each crop: resize 448→224 for the H&E model input, and
           log1p/p99-normalise → resize → 16×16 block mean for the label target.

HDF5 layout (one file per split)
---------------------------------
  /coords   (N, 2)       int16   — (x, y) top-left in resized 896×896 space
  /targets  (N, C, G, G) float32 — normalised mean expression per token cell
  /p99s     (C,)         float32 — global foreground p99 per label channel
  /sources  (N,)         bytes   — input TIF path for each patch
  attrs: marker_names, crop_size, patch_size, token_grid,
         native_mpp, target_mpp, target_mag, n_patches, normalisation,
         resize_to (source resized to this before cropping)
"""

import argparse
import cv2
import h5py
import numpy as np
import tifffile
from pathlib import Path


HEMIT_DATA_DIR   = Path("/mnt/ssd1/virtual_proteomics/data/HEMIT")
HEMIT_NATIVE_MPP = 0.25             # 40x scan
HEMIT_SOURCE_SIZE = 1024
HEMIT_MARKERS    = ["Pan-CK", "CD3", "Dapi"]
OUTPUT_DIR = Path('hemit_patch_dataset')

CROP_SIZE    = 448                  # px in 896-space that map to 224px model input
RESIZE_TO    = 2 * CROP_SIZE        # = 896: resize 1024→896 before cropping
TOKEN_GRID   = 16                   # UNI2 token grid side
TOKEN_PX     = 224 // TOKEN_GRID    # = 14 px per token after resize

# Effective MPP after resize: native * (1024 / 896); crop 448→224 doubles it again
_SCALE       = HEMIT_SOURCE_SIZE / RESIZE_TO          # 1024/896 ≈ 1.143
TARGET_MPP   = HEMIT_NATIVE_MPP * _SCALE * (CROP_SIZE / 224)  # ≈ 0.571 µm/px

# 4 non-overlapping top-left coords (x, y) in resized 896×896 space
TILE_COORDS = [
    (0,         0),
    (CROP_SIZE, 0),
    (0,         CROP_SIZE),
    (CROP_SIZE, CROP_SIZE),
]


# ── p99 ───────────────────────────────────────────────────────────────────────

def compute_global_p99s(label_dir: Path, input_files: list) -> np.ndarray:
    """Compute per-channel p99 from foreground pixels across all label TIFs."""
    C = len(HEMIT_MARKERS)
    accum = [[] for _ in range(C)]
    for inp_path in input_files:
        lbl_path = label_dir / inp_path.name
        if not lbl_path.exists():
            continue
        label = tifffile.imread(str(lbl_path)).astype(np.float32)  # (H, W, C)
        label = cv2.resize(label, (RESIZE_TO, RESIZE_TO), interpolation=cv2.INTER_LINEAR)
        for c in range(C):
            ch = label[:, :, c].ravel()
            fg = ch[ch > 0]
            if len(fg):
                accum[c].append(fg)

    p99s = np.ones(C, dtype=np.float32)
    for c in range(C):
        fg = np.concatenate(accum[c]) if accum[c] else np.array([1.0])
        p99s[c] = float(np.percentile(fg, 99)) if len(fg) else 1.0
    return p99s


# ── target extraction ─────────────────────────────────────────────────────────

def extract_token_targets(
    label: np.ndarray,      # (896, 896, C) float32 — already resized
    x: int, y: int,
    p99s: np.ndarray,       # (C,)
    token_grid: int = TOKEN_GRID,
) -> np.ndarray:
    """
    Crop 448×448 from 896×896 label, log1p/p99-normalise, resize to 224×224,
    block-mean into (token_grid, token_grid) cells.

    Returns (C, token_grid, token_grid) float32.
    """
    token_px = 224 // token_grid
    C = label.shape[2]

    crop = label[y:y + CROP_SIZE, x:x + CROP_SIZE, :]          # (448, 448, C)
    normed = np.clip(
        np.log1p(crop / p99s[None, None, :]), 0.0, 1.0
    )                                                            # (448, 448, C)
    resized = cv2.resize(normed, (224, 224), interpolation=cv2.INTER_LINEAR)
                                                                 # (224, 224, C)
    return (
        resized
        .reshape(token_grid, token_px, token_grid, token_px, C)
        .mean(axis=(1, 3))
        .transpose(2, 0, 1)
    ).astype(np.float32)                                         # (C, G, G)


# ── per-split pipeline ────────────────────────────────────────────────────────

def process_split(
    split: str,
    output_dir: Path,
    token_grid: int = TOKEN_GRID,
    per_slide_p99: bool = False,
) -> int:
    input_dir = HEMIT_DATA_DIR / split / "input"
    label_dir = HEMIT_DATA_DIR / split / "label"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.tif"))
    if not input_files:
        print(f"  [{split}] No TIFs found in {input_dir}, skipping.")
        return 0

    print(f"\n[{split}]  {len(input_files)} source tiles  "
          f"→  {4 * len(input_files)} patches  "
          f"({'per-slide' if per_slide_p99 else 'global'} p99)")

    if per_slide_p99:
        global_p99s = None
    else:
        print(f"  Computing global p99s across {len(input_files)} label images…")
        global_p99s = compute_global_p99s(label_dir, input_files)
        for name, p99 in zip(HEMIT_MARKERS, global_p99s):
            print(f"    {name:<10}  p99={p99:.2f}")

    all_coords  = []
    all_targets = []
    all_sources = []
    split_p99s  = global_p99s  # may be updated in per-slide mode (first slide)

    for inp_path in input_files:
        lbl_path = label_dir / inp_path.name
        if not lbl_path.exists():
            print(f"  Warning: no label for {inp_path.name}, skipping.")
            continue

        label = tifffile.imread(str(lbl_path)).astype(np.float32)  # (1024, 1024, C)
        label = cv2.resize(label, (RESIZE_TO, RESIZE_TO), interpolation=cv2.INTER_LINEAR)

        if per_slide_p99:
            C = label.shape[2]
            p99s = np.ones(C, dtype=np.float32)
            for c in range(C):
                ch = label[:, :, c].ravel()
                fg = ch[ch > 0]
                p99s[c] = float(np.percentile(fg, 99)) if len(fg) else 1.0
            if split_p99s is None:
                split_p99s = p99s
        else:
            p99s = global_p99s

        for x, y in TILE_COORDS:
            all_coords.append((x, y))
            all_targets.append(extract_token_targets(label, x, y, p99s, token_grid))
            all_sources.append(str(inp_path))

    total    = len(all_coords)
    out_path = output_dir / f"{split}.h5"

    targets_arr = np.stack(all_targets, axis=0)   # (N, C, G, G)
    N, C, G, _  = targets_arr.shape
    max_len      = max(len(s) for s in all_sources)

    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("coords",  data=np.array(all_coords, dtype=np.int16), compression="gzip")
        f.create_dataset("targets", data=targets_arr, compression="gzip",
                         chunks=(min(256, N), C, G, G))
        f.create_dataset("p99s",    data=split_p99s)
        f.create_dataset("sources", data=np.array(all_sources, dtype=f"S{max_len}"),
                         compression="gzip")
        f.attrs["marker_names"]  = HEMIT_MARKERS
        f.attrs["crop_size"]     = CROP_SIZE
        f.attrs["patch_size"]    = 224
        f.attrs["token_grid"]    = token_grid
        f.attrs["native_mpp"]    = HEMIT_NATIVE_MPP
        f.attrs["resize_to"]     = RESIZE_TO
        f.attrs["target_mpp"]    = TARGET_MPP
        f.attrs["target_mag"]    = round(10.0 / TARGET_MPP, 1)
        f.attrs["n_patches"]     = total
        f.attrs["p99_mode"]      = "per-slide" if per_slide_p99 else "global"
        f.attrs["normalisation"] = "log1p(x/p99) -> clip[0,1]"

    mb = out_path.stat().st_size / 1e6
    print(f"  /targets  {targets_arr.shape}  mean={targets_arr.mean():.4f}")
    print(f"  Saved → {out_path}  ({mb:.1f} MB)")
    return total


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HEMIT token-grid dataset builder (20x, 4 patches per source tile)"
    )
    parser.add_argument("--splits",        default="train,val,test",
                        help="Comma-separated splits (default: train,val,test)")

    parser.add_argument("--per_slide_p99", action="store_true",
                        help="Compute p99 per slide instead of globally")
    parser.add_argument("--token_grid",    type=int, default=TOKEN_GRID,
                        help=f"Token grid side (default: {TOKEN_GRID}, UNI2-aligned)")
    args = parser.parse_args()

    splits     = [s.strip() for s in args.splits.split(",")]

    print("=" * 60)
    print("  HEMIT token-grid dataset builder")
    print(f"  source size   : {HEMIT_SOURCE_SIZE}×{HEMIT_SOURCE_SIZE} px @ 40x")
    print(f"  resize to     : {RESIZE_TO}×{RESIZE_TO} px  (full FOV, scale {_SCALE:.3f})")
    print(f"  crop_size     : {CROP_SIZE} px (in {RESIZE_TO}-space)  →  224 px model input")
    print(f"  patches/tile  : 4  (2×2 non-overlapping grid)")
    print(f"  token grid    : {args.token_grid}×{args.token_grid}  "
          f"({224 // args.token_grid}×{224 // args.token_grid} px per token)")
    print(f"  markers       : {HEMIT_MARKERS}")
    print("=" * 60)

    total = 0
    for split in splits:
        total += process_split(split, OUTPUT_DIR,
                               token_grid=args.token_grid,
                               per_slide_p99=args.per_slide_p99)

    print(f"\nDone. Total patches: {total:,}")


if __name__ == "__main__":
    main()