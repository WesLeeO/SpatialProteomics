"""
HEMIT token-grid patch dataset builder  (v4 — clean native non-overlap geometry)

Geometry (mirrors cell_cls/pathocell_cls — one coordinate space, no downscale chain)
-----------------------------------------------------------------------------------
Source tiles : 1024×1024 px at 40x (0.25 µm/px). H&E + IF label + nuclei mask all
               live on THIS native grid (same filename), so nothing is resampled
               to align them.
Patch FOV    : the model wants 224 px @ 20x (0.5 µm/px) = 112 µm. At native 0.25
               µm/px that is PATCH_LEVEL0 = round(224·0.5/0.25) = 448 native px.
Tiling       : NON-overlapping grid over the 1024 tile, top-lefts {0, 448, 896}
               → 3×3 = 9 patches / tile. The last row/col only has 128 real px;
               that crop is PADDED to 448 (IF→0) and resized to 224, so the model
               always sees a full, undistorted 112 µm FOV and no cell is dropped
               (vs the old v3 512-downscale + 3×3 stride-144 OVERLAP, which forced a
               per-cell "most-centered framing" dedup — all gone here).
Token target : crop 448 native → pad to 448 → resize to 224 → clip(x/p99_fg,0,1)
               → block-mean into 16×16 (224/16 = 14 px/token).

Edge handling
-------------
A patch is cropped at native res, then padded bottom/right to PATCH_LEVEL0 before
the resize. Because the pad is at the FAR edge, a real pixel keeps its offset
(py-Y, px-X) and maps to token  floor(offset · G / PATCH_LEVEL0)  — i.e. divide by
the FULL patch size, never the clamped size. The cell-feature builder uses the same
rule, so tokens stay pixel-aligned for edge patches too.

The nuclei masks (HEMIT_nuclei_analysis, native 40x 1024²) are NOT touched here;
this builder only emits IF token targets + native patch coords. Cell↔token mapping
happens in the cell-feature builder (never resample labels).

HDF5 layout (one file per split)
---------------------------------
  /coords   (N, 2)       int16   — (x, y) top-left in NATIVE 1024×1024 space
  /targets  (N, C, G, G) float32 — normalised mean expression per token cell
  /p99s     (C,)         float32 — global foreground p99 per channel
  /sources  (N,)         bytes   — input TIF path for each patch
  attrs: marker_names, patch_size_level0, model_input, token_grid, native_mpp,
         target_mpp, target_mag, source_size, n_patches, normalisation
"""

import argparse
import cv2
import h5py
import numpy as np
import tifffile
from pathlib import Path


HEMIT_DATA_DIR    = Path("/mnt/ssd/virtual_proteomics/data/HEMIT")
HEMIT_NATIVE_MPP  = 0.25             # 40x scan
HEMIT_SOURCE_SIZE = 1024
HEMIT_MARKERS     = ["Pan-CK", "CD3", "Dapi"]
OUTPUT_DIR        = Path('datasets/hemit_patch_dataset')

TARGET_MPP   = 0.5                   # ORION/UNI2 training resolution (20x)
MODEL_INPUT  = 224                   # UNI2 model input
TOKEN_GRID   = 16                    # UNI2 token grid side
TOKEN_PX     = MODEL_INPUT // TOKEN_GRID                              # = 14 px per token
# native crop covering the same 112 µm FOV as 224 px @ 0.5 µm/px
PATCH_LEVEL0 = round(MODEL_INPUT * TARGET_MPP / HEMIT_NATIVE_MPP)     # = 448

# native top-left coords (x, y): non-overlapping grid over the 1024 tile
_STARTS     = list(range(0, HEMIT_SOURCE_SIZE, PATCH_LEVEL0))         # [0, 448, 896]
TILE_COORDS = [(x, y) for y in _STARTS for x in _STARTS]             # 9 patches


# ── p99 computation ───────────────────────────────────────────────────────────

def compute_global_p99s(splits: list[str]) -> np.ndarray:
    """Global foreground (>0) p99 per channel across all requested splits (native res)."""
    C = len(HEMIT_MARKERS)
    hists = [np.zeros(256, dtype=np.float64) for _ in range(C)]

    for split in splits:
        label_dir = HEMIT_DATA_DIR / split / "label"
        for lbl_path in sorted(label_dir.glob("*.tif")):
            label = tifffile.imread(str(lbl_path))   # (1024, 1024, C) uint8
            for c in range(C):
                fg = label[:, :, c].ravel()
                fg = fg[fg > 0]
                if len(fg):
                    h, _ = np.histogram(fg, bins=256, range=(0, 256))
                    hists[c] += h

    p99s = np.ones(C, dtype=np.float32)
    for c in range(C):
        total = hists[c].sum()
        if total > 0:
            cdf = np.cumsum(hists[c]) / total
            p99s[c] = float(max(np.searchsorted(cdf, 0.99, side='right'), 1))
        print(f"  {HEMIT_MARKERS[c]:<10} p99={p99s[c]:.0f}")
    return p99s


# ── target extraction ─────────────────────────────────────────────────────────

def crop_pad_resize(img: np.ndarray, x: int, y: int, pad_value: float) -> np.ndarray:
    """
    Crop PATCH_LEVEL0×PATCH_LEVEL0 native at (x, y), pad bottom/right to full size
    (so edge patches keep scale + aspect), then resize to MODEL_INPUT. INTER_AREA
    for the downscale. img: (H, W, C) -> (MODEL_INPUT, MODEL_INPUT, C).
    """
    H, W = img.shape[:2]
    crop = img[y:min(y + PATCH_LEVEL0, H), x:min(x + PATCH_LEVEL0, W)]
    ch, cw = crop.shape[:2]
    if (ch, cw) != (PATCH_LEVEL0, PATCH_LEVEL0):
        crop = cv2.copyMakeBorder(crop, 0, PATCH_LEVEL0 - ch, 0, PATCH_LEVEL0 - cw,
                                  cv2.BORDER_CONSTANT, value=pad_value)
    return cv2.resize(crop, (MODEL_INPUT, MODEL_INPUT), interpolation=cv2.INTER_AREA)


def extract_token_targets(
    label: np.ndarray,      # (1024, 1024, C) float32 — NATIVE 40x
    x: int, y: int,
    p99s: np.ndarray,       # (C,)
    token_grid: int = TOKEN_GRID,
) -> np.ndarray:
    """
    Native crop → pad → resize-to-224 → clip(x/p99_fg,0,1) → block-mean into
    (token_grid, token_grid). Padded edge region is 0 (no IF), so edge tokens read
    as background. Returns (C, token_grid, token_grid) float32.
    """
    token_px = MODEL_INPUT // token_grid
    crop   = crop_pad_resize(label, x, y, pad_value=0.0)             # (224, 224, C)
    normed = np.clip(crop / p99s[None, None, :], 0.0, 1.0)
    return (
        normed
        .reshape(token_grid, token_px, token_grid, token_px, label.shape[2])
        .mean(axis=(1, 3))
        .transpose(2, 0, 1)
    ).astype(np.float32)                                            # (C, G, G)


# ── per-split pipeline ────────────────────────────────────────────────────────

def process_split(
    split: str,
    p99s: np.ndarray,
    output_dir: Path,
    token_grid: int = TOKEN_GRID,
) -> int:
    input_dir = HEMIT_DATA_DIR / split / "input"
    label_dir = HEMIT_DATA_DIR / split / "label"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.tif"))
    if not input_files:
        print(f"  [{split}] No TIFs found in {input_dir}, skipping.")
        return 0

    n_per = len(TILE_COORDS)
    print(f"\n[{split}]  {len(input_files)} source tiles  →  {n_per * len(input_files)} patches")

    all_coords, all_targets, all_sources = [], [], []

    for inp_path in input_files:
        lbl_path = label_dir / inp_path.name
        if not lbl_path.exists():
            print(f"  Warning: no label for {inp_path.name}, skipping.")
            continue

        label = tifffile.imread(str(lbl_path)).astype(np.float32)  # (1024, 1024, C) NATIVE
        for x, y in TILE_COORDS:
            all_coords.append((x, y))                              # NATIVE top-left
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
        f.create_dataset("p99s",    data=p99s)
        f.create_dataset("sources", data=np.array(all_sources, dtype=f"S{max_len}"),
                         compression="gzip")
        f.attrs["marker_names"]      = HEMIT_MARKERS
        f.attrs["patch_size_level0"] = PATCH_LEVEL0
        f.attrs["model_input"]       = MODEL_INPUT
        f.attrs["token_grid"]        = token_grid
        f.attrs["native_mpp"]        = HEMIT_NATIVE_MPP
        f.attrs["target_mpp"]        = TARGET_MPP
        f.attrs["target_mag"]        = round(10.0 / TARGET_MPP, 1)
        f.attrs["source_size"]       = HEMIT_SOURCE_SIZE
        f.attrs["n_patches"]         = total
        f.attrs["normalisation"]     = "clip(x / p99_fg, 0, 1)"

    mb = out_path.stat().st_size / 1e6
    print(f"  /targets  {targets_arr.shape}  mean={targets_arr.mean():.4f}")
    print(f"  Saved → {out_path}  ({mb:.1f} MB)")
    return total


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HEMIT token-grid dataset builder (v4, native non-overlap, padded edges)"
    )
    parser.add_argument("--splits",     default="train,val,test",
                        help="Comma-separated splits (default: train,val,test)")
    parser.add_argument("--token_grid", type=int, default=TOKEN_GRID,
                        help=f"Token grid side (default: {TOKEN_GRID}, UNI2-aligned)")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",")]

    print("=" * 60)
    print("  HEMIT token-grid dataset builder (v4 — clean native geometry)")
    print(f"  source size   : {HEMIT_SOURCE_SIZE}×{HEMIT_SOURCE_SIZE} px @ 40x")
    print(f"  patch level0  : {PATCH_LEVEL0} native px → resize {MODEL_INPUT}  (112 µm FOV @ 20x)")
    print(f"  tiling        : non-overlap top-lefts {_STARTS} → {len(TILE_COORDS)} patches/tile")
    print(f"  edge patches  : padded to {PATCH_LEVEL0} (IF→0), token map ÷ {PATCH_LEVEL0}")
    print(f"  token grid    : {args.token_grid}×{args.token_grid}  "
          f"({MODEL_INPUT // args.token_grid}×{MODEL_INPUT // args.token_grid} px per token)")
    print(f"  target mpp    : {TARGET_MPP:.3f} µm/px  (≈ {round(10.0/TARGET_MPP,1)}x)")
    print(f"  markers       : {HEMIT_MARKERS}")
    print("=" * 60)

    print("\nComputing global foreground p99s across all splits…")
    p99s = compute_global_p99s(splits)

    total = 0
    for split in splits:
        total += process_split(split, p99s, OUTPUT_DIR, token_grid=args.token_grid)

    print(f"\nDone. Total patches: {total:,}")


if __name__ == "__main__":
    main()
