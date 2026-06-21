"""
Visualize a PathoCell patch with cell contours colored by GT cell type.

Shows:
  - H&E patch
  - H&E + cell contours (colored by coarse cell type)
  - Cell-type color map (instance fill)
  - Token grid overlay (optional)

Usage
-----
  python visualize_pathocell_patch.py --file reg001_A
  python visualize_pathocell_patch.py --file reg001_A --x 400 --y 200 --size 600
  python visualize_pathocell_patch.py --file reg001_A --show_tokens
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import h5py
from pathlib import Path
from scipy import ndimage as ndi

HDF_DIR    = Path("/mnt/ssd1/virtual_proteomics/data/pathocell/pathocell_hdf")
NATIVE_MPP = 0.377
TARGET_MPP = 0.5
TOKEN_GRID = 16

COARSE_NAMES = [
    'Background', 'B cells', 'Macrophages', 'Adipocytes',
    'Dendritic cells', 'T cells', 'Granulocytes', 'NK cells', 'Nerves',
    'Plasma cells', 'Smooth muscle', 'Stroma', 'Tumor cells',
    'Vasculature', 'Other',
]

# Distinct colors per cell type (skip index 0 = background)
CELL_COLORS = np.array([
    [0.0,  0.0,  0.0 ],   # 0  background — not drawn
    [0.12, 0.47, 0.71],   # 1  B cells — blue
    [1.00, 0.50, 0.05],   # 2  Macrophages — orange
    [0.17, 0.63, 0.17],   # 3  Adipocytes — green
    [0.84, 0.15, 0.16],   # 4  Dendritic cells — red
    [0.58, 0.40, 0.74],   # 5  T cells — purple
    [0.55, 0.34, 0.29],   # 6  Granulocytes — brown
    [0.89, 0.47, 0.76],   # 7  NK cells — pink
    [0.50, 0.50, 0.50],   # 8  Nerves — gray
    [0.74, 0.74, 0.13],   # 9  Plasma cells — yellow-green
    [0.09, 0.75, 0.81],   # 10 Smooth muscle — cyan
    [0.65, 0.85, 0.33],   # 11 Stroma — light green
    [1.00, 0.00, 0.00],   # 12 Tumor cells — bright red
    [0.00, 0.45, 0.70],   # 13 Vasculature — deep blue
    [0.80, 0.80, 0.80],   # 14 Other — light gray
], dtype=np.float32)


def he_to_uint8(img: np.ndarray) -> np.ndarray:
    """(3, H, W) uint16 → (H, W, 3) uint8."""
    out = np.zeros((3, img.shape[1], img.shape[2]), dtype=np.float32)
    for c in range(3):
        p99 = float(np.percentile(img[c], 99))
        if p99 > 0:
            out[c] = np.clip(img[c] / p99, 0, 1) * 255
    return out.transpose(1, 2, 0).astype(np.uint8)


def crop(arr, y0, x0, size):
    """Crop arr (..., H, W) or (H, W) to (y0:y0+size, x0:x0+size)."""
    if arr.ndim == 2:
        return arr[y0:y0 + size, x0:x0 + size]
    return arr[..., y0:y0 + size, x0:x0 + size]


def draw_contours(he_rgb: np.ndarray, inst: np.ndarray,
                  ct: np.ndarray, thickness: int = 1) -> np.ndarray:
    """Draw per-cell contours on H&E, colored by coarse cell type."""
    out = he_rgb.copy().astype(np.float32) / 255.0
    for cell_id in np.unique(inst):
        if cell_id == 0:
            continue
        mask     = (inst == cell_id).astype(np.uint8)
        ct_val   = int(np.bincount(ct[inst == cell_id]).argmax())
        color    = CELL_COLORS[min(ct_val, len(CELL_COLORS) - 1)]
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1,
                         color=(float(color[0]), float(color[1]), float(color[2])),
                         thickness=thickness)
    return np.clip(out, 0, 1)


def draw_cell_fill(inst: np.ndarray, ct: np.ndarray,
                   alpha: float = 0.6) -> np.ndarray:
    """Filled cell-type map: (H, W, 3) float in [0,1]."""
    H, W   = inst.shape
    canvas = np.ones((H, W, 3), dtype=np.float32)   # white background
    for cell_id in np.unique(inst):
        if cell_id == 0:
            continue
        mask   = inst == cell_id
        ct_val = int(np.bincount(ct[mask]).argmax())
        color  = CELL_COLORS[min(ct_val, len(CELL_COLORS) - 1)]
        canvas[mask] = canvas[mask] * (1 - alpha) + color * alpha
    return canvas


def draw_token_grid(img: np.ndarray, psz: int, color=(1.0, 1.0, 0.0),
                    lw: int = 1) -> np.ndarray:
    """Overlay TOKEN_GRID×TOKEN_GRID grid lines on img (H, W, 3) float."""
    out      = img.copy()
    tok_px   = psz / TOKEN_GRID
    H, W     = img.shape[:2]
    for i in range(1, TOKEN_GRID):
        y = round(i * tok_px)
        x = round(i * tok_px)
        if y < H:
            out[max(0, y - lw):y + lw, :] = color
        if x < W:
            out[:, max(0, x - lw):x + lw] = color
    return out


def legend_patches(present_cts):
    return [mpatches.Patch(color=CELL_COLORS[c], label=COARSE_NAMES[c])
            for c in present_cts if c > 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",        default="reg001_A",
                        help="HDF stem (without .hdf)")
    parser.add_argument("--x",           type=int, default=None,
                        help="Crop left edge in native pixels (default: centre)")
    parser.add_argument("--y",           type=int, default=None,
                        help="Crop top edge in native pixels (default: centre)")
    parser.add_argument("--size",        type=int, default=595,
                        help="Crop size in native pixels (default 595 ≈ 2 patches)")
    parser.add_argument("--show_tokens", action="store_true",
                        help="Overlay UNI2 token grid (one patch = 297px, 16 tokens)")
    parser.add_argument("--out",         default="pathocell_patch_vis.png")
    args = parser.parse_args()

    hdf_path = HDF_DIR / f"{args.file}.hdf"
    with h5py.File(hdf_path, "r") as f:
        img          = f["img"][:]             # (3, H, W) uint16
        gt_inst      = f["gt_inst"][0].astype(np.int32)      # (H, W)
        gt_ct_coarse = f["gt_ct_coarse"][0].astype(np.int32) # (H, W)

    H_full, W_full = gt_inst.shape
    he_full = he_to_uint8(img)   # (H, W, 3)

    # Default crop: centre of the tile
    size = args.size
    x0   = args.x if args.x is not None else (W_full - size) // 2
    y0   = args.y if args.y is not None else (H_full - size) // 2
    x0   = max(0, min(x0, W_full - size))
    y0   = max(0, min(y0, H_full - size))

    he_crop   = he_full[y0:y0 + size, x0:x0 + size]
    inst_crop = gt_inst[y0:y0 + size, x0:x0 + size]
    ct_crop   = gt_ct_coarse[y0:y0 + size, x0:x0 + size]

    n_cells = len(np.unique(inst_crop)) - 1   # exclude background
    present_cts = sorted(c for c in np.unique(ct_crop) if c > 0)
    print(f"Crop ({x0},{y0}) size={size}px  |  {n_cells} cells  |  "
          f"cell types: {[COARSE_NAMES[c] for c in present_cts]}")

    # ── Panels ────────────────────────────────────────────────────────────────
    he_float    = he_crop.astype(np.float32) / 255.0
    contour_img = draw_contours(he_crop, inst_crop, ct_crop, thickness=1)
    fill_img    = draw_cell_fill(inst_crop, ct_crop, alpha=0.55)
    fill_he_img = he_float * 0.5 + fill_img * 0.5   # blend H&E + fill

    if args.show_tokens:
        psz_native = round(224 * TARGET_MPP / NATIVE_MPP)   # 297 px
        contour_img  = draw_token_grid(contour_img,  psz_native)
        fill_he_img  = draw_token_grid(fill_he_img,  psz_native)

    patches = legend_patches(present_cts)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(
        f"{args.file}  —  crop ({x0}, {y0}), {size}×{size} px  "
        f"({n_cells} cells)",
        fontsize=10,
    )

    axes[0].imshow(he_float);            axes[0].set_title("H&E");                        axes[0].axis("off")
    axes[1].imshow(contour_img);         axes[1].set_title("H&E + cell contours");        axes[1].axis("off")
    axes[2].imshow(np.clip(fill_he_img, 0, 1)); axes[2].set_title("H&E + cell-type fill"); axes[2].axis("off")

    fig.legend(handles=patches, loc="lower center", ncol=min(len(patches), 7),
               fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()