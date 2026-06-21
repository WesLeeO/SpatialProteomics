"""
Visualize pancancer patches with GT cell contours colored by cell type.

Layout: rows = patches, cols = H&E | contours | cell-type fill
All panels are 224×224 with an optional 16×16 token grid overlay.

Usage
-----
  python visualize_pancancer_patches.py --tma CRC_TMA_A --core reg001_X01_Y01
  python visualize_pancancer_patches.py --tma CRC_TMA_A --core reg001_X01_Y01 --patches 0,5,10,15
  python visualize_pancancer_patches.py --tma CRC_TMA_A --core reg001_X01_Y01 --hdf_suffix B --n_patches 8
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import h5py
import tifffile
from pathlib import Path

TRIDENT_DIR       = Path("datasets/pancancer_trident_output")
DATASET_DIR       = Path("datasets/pancancer_patch_dataset")
PATHOCELL_HDF_DIR = Path("/mnt/ssd1/virtual_proteomics/data/pathocell/pathocell_hdf")
OUT_DIR           = Path("visualization_out/pancancer")

PATCH_SIZE  = 224
TOKEN_GRID  = 16
TOKEN_PX    = PATCH_SIZE // TOKEN_GRID   # 14 px per token

COARSE_NAMES = [
    'Background', 'B cells', 'Macrophages', 'Adipocytes',
    'Dendritic cells', 'T cells', 'Granulocytes', 'NK cells', 'Nerves',
    'Plasma cells', 'Smooth muscle', 'Stroma', 'Tumor cells',
    'Vasculature', 'Other',
]

CELL_COLORS = np.array([
    [0.0,  0.0,  0.0 ],   # 0  background
    [0.12, 0.47, 0.71],   # 1  B cells
    [1.00, 0.50, 0.05],   # 2  Macrophages
    [0.17, 0.63, 0.17],   # 3  Adipocytes
    [0.84, 0.15, 0.16],   # 4  Dendritic cells
    [0.58, 0.40, 0.74],   # 5  T cells
    [0.55, 0.34, 0.29],   # 6  Granulocytes
    [0.89, 0.47, 0.76],   # 7  NK cells
    [0.50, 0.50, 0.50],   # 8  Nerves
    [0.74, 0.74, 0.13],   # 9  Plasma cells
    [0.09, 0.75, 0.81],   # 10 Smooth muscle
    [0.65, 0.85, 0.33],   # 11 Stroma
    [1.00, 0.00, 0.00],   # 12 Tumor cells
    [0.00, 0.45, 0.70],   # 13 Vasculature
    [0.80, 0.80, 0.80],   # 14 Other
], dtype=np.float32)


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_he_tiff(tma: str, core: str) -> np.ndarray:
    """Load exported uint8 RGB TIFF → (H, W, 3)."""
    path = TRIDENT_DIR / tma / core / f"{core}.tif"
    img  = tifffile.imread(str(path))
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    if img.ndim == 3 and img.shape[0] == 3:
        return img.transpose(1, 2, 0)
    raise ValueError(f"Unexpected H&E shape {img.shape}")


def load_patch_coords(tma: str, core: str) -> tuple[np.ndarray, int]:
    h5 = DATASET_DIR / tma / f"{core}_patch_dataset.h5"
    with h5py.File(h5, "r") as f:
        coords = f["coords"][:]
        psz    = int(f.attrs["patch_size_level0"])
    return coords, psz


def load_gt_seg(core: str, suffix: str = "A") -> tuple[np.ndarray, np.ndarray]:
    """reg001_X01_Y01 → reg001_{suffix}.hdf → (gt_inst, gt_ct_coarse) (H, W)."""
    reg_id = core.split("_X")[0]
    path   = PATHOCELL_HDF_DIR / f"{reg_id}_{suffix}.hdf"
    with h5py.File(path, "r") as f:
        return f["gt_inst"][0].astype(np.int32), f["gt_ct_coarse"][0].astype(np.int32)


# ── Patch extraction ──────────────────────────────────────────────────────────

def crop_and_resize(arr: np.ndarray, x: int, y: int, psz: int,
                    interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    """Crop (y:y+psz, x:x+psz) from arr, resize to PATCH_SIZE×PATCH_SIZE."""
    H = arr.shape[0] if arr.ndim == 2 else arr.shape[0]
    W = arr.shape[1]
    y0, y1 = int(y), min(int(y) + psz, H)
    x0, x1 = int(x), min(int(x) + psz, W)
    crop = arr[y0:y1, x0:x1]
    if crop.shape[0] < psz or crop.shape[1] < psz:
        shape = (psz, psz) if arr.ndim == 2 else (psz, psz, arr.shape[2])
        pad   = np.zeros(shape, dtype=arr.dtype)
        pad[:crop.shape[0], :crop.shape[1]] = crop
        crop  = pad
    return cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE), interpolation=interp)


# ── Token grid overlay ────────────────────────────────────────────────────────

def add_token_grid(img_float: np.ndarray,
                   color: tuple = (1.0, 1.0, 0.0),
                   alpha: float = 0.5) -> np.ndarray:
    """Draw 16×16 token grid lines on a (224, 224, 3) float image."""
    out = img_float.copy()
    for i in range(1, TOKEN_GRID):
        px = i * TOKEN_PX
        # horizontal line
        out[px - 1:px + 1, :] = (
            out[px - 1:px + 1, :] * (1 - alpha) +
            np.array(color, dtype=np.float32) * alpha
        )
        # vertical line
        out[:, px - 1:px + 1] = (
            out[:, px - 1:px + 1] * (1 - alpha) +
            np.array(color, dtype=np.float32) * alpha
        )
    return np.clip(out, 0, 1)


# ── Drawing ───────────────────────────────────────────────────────────────────

def draw_contours(he: np.ndarray, inst: np.ndarray, ct: np.ndarray,
                  thickness: int = 1) -> np.ndarray:
    """Returns (224, 224, 3) float32."""
    out = he.astype(np.float32) / 255.0
    for cell_id in np.unique(inst):
        if cell_id == 0:
            continue
        mask   = (inst == cell_id).astype(np.uint8)
        ct_val = int(np.bincount(ct[inst == cell_id]).argmax())
        color  = CELL_COLORS[min(ct_val, len(CELL_COLORS) - 1)]
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color.tolist(), thickness=thickness)
    return np.clip(out, 0, 1)


def draw_fill(he: np.ndarray, inst: np.ndarray, ct: np.ndarray,
              alpha: float = 0.5) -> np.ndarray:
    """Returns (224, 224, 3) float32."""
    fill = np.ones((*inst.shape, 3), dtype=np.float32)
    for cell_id in np.unique(inst):
        if cell_id == 0:
            continue
        mask   = inst == cell_id
        ct_val = int(np.bincount(ct[mask]).argmax())
        fill[mask] = CELL_COLORS[min(ct_val, len(CELL_COLORS) - 1)]
    base = he.astype(np.float32) / 255.0
    return np.clip(base * (1 - alpha) + fill * alpha, 0, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tma",        default="CRC_TMA_A")
    parser.add_argument("--core",       default="reg001_X01_Y01")
    parser.add_argument("--patches",    default=None,
                        help="Comma-separated patch indices, e.g. 0,5,10")
    parser.add_argument("--n_patches",  type=int, default=8)
    parser.add_argument("--hdf_suffix", default="A", choices=["A", "B"])
    parser.add_argument("--no_grid",    action="store_true",
                        help="Disable 16×16 token grid overlay")
    parser.add_argument("--out",        default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    out_path = args.out or str(OUT_DIR / f"{args.tma}_{args.core}_patches.png")

    print(f"Loading {args.tma}/{args.core}…")
    he_full           = load_he_tiff(args.tma, args.core)
    coords, psz       = load_patch_coords(args.tma, args.core)
    inst_full, ct_full = load_gt_seg(args.core, args.hdf_suffix)

    H_he, W_he   = he_full.shape[:2]
    H_seg, W_seg = inst_full.shape
    print(f"  H&E {H_he}×{W_he}  seg {H_seg}×{W_seg}  psz={psz}  {len(coords)} patches")
    if H_he != H_seg or W_he != W_seg:
        print(f"  WARNING: size mismatch — try --hdf_suffix {'B' if args.hdf_suffix=='A' else 'A'}")

    if args.patches:
        idxs = [int(p) for p in args.patches.split(",")]
    else:
        step = max(1, len(coords) // args.n_patches)
        idxs = list(range(0, len(coords), step))[:args.n_patches]

    n_rows = len(idxs)
    n_cols = 3   # H&E | contours | fill
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.5, n_rows * 2.5))
    if n_rows == 1:
        axes = axes[None, :]

    fig.suptitle(
        f"{args.tma} / {args.core}  (psz={psz}→224, "
        f"token grid 16×16={TOKEN_PX}px/token, GT: {args.core.split('_X')[0]}_{args.hdf_suffix})",
        fontsize=9,
    )

    all_cts = set()

    for row, pidx in enumerate(idxs):
        x, y = int(coords[pidx, 0]), int(coords[pidx, 1])

        he_p   = crop_and_resize(he_full,   x, y, psz, cv2.INTER_LINEAR)
        inst_p = crop_and_resize(inst_full, x, y, psz, cv2.INTER_NEAREST)
        ct_p   = crop_and_resize(ct_full,   x, y, psz, cv2.INTER_NEAREST)

        n_cells   = len(np.unique(inst_p)) - 1
        cts_here  = sorted(c for c in np.unique(ct_p) if c > 0)
        all_cts.update(cts_here)

        he_f       = he_p.astype(np.float32) / 255.0
        contour_f  = draw_contours(he_p, inst_p, ct_p)
        fill_f     = draw_fill(he_p, inst_p, ct_p)

        if not args.no_grid:
            he_f      = add_token_grid(he_f)
            contour_f = add_token_grid(contour_f)
            fill_f    = add_token_grid(fill_f)

        for col, (img, title) in enumerate([
            (he_f,      f"#{pidx}  ({x},{y})"),
            (contour_f, f"{n_cells} cells"),
            (fill_f,    " ".join(COARSE_NAMES[c][:5] for c in cts_here[:4])),
        ]):
            axes[row, col].imshow(np.clip(img, 0, 1), interpolation="nearest")
            axes[row, col].set_title(title, fontsize=7)
            axes[row, col].axis("off")

    axes[0, 0].set_title("H&E", fontsize=8)
    axes[0, 1].set_title("Contours (GT)", fontsize=8)
    axes[0, 2].set_title("Cell-type fill", fontsize=8)

    handles = [mpatches.Patch(color=CELL_COLORS[c], label=COARSE_NAMES[c])
               for c in sorted(all_cts)]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 7), fontsize=7,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()