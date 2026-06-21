"""
Visualize Singular Genomics patches: H&E alongside IF token targets.

Usage:
    python visualize_sg.py --disease lung_cancer
    python visualize_sg.py --disease breast_cancer --n_patches 12 --markers CD3 CD8 PanCK
    python visualize_sg.py --disease kidney_cancer --show_tokens
"""

import argparse
import numpy as np
import cv2
import h5py
import tifffile, zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SG_DIR  = Path("singular_genomics")
HE_ROOT = Path("/mnt/ssd1/virtual_proteomics/data/singular_genomics")
OUT_DIR = Path("visualization_out/sg/patches")
MODEL_SIZE = 224


def open_zarr(tif_path: str):
    tif   = tifffile.TiffFile(tif_path)
    store = tif.aszarr()
    z     = zarr.open(store, mode="r")
    return z["0"] if isinstance(z, zarr.hierarchy.Group) else z


def load_he_patch(zarr_arr, x: int, y: int, psz: int) -> np.ndarray:
    patch = np.array(zarr_arr[y:y+psz, x:x+psz])
    if patch.ndim == 2:
        patch = np.stack([patch]*3, axis=-1)
    elif patch.shape[0] in (1, 3):
        patch = patch.transpose(1, 2, 0)
    return cv2.resize(patch, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)


def visualize(args):
    h5_path = SG_DIR / f"{args.disease}_patch_dataset.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"{h5_path} not found")

    with h5py.File(h5_path) as f:
        coords       = f["coords"][:]
        targets      = f["targets"][:]
        marker_names = list(f.attrs["marker_names"])
        valid_mask   = f["valid_markers"][:]
        psz          = int(f.attrs["patch_size_level0"])
        token_grid   = int(f.attrs.get("token_grid", 16))

    # resolve H&E OME-TIFF
    he_matches = list((HE_ROOT / args.disease).rglob("*_HE.ome.tiff"))
    if not he_matches:
        raise FileNotFoundError(f"No *_HE.ome.tiff under {HE_ROOT / args.disease}")
    zarr_arr = open_zarr(str(he_matches[0]))
    print(f"H&E: {he_matches[0].name}  WSI shape: {zarr_arr.shape}")
    print(f"Dataset: {args.disease}  {len(coords):,} patches  psz={psz}px")

    # resolve markers to display
    if args.markers:
        sel = [(i, m) for i, m in enumerate(marker_names)
               if m in args.markers and valid_mask[i]]
        missing = [m for m in args.markers if m not in marker_names]
        if missing:
            print(f"  [warn] not in panel: {missing}")
    else:
        sel = [(i, m) for i, m in enumerate(marker_names) if valid_mask[i]]

    if not sel:
        raise ValueError("No valid markers to display")

    n_cols = 1 + len(sel) * (2 if args.show_tokens else 1)
    rng    = np.random.default_rng(args.seed)
    n      = min(args.n_patches, len(coords))
    pick   = np.sort(rng.choice(len(coords), n, replace=False))

    fig, axes = plt.subplots(n, n_cols,
                             figsize=(n_cols * 2.5, n * 2.5),
                             squeeze=False)

    axes[0, 0].set_title("H&E", fontsize=9, fontweight="bold")
    col = 1
    for _, name in sel:
        axes[0, col].set_title(name, fontsize=9, fontweight="bold")
        col += 1
        if args.show_tokens:
            axes[0, col].set_title(f"{name}\n(tokens)", fontsize=9, fontweight="bold")
            col += 1

    for row, idx in enumerate(pick):
        x, y = int(coords[idx, 0]), int(coords[idx, 1])

        he_patch = load_he_patch(zarr_arr, x, y, psz)
        axes[row, 0].imshow(he_patch)
        axes[row, 0].set_ylabel(f"#{idx}", fontsize=7)

        col = 1
        for mi, name in sel:
            token_img = targets[idx, mi]   # (token_grid, token_grid)
            axes[row, col].imshow(
                np.clip(token_img, 0, 1), cmap="gray", vmin=0, vmax=1,
                interpolation="nearest" if args.show_tokens else "bilinear",
            )
            col += 1
            if args.show_tokens:
                axes[row, col].imshow(
                    np.clip(token_img, 0, 1), cmap="inferno", vmin=0, vmax=1,
                    interpolation="nearest",
                )
                col += 1

    for ax in axes.ravel():
        ax.axis("off")

    marker_tag = "-".join(m for _, m in sel[:6])
    suffix = "_tokens" if args.show_tokens else ""
    plt.suptitle(f"{args.disease}  |  Singular Genomics", fontsize=11, y=1.002)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.disease}_{marker_tag}{suffix}_n{n}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize SG patches (H&E + IF tokens)")
    parser.add_argument("--disease",    required=True,
                        help="e.g. lung_cancer, breast_cancer, kidney_cancer")
    parser.add_argument("--n_patches",  type=int, default=12)
    parser.add_argument("--markers",    nargs="*", default=None)
    parser.add_argument("--show_tokens", action="store_true",
                        help="Show token grid alongside the upsampled view")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()
    visualize(args)


if __name__ == "__main__":
    main()