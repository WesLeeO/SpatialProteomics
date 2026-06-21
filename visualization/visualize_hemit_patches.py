"""
Visualize HEMIT patches: H&E alongside IF token targets.

Each source TIF is a 1024×1024 H&E crop that gets resized to resize_to before
cropping at (x, y, crop_size). Targets are already normalised token grids.

Usage:
    python visualize_hemit.py --split train
    python visualize_hemit.py --split val --n_patches 16
    python visualize_hemit.py --split train --show_tokens --markers Pan-CK CD3
"""

import argparse
import numpy as np
import cv2
import h5py
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

H5_DIR    = Path("datasets/hemit_patch_dataset")
OUT_DIR   = Path("visualization_out/hemit/patches")
MODEL_SIZE = 224


def load_he_patch(src_path: str, x: int, y: int,
                  crop_size: int, resize_to: int) -> np.ndarray:
    img = tifffile.imread(src_path)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.shape[0] in (1, 3):
        img = img.transpose(1, 2, 0)
        if img.shape[2] == 1:
            img = np.concatenate([img]*3, axis=2)
    if resize_to > 0 and img.shape[0] != resize_to:
        img = cv2.resize(img, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
    patch = img[y:y+crop_size, x:x+crop_size]
    return cv2.resize(patch, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)


def visualize(args):
    h5_path = H5_DIR / f"{args.split}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"{h5_path} not found")

    with h5py.File(h5_path) as f:
        coords       = f["coords"][:]
        targets      = f["targets"][:]
        sources      = [s.decode() if isinstance(s, bytes) else s for s in f["sources"][:]]
        marker_names = list(f.attrs["marker_names"])
        crop_size    = int(f.attrs["crop_size"])
        resize_to    = int(f.attrs.get("resize_to", 0))
        token_grid   = int(f.attrs.get("token_grid", 16))

    print(f"Split: {args.split}  {len(coords):,} patches  "
          f"crop_size={crop_size}px  resize_to={resize_to}px")
    print(f"Markers: {marker_names}")

    # marker selection
    if args.markers:
        sel = [(i, m) for i, m in enumerate(marker_names) if m in args.markers]
        missing = [m for m in args.markers if m not in marker_names]
        if missing:
            print(f"  [warn] not in panel: {missing}")
    else:
        sel = list(enumerate(marker_names))

    if not sel:
        raise ValueError("No markers to display")

    n_cols = 1 + len(sel) * (2 if args.show_tokens else 1)
    rng    = np.random.default_rng(args.seed)
    n      = min(args.n_patches, len(coords))
    pick   = np.sort(rng.choice(len(coords), n, replace=False))

    # group by source to show which TIF each patch comes from
    unique_srcs = sorted(set(sources[i] for i in pick))
    print(f"Source TIFs represented ({len(unique_srcs)}): "
          + ", ".join(Path(s).name for s in unique_srcs[:5])
          + ("…" if len(unique_srcs) > 5 else ""))

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
        src  = sources[idx]

        he_patch = load_he_patch(src, x, y, crop_size, resize_to)
        axes[row, 0].imshow(he_patch)
        axes[row, 0].set_ylabel(f"#{idx}\n{Path(src).name[:20]}", fontsize=5)

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

    marker_tag = "-".join(m for _, m in sel)
    suffix = "_tokens" if args.show_tokens else ""
    plt.suptitle(f"HEMIT — {args.split} split", fontsize=11, y=1.002)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.split}_{marker_tag}{suffix}_n{n}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize HEMIT patches (H&E + IF tokens)")
    parser.add_argument("--split",      default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--n_patches",  type=int, default=12)
    parser.add_argument("--markers",    nargs="*", default=None,
                        help="Subset of markers (default: all). Panel: Pan-CK, CD3, Dapi")
    parser.add_argument("--show_tokens", action="store_true")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()
    visualize(args)


if __name__ == "__main__":
    main()