"""
Visualize NSCLC Charité patches: H&E side-by-side with IF channels.

Reads the pre-built patch dataset HDF5, loads the corresponding H&E and IF
JPEGs from spot_center_crops, then renders random patches for a chosen spot.

Usage:
    python visualize_nsclc_charite.py --spot 00878e3e-8b44-4756-9394-87fbac91e8b3
    python visualize_nsclc_charite.py --spot 00878e3e --n_patches 16 --markers CD3 CD8 CK
    python visualize_nsclc_charite.py --spot 00878e3e --show_tokens
"""

import argparse
import numpy as np
import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

CROPS_DIR = Path("/mnt/ssd1/virtual_proteomics/data/nsclc_charite/extracted/spot_center_crops")
H5_PATH   = Path("nsclc_charite_patch_dataset.h5")
OUT_DIR   = Path("visualize_nsclc_charite_out")
MODEL_SIZE = 224


def load_jpeg(path: Path) -> np.ndarray:
    return np.array(Image.open(path)).astype(np.float32)


def crop_patch(img: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    return cv2.resize(img[y:y + size, x:x + size], (MODEL_SIZE, MODEL_SIZE),
                      interpolation=cv2.INTER_LINEAR)


def resolve_spot(all_spot_ids: np.ndarray, query: str) -> str:
    """Accept full UUID or unambiguous prefix."""
    matches = [s for s in np.unique(all_spot_ids) if s.startswith(query)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        available = sorted(np.unique(all_spot_ids))
        raise ValueError(f"Spot '{query}' not found. Available:\n  " +
                         "\n  ".join(available[:10]) + ("…" if len(available) > 10 else ""))
    raise ValueError(f"Prefix '{query}' is ambiguous: {matches}")


def visualize(args, h5_path=None, out_dir=None):
    h5_path = h5_path or H5_PATH
    out_dir = out_dir or OUT_DIR

    with h5py.File(h5_path) as f:
        all_coords   = f["coords"][:]
        all_spot_ids = np.array([s.decode() if isinstance(s, bytes) else s
                                 for s in f["spot_ids"][:]])
        p99s         = f["p99s"][:]          # (C,) global
        marker_names = list(f.attrs["marker_names"])
        patch_size_level0 = int(f.attrs["patch_size_level0"])
        token_grid        = int(f.attrs.get("token_grid", 16))
        targets_all  = f["targets"][:] if args.show_tokens else None

    spot = resolve_spot(all_spot_ids, args.spot)
    mask = all_spot_ids == spot
    coords = all_coords[mask]
    print(f"Spot {spot}: {len(coords)} patches")

    # resolve markers
    if args.markers:
        sel_names = args.markers
    else:
        sel_names = marker_names
    sel = []
    for name in sel_names:
        if name in marker_names:
            sel.append((marker_names.index(name), name))
        else:
            print(f"  [warn] marker '{name}' not in dataset — skipping")
    if not sel:
        raise ValueError("No valid markers selected")

    # load full H&E and IF images for this spot
    he_path = CROPS_DIR / f"{spot}_he.jpg"
    if not he_path.exists():
        raise FileNotFoundError(f"H&E not found: {he_path}")
    he_img = load_jpeg(he_path).astype(np.uint8)

    marker_imgs = {}
    for mi, name in sel:
        path = CROPS_DIR / f"{spot}_{name}.jpg"
        if path.exists():
            marker_imgs[name] = load_jpeg(path)
        else:
            print(f"  [warn] IF image for '{name}' not found — will show zeros")
            marker_imgs[name] = None

    # random patch selection
    rng  = np.random.default_rng(args.seed)
    n    = min(args.n_patches, len(coords))
    pick = rng.choice(len(coords), n, replace=False)
    pick.sort()
    sel_coords = coords[pick]

    n_marker_cols = len(sel) * (2 if args.show_tokens else 1)
    n_cols = 1 + n_marker_cols

    fig, axes = plt.subplots(n, n_cols,
                             figsize=(n_cols * 2.5, n * 2.5),
                             squeeze=False)

    # column headers
    axes[0, 0].set_title("H&E", fontsize=9, fontweight="bold")
    col = 1
    for _, name in sel:
        axes[0, col].set_title(name, fontsize=9, fontweight="bold")
        col += 1
        if args.show_tokens:
            axes[0, col].set_title(f"{name}\n(tokens)", fontsize=9, fontweight="bold")
            col += 1

    ps = patch_size_level0

    if args.show_tokens:
        core_indices = np.where(mask)[0]
        pick_global  = core_indices[pick]

    for row, (orig_idx, (px, py)) in enumerate(zip(pick, sel_coords)):
        x, y = int(px), int(py)

        # H&E patch
        he_patch = crop_patch(he_img, x, y, ps)
        axes[row, 0].imshow(he_patch)
        axes[row, 0].set_ylabel(f"#{orig_idx}", fontsize=7)

        col = 1
        for mi, name in sel:
            img = marker_imgs[name]
            p99 = max(float(p99s[mi]), 1.0)

            if img is not None:
                H_i, W_i = img.shape[:2]
                region = img[y:min(y + ps, H_i), x:min(x + ps, W_i)]
                patch  = cv2.resize(region, (MODEL_SIZE, MODEL_SIZE),
                                    interpolation=cv2.INTER_LINEAR)
                normed = np.clip(np.log1p(patch / p99), 0.0, 1.0)
            else:
                normed = np.zeros((MODEL_SIZE, MODEL_SIZE), dtype=np.float32)

            axes[row, col].imshow(normed, cmap="gray", vmin=0, vmax=1)
            col += 1

            if args.show_tokens:
                token_img = targets_all[pick_global[row], mi]   # (token_grid, token_grid)
                axes[row, col].imshow(
                    np.clip(token_img, 0.0, 1.0), cmap="gray", vmin=0, vmax=1,
                    interpolation="nearest",
                )
                col += 1

    for ax in axes.ravel():
        ax.axis("off")

    marker_tag = "-".join(name for _, name in sel[:8])
    plt.suptitle(f"{spot[:8]}…  |  nsclc_charite", fontsize=11, y=1.002)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_tokens" if args.show_tokens else ""
    out_path = out_dir / f"{spot[:8]}_{marker_tag}{suffix}_n{n}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize NSCLC Charité patches (H&E + IF)")
    parser.add_argument("--spot",        required=True,
                        help="Spot UUID or unambiguous prefix")
    parser.add_argument("--n_patches",   type=int, default=16)
    parser.add_argument("--markers",     nargs="*", default=None,
                        help="IF markers to display (default: all)")
    parser.add_argument("--show_tokens", action="store_true",
                        help="Add a column showing the 16×16 token targets")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--h5",          default=str(H5_PATH))
    parser.add_argument("--out_dir",     default=str(OUT_DIR))
    args = parser.parse_args()

    visualize(args, h5_path=Path(args.h5), out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()