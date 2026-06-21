"""
Visualize immunoatlas patches: H&E side-by-side with IF channels.

Reads the pre-built patch dataset HDF5, loads the corresponding H&E core PNG
and WebP marker images, then renders random patches for a chosen core.

Usage:
    python visualize_immunoatlas.py --core core001
    python visualize_immunoatlas.py --core core001 --n_patches 16 --markers Hoechst CD3 CD8 CD45
    python visualize_immunoatlas.py --core core001 --show_tokens
"""

import argparse
import csv
import numpy as np
import cv2
import h5py
import imageio.v3 as iio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATASET      = "immunoatlas_NOLN210920"
ROOT         = Path(f"/mnt/ssd1/virtual_proteomics/data/{DATASET}")
MANIFEST     = ROOT / "manifest.tsv"
CORE_PNG_DIR = Path("datasets/immunoatlas_70_png")
H5_PATH      = Path(f"{DATASET}_patch_dataset.h5")
OUT_DIR      = Path("visualization_out/immunoatlas")

WEBP_SCALE  = 2
MODEL_SIZE  = 224
EXCLUDE_CHANNELS = {"DRAQ5", "composite"}


def parse_manifest():
    core_channels = {}
    with open(MANIFEST) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            core  = row["core_name"]
            cname = row["channel_name"]
            cidx  = row["channel_index"]
            fpath = ROOT / row["filename"]
            if cname in EXCLUDE_CHANNELS or cidx == "-":
                continue
            if core not in core_channels:
                core_channels[core] = {}
            core_channels[core][cname] = fpath
    return core_channels


def load_webp_channel(path: Path) -> np.ndarray:
    img = iio.imread(str(path)).astype(np.float32)
    if img.ndim == 3:
        img = img[:, :, 0]   # R=B=true value; G has ±1 WebP YCbCr artifact
    return img


def crop_he(he_img: np.ndarray, x: int, y: int, size: int) -> np.ndarray:
    H, W = he_img.shape[:2]
    patch = he_img[y:min(y + size, H), x:min(x + size, W)]
    if patch.ndim == 3 and patch.shape[2] > 3:
        patch = patch[:, :, :3]
    # trim black (all-zero) border columns/rows before resize
    occupied = patch.any(axis=2) if patch.ndim == 3 else patch > 0
    rows = np.where(occupied.any(axis=1))[0]
    cols = np.where(occupied.any(axis=0))[0]
    if rows.size and cols.size:
        patch = patch[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return cv2.resize(patch.astype(np.uint8), (MODEL_SIZE, MODEL_SIZE),
                      interpolation=cv2.INTER_LINEAR)


def visualize(args, h5_path=None, out_dir=None):
    h5_path = h5_path or H5_PATH
    out_dir = out_dir or OUT_DIR
    with h5py.File(h5_path) as f:
        all_coords   = f["coords"][:]
        all_core_ids = np.array([c.decode() if isinstance(c, bytes) else c
                                 for c in f["core_ids"][:]])
        p99s_raw     = f["p99s"][:]
        marker_names = list(f.attrs["marker_names"])
        patch_size_level0 = int(f.attrs["patch_size_level0"])
        token_grid        = int(f.attrs.get("token_grid", 16))
        targets_all  = f["targets"][:] if args.show_tokens else None

        # support both old (n_cores, C) and new (C,) p99s shapes
        if p99s_raw.ndim == 2:
            unique_cores = list(dict.fromkeys(all_core_ids))
            core_to_row  = {c: i for i, c in enumerate(unique_cores)}
            p99s_global  = None   # handled per-core below
        else:
            p99s_global  = p99s_raw   # (C,)
            core_to_row  = None

    core = args.core
    mask = all_core_ids == core
    if not mask.any():
        raise ValueError(f"Core '{core}' not found in dataset. "
                         f"Available: {sorted(set(all_core_ids))[:10]}…")

    coords  = all_coords[mask]
    if p99s_global is not None:
        p99s = p99s_global            # (C,) global
    else:
        p99s = p99s_raw[core_to_row[core]]  # (C,) per-core fallback

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

    # load H&E core PNG
    core_png = CORE_PNG_DIR / f"{core}.png"
    if not core_png.exists():
        raise FileNotFoundError(f"H&E PNG not found: {core_png}")
    he_img = np.array(iio.imread(str(core_png)))

    # load WebP marker images for selected channels
    core_channels = parse_manifest()
    ch_map = core_channels.get(core, {})
    marker_imgs = {}
    for mi, name in sel:
        path = ch_map.get(name)
        if path and path.exists():
            marker_imgs[name] = load_webp_channel(path)
        else:
            print(f"  [warn] WebP for '{name}' not found — will show zeros")
            marker_imgs[name] = None

    # random patch selection
    rng   = np.random.default_rng(args.seed)
    n     = min(args.n_patches, len(coords))
    pick  = rng.choice(len(coords), n, replace=False)
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

    ps_he   = patch_size_level0
    ps_webp = ps_he * WEBP_SCALE

    # get targets indices relative to the per-core slice if showing tokens
    if args.show_tokens:
        core_indices = np.where(mask)[0]
        pick_global  = core_indices[pick]

    for row, (orig_idx, (x_he, y_he)) in enumerate(zip(pick, sel_coords)):
        x_he, y_he = int(x_he), int(y_he)

        # H&E patch
        he_patch = crop_he(he_img, x_he, y_he, ps_he)
        axes[row, 0].imshow(he_patch)
        axes[row, 0].set_ylabel(f"#{orig_idx}", fontsize=7)

        x_webp = x_he * WEBP_SCALE
        y_webp = y_he * WEBP_SCALE

        col = 1
        for mi, name in sel:
            img = marker_imgs[name]
            p99 = max(float(p99s[mi]), 1.0)

            if img is not None:
                H_w, W_w = img.shape
                x1 = min(x_webp + ps_webp, W_w)
                y1 = min(y_webp + ps_webp, H_w)
                region = img[y_webp:y1, x_webp:x1]
                patch  = cv2.resize(region, (MODEL_SIZE, MODEL_SIZE),
                                    interpolation=cv2.INTER_LINEAR)
                normed = np.clip(patch / p99, 0.0, 1.0)
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
    plt.suptitle(f"{core}  |  {DATASET}", fontsize=11, y=1.002)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_tokens" if args.show_tokens else ""
    out_path = out_dir / f"{core}_{marker_tag}{suffix}_n{n}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize immunoatlas patches (H&E + IF)")
    parser.add_argument("--core",        required=True,
                        help="Core name, e.g. core001")
    parser.add_argument("--n_patches",   type=int, default=16,
                        help="Number of random patches to show")
    parser.add_argument("--markers",     nargs="*", default=None,
                        help="IF markers to display (default: all)")
    parser.add_argument("--show_tokens", action="store_true",
                        help="Add a column showing the pre-computed 16×16 token targets")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--h5",          default=str(H5_PATH),
                        help="Path to patch dataset HDF5")
    parser.add_argument("--out_dir",     default=str(OUT_DIR))
    args = parser.parse_args()

    visualize(args, h5_path=Path(args.h5), out_dir=Path(args.out_dir))


if __name__ == "__main__":
    main()