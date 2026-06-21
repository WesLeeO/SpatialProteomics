"""
Visualize random JEDI patches: H&E side-by-side with MxIF channels.

Reads coords from jedi_patch_dataset HDF5, opens H&E and the 5 single-channel
MxIF TIFFs, maps H&E patch corners through the Valis transform to MxIF space,
then uses cv2.warpPerspective to extract the correctly-aligned region.

Usage:
    python visualize_jedi_patches.py
    python visualize_jedi_patches.py --n_patches 8 --markers DNA CD3 CD68
"""

import argparse
import sys
import numpy as np
import cv2
import h5py
import tifffile
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR  = Path("/mnt/ssd1/virtual_proteomics/data/JEDI_201207")
H5_PATH   = Path("datasets/jedi_patch_dataset/JEDI20034_patch_dataset.h5")
VALIS_DIR = Path("datasets/jedi_valis")
OUT_DIR   = Path("visualization_out/jedi")

HE_TIF    = DATA_DIR / "JEDI20033.tif"

CHANNELS = [
    ("c0_DNA",  "DNA"),
    ("c1_CD20", "CD20"),
    ("c2_CD45", "CD45"),
    ("c3_CD3",  "CD3"),
    ("c4_CD68", "CD68"),
]

MODEL_SIZE = 224


def mxif_path(suffix: str) -> Path:
    return DATA_DIR / f"JEDI20034_{suffix}.tif"


def open_he_zarr(path: Path):
    """Open H&E TIFF as zarr. Returns (arr, h_ax, w_ax) for (H,W,C) or (C,H,W)."""
    tif   = tifffile.TiffFile(str(path))
    store = zarr.LRUStoreCache(tif.aszarr(), max_size=2 * 2**30)
    z     = zarr.open(store, mode="r")
    arr   = z["0"] if isinstance(z, zarr.hierarchy.Group) else z
    if arr.ndim == 3 and arr.shape[2] <= 4:   # (H, W, C) RGB
        return arr, 0, 1
    if arr.ndim == 3 and arr.shape[0] <= 4:   # (C, H, W)
        return arr, 1, 2
    raise ValueError(f"Unexpected H&E zarr shape {arr.shape} for {path.name}")


def open_mxif_zarr(suffix: str) -> zarr.Array:
    """Open a single-channel MxIF TIFF as a zarr array (H, W)."""
    path = mxif_path(suffix)
    return zarr.open(tifffile.TiffFile(str(path)).aszarr(), mode="r")


def load_valis(valis_dir: Path, he_name: str, dna_name: str):
    sys.path.insert(0, str(Path(__file__).parent))
    from valis import registration as valis_reg
    pickles = list(valis_dir.rglob("*.pickle"))
    if not pickles:
        raise FileNotFoundError(f"No Valis pickle under {valis_dir}")
    reg = valis_reg.load_registrar(str(pickles[0]))
    he_slide = dna_slide = None
    for slide in reg.slide_dict.values():
        name = Path(slide.src_f).name
        if name == he_name:
            he_slide = slide
        elif name == dna_name:
            dna_slide = slide
    if he_slide is None:
        raise KeyError(f"{he_name} not in registrar "
                       f"(have: {[Path(s.src_f).name for s in reg.slide_dict.values()]})")
    if dna_slide is None:
        raise KeyError(f"{dna_name} not in registrar")
    return he_slide, dna_slide


def visualize(args):
    h5_path = Path(args.h5)
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        coords            = f["coords"][:]
        p99s              = f["p99s"][:]
        p20s              = f["p20s"][:] if "p20s" in f else np.zeros_like(p99s)
        marker_names      = list(f.attrs["marker_names"])
        patch_size_level0 = int(f.attrs["patch_size_level0"])

    print(f"Patches        : {len(coords)}")
    print(f"H&E patch size : {patch_size_level0} px")
    print(f"Markers        : {marker_names}")

    # marker selection
    sel_names = args.markers if args.markers else marker_names
    sel = []  # (channel_idx, marker_name)
    for name in sel_names:
        if name in marker_names:
            sel.append((marker_names.index(name), name))
        else:
            print(f"  [warn] '{name}' not in dataset, skipping")
    if not sel:
        raise ValueError("No valid markers selected")

    # open images
    print("Opening H&E…")
    he_arr, h_ax, w_ax = open_he_zarr(HE_TIF)
    H_he = he_arr.shape[h_ax]
    W_he = he_arr.shape[w_ax]
    print(f"  H&E shape : {he_arr.shape}")

    print("Opening MxIF channels…")
    mxif_arrs = [open_mxif_zarr(suffix) for suffix, _ in CHANNELS]
    H_mxif, W_mxif = mxif_arrs[0].shape[:2]
    print(f"  MxIF shape: {mxif_arrs[0].shape}")

    # Valis
    valis_dir = Path(args.valis_dir)
    he_clean  = valis_dir / "JEDI20033_clean.tif"
    dna_local = valis_dir / "JEDI20034_c0_DNA.tif"
    print("Loading Valis registrar…")
    he_slide, dna_slide = load_valis(valis_dir, he_clean.name, dna_local.name)

    # random patch selection
    rng  = np.random.default_rng(args.seed)
    n    = min(args.n_patches, len(coords))
    pick = rng.choice(len(coords), n, replace=False)
    pick.sort()
    sel_coords = coords[pick]

    # map all 4 corners of each H&E patch → MxIF space in one batched call
    ps_he = patch_size_level0
    tl = sel_coords.astype(float)
    all_corners = np.vstack([
        tl,
        tl + [ps_he, 0    ],
        tl + [0,     ps_he],
        tl + [ps_he, ps_he],
    ])  # (4N, 2): TL×N, TR×N, BL×N, BR×N
    mapped_all     = he_slide.warp_xy_from_to(all_corners, dna_slide)  # (4N, 2)
    mapped_corners = mapped_all.reshape(4, n, 2).transpose(1, 0, 2)    # (N, 4, 2)

    # fixed destination for warpPerspective (order matches vstack above)
    dst_pts = np.float32([
        [0,          0         ],
        [MODEL_SIZE, 0         ],
        [0,          MODEL_SIZE],
        [MODEL_SIZE, MODEL_SIZE],
    ])

    # figure
    n_cols = 1 + len(sel)
    fig, axes = plt.subplots(n, n_cols,
                             figsize=(n_cols * 2.5, n * 2.5),
                             squeeze=False)
    col_titles = ["H&E"] + [name for _, name in sel]
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=9, fontweight="bold")

    for row, (orig_idx, (x_he, y_he), mc) in enumerate(
        zip(pick, sel_coords, mapped_corners)
    ):
        x_he, y_he = int(x_he), int(y_he)

        # H&E — axis-aligned crop, no warp needed
        if h_ax == 0:   # (H, W, C)
            patch_he = np.array(
                he_arr[y_he:min(y_he+ps_he, H_he), x_he:min(x_he+ps_he, W_he), :3],
                dtype=np.uint8,
            )
        else:           # (C, H, W)
            patch_he = np.array(
                he_arr[:3, y_he:min(y_he+ps_he, H_he), x_he:min(x_he+ps_he, W_he)],
                dtype=np.uint8,
            ).transpose(1, 2, 0)
        patch_he = cv2.resize(patch_he, (MODEL_SIZE, MODEL_SIZE),
                              interpolation=cv2.INTER_LINEAR)
        axes[row, 0].imshow(patch_he)
        axes[row, 0].set_ylabel(f"#{orig_idx}", fontsize=7)

        if np.any(np.isnan(mc)):
            for col in range(1, n_cols):
                axes[row, col].text(0.5, 0.5, "NaN", ha="center", va="center",
                                    transform=axes[row, col].transAxes, color="red")
            continue

        # AABB in MxIF pixel space
        x0 = max(int(np.floor(mc[:, 0].min())), 0)
        y0 = max(int(np.floor(mc[:, 1].min())), 0)
        x1 = min(int(np.ceil (mc[:, 0].max())), W_mxif)
        y1 = min(int(np.ceil (mc[:, 1].max())), H_mxif)

        if x1 <= x0 or y1 <= y0:
            for col in range(1, n_cols):
                axes[row, col].text(0.5, 0.5, "OOB", ha="center", va="center",
                                    transform=axes[row, col].transAxes, color="orange")
            continue

        src_pts = np.float32([[mc[j, 0] - x0, mc[j, 1] - y0] for j in range(4)])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        for col, (ci, _) in enumerate(sel, start=1):
            ch = np.array(mxif_arrs[ci][y0:y1, x0:x1], dtype=np.float32)
            warped = cv2.warpPerspective(
                ch, M, (MODEL_SIZE, MODEL_SIZE),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            )
            p99  = max(float(p99s[ci]), 1.0)
            p20  = float(p20s[ci])
            rang = max(p99 - p20, 1.0)
            normed = np.clip(np.log1p(np.maximum(warped - p20, 0.0) / rang), 0.0, 1.0)
            axes[row, col].imshow(normed, cmap="hot", vmin=0, vmax=1)

    for ax in axes.ravel():
        ax.axis("off")

    plt.suptitle("JEDI20033 H&E  /  JEDI20034 MxIF  — registration check",
                 fontsize=11, y=1.002)
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker_tag = "-".join(name for _, name in sel)
    out_path   = out_dir / f"JEDI_{marker_tag}_n{n}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize JEDI patches: H&E + MxIF registration check"
    )
    parser.add_argument("--n_patches", type=int, default=16,
                        help="Number of random patches to visualize")
    parser.add_argument("--markers",   nargs="*", default=None,
                        help="MxIF markers to show (default: all 5)")
    parser.add_argument("--seed",      type=int,  default=42)
    parser.add_argument("--h5",        default=str(H5_PATH),
                        help="Path to JEDI patch dataset HDF5")
    parser.add_argument("--valis_dir", default=str(VALIS_DIR))
    parser.add_argument("--out_dir",   default=str(OUT_DIR))
    args = parser.parse_args()
    visualize(args)


if __name__ == "__main__":
    main()