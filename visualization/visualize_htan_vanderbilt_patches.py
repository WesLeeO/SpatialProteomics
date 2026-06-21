"""
Visualize HTAN Vanderbilt patches: H&E side-by-side with MxIF channels.

Loads coords from the HDF5, reads the full H&E and MxIF TIFF, maps each H&E
patch's 4 corners through the Valis transform (crop H&E → MxIF space), then
uses cv2.warpPerspective to extract the exact rotated patch region.

Usage:
    python visualize_htan_vanderbilt_patches.py --sample HTA11_1938_tile001
    python visualize_htan_vanderbilt_patches.py --sample HTA11_1938_tile001 \
        --n_patches 16 --markers DAPI CD3 Pan-CK
"""

import argparse
import re
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

DATA_ROOT  = Path("/mnt/ssd1/virtual_proteomics/data/HTAN/vanderbilt")
H5_DIR     = Path("htan_vanderbilt_patch_dataset")
VALIS_DIR  = Path("htan_vanderbilt_valis")
OUT_DIR    = Path("visualization_out/htan_vanderbilt")
MODEL_SIZE = 224


# ── file discovery ─────────────────────────────────────────────────────────────

def parse_sample(sample: str):
    """'HTA11_1938_tile001' → ('HTA11_1938', 1)"""
    m = re.match(r'^(.+)_tile(\d+)$', sample)
    if not m:
        raise ValueError(f"Cannot parse sample name: {sample!r}  "
                         f"(expected <participant>_tile<N>)")
    return m.group(1), int(m.group(2))


def find_files(participant: str, tile_idx: int, data_root: Path):
    """Return (he_path, mxif_path, he_crop_path, dapi_path)."""
    data_dir = data_root / participant

    # H&E: HTAN naming convention — filename ends with 9999
    he_files = sorted(data_dir.glob("*9999.tif"))
    if not he_files:
        raise FileNotFoundError(f"No H&E (*9999.tif) found in {data_dir}")
    he_path = he_files[0]

    mxif_files = sorted(f for f in data_dir.glob("*.tif")
                        if f != he_path
                        and "_he_crop" not in f.name
                        and "_dapi" not in f.name)
    if tile_idx < 1 or tile_idx > len(mxif_files):
        raise ValueError(f"tile_idx {tile_idx} out of range "
                         f"(found {len(mxif_files)} MxIF tiles)")
    mxif_path = mxif_files[tile_idx - 1]

    sample       = f"{participant}_tile{tile_idx:03d}"
    he_crop_path = data_dir / f"{sample}_he_crop.tif"
    dapi_path    = data_dir / f"{mxif_path.stem}_dapi.tif"

    for p in (he_crop_path, dapi_path):
        if not p.exists():
            raise FileNotFoundError(f"Expected file not found: {p}")

    return he_path, mxif_path, he_crop_path, dapi_path


# ── Valis ──────────────────────────────────────────────────────────────────────

def load_valis_slides(valis_dir: Path, he_crop_name: str, dapi_name: str):
    from valis import registration as valis_reg
    pickles = list(valis_dir.rglob("*.pickle"))
    if not pickles:
        raise FileNotFoundError(f"No Valis pickle under {valis_dir}")
    reg = valis_reg.load_registrar(str(pickles[0]))
    he_slide = dapi_slide = None
    for slide in reg.slide_dict.values():
        name = Path(slide.src_f).name
        if name == he_crop_name:
            he_slide = slide
        elif name == dapi_name:
            dapi_slide = slide
    available = [Path(s.src_f).name for s in reg.slide_dict.values()]
    if he_slide is None:
        raise KeyError(f"{he_crop_name} not in registrar (available: {available})")
    if dapi_slide is None:
        raise KeyError(f"{dapi_name} not in registrar (available: {available})")
    return he_slide, dapi_slide


# ── image IO ───────────────────────────────────────────────────────────────────

def open_he_zarr(he_path: Path):
    """Open flat RGB TIFF as zarr → (H, W, 3) proxy."""
    tif   = tifffile.TiffFile(str(he_path))
    store = zarr.LRUStoreCache(tif.aszarr(), max_size=2 * 2**30)
    z     = zarr.open(store, mode="r")
    if isinstance(z, zarr.hierarchy.Group):
        z = z["0"]
    return z


def open_mxif_zarr(mxif_path: Path):
    """Open 28-page MxIF TIFF as zarr → (C, H, W) proxy."""
    tif   = tifffile.TiffFile(str(mxif_path))
    store = zarr.LRUStoreCache(tif.aszarr(), max_size=2 * 2**30)
    z     = zarr.open(store, mode="r")
    if isinstance(z, zarr.hierarchy.Group):
        z = z["0"]
    return z


def crop_he_patch(he_arr, x: int, y: int, size: int) -> np.ndarray:
    """Read an axis-aligned H&E patch → (MODEL_SIZE, MODEL_SIZE, 3) uint8."""
    H, W = he_arr.shape[0], he_arr.shape[1]
    patch = np.array(he_arr[y:min(y + size, H), x:min(x + size, W), :3],
                     dtype=np.uint8)
    return cv2.resize(patch, (MODEL_SIZE, MODEL_SIZE),
                      interpolation=cv2.INTER_LINEAR)


# ── main ───────────────────────────────────────────────────────────────────────

def visualize(args):
    participant, tile_idx = parse_sample(args.sample)

    h5_path = Path(args.h5_dir) / f"{args.sample}_patch_dataset.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {h5_path}")

    with h5py.File(h5_path) as f:
        coords            = f["coords"][:]
        p99s              = f["p99s"][:]
        p10s              = f["p10s"][:] if "p10s" in f else np.zeros_like(p99s)
        marker_names      = list(f.attrs["marker_names"])
        channel_indices   = list(f["channel_indices"][:]) if "channel_indices" in f \
                            else list(range(len(marker_names)))
        patch_size_level0 = int(f.attrs["patch_size_level0"])
        ps_mxif           = int(f.attrs["ps_mxif"])

    print(f"Sample      : {args.sample}")
    print(f"Patches     : {len(coords)}")
    print(f"H&E size    : {patch_size_level0}px  →  {MODEL_SIZE}px")
    print(f"MxIF size   : {ps_mxif}px")
    print(f"Markers     : {marker_names}")

    # resolve marker selection
    sel_names = args.markers if args.markers else marker_names[:min(5, len(marker_names))]
    sel = []   # list of (marker_list_idx, mxif_channel_idx, marker_name)
    for name in sel_names:
        if name in marker_names:
            mi = marker_names.index(name)
            sel.append((mi, int(channel_indices[mi]), name))
        else:
            print(f"  [warn] marker '{name}' not in dataset, skipping")
    if not sel:
        raise ValueError("No valid markers selected")

    # files
    he_path, mxif_path, he_crop_path, dapi_path = find_files(
        participant, tile_idx, Path(args.data_root))
    valis_sample_dir = Path(args.valis_dir) / args.sample

    print(f"H&E         : {he_path.name}")
    print(f"MxIF        : {mxif_path.name}")
    print(f"Valis dir   : {valis_sample_dir}")

    print("Loading Valis registrar…")
    he_slide, dapi_slide = load_valis_slides(
        valis_sample_dir, he_crop_path.name, dapi_path.name)

    he_arr   = open_he_zarr(he_crop_path)   # coords are in crop space
    mxif_arr = open_mxif_zarr(mxif_path)

    # MxIF is (C, H, W)
    H_mxif, W_mxif = int(mxif_arr.shape[-2]), int(mxif_arr.shape[-1])

    # random patch selection
    rng  = np.random.default_rng(args.seed)
    n    = min(args.n_patches, len(coords))
    pick = rng.choice(len(coords), n, replace=False)
    pick.sort()
    sel_coords = coords[pick]

    # map H&E patch corners → MxIF space
    # coords are in crop H&E space (TRIDENT ran on he_crop), no offset needed
    ps_he = patch_size_level0
    tl = sel_coords.astype(float)
    all_corners = np.vstack([
        tl,
        tl + [ps_he, 0    ],
        tl + [0,     ps_he],
        tl + [ps_he, ps_he],
    ])   # (4N, 2)  order: TL, TR, BL, BR
    mapped_all     = he_slide.warp_xy_from_to(all_corners, dapi_slide)  # (4N, 2)
    mapped_corners = mapped_all.reshape(4, n, 2).transpose(1, 0, 2)     # (N, 4, 2)

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

    for c, title in enumerate(["H&E"] + [nm for _, _, nm in sel]):
        axes[0, c].set_title(title, fontsize=9, fontweight="bold")

    for row, (orig_idx, (x_he, y_he), mc) in enumerate(
        zip(pick, sel_coords, mapped_corners)
    ):
        x_he, y_he = int(x_he), int(y_he)

        # H&E patch (axis-aligned, read from full H&E)
        he_patch = crop_he_patch(he_arr, x_he, y_he, ps_he)
        axes[row, 0].imshow(he_patch)
        axes[row, 0].set_ylabel(f"#{orig_idx}", fontsize=7)

        if np.any(np.isnan(mc)):
            for col in range(1, n_cols):
                axes[row, col].text(0.5, 0.5, "NaN", ha="center", va="center",
                                    transform=axes[row, col].transAxes, color="red")
            continue

        # AABB in MxIF space
        x0 = max(int(np.floor(mc[:, 0].min())), 0)
        y0 = max(int(np.floor(mc[:, 1].min())), 0)
        x1 = min(int(np.ceil (mc[:, 0].max())), W_mxif)
        y1 = min(int(np.ceil (mc[:, 1].max())), H_mxif)

        if x1 <= x0 or y1 <= y0:
            for col in range(1, n_cols):
                axes[row, col].text(0.5, 0.5, "OOB", ha="center", va="center",
                                    transform=axes[row, col].transAxes, color="orange")
            continue

        src_pts = np.float32([
            [mc[j, 0] - x0, mc[j, 1] - y0] for j in range(4)
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        for col, (mi, ch_idx, _) in enumerate(sel, start=1):
            # mxif_arr is (C, H, W); ch_idx from channel_indices is direct
            ch_data = np.array(mxif_arr[ch_idx, y0:y1, x0:x1], dtype=np.float32)
            warped  = cv2.warpPerspective(
                ch_data, M, (MODEL_SIZE, MODEL_SIZE),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            )
            p99    = max(float(p99s[mi]), 1.0)
            p10    = float(p10s[mi])
            rang   = max(p99 - p10, 1.0)
            normed = np.clip(np.log1p(np.maximum(warped - p10, 0.0) / rang), 0.0, 1.0)
            axes[row, col].imshow(normed, cmap="hot", vmin=0, vmax=1)

    for ax in axes.ravel():
        ax.axis("off")

    plt.suptitle(args.sample, fontsize=11, y=1.002)
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker_tag = "-".join(nm for _, _, nm in sel)
    out_path   = out_dir / f"{args.sample}_{marker_tag}_n{n}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize HTAN Vanderbilt patches (H&E + MxIF)")
    parser.add_argument("--sample",    required=True,
                        help="Sample name, e.g. HTA11_1938_tile001")
    parser.add_argument("--n_patches", type=int, default=16)
    parser.add_argument("--markers",   nargs="*", default=None,
                        help="MxIF markers to display (default: first 5)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--data_root", default=str(DATA_ROOT))
    parser.add_argument("--h5_dir",    default=str(H5_DIR))
    parser.add_argument("--valis_dir", default=str(VALIS_DIR))
    parser.add_argument("--out_dir",   default=str(OUT_DIR))
    args = parser.parse_args()
    visualize(args)


if __name__ == "__main__":
    main()