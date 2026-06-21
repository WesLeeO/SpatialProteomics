"""
Visualize random melanoma patches: H&E side-by-side with IF channels.

Loads coords from melanoma_patch_dataset HDF5, reads H&E + IF images,
maps all 4 H&E patch corners through the Valis transform to IF space,
then uses cv2.warpPerspective to extract the exact rotated patch region.

Usage:
    python visualize_melanoma_patches.py --sample MEL01-3-1-ROI1
    python visualize_melanoma_patches.py --sample MEL01-3-1-ROI1 --n_patches 32 --markers DNA_1 CD3 Pan-CK CD8a
"""

import argparse
import sys
import re
import json
import numpy as np
import cv2
import h5py
import tifffile
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from build_patch_dataset_melanoma import parse_csv_channels, build_af_arrays

DATA_DIR    = Path("/mnt/ssd1/virtual_proteomics/data/melanoma")

# Must stay in sync with IF_SPLIT_AXIS in build_patch_dataset_melanoma.py
IF_SPLIT_AXIS: dict[str, str] = {
    "MEL02-1-1": "h",
    "MEL03-1-1": "w",
    "MEL04-1-1": "w",
    "MEL05-1-1": "h",
    "MEL07-1-1": "w_ltr",
    "MEL08-1-1": "h",
    "MEL09-1-1": "h",
    "MEL10-1-1": "h",
    "MEL11-1-1": "w_ltr",
    "MEL12-1-1": "h",
    "MEL13-2-1": "w_ltr"
}

H5_DIR      = Path("datasets/melanoma_patch_dataset")
VALIS_DIR   = Path("datasets/melanoma_valis")
OUT_DIR     = Path("visualization_out/melanoma")

MODEL_SIZE  = 224


# ── file discovery ────────────────────────────────────────────────────────────

def sample_paths(sample: str, data_dir: Path, valis_dir: Path):
    """
    Returns (active_he, if_path, active_if, y_if_offset, valis_sample_dir).

    active_he   : rotated H&E copy if it exists, otherwise original.
    active_if   : IF crop used during VALIS registration (multi-ROI), or full IF.
    y_if_offset : pixels to add to VALIS IF coords to get full-IF coordinates.
    """
    m = re.match(r'^(.*)-ROI(\d+)$', sample)
    if not m:
        raise ValueError(f"Cannot parse sample name: {sample!r}")
    if_stem   = m.group(1)
    roi_n     = m.group(2)
    roi_n_int = int(roi_n)
    print(f"ROI num: {roi_n_int}")
    prefix    = "-".join(if_stem.split("-")[:-1])

    if_path = data_dir / f"{if_stem}.ome.tif"
    he_orig = data_dir / f"{prefix}-0-HE-ROI{roi_n}.ome.tif"

    # rotated H&E copy
    base = he_orig.name.split(".")[0]
    rotated = None
    for suffix in ("_rot90cw", "_rot90ccw", "_rot180"):
        cand = data_dir / f"{base}{suffix}.tif"
        if cand.exists():
            rotated = cand
            break
    active_he = rotated if rotated else he_orig
    if not active_he.exists():
        raise FileNotFoundError(f"H&E not found: {active_he}")
    if not if_path.exists():
        raise FileNotFoundError(f"IF not found: {if_path}")

    # IF crop used for VALIS registration (produced by write_if_valis_crop).
    # xcrop variant takes priority when the build script applied an extra x-crop.
    if_crop = data_dir / f"{if_stem}_roi{roi_n}_regcrop_xcrop.tif"
    if not if_crop.exists():
        if_crop = data_dir / f"{if_stem}_roi{roi_n}_regcrop.tif"
    if if_crop.exists():
        active_if = if_crop
        # Reproduce offsets exactly as the build script does.
        he_rois    = sorted(data_dir.glob(f"{prefix}-0-HE-ROI*.ome.tif"))
        n_rois     = len(he_rois)
        split_axis = IF_SPLIT_AXIS.get(if_stem, "h")
        with tifffile.TiffFile(str(if_path)) as tif:
            shape = tif.series[0].shape   # (C, H, W) for multiplex IF
        H_if = shape[-2]
        W_if = shape[-1]
        if split_axis == "h":
            h_slice     = H_if // n_rois
            x_if_offset = 0
            y_if_offset = (roi_n_int - 1) * h_slice
        else:   # "w" / "w_ltr"
            w_slice     = W_if // n_rois
            chunk       = (roi_n_int - 1) if split_axis == "w_ltr" else (n_rois - roi_n_int)
            x_if_offset = chunk * w_slice
            y_if_offset = 0
    else:
        active_if   = if_path
        x_if_offset = 0
        y_if_offset = 0

    return active_he, if_path, active_if, x_if_offset, y_if_offset, valis_dir / sample


def load_valis_slides(valis_sample_dir: Path, he_name: str, if_name: str):
    sys.path.insert(0, str(Path(__file__).parent))
    from valis import registration as valis_reg

    pickles = list(valis_sample_dir.rglob("*.pickle"))
    if not pickles:
        raise FileNotFoundError(f"No Valis pickle under {valis_sample_dir}")
    registrar = valis_reg.load_registrar(str(pickles[0]))

    he_slide = if_slide = None
    for slide in registrar.slide_dict.values():
        name = Path(slide.src_f).name
        if name == he_name:
            he_slide = slide
        elif name == if_name:
            if_slide = slide
    if he_slide is None:
        raise KeyError(f"{he_name} not found in registrar "
                       f"(available: {[Path(s.src_f).name for s in registrar.slide_dict.values()]})")
    if if_slide is None:
        raise KeyError(f"{if_name} not found in registrar")
    return he_slide, if_slide


# ── image IO ──────────────────────────────────────────────────────────────────

def open_zarr(path: Path):
    """Open OME-TIFF as zarr. Returns (arr, c_ax, h_ax, w_ax)."""
    tif   = tifffile.TiffFile(str(path))
    store = zarr.LRUStoreCache(tif.aszarr(), max_size=2 * 2**30)
    z     = zarr.open(store, mode="r")
    arr   = z["0"] if isinstance(z, zarr.hierarchy.Group) else z
    if arr.ndim == 3:
        if arr.shape[2] <= 4:        # (H, W, C) RGB H&E
            return arr, 2, 0, 1
        return arr, 0, 1, 2          # (C, H, W) IF
    raise ValueError(f"Unexpected zarr shape {arr.shape} for {path.name}")


def crop_he(arr, x: int, y: int, size: int) -> np.ndarray:
    """Returns (MODEL_SIZE, MODEL_SIZE, 3) uint8."""
    H, W = arr.shape[0], arr.shape[1]
    patch = np.array(arr[y:min(y+size, H), x:min(x+size, W), :3], dtype=np.uint8)
    return cv2.resize(patch, (MODEL_SIZE, MODEL_SIZE), interpolation=cv2.INTER_LINEAR)


# ── main visualisation ────────────────────────────────────────────────────────

def visualize(args):
    h5_path = Path(args.h5_dir) / f"{args.sample}_patch_dataset.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 not found: {h5_path}")

    with h5py.File(h5_path) as f:
        coords            = f["coords"][:]
        p99s              = f["p99s"][:]
        p10s              = f["p10s"][:] if "p10s" in f else np.zeros_like(p99s)
        marker_names      = list(f.attrs["marker_names"])
        patch_size_level0 = int(f.attrs["patch_size_level0"])
        ps_if             = int(f.attrs["ps_if"])
        x_if_offset_h5    = int(f.attrs["x_if_offset"]) if "x_if_offset" in f.attrs else None
        y_if_offset_h5    = int(f.attrs["y_if_offset"]) if "y_if_offset" in f.attrs else None
        if "channel_indices" in f:
            channel_indices = list(f["channel_indices"][:])
        else:
            channel_indices = list(range(len(marker_names)))

    # ── AF subtraction setup ──────────────────────────────────────────────────
    # Derive IF stem from sample name (e.g. MEL01-3-1-ROI1 → MEL01-3-1)
    m_stem = re.match(r'^(.*)-ROI\d+$', args.sample)
    if_stem = m_stem.group(1) if m_stem else args.sample
    zip_path = Path(args.data_dir) / f"{if_stem}-features.zip"
    if zip_path.exists():
        all_channels = parse_csv_channels(zip_path)
        lams, biases = build_af_arrays(all_channels, args.af_params, slide=args.sample)
        # tif_ch → (af_ch, lam, bias)
        af_info = {ch_idx: (af_ch, float(lams[ci]), float(biases[ci]))
                   for ci, (ch_idx, _, af_ch) in enumerate(all_channels)}
    else:
        print(f"  [af] features zip not found ({zip_path.name}) — skipping AF subtraction")
        af_info = {}

    print(f"Sample      : {args.sample}")
    print(f"Patches     : {len(coords)}")
    print(f"H&E size    : {patch_size_level0}px  →  {MODEL_SIZE}px")
    print(f"IF ref size : {ps_if}px")
    print(f"Markers     : {marker_names}")

    # resolve marker indices
    if args.markers:
        sel_markers = args.markers
    else:
        sel_markers = marker_names
    sel = []   # list of (marker_list_idx, tif_channel_idx, marker_name, af_ch, lam, bias)
    for name in sel_markers:
        if name in marker_names:
            mi      = marker_names.index(name)
            tif_ch  = channel_indices[mi]
            af_ch, lam, bias = af_info.get(tif_ch, (None, 0.0, 0.0))
            sel.append((mi, tif_ch, name, af_ch, lam, bias))
        else:
            print(f"  [warn] marker '{name}' not in dataset, skipping")
    if not sel:
        raise ValueError("No valid markers selected")

    # file paths
    active_he, if_path, active_if, x_if_offset_d, y_if_offset_d, valis_sample_dir = (
        sample_paths(args.sample, Path(args.data_dir), Path(args.valis_dir))
    )
    # HDF5-stored offsets are authoritative; fall back to derived only if absent
    x_if_offset = x_if_offset_h5 if x_if_offset_h5 is not None else x_if_offset_d
    y_if_offset = y_if_offset_h5 if y_if_offset_h5 is not None else y_if_offset_d

    print(f"H&E         : {active_he.name}")
    print(f"IF (full)   : {if_path.name}")
    print(f"IF (valis)  : {active_if.name}")
    print(f"x_offset    : {x_if_offset}  y_offset: {y_if_offset}")

    print("Loading Valis registrar…")
    he_slide, if_slide = load_valis_slides(
        valis_sample_dir, active_he.name, active_if.name
    )

    he_arr, _, _, _          = open_zarr(active_he)
    if_arr, c_ax, h_ax, w_ax = open_zarr(if_path)
    H_if = if_arr.shape[h_ax]
    W_if = if_arr.shape[w_ax]

    # random patch selection
    rng = np.random.default_rng(args.seed)
    n   = min(args.n_patches, len(coords))
    pick = rng.choice(len(coords), n, replace=False)
    pick.sort()
    sel_coords = coords[pick]

    # map all 4 corners of each selected H&E patch to VALIS IF space
    ps_he = patch_size_level0
    tl = sel_coords.astype(float)
    all_corners = np.vstack([
        tl,
        tl + [ps_he, 0],
        tl + [0,     ps_he],
        tl + [ps_he, ps_he],
    ])  # (4N, 2): TL×N, TR×N, BL×N, BR×N
    mapped_all = he_slide.warp_xy_from_to(all_corners, if_slide)  # (4N, 2)
    mapped_corners = mapped_all.reshape(4, n, 2).transpose(1, 0, 2)  # (N, 4, 2)

    # destination corners for warpPerspective: canonical MODEL_SIZE square
    # order matches vstack above: TL, TR, BL, BR
    dst_pts = np.float32([
        [0,          0         ],
        [MODEL_SIZE, 0         ],
        [0,          MODEL_SIZE],
        [MODEL_SIZE, MODEL_SIZE],
    ])

    # ── figure ────────────────────────────────────────────────────────────────
    # Columns per marker: 1 base + 1 if --show_raw + 1 if --show_af (and af exists)
    def n_marker_cols(af_ch):
        return 1 + (1 if args.show_raw else 0) + (1 if args.show_af and af_ch is not None else 0)

    n_cols = 1 + sum(n_marker_cols(af_ch) for _, _, _, af_ch, _, _ in sel)
    fig, axes = plt.subplots(n, n_cols,
                             figsize=(n_cols * 2.5, n * 2.5),
                             squeeze=False)

    col_titles = ["H&E"]
    for _, _, name, af_ch, _, _ in sel:
        if args.show_raw:
            col_titles.append(f"{name}\n(raw)")
        col_titles.append(name)
        if args.show_af and af_ch is not None:
            col_titles.append(f"{name}\n(AF)")
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=9, fontweight="bold")

    for row, (orig_idx, (x_he, y_he), mc) in enumerate(
        zip(pick, sel_coords, mapped_corners)
    ):
        x_he, y_he = int(x_he), int(y_he)

        # H&E patch (axis-aligned in H&E space — no warp needed)
        he_patch = crop_he(he_arr, x_he, y_he, ps_he)
        axes[row, 0].imshow(he_patch)
        axes[row, 0].set_ylabel(f"#{orig_idx}", fontsize=7)

        if np.any(np.isnan(mc)):
            for col in range(1, n_cols):
                axes[row, col].text(0.5, 0.5, "NaN", ha="center", va="center",
                                    transform=axes[row, col].transAxes, color="red")
            continue

        # AABB in full IF coordinate space (offsets convert VALIS crop→full-IF)
        x0 = max(int(np.floor(mc[:, 0].min())) + x_if_offset, 0)
        y0 = max(int(np.floor(mc[:, 1].min())) + y_if_offset, 0)
        x1 = min(int(np.ceil (mc[:, 0].max())) + x_if_offset, W_if)
        y1 = min(int(np.ceil (mc[:, 1].max())) + y_if_offset, H_if)

        if x1 <= x0 or y1 <= y0:
            for col in range(1, n_cols):
                axes[row, col].text(0.5, 0.5, "OOB", ha="center", va="center",
                                    transform=axes[row, col].transAxes, color="orange")
            continue

        # Read the full AABB from IF (all channels at once)
        idx_sl = [slice(None)] * if_arr.ndim
        idx_sl[h_ax] = slice(y0, y1)
        idx_sl[w_ax] = slice(x0, x1)
        region = if_arr[tuple(idx_sl)].astype(np.float32)
        if c_ax != 0:
            region = region.transpose(2, 0, 1)  # → (C, H_aabb, W_aabb)

        # Source corners in AABB-local coordinates.
        # mc[:,1] is in VALIS IF space; adding y_if_offset and subtracting y0
        # (which already includes y_if_offset) gives the local coordinate.
        src_pts = np.float32([
            [mc[j, 0] + x_if_offset - x0, mc[j, 1] + y_if_offset - y0]
            for j in range(4)
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        col = 1
        for mi, tif_ch, _, af_ch, lam, bias in sel:
            sig = region[tif_ch]
            if af_ch is not None and lam > 0:
                ch_data = np.maximum(sig - lam * region[af_ch] - bias, 0.0)
            else:
                ch_data = np.maximum(sig, 0.0)

            warped = cv2.warpPerspective(
                ch_data, M, (MODEL_SIZE, MODEL_SIZE),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            )
            p99 = max(float(p99s[mi]), 1.0)

            if args.show_raw:
                raw = cv2.resize(ch_data, (MODEL_SIZE, MODEL_SIZE),
                                 interpolation=cv2.INTER_LINEAR)
                axes[row, col].imshow(
                    np.clip(np.log1p(raw / p99), 0.0, 1.0),
                    cmap="gray", vmin=0, vmax=1)
                col += 1

            normed = np.clip(np.log1p(warped / p99), 0.0, 1.0)
            axes[row, col].imshow(normed, cmap="gray", vmin=0, vmax=1)
            col += 1

            if args.show_af and af_ch is not None:
                af_raw = region[af_ch]
                af_warped = cv2.warpPerspective(
                    af_raw, M, (MODEL_SIZE, MODEL_SIZE),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                )
                af_p99 = max(float(np.percentile(af_raw[af_raw > 0], 99.9))
                             if np.any(af_raw > 0) else 1.0, 1.0)
                axes[row, col].imshow(
                    np.clip(np.log1p(af_warped / af_p99), 0.0, 1.0),
                    cmap="gray", vmin=0, vmax=1)
                col += 1

    for ax in axes.ravel():
        ax.axis("off")

    plt.suptitle(args.sample, fontsize=11, y=1.002)
    plt.tight_layout()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker_tag = "-".join(name for _, _, name, *_ in sel)
    out_path = out_dir / f"{args.sample}_{marker_tag}_{'show_raw' if args.show_raw else ''}_n{n}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize melanoma patches (H&E + IF)")
    parser.add_argument("--sample",     required=True,
                        help="Sample name, e.g. MEL01-3-1-ROI1")
    parser.add_argument("--n_patches",  type=int, default=8,
                        help="Number of random patches to show")
    parser.add_argument("--markers",    nargs="*", default=None,
                        help="IF markers to display (default: all)")
    parser.add_argument("--show_raw",   action="store_true",
                        help="Add a raw (axis-aligned, un-warped) column per marker "
                             "to distinguish measurement noise from warp artifacts")
    parser.add_argument("--show_af",    action="store_true",
                        help="Add an AF channel column after each marker that has one")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--data_dir",   default=str(DATA_DIR))
    parser.add_argument("--h5_dir",     default=str(H5_DIR))
    parser.add_argument("--valis_dir",  default=str(VALIS_DIR))
    parser.add_argument("--out_dir",    default=str(OUT_DIR))
    parser.add_argument("--af_params",  default="af_params_melanoma.json",
                        help="AF subtraction JSON (default: af_params_melanoma.json)")
    args = parser.parse_args()
    visualize(args)


if __name__ == "__main__":
    main()