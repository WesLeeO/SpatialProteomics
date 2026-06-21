"""
Inference + evaluation + visualisation for pancancer CODEX TMA data.

Loads pancancer H5 datasets, runs the ORION-trained model on H&E patches,
finds overlapping markers between ORION outputs and each TMA's panel,
computes metrics, and saves per-core visualisations.

For token intensity distribution analysis see biomarker_expression_dist_pancancer.py.

Usage examples
--------------
# All TMA cores, composite view
python visualize_pancancer_predictions.py

# Specific TMA and cores, per-marker grid
python visualize_pancancer_predictions.py --tma CRC_TMA_A \
    --cores reg001_X01_Y01 reg002_X01_Y01 --mode pred

# Print cross-TMA marker overlap table and exit
python visualize_pancancer_predictions.py --overlap_summary
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import csv
import math
import re

import cv2
import h5py
import numpy as np
import tifffile
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import defaultdict

from visualize_orion_predictions import (
    run_inference, load_model,
    make_grid_figure, compute_metrics, compute_canvas_ssim,
    print_metrics, save_metrics_csv,
    IMAGENET_MEAN, IMAGENET_STD, TOKEN_GRID,
)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
DATASET_DIR     = Path("datasets/pancancer_patch_dataset")
HE_CACHE_DIR    = Path("datasets/pancancer_trident_output")
MODEL_DIR       = Path("training_outputs/outputs_orion_token_UNI2_finetuning_full")
DEFAULT_OUT_DIR = Path("visualization_out/pancancer")

TMAS = ["CRC_TMA_A", "CRC_TMA_B", "Multi-tumor", "Tonsil"]

ORION_MARKER_NAMES = [
    "Hoechst", "CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a",
    "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "E-Cadherin",
    "Ki-67", "Pan-CK", "SMA",
]

# Per ORION marker: list of normalised aliases to search in pancancer panels.
# Normalisation strips non-alphanumeric chars and lowercases.
ORION_ALIASES: dict[str, list[str]] = {
    "Hoechst":    ["hoechst", "dapi"],
    "CD31":       ["cd31"],
    "CD45":       ["cd45"],
    "CD68":       ["cd68"],
    "CD4":        ["cd4"],
    "FOXP3":      ["foxp3"],
    "CD8a":       ["cd8a", "cd8"],
    "CD45RO":     ["cd45ro"],
    "CD20":       ["cd20"],
    "PD-L1":      ["pdl1", "pdl1"],
    "CD3e":       ["cd3e", "cd3"],
    "CD163":      ["cd163"],
    "E-Cadherin": ["ecadherin", "ecad"],
    "Ki-67":      ["ki67", "ki67"],
    "Pan-CK":     ["pancytokeratin", "cytokeratin7", "cytokeratin", "panck"],
    "SMA":        ["sma", "asma"],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Marker overlap
# ---------------------------------------------------------------------------

def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_overlap(pancancer_markers: list[str]) -> list[tuple[int, str, int, str]]:
    """
    Returns [(orion_idx, orion_name, pc_idx, pc_name), ...] for each ORION
    marker that has a matching marker in the pancancer panel.
    """
    pc_norm = {_norm(m): (i, m) for i, m in enumerate(pancancer_markers)}
    overlap = []
    for orion_idx, orion_name in enumerate(ORION_MARKER_NAMES):
        aliases = ORION_ALIASES.get(orion_name, [_norm(orion_name)])
        for alias in aliases:
            if alias in pc_norm:
                pc_idx, pc_name = pc_norm[alias]
                overlap.append((orion_idx, orion_name, pc_idx, pc_name))
                break
    return overlap


def print_overlap_table(tma: str, pancancer_markers: list[str]) -> None:
    overlap = find_overlap(pancancer_markers)
    matched = {pc_name for *_, pc_name in overlap}
    unmatched = [m for m in ORION_MARKER_NAMES
                 if not any(on == m for _, on, *_ in overlap)]
    print(f"\n  {tma} — {len(overlap)}/{len(ORION_MARKER_NAMES)} ORION markers matched")
    print(f"  {'ORION marker':<16} → pancancer name")
    print("  " + "-" * 38)
    for orion_idx, orion_name, pc_idx, pc_name in overlap:
        print(f"  {orion_name:<16} → {pc_name}  (ch {pc_idx})")
    if unmatched:
        print(f"  Unmatched ORION: {', '.join(unmatched)}")
    pc_only = [m for m in pancancer_markers if m not in matched and m != "Hoechst"]
    print(f"  Pancancer-only ({len(pc_only)}): {', '.join(pc_only[:8])}"
          + (" …" if len(pc_only) > 8 else ""))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_core(h5_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Returns coords, targets, marker_names, patch_size_level0."""
    with h5py.File(h5_path) as f:
        coords            = f["coords"][:]
        targets           = f["targets"][:]
        marker_names      = list(f.attrs["marker_names"])
        patch_size_level0 = int(f.attrs["patch_size_level0"])
    return coords, targets, marker_names, patch_size_level0


def load_he_rgb(tma: str, core_id: str) -> np.ndarray:
    """Load cached uint8 RGB H&E TIF from TRIDENT export directory."""
    he_path = HE_CACHE_DIR / tma / core_id / f"{core_id}.tif"
    if not he_path.exists():
        raise FileNotFoundError(f"H&E TIF not found: {he_path}")
    arr = tifffile.imread(str(he_path))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return arr[:, :, :3]   # (H, W, 3) uint8


def extract_patches_224(he_arr: np.ndarray, coords: np.ndarray,
                         patch_size_level0: int) -> np.ndarray:
    """Crop → resize → ImageNet-normalise. Returns (N, 3, 224, 224) float32."""
    H, W = he_arr.shape[:2]
    patches = []
    for i, (x, y) in enumerate(coords):
        if i % 500 == 0:
            print(f"    extracting patches {i}/{len(coords)}", flush=True)
        x, y = int(x), int(y)
        crop = he_arr[y:min(y + patch_size_level0, H),
                      x:min(x + patch_size_level0, W), :].astype(np.float32)
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        crop = (crop / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        patches.append(crop.transpose(2, 0, 1))
    return np.stack(patches).astype(np.float32)


def extract_he_tokens(he_arr: np.ndarray, coords: np.ndarray,
                       patch_size_level0: int,
                       token_grid: int = TOKEN_GRID) -> np.ndarray:
    """Block-average H&E crops to token resolution. Returns (N, 3, G, G) float32 [0,1]."""
    G   = token_grid
    ppb = 224 // G
    H, W = he_arr.shape[:2]
    tokens = np.zeros((len(coords), 3, G, G), dtype=np.float32)
    for i, (x, y) in enumerate(coords):
        x, y = int(x), int(y)
        crop = he_arr[y:min(y + patch_size_level0, H),
                      x:min(x + patch_size_level0, W), :].astype(np.float32)
        p224 = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        tokens[i] = (
            p224.reshape(G, ppb, G, ppb, 3)
                .mean(axis=(1, 3))
                .transpose(2, 0, 1)
            / 255.0
        )
    return tokens


def remap_targets(targets_pc: np.ndarray,
                  overlap: list[tuple[int, str, int, str]]) -> np.ndarray:
    """
    Remap (N, C_pc, G, G) targets into ORION channel order (N, 16, G, G).
    Channels without a match are left as zero.
    """
    N, _, G, _ = targets_pc.shape
    out = np.zeros((N, len(ORION_MARKER_NAMES), G, G), dtype=np.float32)
    for orion_idx, _, pc_idx, _ in overlap:
        out[:, orion_idx] = targets_pc[:, pc_idx]
    return out


# ---------------------------------------------------------------------------
# Per-marker grid visualisation (mirrors visualize_sg_predictions.py)
# ---------------------------------------------------------------------------

def visualize_per_marker(
    coords: np.ndarray,
    he_tokens: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    H: int, W: int,
    patch_size_level0: int,
    sel: list,
    mode: str = "pred",
    token_grid: int = TOKEN_GRID,
    title: str = "",
    canvas_px: int = 2400,
    ncols: int = 4,
    dpi: int = 150,
    cmap: str = "inferno",
) -> plt.Figure:
    G      = token_grid
    scale1 = G / patch_size_level0

    canvas_h = math.ceil(H * scale1)
    canvas_w = math.ceil(W * scale1)

    he_canvas = np.full((canvas_h, canvas_w, 3), np.nan, dtype=np.float32)
    n_sel     = len(sel)
    mcanvases = np.full((n_sel, canvas_h, canvas_w), np.nan, dtype=np.float32)

    data = preds if mode == "pred" else targets

    for i, (x, y) in enumerate(coords):
        r0 = round(int(y) * scale1)
        c0 = round(int(x) * scale1)
        r1 = min(r0 + G, canvas_h)
        c1 = min(c0 + G, canvas_w)
        gr, gc = r1 - r0, c1 - c0
        he_canvas[r0:r1, c0:c1] = he_tokens[i].transpose(1, 2, 0)[:gr, :gc]
        for k, (channel, _) in enumerate(sel):
            mcanvases[k, r0:r1, c0:c1] = data[i, channel, :gr, :gc]

    bg_mask = np.isnan(he_canvas[:, :, 0])
    he_rgb  = np.clip(np.nan_to_num(he_canvas, nan=1.), 0., 1.)

    scale2 = canvas_px / max(canvas_h, canvas_w)
    disp_h = max(1, int(round(canvas_h * scale2)))
    disp_w = max(1, int(round(canvas_w * scale2)))

    he_disp = cv2.resize(he_rgb, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

    n_panels = 1 + n_sel
    nrows    = math.ceil(n_panels / ncols)
    panel_w  = min(canvas_px / (dpi * ncols), 4.0)
    panel_h  = panel_w * disp_h / max(disp_w, 1)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(panel_w * ncols, panel_h * nrows),
                              dpi=dpi, squeeze=False)
    axes_flat = axes.ravel()

    axes_flat[0].imshow(he_disp, interpolation="nearest")
    axes_flat[0].set_title("H&E", fontsize=8)
    axes_flat[0].axis("off")

    for k, (_, mname) in enumerate(sel):
        ch = mcanvases[k].copy()
        ch[bg_mask] = 0.
        ch_disp = cv2.resize(np.clip(ch, 0., 1.).astype(np.float32),
                              (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
        ax = axes_flat[k + 1]
        ax.imshow(ch_disp, cmap=cmap, vmin=0., vmax=1., interpolation="nearest")
        ax.set_title(mname, fontsize=8)
        ax.axis("off")

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    mode_label = "Predicted" if mode == "pred" else "Ground truth"
    fig.suptitle(f"{title} — {mode_label}" if title else mode_label, fontsize=10)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-core pipeline
# ---------------------------------------------------------------------------

def process_core(
    tma: str,
    core_id: str,
    model,
    out_dir: Path,
    mode: str,
    panel_px: int,
    ncols: int,
) -> None:
    h5_path = DATASET_DIR / tma / f"{core_id}_patch_dataset.h5"
    if not h5_path.exists():
        print(f"  Missing H5: {h5_path}, skipping.")
        return

    coords, targets_pc, marker_names, patch_size_level0 = load_core(h5_path)
    overlap = find_overlap(marker_names)

    if not overlap:
        print(f"  No overlapping markers found for {tma}/{core_id}, skipping.")
        return

    print(f"  {len(coords)} patches, patch_size_level0={patch_size_level0}, "
          f"{len(overlap)} overlapping markers")

    try:
        he_arr = load_he_rgb(tma, core_id)
    except FileNotFoundError as e:
        print(f"  {e}, skipping.")
        return

    H, W = he_arr.shape[:2]

    print("  Extracting H&E patches…")
    patches_224 = extract_patches_224(he_arr, coords, patch_size_level0)
    print(f"  Running inference (device={device})…")
    preds            = run_inference(model, patches_224)
    targets_remapped = remap_targets(targets_pc, overlap)

    print("  Building H&E token canvas…")
    he_tokens = extract_he_tokens(he_arr, coords, patch_size_level0, TOKEN_GRID)

    # sel uses ORION indices; labels show ORION name (pc name in parens if different)
    sel = []
    for orion_idx, orion_name, _, pc_name in overlap:
        label = orion_name if orion_name == pc_name else f"{orion_name} ({pc_name})"
        sel.append((orion_idx, label))

    title = f"{tma} / {core_id}"

    if mode == "composite":
        metrics = compute_metrics(preds, targets_remapped, sel)
        fig, pred_canvas, tgt_canvas = make_grid_figure(
            coords, he_tokens, preds, targets_remapped,
            H, W, patch_size_level0,
            sel=sel, token_grid=TOKEN_GRID,
            title=title, canvas_px=panel_px,
        )
        ssim_map = compute_canvas_ssim(pred_canvas, tgt_canvas, sel)
        for row in metrics:
            row["ssim"] = ssim_map.get(row["marker"], np.nan)
        print_metrics(metrics, title="Metrics")
        save_metrics_csv(metrics, out_dir / f"{tma}_{core_id}_metrics.csv")
        out_path = out_dir / f"{tma}_{core_id}_composite.png"
    else:
        fig = visualize_per_marker(
            coords, he_tokens, preds, targets_remapped,
            H, W, patch_size_level0,
            sel=sel, mode=mode,
            token_grid=TOKEN_GRID, title=title,
            canvas_px=panel_px, ncols=ncols,
        )
        out_path = out_dir / f"{tma}_{core_id}_{mode}.png"

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Cross-TMA marker overlap summary
# ---------------------------------------------------------------------------

def print_cross_tma_overlap() -> None:
    """Print which ORION markers overlap across all TMAs and their coverage."""
    print("\n" + "=" * 60)
    print("  Cross-TMA ORION marker overlap")
    print("=" * 60)

    tma_overlaps: dict[str, set[str]] = {}
    for tma in TMAS:
        h5_files = sorted((DATASET_DIR / tma).glob("*_patch_dataset.h5"))
        if not h5_files:
            continue
        with h5py.File(h5_files[0]) as f:
            markers = list(f.attrs["marker_names"])
        overlap = find_overlap(markers)
        tma_overlaps[tma] = {on for _, on, *_ in overlap}
        print_overlap_table(tma, markers)

    if len(tma_overlaps) > 1:
        common = set.intersection(*tma_overlaps.values())
        print(f"\n  Markers present in ALL {len(tma_overlaps)} TMAs: {sorted(common)}")
        for tma, names in tma_overlaps.items():
            unique = names - set.union(*(v for k, v in tma_overlaps.items() if k != tma))
            if unique:
                print(f"  Unique to {tma}: {sorted(unique)}")


# ---------------------------------------------------------------------------
# Aggregate metrics summary across all cores
# ---------------------------------------------------------------------------

def aggregate_metrics(out_dir: Path) -> None:
    csv_files = sorted(out_dir.glob("*_metrics.csv"))
    if not csv_files:
        return

    by_marker: dict[str, list] = defaultdict(list)
    for f in csv_files:
        with open(f) as fh:
            for row in csv.DictReader(fh):
                by_marker[row["marker"]].append({
                    k: float(v) for k, v in row.items() if k != "marker"
                })

    print(f"\n  Aggregate metrics ({len(csv_files)} cores)")
    print(f"  {'Marker':<20} {'n_cores':>8} {'Pearson r':>10} {'PSNR':>10} {'SSIM':>8}")
    print("  " + "-" * 58)

    summary_rows = []
    for marker, rows in sorted(by_marker.items()):
        vals = {k: [r[k] for r in rows if np.isfinite(r[k])] for k in rows[0]}
        pr   = np.mean(vals["pearson_r"]) if vals["pearson_r"] else np.nan
        psnr = np.mean(vals["psnr"])      if vals["psnr"]      else np.nan
        ssim = np.mean(vals["ssim"])      if vals["ssim"]      else np.nan
        n    = len(rows)
        print(f"  {marker:<20} {n:>8} {pr:>10.4f} {psnr:>10.2f} {ssim:>8.4f}")
        summary_rows.append(dict(marker=marker, n_cores=n,
                                  mean_pearson_r=pr, mean_psnr=psnr, mean_ssim=ssim))

    summary_path = out_dir / "aggregate_metrics.csv"
    with open(summary_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\n  Aggregate metrics saved → {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pancancer CODEX inference + evaluation + visualisation."
    )
    parser.add_argument("--tma",       default="all",
                        help=f"'all' or one of: {TMAS}")
    parser.add_argument("--cores",     nargs="*", default=None,
                        help="Core IDs to process (default: all). E.g. reg001_X01_Y01")
    parser.add_argument("--mode",      default="composite",
                        choices=["composite", "pred", "gt"],
                        help="composite → H&E / pred / GT blended; "
                             "pred/gt → per-marker grid")

    parser.add_argument("--model_dir", default=str(MODEL_DIR))
    parser.add_argument("--panel_px",  type=int, default=2400)
    parser.add_argument("--ncols",     type=int, default=4,
                        help="Columns in per-marker grid (mode=pred/gt)")
    parser.add_argument("--overlap_summary", action="store_true",
                        help="Print cross-TMA overlap table and exit")
    args = parser.parse_args()


    tma_keys = TMAS if args.tma == "all" else [args.tma]

    if args.overlap_summary:
        print_cross_tma_overlap()
        return

    out_dir = DEFAULT_OUT_DIR / args.tma
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model_path = Path(args.model_dir) / "best_model.pt"
    print(f"\nLoading model from {model_path}…")
    model = load_model(model_path)

    # Print overlap summary
    print_cross_tma_overlap()

    # Per-core inference + visualisation
    for tma in tma_keys:
        h5_dir   = DATASET_DIR / tma
        h5_files = sorted(h5_dir.glob("*_patch_dataset.h5"))
        if not h5_files:
            print(f"\nNo H5 files found in {h5_dir}, skipping {tma}.")
            continue

        # Filter to requested cores
        if args.cores:
            h5_files = [f for f in h5_files
                        if f.stem.replace("_patch_dataset", "") in args.cores]

        print(f"\n{'#'*60}\n  {tma}  —  {len(h5_files)} cores\n{'#'*60}")

        for h5_path in h5_files:
            core_id = h5_path.stem.replace("_patch_dataset", "")
            print(f"\n── {tma} / {core_id} ─────────────────────────")
            try:
                process_core(
                    tma, core_id, model, out_dir,
                    mode=args.mode, panel_px=args.panel_px,
                    ncols=args.ncols,
                )
            except Exception as exc:
                import traceback
                print(f"  Error: {exc}")
                traceback.print_exc()

    aggregate_metrics(out_dir)
    print(f"\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    main()