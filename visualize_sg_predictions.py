import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from build_patch_dataset_orion_crc_reg import open_zarr_level0
from build_patch_dataset_singular_genomics import SG_DATA_ROOT, SINGULAR_GENOMICS_BIOMARKERS
from visualize_orion_predictions import (
    extract_patches, extract_he_tokens, make_grid_figure, run_inference, load_model,
    compute_metrics, print_metrics, save_metrics_csv,
    TOKEN_GRID,
)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
SG_H5_DIR       = Path("singular_genomics")
SG_MODEL_DIR    = Path("outputs_orion_token_UNI2_finetuning")
DEFAULT_OUT_DIR = Path("visualize_sg_out")

# ORION channel order — index must match the model's output dimension
ORION_MARKER_NAMES = [
    "Hoechst", "CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a",
    "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "E-Cadherin",
    "Ki-67", "Pan-CK", "SMA",
]

# Maps each ORION marker to its name in the SG panel (None = no SG equivalent)
ORION_TO_SG = {
    "Hoechst":    None,
    "CD31":       "CD31",
    "CD45":       "CD45",
    "CD68":       "CD68",
    "CD4":        "CD4",
    "FOXP3":      "FOXP3",
    "CD8a":       "CD8",
    "CD45RO":     None,
    "CD20":       "CD20",
    "PD-L1":      "PDL1",
    "CD3e":       "CD3",
    "CD163":      None,
    "E-Cadherin": None,
    "Ki-67":      "KI67",
    "Pan-CK":     "PanCK",
    "SMA":        "aSMA",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sg_he_path(disease: str) -> Path:
    he_dir = SG_DATA_ROOT / disease / "g4x_viewer"
    slides = sorted(he_dir.glob("*_HE.ome.tiff"))
    if not slides:
        raise FileNotFoundError(f"No *_HE.ome.tiff in {he_dir}")
    return slides[0]


def remap_targets(targets_sg: np.ndarray,
                  sg_marker_names: list,
                  sg_valid_names: set) -> np.ndarray:
    """
    Remap SG ground-truth targets from SG channel order to ORION channel order
    so they align with the model's output indices for ground-truth comparison.

    Only copies channels whose JP2 was actually present (sg_valid_names).
    Missing channels stay zero so they are excluded from sel before display.

    targets_sg : (N, C_sg, G, G)
    Returns     : (N, 16, G, G) — ORION-indexed; absent/unmapped channels = 0.
    """
    N, _, G, _ = targets_sg.shape
    remapped = np.zeros((N, len(ORION_MARKER_NAMES), G, G), dtype=np.float32)
    for orion_idx, orion_name in enumerate(ORION_MARKER_NAMES):
        sg_name = ORION_TO_SG.get(orion_name)
        if sg_name and sg_name in sg_valid_names:
            sg_idx = sg_marker_names.index(sg_name)
            remapped[:, orion_idx] = targets_sg[:, sg_idx]
    return remapped


# ---------------------------------------------------------------------------
# Per-biomarker grid visualisation
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
    token_grid: int = 16,
    title: str = "",
    canvas_px: int = 2400,
    ncols: int = 4,
    dpi: int = 150,
    cmap: str = "gray",
) -> plt.Figure:
    """
    Per-biomarker slide grid.

    H&E is shown as the first panel, followed by one grayscale/colormap panel
    per selected marker — each rendered independently at slide level.

    Parameters
    ----------
    mode    : "pred" → model predictions, "gt" → ground-truth IF expression.
    ncols   : number of columns in the panel grid (H&E counts as one panel).
    cmap    : matplotlib colormap for IF channels (default: "inferno").
    """
    import math
    import cv2

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

    def _resize2(arr):
        return cv2.resize(arr.astype(np.float32), (disp_w, disp_h),
                          interpolation=cv2.INTER_NEAREST)

    he_disp = cv2.resize(he_rgb, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

    n_panels  = 1 + n_sel
    nrows     = math.ceil(n_panels / ncols)
    panel_w   = min(canvas_px / (dpi * ncols), 4.0)
    panel_h   = panel_w * disp_h / max(disp_w, 1)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_w * ncols, panel_h * nrows),
        dpi=dpi,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    axes_flat[0].imshow(he_disp, interpolation="nearest")
    axes_flat[0].set_title("H&E", fontsize=8)
    axes_flat[0].axis("off")

    for k, (_, mname) in enumerate(sel):
        ch         = mcanvases[k].copy()
        ch[bg_mask] = 0.                      # black background for unvisited regions
        ch_disp    = _resize2(np.clip(ch, 0., 1.))

        ax = axes_flat[k + 1]
        ax.imshow(ch_disp, cmap=cmap, vmin=0., vmax=1., interpolation="nearest")
        ax.set_title(mname, fontsize=8)
        ax.axis("off")

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    mode_label = "Predicted" if mode == "pred" else "Ground truth"
    suptitle   = f"{title} — {mode_label}" if title else mode_label
    fig.suptitle(suptitle, fontsize=10)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Per-slide pipeline
# ---------------------------------------------------------------------------

def process_slide(disease: str, model, out_dir: Path,
                  markers: list, rerun: bool, panel_px: int,
                  mode: str = "composite", ncols: int = 4) -> None:
    h5_path = SG_H5_DIR / f"{disease}_patch_dataset.h5"
    he_path = sg_he_path(disease)

    for p in [h5_path, he_path]:
        if not p.exists():
            print(f"  Missing {p}, skipping.")
            return

    preds_path   = out_dir / f"{disease}_preds.npy"
    targets_path = out_dir / f"{disease}_targets.npy"

    with h5py.File(h5_path) as f:
        coords            = f["coords"][:]
        targets_sg        = f["targets"][:]              # (N, C_sg, G, G)
        patch_size_level0 = int(f.attrs["patch_size_level0"])
        sg_marker_names   = list(f.attrs.get("marker_names", SINGULAR_GENOMICS_BIOMARKERS))
        valid_mask        = f["valid_markers"][:]        # (C_sg,) bool

    # Only markers whose JP2 was present; missing channels are zero-filled noise
    sg_valid_names = {name for name, ok in zip(sg_marker_names, valid_mask) if ok}
    print(f"  {len(coords)} patches, {valid_mask.sum()}/{len(sg_marker_names)} SG markers present, "
          f"patch_size_level0={patch_size_level0}")

    he_arr, _, h_ax, w_ax = open_zarr_level0(he_path)
    H, W = int(he_arr.shape[h_ax]), int(he_arr.shape[w_ax])

    if not rerun and preds_path.exists():
        print("  Loading cached predictions…")
        preds = np.load(preds_path)
    else:
        print("  Extracting H&E patches for inference…")
        patches_224 = extract_patches(he_arr, coords, patch_size_level0)
        print(f"  Running inference (device={device})…")
        preds = run_inference(model, patches_224)
        targets_remapped = remap_targets(targets_sg, sg_marker_names, sg_valid_names)
        np.save(preds_path,   preds)
        np.save(targets_path, targets_remapped)
        print(f"  Cached → {preds_path}")

    targets_remapped = remap_targets(targets_sg, sg_marker_names, sg_valid_names)

    print("  Building H&E token canvas…")
    he_tokens = extract_he_tokens(he_arr, coords, patch_size_level0, TOKEN_GRID)

    # Default: only show ORION markers that have an SG ground-truth equivalent
    if markers:
        sel = []
        for m in markers:
            if m in ORION_MARKER_NAMES:
                sel.append((ORION_MARKER_NAMES.index(m), m))
            else:
                print(f"  Warning: '{m}' not in ORION marker list, skipping.")
    else:
        sel = [
            (i, name) for i, name in enumerate(ORION_MARKER_NAMES)
            if ORION_TO_SG.get(name) and ORION_TO_SG[name] in sg_valid_names
        ]



    if mode == "composite":
        metrics = compute_metrics(preds, targets_remapped, sel)
        print_metrics(metrics, title="Metrics (token-level)")
        save_metrics_csv(metrics, out_dir / f"{disease}_metrics.csv")

        fig = make_grid_figure(
            coords, he_tokens, preds, targets_remapped,
            H, W, patch_size_level0,
            token_grid=TOKEN_GRID,
            sel=sel,
            title=disease,
            canvas_px=panel_px,
        )
        out_path = out_dir / f"{disease}-{'-'.join(s[1] for s in sel)}.png"
    else:
        fig = visualize_per_marker(
            coords, he_tokens, preds, targets_remapped,
            H, W, patch_size_level0,
            sel=sel,
            mode=mode,
            token_grid=TOKEN_GRID,
            title=disease,
            canvas_px=panel_px,
            ncols=ncols,
        )
        out_path = out_dir / f"{disease}_per_marker_{mode}.png"

    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Inference + composite visualisation for Singular Genomics slides "
                    "(ORION-trained model, ORION→SG marker remapping)."
    )
    parser.add_argument("--diseases",  nargs="+", default=["breast_cancer"],
                        metavar="DISEASE",
                        help="Disease folders to process (must have h5 and HE slide)")
    parser.add_argument("--out_dir",   default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--panel_px",  type=int, default=2400,
                        help="Canvas longest side in pixels (default: 2400)")
    parser.add_argument("--markers",   nargs="*", default=None,
                        help="ORION marker names to show (default: all with SG equivalent). "
                             f"Options: {ORION_MARKER_NAMES}")
    parser.add_argument("--rerun",     action="store_true", default=False,
                        help="Re-run inference even if cached .npy files exist")
    parser.add_argument("--model_dir", default=str(SG_MODEL_DIR))
    parser.add_argument("--mode",      default="composite",
                        choices=["composite", "pred", "gt"],
                        help="composite → colour-blended H&E/pred/GT (default); "
                             "pred → per-marker grid of predictions + H&E; "
                             "gt   → per-marker grid of ground truth + H&E")
    parser.add_argument("--ncols",     type=int, default=4,
                        help="Columns in the per-marker grid (only used with --mode pred/gt, default: 4)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_dir) / "best_model.pt"
    print(f"Loading model from {model_path}…")
    model = load_model(model_path)

    for disease in args.diseases:
        print(f"\n── {disease} ────────────────────────────────────────────")
        try:
            process_slide(disease, model, out_dir,
                          markers=args.markers, rerun=args.rerun, panel_px=args.panel_px,
                          mode=args.mode, ncols=args.ncols)
        except Exception as exc:
            import traceback
            print(f"  Error: {exc}")
            traceback.print_exc()

    print(f"\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    main()