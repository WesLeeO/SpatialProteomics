import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import math
import cv2
import h5py
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from build_patch_dataset_orion_crc_reg import crc_paths, open_zarr_level0

from model import SpatialModel

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
ORION_DATA_DIR    = Path("/mnt/ssd1/virtual_proteomics/data/ORION_CRC/data")
PATCH_DATASET_DIR = Path("orion_crc_patch_dataset_reg")
MODEL_DIR         = Path("outputs_orion_token_UNI2_finetuning")
DEFAULT_OUT_DIR   = Path("visualize_orion_out")

MODEL_NAME  = "UNI2"
NUM_OUTPUTS = 16
TOKEN_GRID  = 16
BATCH_SIZE  = 2048

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def extract_patches(he_arr, coords: np.ndarray,
                    patch_size_level0: int, model_input_size: int = 224) -> np.ndarray:
    """Crop + resize + ImageNet-normalise HE patches. Returns (N, 3, 224, 224) float32.

    he_arr is (H, W, 3) as returned by open_zarr_level0 for H&E slides.
    """
    H, W = he_arr.shape[0], he_arr.shape[1]
    patches = []
    for i, (x, y) in enumerate(coords):
        if i % 500 == 0:
            print(f'{i}/{len(coords)} done')
        x, y = int(x), int(y)
        crop = np.array(
            he_arr[y:min(y + patch_size_level0, H), x:min(x + patch_size_level0, W), :],
            dtype=np.float32,
        )
        crop = cv2.resize(crop, (model_input_size, model_input_size),
                          interpolation=cv2.INTER_LINEAR)
        crop = (crop / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        patches.append(crop.transpose(2, 0, 1))
    return np.stack(patches, axis=0).astype(np.float32)


def extract_he_tokens(he_arr, coords: np.ndarray,
                      patch_size_level0: int, token_grid: int = 16,
                      model_size: int = 224) -> np.ndarray:
    """
    Read H&E at token resolution, using the same resize path as the build script.

    The build script computes IF targets by:
      level-0 patch (patch_size_level0 × patch_size_level0)
        → cv2.resize to (224, 224)
        → reshape (16, 14, 16, 14, C).mean(axes 1,3)   [14 = 224//16]

    We apply the identical resize here so each H&E token pixel covers exactly
    the same spatial region as the corresponding prediction / ground-truth token.

    Returns (N, 3, G, G) float32 in [0, 1].
    """
    G   = token_grid
    ppb = model_size // G               # model pixels per token (= 14 for UNI2, G=16)
    H, W = he_arr.shape[0], he_arr.shape[1]
    N = len(coords)
    tokens = np.zeros((N, 3, G, G), dtype=np.float32)

    for i, (x, y) in enumerate(coords):
        if i % 500 == 0:
            print(f'{i}/{len(coords)} done')
        x, y = int(x), int(y)
        crop = np.array(
            he_arr[y:min(y + patch_size_level0, H), x:min(x + patch_size_level0, W), :],
            dtype=np.float32,
        )
        # Resize to model input size — same interpolation as the build script and extract_patches.
        # This makes the H&E display pixel-aligned with both predictions and IF targets.
        patch_224 = cv2.resize(crop, (model_size, model_size), interpolation=cv2.INTER_LINEAR)

        # Block-average into G×G tokens of ppb×ppb model pixels each.
        # Matches the exact reshape used in compute_token_grid_targets.
        tokens[i] = (
            patch_224.reshape(G, ppb, G, ppb, 3)
                     .mean(axis=(1, 3))    # (G, G, 3)
                     .transpose(2, 0, 1)   # (3, G, G)
            / 255.0
        )
    return tokens   # (N, 3, G, G) in [0, 1]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model(checkpoint: Path) -> torch.nn.Module:
    model = SpatialModel(MODEL_NAME, num_outputs=NUM_OUTPUTS,
                         token_grid=TOKEN_GRID, freeze=True, fds_cfg=None)
    state = torch.load(checkpoint, map_location=device)
    # strict=False: ignores FDS buffers present in a training checkpoint
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    print(f"  Model loaded from {checkpoint}")
    return model


def run_inference(model, patches_np: np.ndarray) -> np.ndarray:
    """Run model on (N, 3, 224, 224) float32 array. Returns (N, C, G, G) predictions."""
    dataset = TensorDataset(torch.from_numpy(patches_np))
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    preds   = []
    with torch.no_grad():
        for (batch,) in loader:
            pred, _ = model(batch.to(device))   # forward returns (preds, h)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)        # (N, C, G, G)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray, targets: np.ndarray, sel: list) -> list:
    """
    preds / targets: (N, C, G, G) float32 in [0, 1]
    sel: [(channel_idx, name), ...]
    Returns list of dicts: marker, pearson_r, psnr, ssim.
    """
    from scipy.stats import pearsonr
    from skimage.metrics import structural_similarity

    N, C, G, _ = preds.shape
    win_size = 7 if G >= 8 else (G - 1) | 1   # largest odd ≤ G-1, min 1

    rows = []
    for ch, name in sel:
        p = preds[:, ch].astype(np.float64)    # (N, G, G)
        t = targets[:, ch].astype(np.float64)  # (N, G, G)

        p_flat, t_flat = p.ravel(), t.ravel()
        if len(p_flat) < 2 or t_flat.std() < 1e-8:
            rows.append(dict(marker=name, pearson_r=np.nan, psnr=np.nan, ssim=np.nan))
            continue

        pr, _ = pearsonr(p_flat, t_flat)

        mse = float(np.mean((p_flat - t_flat) ** 2))
        psnr = 20.0 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else np.inf

        ssim_vals = [
            structural_similarity(t[i], p[i], data_range=1.0, win_size=win_size)
            for i in range(N)
        ]

        rows.append(dict(marker=name, pearson_r=float(pr),
                         psnr=float(psnr), ssim=float(np.mean(ssim_vals))))
    return rows


def print_metrics(rows: list, title: str = ""):
    if title:
        print(f"\n  {title}")
    print(f"  {'Marker':<15} {'Pearson r':>10} {'PSNR (dB)':>10} {'SSIM':>8}")
    print("  " + "-" * 45)
    for r in rows:
        pr   = f"{r['pearson_r']:.4f}" if np.isfinite(r['pearson_r']) else "    nan"
        psnr = f"{r['psnr']:.2f}"      if np.isfinite(r['psnr'])      else "    nan"
        ssim = f"{r['ssim']:.4f}"      if np.isfinite(r['ssim'])      else "    nan"
        print(f"  {r['marker']:<15} {pr:>10} {psnr:>10} {ssim:>8}")


def save_metrics_csv(rows: list, path: Path):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["marker", "pearson_r", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _marker_colors(n: int) -> np.ndarray:
    """Return (n, 3) float32 RGB colours, evenly spaced in hue via tab20."""
    cmap = plt.get_cmap("tab20", n)
    return np.array([cmap(k)[:3] for k in range(n)], dtype=np.float32)


def make_grid_figure(
    coords: np.ndarray,      # (N, 2) — (x, y) top-left at level 0
    he_tokens: np.ndarray,   # (N, 3, G, G) H&E at token resolution, [0, 1]
    preds: np.ndarray,       # (N, C, G, G) model predictions
    targets: np.ndarray,     # (N, C, G, G) ground-truth IF expression
    H: int, W: int,          # slide height / width in level-0 pixels
    patch_size_level0: int,
    sel: list,    # indices of markers to include; None → all
    token_grid: int = 16,
    title: str = "",
    canvas_px: int = 2400,
    dpi: int = 150,
) -> plt.Figure:
    """
    Build a 1-row × 3-col figure:
      col 0 — H&E downsampled to token resolution
      col 1 — predicted IF composite (selected markers, each a distinct colour)
      col 2 — ground-truth IF composite

    Each selected marker is assigned a unique colour; its expression intensity
    scales the brightness of that colour.  Markers are additively blended.

    Each canvas pixel = one UNI2 patch token = (patch_size_level0 // token_grid)
    level-0 pixels per side.
    """
    G  = token_grid
    N  = preds.shape[0]
    C  = preds.shape[1]

    # Scale factor: level-0 pixels → canvas pixels (tokens).
    # Each token covers (patch_size_level0 / G) level-0 pixels after the
    # level-0→224 resize, but that ratio is 345/16 = 21.5625 — not an integer.
    # Using integer division (// 21) would accumulate a 9-px drift by the patch
    # edge.  The exact scale G/patch_size_level0 avoids this.
    scale1 = G / patch_size_level0   # = 16/345 ≈ 0.04638

    n_sel = len(sel)
    colors = _marker_colors(n_sel)          # (num_markers, 3)


    # ── Build slide-level canvases at token resolution ───────────────────────
    canvas_h = math.ceil(H * scale1)
    canvas_w = math.ceil(W * scale1)

    he_canvas   = np.full((canvas_h, canvas_w, 3), np.nan, dtype=np.float32)
    pred_canvas = np.full((canvas_h, canvas_w, n_sel), np.nan, dtype=np.float32)
    tgt_canvas  = np.full((canvas_h, canvas_w, n_sel), np.nan, dtype=np.float32)

    for i, (x, y) in enumerate(coords):
        # Convert level-0 patch origin to canvas token coordinates.
        # round() rather than int() so sub-patch-size offsets land on the
        # nearest token rather than always rounding down.
        r0 = round(int(y) * scale1)
        c0 = round(int(x) * scale1)
        r1 = min(r0 + G, canvas_h)
        c1 = min(c0 + G, canvas_w)
        gr, gc = r1 - r0, c1 - c0   # tokens that fit (handles boundary patches)

        # he_tokens[i]: (3, G, G) → (G, G, 3)
        he_canvas[r0:r1, c0:c1, :] = he_tokens[i].transpose(1, 2, 0)[:gr, :gc]

        for k, (channel, _) in enumerate(sel):
            # preds[i, j]: (G, G)
            pred_canvas[r0:r1, c0:c1, k] = preds[i, channel, :gr, :gc]
            tgt_canvas[r0:r1, c0:c1, k]  = targets[i, channel, :gr, :gc]

    # Background mask: canvas cells never visited by any patch
    bg_mask = np.isnan(he_canvas[:, :, 0])

    # ── Build RGB composites ──────────────────────────────────────────────────
    def composite(canvas_hwk):
        """(H, W, n_sel) → (H, W, 3) additive RGB blend, white background."""
        rgb = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
        for k in range(n_sel):
            intensity = np.nan_to_num(canvas_hwk[:, :, k], nan=0.)   # (H, W)
            rgb += intensity[:, :, None] * colors[k]
        rgb = np.clip(rgb, 0., 1.)
        rgb[bg_mask] = 0.   # black for unvisited regions (protein)
        return rgb

    pred_if = composite(pred_canvas)
    tgt_if  = composite(tgt_canvas)

    he_rgb = np.nan_to_num(he_canvas, nan=1.)   # white background in H&E
    he_rgb = np.clip(he_rgb, 0., 1.)

    # ── Scale to canvas_px for display ───────────────────────────────────────
    scale2  = canvas_px / max(canvas_h, canvas_w)
    disp_h = max(1, int(round(canvas_h * scale2)))
    disp_w = max(1, int(round(canvas_w * scale2)))

    def resize3(arr_hw3):
        return cv2.resize(arr_hw3, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

    he_disp   = resize3(he_rgb)
    pred_disp = resize3(pred_if)
    tgt_disp  = resize3(tgt_if)

    # ── Figure: 1 image row + 1 legend row ───────────────────────────────────
    col_w = min(canvas_px / (dpi * 3), 6.)     # inches per panel, max 6
    col_h = col_w * disp_h / disp_w

    # Leave a little height for the legend strip
    legend_h = max(0.6, n_sel / 8)
    fig = plt.figure(figsize=(col_w * 3, col_h + legend_h), dpi=dpi)

    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[col_h, legend_h],
        hspace=0.05, wspace=0.02,
    )

    ax_he   = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_tgt  = fig.add_subplot(gs[0, 2])
    ax_leg  = fig.add_subplot(gs[1, :])

    for ax, img, ttl in [
        (ax_he,   he_disp,   "H&E"),
        (ax_pred, pred_disp, "Predicted"),
        (ax_tgt,  tgt_disp,  "Ground truth"),
    ]:
        ax.imshow(img, interpolation='nearest')
        ax.set_title(ttl, fontsize=9)
        ax.axis('off')

    # Legend: coloured swatch + marker name for each selected marker
    handles = [
        mpatches.Patch(facecolor=colors[i], label=marker_name)
        for i , (_, marker_name) in enumerate(sel)
    ]
    ax_leg.legend(
        handles=handles,
        loc='center',
        ncol=min(n_sel, 8),
        fontsize=7,
        frameon=False,
        handleheight=1.2,
        handlelength=1.5,
    )
    ax_leg.axis('off')

    if title:
        fig.suptitle(title, fontsize=9, y=1.01)

    return fig


# ---------------------------------------------------------------------------
# Per-slide pipeline
# ---------------------------------------------------------------------------

def process_slide(slide: str, model, out_dir: Path,
                  markers: list,
                  rerun: bool, panel_px: int):

    h5_path          = PATCH_DATASET_DIR / f"{slide}_patch_dataset.h5"
    he_path, if_path = crc_paths(slide)

    for p in [h5_path, he_path, if_path]:
        if not p.exists():
            print(f"  Missing {p}, skipping.")
            return

    preds_path   = out_dir / f"{slide}_preds.npy"
    targets_path = out_dir / f"{slide}_targets.npy"
    names_path   = out_dir / f"{slide}_names.npy"

    with h5py.File(h5_path) as f:
        coords            = f["coords"][:]
        targets           = f["targets"][:]              # (N, C, G, G)
        patch_size_level0 = int(f.attrs["patch_size_level0"])
        marker_names      = list(f.attrs.get("marker_names", []))

    names = marker_names
    print(f"  {len(coords)} patches, {len(names)} markers, "
          f"patch_size_level0={patch_size_level0}")

    he_arr, _, h_ax, w_ax = open_zarr_level0(he_path)
    H, W = int(he_arr.shape[h_ax]), int(he_arr.shape[w_ax])

    # ── Inference (cached) ────────────────────────────────────────────────────
    if not rerun and preds_path.exists():
        print("  Loading cached predictions …")
        preds = np.load(preds_path)
    else:
        print("  Extracting H&E patches for inference …")
        patches_224 = extract_patches(he_arr, coords, patch_size_level0)
        print(f"  Running inference (device={device}) …")
        preds = run_inference(model, patches_224)
        np.save(preds_path,   preds)
        np.save(targets_path, targets)
        np.save(names_path,   np.array(names))
        print(f"  Cached → {preds_path}")

    # ── H&E tokens from level-0 (avoids double-resize artefact) ──────────────
    print("  Building H&E token canvas …")
    he_tokens = extract_he_tokens(he_arr, coords, patch_size_level0, TOKEN_GRID)

    if markers:
        sel = []
        for m in markers:
            sel.append((names.index(m), m))
    else:
        sel = [(names.index(n), n) for n in names]

    metrics = compute_metrics(preds, targets, sel)
    print_metrics(metrics, title="Metrics (token-level)")
    save_metrics_csv(metrics, out_dir / f"{slide}_metrics.csv")

    fig = make_grid_figure(
        coords, he_tokens, preds, targets,
        H, W, patch_size_level0,
        token_grid=TOKEN_GRID,
        sel=sel,
        title=slide,
        canvas_px=panel_px
    )

    out_path = out_dir / f"{slide}-{'-'.join([s[1] for s in sel])}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference + composite visualisation for ORION slides."
    )
    parser.add_argument("--slides",   nargs="+", default=["CRC02"], metavar="SLIDE")
    parser.add_argument("--out_dir",  default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--panel_px", type=int, default=2400,
                        help="Canvas longest side in pixels (default: 2400)")
    parser.add_argument("--markers",  nargs="*", default=None,
                        help="Marker names or indices to show in composite "
                             "(default: all).  E.g. --markers CD45 CD8a Pan-CK")
    parser.add_argument("--rerun",    action="store_true", default=False,
                        help="Re-run inference even if cached .npy files exist")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {MODEL_DIR / 'best_model.pt'} …")
    model = load_model(MODEL_DIR / "best_model.pt")

    print(args.markers)

    for slide in args.slides:
        print(f"\n── {slide} ──────────────────────────────────────────────────")
        try:
            process_slide(
                slide, model, out_dir,
                markers=args.markers,
                rerun=args.rerun,
                panel_px=args.panel_px,
            )
        except Exception as exc:
            import traceback
            print(f"  Error: {exc}")
            traceback.print_exc()

    print(f"\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    main()