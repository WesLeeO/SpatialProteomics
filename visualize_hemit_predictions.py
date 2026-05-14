import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import cv2
import h5py
import re
import numpy as np
import tifffile
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict
from pathlib import Path

from visualize_orion_predictions import (
    run_inference, load_model,
    IMAGENET_MEAN, IMAGENET_STD, TOKEN_GRID,
)

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
HEMIT_H5_DIR    = Path("hemit_patch_dataset")
MODEL_DIR       = Path("outputs_orion_token_UNI2_finetuning_full")
DEFAULT_OUT_DIR = Path("visualize_hemit_out/test")

HEMIT_SOURCE_SIZE = 1024
RESIZE_TO         = 896          # resize source to this before cropping
CROP_SIZE         = 448          # px in 896-space → 224px model input
HEMIT_MARKERS     = ["Pan-CK", "CD3", "Dapi"]

ORION_MARKER_NAMES = [
    "Hoechst", "CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a",
    "CD45RO", "CD20", "PD-L1", "CD3e", "CD163", "E-Cadherin",
    "Ki-67", "Pan-CK", "SMA",
]

HEMIT_TO_ORION = {"CD3": "CD3e", "Pan-CK": "Pan-CK"}

DEFAULT_SEL = [
    (ORION_MARKER_NAMES.index("CD3e"),   "CD3e"),
    (ORION_MARKER_NAMES.index("Pan-CK"), "Pan-CK"),
]

_OFF = 0   # no offset: coords are in resized 896×896 space, origin at (0,0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

_TILE_RE = re.compile(r'^(\[[\d,]+\])_patch_(\d+)_(\d+)$')

def parse_tile_name(src_path: str) -> tuple[str, int, int]:
    """
    '[12146,53552]_patch_3_7.tif' → ('[12146,53552]', 3, 7).
    Falls back to (stem, 0, 0) if pattern doesn't match.
    """
    stem = Path(src_path).stem
    m = _TILE_RE.match(stem)
    if m:
        return m.group(1), int(m.group(2)), int(m.group(3))
    return stem, 0, 0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_split(h5_path: Path):
    """
    Returns:
      coords       (N, 2) int16
      targets      (N, C, G, G) float32
      by_slide     {slide_id: {(tile_row, tile_col): list[patch_idx]}}
      crop_size    int
    """
    with h5py.File(h5_path) as f:
        coords    = f["coords"][:]
        targets   = f["targets"][:]
        sources   = [s.decode() for s in f["sources"][:]]
        crop_size = int(f.attrs["crop_size"])

    by_slide: dict[str, dict[tuple, list]] = defaultdict(lambda: defaultdict(list))
    for idx, src in enumerate(sources):
        slide_id, tile_row, tile_col = parse_tile_name(src)
        by_slide[slide_id][(tile_row, tile_col)].append(idx)

    return coords, targets, dict(by_slide), crop_size


# ---------------------------------------------------------------------------
# Per-patch helpers
# ---------------------------------------------------------------------------

def remap_targets(targets_hemit: np.ndarray) -> np.ndarray:
    """HEMIT (N, 3, G, G) → ORION channel order (N, 16, G, G)."""
    N, _, G, _ = targets_hemit.shape
    out = np.zeros((N, len(ORION_MARKER_NAMES), G, G), dtype=np.float32)
    for hi, hname in enumerate(HEMIT_MARKERS):
        oname = HEMIT_TO_ORION.get(hname)
        if oname:
            out[:, ORION_MARKER_NAMES.index(oname)] = targets_hemit[:, hi]
    return out


def extract_patches_224(he_arr: np.ndarray, coords: np.ndarray,
                         crop_size: int) -> np.ndarray:
    patches = []
    for x, y in coords:
        x, y = int(x), int(y)
        crop = he_arr[y:y + crop_size, x:x + crop_size, :].astype(np.float32)
        crop = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        crop = (crop / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        patches.append(crop.transpose(2, 0, 1))
    return np.stack(patches, axis=0)


def he_tokens_from_arr(he_arr: np.ndarray, coords: np.ndarray,
                        crop_size: int, token_grid: int = TOKEN_GRID) -> np.ndarray:
    ppb = 224 // token_grid
    tokens = np.zeros((len(coords), 3, token_grid, token_grid), dtype=np.float32)
    for i, (x, y) in enumerate(coords):
        x, y = int(x), int(y)
        crop = he_arr[y:y + crop_size, x:x + crop_size, :].astype(np.float32)
        p224 = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR)
        tokens[i] = (
            p224.reshape(token_grid, ppb, token_grid, ppb, 3)
                .mean(axis=(1, 3))
                .transpose(2, 0, 1)
            / 255.0
        )
    return tokens


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray, targets: np.ndarray, sel: list) -> list:
    """
    preds / targets: (N, C, G, G) float32 in [0, 1]
    Returns list of dicts: marker, pearson_r, psnr.  SSIM computed on canvas separately.
    """
    from scipy.stats import pearsonr

    rows = []
    for ch, name in sel:
        p_flat = preds[:, ch].ravel().astype(np.float64)
        t_flat = targets[:, ch].ravel().astype(np.float64)
        if len(p_flat) < 2 or t_flat.std() < 1e-8:
            rows.append(dict(marker=name, pearson_r=np.nan, psnr=np.nan, ssim=np.nan))
            continue
        pr, _ = pearsonr(p_flat, t_flat)
        mse   = float(np.mean((p_flat - t_flat) ** 2))
        psnr  = 20.0 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else np.inf
        rows.append(dict(marker=name, pearson_r=float(pr), psnr=float(psnr), ssim=np.nan))
    return rows


def compute_canvas_ssim(
    pred_canvas: np.ndarray,   # (H, W, n_sel) float32
    tgt_canvas:  np.ndarray,   # (H, W, n_sel) float32
    sel: list,
) -> dict[str, float]:
    """Compute SSIM per marker on the assembled slide canvas."""
    from skimage.metrics import structural_similarity
    out = {}
    for k, (_, name) in enumerate(sel):
        p = pred_canvas[:, :, k].astype(np.float64)
        t = tgt_canvas[:,  :, k].astype(np.float64)
        if t.std() < 1e-8:
            out[name] = np.nan
            continue
        H, W = p.shape
        out[name] = float(structural_similarity(t, p, data_range=1.0, win_size=7))
    return out


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
# Slide-level canvas assembly
# ---------------------------------------------------------------------------

def _marker_colors(n: int) -> np.ndarray:
    cmap = plt.get_cmap("tab20", n)
    return np.array([cmap(k)[:3] for k in range(n)], dtype=np.float32)


def assemble_slide_canvas(
    tile_data: dict,          # {(tile_row, tile_col): (patch_coords, he_tok, preds, tgts)}
    sel: list,
    token_grid: int = TOKEN_GRID,
    crop_size: int = CROP_SIZE,
):
    """
    Build a slide-level token canvas from all tiles.

    Each 1024×1024 source tile contains 4 sub-patches (2×2) → 2G×2G token block.
    Tile at (tile_row, tile_col) occupies canvas rows [tile_row*2G : (tile_row+1)*2G]
    and canvas cols [tile_col*2G : (tile_col+1)*2G].
    Sub-patch placement within the block is derived from the stored (x, y) coords.

    Returns: he_canvas (H,W,3), pred_composite (H,W,3), tgt_composite (H,W,3),
             pred_canvas (H,W,n_sel), tgt_canvas (H,W,n_sel)
    """
    G  = token_grid
    G2 = 2 * G
    n_sel = len(sel)
    colors = _marker_colors(n_sel)

    tile_rows = [r for r, _ in tile_data]
    tile_cols = [c for _, c in tile_data]
    n_tile_rows = max(tile_rows) + 1
    n_tile_cols = max(tile_cols) + 1

    H_canvas = n_tile_rows * G2
    W_canvas = n_tile_cols * G2

    he_canvas   = np.zeros((H_canvas, W_canvas, 3),     dtype=np.float32)
    pred_canvas = np.zeros((H_canvas, W_canvas, n_sel), dtype=np.float32)
    tgt_canvas  = np.zeros((H_canvas, W_canvas, n_sel), dtype=np.float32)

    # preds 4x3xGxG
    # targets 4x3xGxG
    for (tile_row, tile_col), (patch_coords, he_tok, preds, tgts) in tile_data.items():
        # Top-left token of this tile's block in the slide canvas
        r_base = tile_row * G2
        c_base = tile_col * G2

        # Within the tile, derive each sub-patch's 2×2 position from its (x,y) coords
        x_min = int(patch_coords[:, 0].min())
        y_min = int(patch_coords[:, 1].min())

        for pi, (x, y) in enumerate(patch_coords):
            sub_col = round((int(x) - x_min) / crop_size)   # 0 or 1
            sub_row = round((int(y) - y_min) / crop_size)   # 0 or 1
            r0 = r_base + sub_row * G
            c0 = c_base + sub_col * G

            he_canvas[r0:r0+G, c0:c0+G, :] = he_tok[pi].transpose(1, 2, 0)
            for k, (ch, _) in enumerate(sel):
                pred_canvas[r0:r0+G, c0:c0+G, k] = preds[pi, ch]
                tgt_canvas[r0:r0+G, c0:c0+G, k]  = tgts[pi, ch]

    def composite(canvas_hwk):
        rgb = np.zeros((H_canvas, W_canvas, 3), dtype=np.float32)
        for k in range(n_sel):
            rgb += canvas_hwk[:, :, k:k+1] * colors[k]
        return np.clip(rgb, 0., 1.)

    return he_canvas, composite(pred_canvas), composite(tgt_canvas), pred_canvas, tgt_canvas


def make_slide_figure(
    he_canvas: np.ndarray,
    pred_composite: np.ndarray,
    tgt_composite: np.ndarray,
    sel: list,
    title: str = "",
    upscale: int = 14,
    dpi: int = 150,
) -> plt.Figure:
    colors = _marker_colors(len(sel))
    H, W = he_canvas.shape[:2]
    dH, dW = H * upscale, W * upscale

    def up(arr):
        return cv2.resize(arr, (dW, dH), interpolation=cv2.INTER_NEAREST)

    panels = [
        (up(he_canvas),      "H&E"),
        (up(pred_composite), "Predicted"),
        (up(tgt_composite),  "Ground truth"),
    ]

    fig_w = dW * 3 / dpi
    fig_h = dH / dpi
    legend_h = max(0.5, len(sel) / 8)

    fig = plt.figure(figsize=(fig_w, fig_h + legend_h), dpi=dpi)
    gs  = fig.add_gridspec(2, 3, height_ratios=[fig_h, legend_h],
                           hspace=0.05, wspace=0.02)

    for col, (img, ttl) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(img, interpolation="nearest")
        ax.set_title(ttl, fontsize=9)
        ax.axis("off")

    ax_leg = fig.add_subplot(gs[1, :])
    handles = [mpatches.Patch(facecolor=colors[i], label=name)
               for i, (_, name) in enumerate(sel)]
    ax_leg.legend(handles=handles, loc="center",
                  ncol=min(len(sel), 8), fontsize=8, frameon=False)
    ax_leg.axis("off")

    if title:
        fig.suptitle(title, fontsize=10, y=1.01)

    return fig


# ---------------------------------------------------------------------------
# Per-slide pipeline
# ---------------------------------------------------------------------------

def process_slide(
    slide_id: str,
    tile_positions: dict,   # {(tile_row, tile_col): [patch_idx, ...]}
    coords: np.ndarray,
    targets_hemit: np.ndarray,
    crop_size: int,
    model,
    out_dir: Path,
    sel: list,
    upscale: int,
    tile_step: int = 2,     # stride in tile-index space; 2 = use every 2nd tile (50% source overlap)
) -> None:
    # Keep only non-overlapping tiles: (row % tile_step == 0) and (col % tile_step == 0)
    # Value: (original_row, original_col, patch_indices) so tile stem stays stable for caching
    filtered = {
        (r // tile_step, c // tile_step): (r, c, idxs)
        for (r, c), idxs in tile_positions.items()
        if r % tile_step == 0 and c % tile_step == 0
    }

    n_tiles = len(filtered)
    print(f"  {n_tiles} non-overlapping tiles (step={tile_step})  "
          f"({max(r for r,_ in filtered)+1} rows × "
          f"{max(c for _,c in filtered)+1} cols)")

    tile_data = {}

    for (tile_row, tile_col), (orig_row, orig_col, patch_indices) in sorted(filtered.items()):
        patch_coords  = coords[patch_indices]           # (4, 2)
        patch_targets = targets_hemit[patch_indices]    # (4, 3, G, G)

        src_path = None
        for split_dir in (Path("/mnt/ssd1/virtual_proteomics/data/HEMIT") /
                          d for d in ("train", "val", "test")):
            candidate = split_dir / "input" / f"{slide_id}_patch_{orig_row}_{orig_col}.tif"
            if candidate.exists():
                src_path = str(candidate)
                break

        if src_path is None:
            print(f"    [{orig_row},{orig_col}] source TIF not found, skipping.")
            continue

        he_arr = tifffile.imread(src_path)
        if he_arr.ndim == 2:
            he_arr = np.stack([he_arr] * 3, axis=-1)
        he_arr = he_arr[:, :, :3]
        he_arr = cv2.resize(he_arr, (RESIZE_TO, RESIZE_TO), interpolation=cv2.INTER_LINEAR)

        patches_224 = extract_patches_224(he_arr, patch_coords, crop_size)
        preds  = run_inference(model, patches_224)       # (4, 16, G, G)
        tgts   = remap_targets(patch_targets)            # (4, 16, G, G)
        he_tok = he_tokens_from_arr(he_arr, patch_coords, crop_size, TOKEN_GRID)

        tile_data[(tile_row, tile_col)] = (patch_coords, he_tok, preds, tgts)

    if not tile_data:
        print("  No tiles found, skipping slide.")
        return

    all_preds = np.concatenate([d[2] for d in tile_data.values()])  # (N, 16, G, G)
    all_tgts  = np.concatenate([d[3] for d in tile_data.values()])  # (N, 16, G, G)
    metrics = compute_metrics(all_preds, all_tgts, sel)

    he_canvas, pred_comp, tgt_comp, pred_canvas, tgt_canvas = assemble_slide_canvas(
        tile_data, sel, TOKEN_GRID, crop_size
    )

    ssim_map = compute_canvas_ssim(pred_canvas, tgt_canvas, sel)
    for row in metrics:
        row['ssim'] = ssim_map.get(row['marker'], np.nan)

    print_metrics(metrics, title="Metrics (token-level, SSIM on canvas)")
    save_metrics_csv(metrics, out_dir / f"{slide_id}_metrics.csv")

    fig = make_slide_figure(
        he_canvas, pred_comp, tgt_comp,
        sel=sel, title=slide_id, upscale=upscale,
    )
    out_path = out_dir / f"{slide_id}-{'-'.join(s[1] for s in sel)}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HEMIT inference + slide-level visualisation "
                    "(ORION model, shared markers: CD3e & Pan-CK)."
    )
    parser.add_argument("--split",     default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--slides",    nargs="*", default=None,
                        help="Slide IDs to visualise, e.g. '[12146,53552]' "
                             "(default: all slides in the split)")
    parser.add_argument("--out_dir",   default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--upscale",   type=int, default=7,
                        help="Display pixels per token (default: 14 → 1px per token_px)")
    parser.add_argument("--markers",   nargs="*", default=None,
                        help=f"ORION markers to show (default: CD3e Pan-CK). "
                             f"Options: {ORION_MARKER_NAMES}")
    parser.add_argument("--tile_step",  type=int, default=2,
                        help="Use every N-th tile to avoid overlapping sources "
                             "(default 2 for HEMIT 50%% stride; set 1 for non-overlapping datasets)")
    parser.add_argument("--model_dir", default=str(MODEL_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    h5_path = HEMIT_H5_DIR / f"{args.split}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"H5 not found: {h5_path}. Run build_patch_dataset_hemit_token.py first."
        )

    coords, targets_hemit, by_slide, crop_size = load_split(h5_path)
    print(f"Loaded {args.split}: {len(coords)} patches from "
          f"{len(by_slide)} slides  (crop_size={crop_size})")

    model_path = Path(args.model_dir) / "best_model.pt"
    print(f"Loading model from {model_path}…")
    model = load_model(model_path)

    sel = DEFAULT_SEL
    if args.markers:
        sel = [(ORION_MARKER_NAMES.index(m), m) for m in args.markers
               if m in ORION_MARKER_NAMES]

    slide_ids = args.slides if args.slides else sorted(by_slide.keys())

    for slide_id in slide_ids:
        if slide_id not in by_slide:
            print(f"Slide '{slide_id}' not found in split, skipping.")
            continue
        print(f"\n── {slide_id} ─────────────────────────────────────────")
        try:
            process_slide(
                slide_id, by_slide[slide_id],
                coords, targets_hemit, crop_size,
                model, out_dir, sel,
                upscale=args.upscale,
                tile_step=args.tile_step,
            )
        except Exception as exc:
            import traceback
            print(f"  Error: {exc}")
            traceback.print_exc()

    print(f"\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    main()


# val [15379,50610] [16374,46927] [15407,49420] [16589,45753] [16770,48227] [17177,49472] [17681,52463] [17780,53623]
# train 