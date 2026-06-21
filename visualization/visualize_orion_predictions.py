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

from model import SpatialModel, NeighbourCLSModel, NeighbourhoodSpatialModel
from dataset_orion_reg import OrionSpatialDataset

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
ORION_DATA_DIR    = Path("/mnt/ssd1/virtual_proteomics/data/ORION_CRC/data")
PATCH_DATASET_DIR = Path("datasets/orion_crc_patch_dataset_benchmark")
MODEL_DIR         = Path("training_outputs/outputs_orion_token_UNI2_baseline_lora8x16mlp_2loss_lbg8_fg0")
OUTPUT_DIR = Path("training_outputs/outputs_orion_token_UNI2_baseline_lora8x16mlp_2loss_lbg8_fg0")

MODEL_NAME  = "UNI2"
NUM_OUTPUTS = 16
TOKEN_GRID  = 16
BATCH_SIZE  = 1024

# ── Neighbour model config — must match training_orion_reg.py ────────────────
# When USE_NEIGHBOURS, inference is driven through OrionSpatialDataset so the
# H&E preprocessing and the per-slide neighbour-CLS cache are byte-identical to
# training (the script's own extract_patches path can't supply neighbour CLS).
USE_NEIGHBOURS = False
NEIGHBOUR_ARCH = "cls"                       # "cls" → NeighbourCLSModel
# Set True for a model trained with MASK_NEIGHBOURS=1 (e.g. *_no_neighbours_*): neighbours
# are masked out of attention at inference too, so the run matches training. Feeding real
# neighbours to a no-neighbours model is a train/test mismatch → invalid predictions.
MASK_NEIGHBOURS = True
TIFF_DIR       = Path("/mnt/ssd1/virtual_proteomics/data/ORION_CRC")
CLS_CACHE_DIR  = str(PATCH_DATASET_DIR / "cls_cache")
NEIGHBOUR_BATCH_SIZE = 64   # forward-only UNI2 (ViT-g); keep small for inference

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

print(MASK_NEIGHBOURS)

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
    if USE_NEIGHBOURS:
        # Defaults (d_model=512, n_layers=4, n_heads=8, detach_neighbours=True) match
        # the training-time construction in training_orion_reg.py.
        NbModel = NeighbourCLSModel if NEIGHBOUR_ARCH == "cls" else NeighbourhoodSpatialModel
        model = NbModel(MODEL_NAME, num_outputs=NUM_OUTPUTS,
                        token_grid=TOKEN_GRID, unfreeze_last_n=0)
    else:
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
        for i, (batch,) in enumerate(loader):
            print(f'{i}/{len(loader)}')
            pred, _ = model(batch.to(device))   # forward returns (preds, h)
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0)        # (N, C, G, G)


def run_inference_neighbours(model, slide: str) -> np.ndarray:
    """
    Faithful neighbour-model inference for one slide.

    Drives OrionSpatialDataset (use_neighbours=True + CLS cache) so the H&E
    preprocessing and the 8 neighbour-CLS vectors are identical to training.
    The dataset iterates a single slide in h5-coords order with shuffle=False,
    so the returned (N, C, G, G) preds align row-for-row with the h5 `coords`
    and `targets` the caller reads separately.
    """
    ds = OrionSpatialDataset(
        str(PATCH_DATASET_DIR), str(TIFF_DIR),
        slide_names=[slide], augment=False,
        use_neighbours=True, cls_cache_dir=CLS_CACHE_DIR,
    )
    loader = DataLoader(ds, batch_size=NEIGHBOUR_BATCH_SIZE, shuffle=False,
                        num_workers=8, pin_memory=True)
    preds = []
    with torch.no_grad():
        for patch, _target, _mask, nbr_cls, present in loader:
            if MASK_NEIGHBOURS:
                present = torch.zeros_like(present)      # mirror MASK_NEIGHBOURS=1 training
            with torch.autocast("cuda", enabled=torch.cuda.is_available()):
                pred, _ = model(patch.to(device),
                                neighbour_cls=nbr_cls.to(device),
                                neighbour_present=present.to(device))
            preds.append(pred.float().cpu().numpy())
    return np.concatenate(preds, axis=0)        # (N, C, G, G)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(preds: np.ndarray, targets: np.ndarray, sel: list) -> list:
    """
    preds / targets: (N, C, G, G) float32 in [0, 1]
    Returns list of dicts: marker, pearson_r, psnr, ssim.  SSIM filled later from canvas.
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
        mse  = float(np.mean((p_flat - t_flat) ** 2))
        psnr = 20.0 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else np.inf
        rows.append(dict(marker=name, pearson_r=float(pr), psnr=float(psnr), ssim=np.nan))
    return rows


def compute_canvas_ssim(
    pred_canvas: np.ndarray,   # (H, W, n_sel) float32, NaN for unvisited
    tgt_canvas:  np.ndarray,   # (H, W, n_sel) float32, NaN for unvisited
    sel: list,
) -> dict[str, float]:
    """Compute SSIM per marker on the assembled slide canvas."""
    from skimage.metrics import structural_similarity
    out = {}
    for k, (_, name) in enumerate(sel):
        p = np.nan_to_num(pred_canvas[:, :, k], nan=0.).astype(np.float64)
        t = np.nan_to_num(tgt_canvas[:,  :, k], nan=0.).astype(np.float64)
        if t.std() < 1e-8:
            out[name] = np.nan
            continue
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

    he_rgb = np.nan_to_num(he_canvas, nan=1.)
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

    return fig, pred_canvas, tgt_canvas


# ---------------------------------------------------------------------------
# High-res token-pixel export (1 token → 1 pixel), per marker, per source
# ---------------------------------------------------------------------------
# All three sources are cached at token resolution (N, C, G, G), row-aligned to
# the h5 `coords`, so no inference is needed here:
#   GT     — h5 `targets`
#   mine   — MINE_PRED_DIR/<slide>_preds.npy (= MODEL_DIR, the loaded checkpoint's folder)
#   miphei — MIPHEI_CACHE_DIR/<slide>_preds.npy (MIPHEI-ViT, already token-averaged)
# `mine` preds are cached in MODEL_DIR so they always live next to (and match) the
# checkpoint that produced them — auto-computed missing preds are written here too.
MINE_PRED_DIR    = MODEL_DIR
MIPHEI_CACHE_DIR = Path("benchmarking/preds_cache/miphei-vit")
# Full (unfiltered) benchmark patch grid. When the visualised dataset is a
# *filtered* set (e.g. orion_crcv2, which drops artifact patches), this provides
# the full benchmark coords so we can render the full H&E alongside the filtered one.
BENCHMARK_DIR    = Path("datasets/orion_crc_patch_dataset_benchmark")


def _assemble_token_canvas(coords, vals, scale1, G, canvas_h, canvas_w):
    """vals: (N, G, G) one marker → (canvas_h, canvas_w) float32, NaN where unvisited.
    Each token lands on exactly one canvas pixel (token→pixel)."""
    canvas = np.full((canvas_h, canvas_w), np.nan, dtype=np.float32)
    for i, (x, y) in enumerate(coords):
        r0 = round(int(y) * scale1)
        c0 = round(int(x) * scale1)
        r1, c1 = min(r0 + G, canvas_h), min(c0 + G, canvas_w)
        canvas[r0:r1, c0:c1] = vals[i, :r1 - r0, :c1 - c0]
    return canvas


def all_benchmark_slides() -> list:
    """All slides with a benchmark h5, e.g. ['CRC01', 'CRC02', ...]."""
    return sorted(p.name[: -len("_patch_dataset.h5")]
                  for p in PATCH_DATASET_DIR.glob("*_patch_dataset.h5"))


def _compute_mine_preds(slide: str, coords: np.ndarray, psz: int,
                        model) -> np.ndarray:
    """Run inference to produce my model's (N, C, G, G) token preds for one slide."""
    if USE_NEIGHBOURS:
        return run_inference_neighbours(model, slide)
    he_path, _ = crc_paths(slide)
    he_arr, _, _, _ = open_zarr_level0(he_path)
    patches = extract_patches(he_arr, coords, psz)
    return run_inference(model, patches)


def _render_he_canvas(coords, he_arr, psz, scale1, G, canvas_h, canvas_w):
    """Assemble an H&E token canvas (1 token → 1 px) for the given coords.
    Unvisited cells stay white, so a *filtered* coord set leaves holes where
    patches were dropped. Returns (canvas_h, canvas_w, 3) float32 RGB in [0,1]."""
    he_tokens = extract_he_tokens(he_arr, coords, psz, G)              # (N, 3, G, G) in [0,1]
    canvas = np.full((canvas_h, canvas_w, 3), 1.0, dtype=np.float32)   # white bg
    for i, (x, y) in enumerate(coords):
        r0, c0 = round(int(y) * scale1), round(int(x) * scale1)
        r1, c1 = min(r0 + G, canvas_h), min(c0 + G, canvas_w)
        canvas[r0:r1, c0:c1] = he_tokens[i].transpose(1, 2, 0)[:r1 - r0, :c1 - c0]
    return canvas


def save_highres_token_images(slide: str, out_dir: Path, markers: list,
                              raw: bool = False, which: list = None,
                              get_model=None, G: int = TOKEN_GRID):
    """Write per-marker grayscale PNGs (token resolution, 1px/token) for GT,
    my model, and MIPHEI under out_dir/<slide>/{gt,mine,miphei}/<marker>.png, plus
    the H&E at the same token resolution → out_dir/<slide>/he.png.

    `which` restricts the exported sources to a subset of {gt, mine, miphei};
    None → all available.

    `get_model` is a zero-arg callable returning the loaded model. If `mine`
    preds are missing, they are computed via inference and cached to
    MINE_PRED_DIR instead of skipping the source. None → skip when missing."""
    h5_path = PATCH_DATASET_DIR / f"{slide}_patch_dataset.h5"
    if not h5_path.exists():
        print(f"  Missing {h5_path}, skipping.")
        return
    with h5py.File(h5_path) as f:
        coords  = f["coords"][:]
        targets = f["targets"][:]                       # (N, C, G, G)
        psz     = int(f.attrs["patch_size_level0"])
        names   = list(f.attrs["marker_names"])

    sources = {}
    if which is None or "gt" in which:
        sources["gt"] = targets
    for tag, d in [("mine", MINE_PRED_DIR), ("miphei", MIPHEI_CACHE_DIR)]:
        if which is not None and tag not in which:
            continue
        p = d / f"{slide}_preds.npy"
        if not p.exists():
            if tag == "mine" and get_model is not None:
                print(f"  [mine] {p} missing — running inference …")
                arr = _compute_mine_preds(slide, coords, psz, get_model())
                d.mkdir(parents=True, exist_ok=True)
                np.save(p, arr)
                print(f"  [mine] cached → {p}")
            else:
                # miphei can't be computed here; mine only skips if no model given
                print(f"  [{tag}] missing {p} — skipping that source.")
                continue
        else:
            arr = np.load(p)
        if len(arr) != len(coords):
            print(f"  [{tag}] {p} has {len(arr)} rows != {len(coords)} coords — skipping.")
            continue
        sources[tag] = arr
    print(f"  {slide}: {len(coords)} patches, sources={list(sources)}, psz_level0={psz}")

    # Full (unfiltered) benchmark coords for the second H&E version. Only loaded
    # when the visualised dataset is NOT the benchmark itself (i.e. a filtered set
    # like orion_crcv2). crcv2 coords share the benchmark level-0 space, so both
    # H&E renders use the same slide pixels and the same canvas → pixel-aligned.
    bench_coords = None
    bench_h5 = BENCHMARK_DIR / f"{slide}_patch_dataset.h5"
    if PATCH_DATASET_DIR.resolve() != BENCHMARK_DIR.resolve() and bench_h5.exists():
        with h5py.File(bench_h5) as bf:
            bench_coords = bf["coords"][:]

    # token-resolution canvas size, derived from coords (no slide I/O needed).
    # Span the union of filtered + full coords so the full H&E isn't clipped.
    scale1   = G / psz
    extent   = coords if bench_coords is None else np.vstack([coords, bench_coords])
    canvas_h = int(round(int(extent[:, 1].max()) * scale1)) + G
    canvas_w = int(round(int(extent[:, 0].max()) * scale1)) + G

    sel = ([(names.index(m), m) for m in markers] if markers
           else [(i, n) for i, n in enumerate(names)])

    # per-marker contrast scale: shared across sources so mine/miphei/gt are
    # directly comparable. Use GT's 99.5th percentile over visited tokens.
    vmax = {}
    for ch, name in sel:
        gt_canvas = _assemble_token_canvas(coords, targets[:, ch], scale1, G, canvas_h, canvas_w)
        visited = ~np.isnan(gt_canvas)
        hi = np.percentile(gt_canvas[visited], 99.5) if visited.any() else 1.0
        vmax[name] = max(float(hi), 1e-6)

    for tag, arr in sources.items():
        sdir = out_dir / slide / tag
        sdir.mkdir(parents=True, exist_ok=True)
        for ch, name in sel:
            canvas = _assemble_token_canvas(coords, arr[:, ch], scale1, G, canvas_h, canvas_w)
            img = np.nan_to_num(canvas, nan=0.0)
            img = np.clip(img if raw else img / vmax[name], 0.0, 1.0)
            fname = name.replace("/", "_") + ".png"
            cv2.imwrite(str(sdir / fname), (img * 255).astype(np.uint8))
        print(f"    saved {len(sel)} markers → {sdir}  ({canvas_w}×{canvas_h}px)")

    # H&E at the same token resolution (1 token → 1 pixel). Needs the slide
    # pixels, so this is the only part that touches the H&E zarr. We render up to
    # two versions, both on the shared canvas (so they overlay pixel-for-pixel):
    #   he_filtered.png — current dataset coords (holes where artifact patches dropped)
    #   he_full.png     — full benchmark coords (every patch / "trident output")
    # When the current dataset IS the benchmark, only he.png (full) is written.
    sdir = out_dir / slide
    sdir.mkdir(parents=True, exist_ok=True)
    he_path, _ = crc_paths(slide)
    he_arr, _, h_ax, w_ax = open_zarr_level0(he_path)

    cur_name   = "he_filtered.png" if bench_coords is not None else "he.png"
    cur_canvas = _render_he_canvas(coords, he_arr, psz, scale1, G, canvas_h, canvas_w)
    cv2.imwrite(str(sdir / cur_name), (cur_canvas[:, :, ::-1] * 255).astype(np.uint8))
    print(f"    saved H&E ({len(coords)} patches) → {sdir / cur_name}  ({canvas_w}×{canvas_h}px)")

    if bench_coords is not None:
        full_canvas = _render_he_canvas(bench_coords, he_arr, psz, scale1, G, canvas_h, canvas_w)
        cv2.imwrite(str(sdir / "he_full.png"), (full_canvas[:, :, ::-1] * 255).astype(np.uint8))
        print(f"    saved full H&E ({len(bench_coords)} patches) → {sdir / 'he_full.png'}  ({canvas_w}×{canvas_h}px)")


# ---------------------------------------------------------------------------
# Per-slide pipeline
# ---------------------------------------------------------------------------

def process_slide(slide: str, model, out_dir: Path,
                  markers: list,
                  panel_px: int):

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

    # ── Inference (cached: reuse saved preds, else compute) ───────────────────
    # Delete the slide's *_preds.npy to force re-inference (e.g. after changing
    # the model / checkpoint).
    if preds_path.exists():
        print(f"  Loading cached predictions ← {preds_path}")
        preds = np.load(preds_path)
        assert len(preds) == len(coords), (
            f"{slide}: cached preds {len(preds)} != coords {len(coords)} — stale cache, delete {preds_path}")
    elif USE_NEIGHBOURS:
        print(f"  Running neighbour inference via OrionSpatialDataset (device={device}) …")
        preds = run_inference_neighbours(model, slide)
        assert len(preds) == len(coords), (
            f"{slide}: preds {len(preds)} != coords {len(coords)} — dataset/h5 order mismatch")
        np.save(preds_path,   preds)
        np.save(targets_path, targets)
        np.save(names_path,   np.array(names))
        print(f"  Cached → {preds_path}")
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

    fig, pred_canvas, tgt_canvas = make_grid_figure(
        coords, he_tokens, preds, targets,
        H, W, patch_size_level0,
        token_grid=TOKEN_GRID,
        sel=sel,
        title=slide,
        canvas_px=panel_px
    )

    ssim_map = compute_canvas_ssim(pred_canvas, tgt_canvas, sel)
    for row in metrics:
        row['ssim'] = ssim_map.get(row['marker'], np.nan)

    print_metrics(metrics, title="Metrics (SSIM on slide canvas)")
    tag = '-'.join(markers) if markers else "all"
    save_metrics_csv(metrics, out_dir / f"{slide}_{tag}_metrics.csv")

    out_path = out_dir / f"{slide}-{tag}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference + composite visualisation for ORION slides."
    )
    parser.add_argument("--slides",   nargs="+", default=None, metavar="SLIDE",
                        help="Slides to process. Default: CRC02 for the composite "
                             "figure; ALL benchmark slides for --highres.")
    parser.add_argument("--out_dir",  default=OUTPUT_DIR,
                        help="Output directory (default: visualize_orion_out/full/<slide> "
                             "when a single slide is given, else visualize_orion_out/full)")
    parser.add_argument("--panel_px", type=int, default=2400,
                        help="Canvas longest side in pixels (default: 2400)")
    parser.add_argument("--markers",  nargs="*", default=None,
                        help="Marker names or indices to show in composite "
                             "(default: all).  E.g. --markers CD45 CD8a Pan-CK")
    parser.add_argument("--highres", action="store_true",
                        help="Export per-marker grayscale token-pixel images (1 token → 1 "
                             "pixel) for GT, my model, and MIPHEI from cached token preds "
                             "(no inference). Saves to <out_dir>/<slide>/{gt,mine,miphei}/.")
    parser.add_argument("--raw", action="store_true",
                        help="With --highres: skip the per-marker contrast stretch "
                             "(write clipped [0,1]→[0,255] values directly).")
    parser.add_argument("--sources", nargs="+", default=None,
                        choices=["gt", "mine", "miphei"],
                        help="With --highres: which sources to export "
                             "(default: all available). E.g. --sources gt mine")
    parser.add_argument("--patch_dataset_dir", default=None,
                        help="Override PATCH_DATASET_DIR (e.g. orion_crcv2_patch_dataset "
                             "to QC the new tile20x GT).")
    args = parser.parse_args()

    if args.patch_dataset_dir:
        global PATCH_DATASET_DIR
        PATCH_DATASET_DIR = Path(args.patch_dataset_dir)
        print(f"PATCH_DATASET_DIR -> {PATCH_DATASET_DIR}")

    if args.highres:
        # use a dedicated default dir unless the user explicitly overrode --out_dir
        out_dir = (Path("highres_token_out") if args.out_dir in (None, OUTPUT_DIR)
                   else Path(args.out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        slides = args.slides if args.slides is not None else all_benchmark_slides()
        print(f"Slides ({len(slides)}): {', '.join(slides)}")

        # Lazy model loader: only construct/load the model the first time a slide
        # is actually missing its `mine` preds (otherwise this is a no-inference run).
        _model_cache = {}
        def get_model():
            if "m" not in _model_cache:
                print(f"Loading model from {MODEL_DIR / 'best_model.pt'} …")
                _model_cache["m"] = load_model(MODEL_DIR / "best_model.pt")
            return _model_cache["m"]

        for slide in slides:
            print(f"\n── {slide} (highres token-pixel export) ──────────────────────")
            save_highres_token_images(slide, out_dir, markers=args.markers,
                                      raw=args.raw, which=args.sources,
                                      get_model=get_model)
        print(f"\nDone. Output in: {out_dir}")
        return

    # composite-figure mode keeps its single-slide default
    if args.slides is None:
        args.slides = ["CRC02"]

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif len(args.slides) == 1:
        out_dir = Path("visualization_out/orion/full") / args.slides[0]
    else:
        out_dir = Path("visualization_out/orion/full")
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
                panel_px=args.panel_px,
            )
        except Exception as exc:
            import traceback
            print(f"  Error: {exc}")
            traceback.print_exc()

    print(f"\nDone. Output in: {out_dir}")


if __name__ == "__main__":
    main()