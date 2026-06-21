"""
Multi-model token-level benchmark on one or more ORION slides.

Every model is compared at YOUR token resolution (16×16 grid per patch) against
the same token-grid GT from the H5 dataset:

  • Yours (UNI2 finetune-4) — token preds reused from the cached
                             outputs_orion_token_UNI2_baseline_bg0.2/<SLIDE>_preds.npy
                             (override dir/label with --your_preds / --your_label;
                             dump them first via visualize_orion_predictions.py)
  • Pixel baselines        — run at their NATIVE 256 px / 0.5 µm/px: feed the 128 µm
    (MIPHEI-vit, Pix2Pix,    context window, center-crop the 256 px output back to the
     HEMIT, DiffusionFT)     patch's 112 µm FOV, average 14×14 blocks → 16×16 tokens.
  • Rosie-ORION            — patch-level scalar output, broadcast over the 16×16 grid.
  • MIPHEI-ConvNeXt        — needs segmentation_models_pytorch==0.4.0, which conflicts
                             with this env's smp 0.5.0, so it can't be loaded in-process.
                             It runs in a dedicated conda env (`smp040`) via
                             gen_convnext_preds.py, which reproduces the same pixel→token
                             pipeline and dumps convnext_preds/<SLIDE>_preds.npy. When you
                             request `miphei-convnext`, this script auto-invokes that
                             generator (conda run -n smp040 …) for any missing slides, then
                             loads the cached preds as an extra bar — no manual step needed.

Metrics: per-marker Pearson r on ALL tokens (PCC) AND on GT-positive tokens only
(PCC+, gt > 0). Global PCC is dominated by the near-0 background mass for sparse markers
— where the model's inevitable noise-around-0 lives — so it mostly measures bg↔fg
separation; PCC+ drops the exactly-0 background and scores signal fidelity where the
marker is actually present. The `> 0` cut is GT-only and not the `> token-mean` boundary
the model trained on, so the comparison is fair (no circularity). Each slide produces
its own PCC and PCC+ plot + csv (the
latter suffixed `_pccplus`); when more than one slide is given, pooled summaries (over
all slides' tokens concatenated) are also written for both. Each baseline is loaded in
its own try/except: a model that fails to load/run is skipped and noted, the rest still
produce results.

Caching: every model's full-slide token preds are written to
preds_cache/<model_key>/<slide>_preds.npy on first run and reloaded on later runs, so
re-benchmarking (or adding one new model) is near-instant — only cache-miss models run
inference. Caches are full N in coords order and independent of --max_patches, so they
stay valid across subsamples. Use --refresh to ignore + overwrite them (e.g. after
retraining a model or changing its checkpoint).

Timing: when a model actually runs inference, the whole-slide end-to-end wall-clock
(read + preprocess + forward + tokenize) is saved as a sidecar next to its cache
(preds_cache/<key>/<slide>_time.txt). At the end of every run a per-model inference-time
table is printed and written to results/timing.csv — it persists across cached re-runs,
so you don't have to re-infer to see it (use --refresh to re-measure).

Usage
-----
  python benchmark_all_slides.py --slides CRC19
  python benchmark_all_slides.py --slides CRC19,CRC18,CRC17 --max_patches 3000
  python benchmark_all_slides.py --slides CRC19 --models miphei-vit,pix2pix,hemit
  python benchmark_all_slides.py --slides CRC19 --models miphei-convnext   # auto-runs smp040 env
  python benchmark_all_slides.py --slides CRC19 --refresh                  # force re-inference

The dedicated env is created once with (see gen_convnext_preds.py for the full list):
  conda create -y -n smp040 python=3.10
  conda run -n smp040 pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  conda run -n smp040 pip install "segmentation_models_pytorch==0.4.0" timm safetensors \
              tifffile imagecodecs zarr h5py opencv-python-headless numpy
"""

import os
import sys
import time
import argparse
import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import h5py
import cv2
import torch
import tifffile
import zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "MIPHEI-ViT"))

BENCHMARKING_DIR = Path(__file__).resolve().parent.parent
H5_DIR        = REPO / "datasets/orion_crc_patch_dataset_benchmark"
TIFF_DIR      = Path("/mnt/ssd/virtual_proteomics/data/ORION_CRC")
# "Ours" = the UNI2 finetune-last-4 baseline (matched-comparison winner; cls-neighbours
# were ruled out as null). Overridable with --your_preds / --your_label. Per-slide
# {SLIDE}_preds.npy must already be dumped here (visualize_orion_predictions.py).
PRED_DIR = REPO / "training_outputs/outputs_orion_token_UNI2_baseline_bg0.2"
LABEL    = "UNI2 finetune-4 (ours)"

# Every model's token preds are cached to disk so re-runs (and the slow pixel /
# diffusion models) only pay inference ONCE. Layout: preds_cache/<key>/<slide>_preds.npy,
# full N in coords order (independent of --max_patches, so a cache is reusable across
# any subsample). In-harness models (REGISTRY) write their cache after inference;
# the separate-env model (convnext) writes it from gen_convnext_preds.py. Pass
# --refresh to ignore + overwrite existing caches.
PRED_CACHE    = BENCHMARKING_DIR / "preds_cache"

# Models whose deps conflict with this env are run in an ISOLATED conda env and their
# cache is generated there. MIPHEI-ConvNeXt needs segmentation_models_pytorch==0.4.0
# (this env has 0.5.0; the CenterBlock/Conv2dReLU API changed), so it runs in the
# `smp040` env via gen_convnext_preds.py. Each entry maps a display label to:
#   pred_dir : where {slide}_preds.npy land (under PRED_CACHE)
#   env      : conda env that has the right deps (None → no auto-generation)
#   script   : generator script (run as `conda run -n <env> python <script> --slides …`)
CACHED_PREDS = {
    "MIPHEI-ConvNeXt": dict(
        pred_dir = PRED_CACHE / "miphei-convnext",
        env      = "smp040",
        script   = BENCHMARKING_DIR / "gen_convnext_preds.py",
    ),
}
RESULTS_DIR   = BENCHMARKING_DIR / "results_baseline"


def ensure_cached_preds(label, cfg, slides):
    """Make sure {slide}_preds.npy exists for every requested slide, generating any
    that are missing by invoking the model's generator script inside its dedicated
    conda env. Returns the list of slides whose preds are available afterwards."""
    pred_dir = Path(cfg["pred_dir"])
    missing  = [s for s in slides if not (pred_dir / f"{s}_preds.npy").exists()]
    if not missing:
        return slides
    if not cfg.get("env") or not cfg.get("script"):
        print(f"  [cached] {label}: missing {missing} and no env/script configured — skipping those")
        return [s for s in slides if s not in missing]

    cmd = ["conda", "run", "--no-capture-output", "-n", cfg["env"],
           "python", str(cfg["script"]), "--slides", ",".join(missing)]
    print(f"\n[{label}] cached preds missing for {missing} → generating in env "
          f"'{cfg['env']}':\n  {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [SKIP] {label}: generation failed ({type(e).__name__}: {e}) — "
              f"is the '{cfg['env']}' env created? See gen_convnext_preds.py header.")
    return [s for s in slides if (pred_dir / f"{s}_preds.npy").exists()]

MPP_HE     = 0.325
TOKEN_GRID = 16
NUM_OUTPUTS = 16
NATIVE_IMG = 256          # all ORION pixel baselines are 256 px @ 0.5 µm/px
NATIVE_MPP = 0.5

# Rosie: predicts an 8×8-px output cell from a 128×128-px context window (sliding
# window, ConvNeXt). ROSIE_MPP is its working resolution (20× ≈ 0.5 µm/px, same
# ORION tiling as the other baselines) — adjust if the paper used a different MPP.
ROSIE_MPP = 0.5
ROSIE_P   = 128
ROSIE_S   = 8

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PIX2PIX_MEAN  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
PIX2PIX_STD   = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# key → (display, ckpt_subdir, mean, std, output_type)
#   output_type: "pixel" (B,C,256,256) Tanh | "scalar" (B,C) patch-level
REGISTRY = {
    "miphei-vit":      ("MIPHEI-ViT",      "MIPHEI-vit",      IMAGENET_MEAN, IMAGENET_STD, "pixel"),
    "pix2pix":         ("Pix2Pix",         "Pix2Pix",         PIX2PIX_MEAN,  PIX2PIX_STD,  "pixel"),
    "hemit":           ("HEMIT",           "HEMIT",           PIX2PIX_MEAN,  PIX2PIX_STD,  "pixel"),
    #"rosie":           ("Rosie-ORION",     "Rosie-ORION",     IMAGENET_MEAN, IMAGENET_STD, "scalar"),
    # DiffusionFT: 1-step HE→MIF Stable-Diffusion pipeline; fed in [-1,1], outputs (B,16,256,256).
    #"diffusion":       ("DiffusionFT",     "DiffusionFT",     PIX2PIX_MEAN,  PIX2PIX_STD,  "pixel"),
    # NOTE: MIPHEI-ConvNeXt is NOT here — it needs segmentation_models_pytorch==0.4.0
    # (this env has 0.5.0; CenterBlock/Conv2dReLU API changed), so it can't be loaded
    # in-process. It runs via CACHED_PREDS["MIPHEI-ConvNeXt"] in the smp040 env instead
    # (model key "miphei-convnext"). The load_model() branch below is kept only as a
    # reference implementation for that separate-env script.
}


# ── Loaders ─────────────────────────────────────────────────────────────────--
# Each model is loaded directly from its source file via importlib, bypassing the
# MIPHEI-ViT `benchmark.models` package __init__ (which drags in pytorch_lightning,
# an incompatible segmentation_models_pytorch, xgboost, diffusers, …). The
# individual model files only need torch/timm/torchvision/einops.

def _load_file(name, relpath):
    spec = importlib.util.spec_from_file_location(name, str(REPO / "MIPHEI-ViT" / relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _adapt_hemit_statedict(state_dict, model):
    """Inlined from MIPHEI benchmark/evaluators/utils.adapt_checkpoint_hemit (which
    otherwise pulls in xgboost/sklearn). Renames downsample keys, drops index/mask
    buffers, and resamples patch-embed / rel-pos-bias for the target img_size."""
    from timm.layers import resample_patch_embed, resize_rel_pos_bias_table
    sd = {}
    for k, v in state_dict.items():
        if ".downsample.norm" in k or "downsample.reduction" in k:
            ks = k.split("."); ks[2] = str(int(ks[2]) + 1); nk = ".".join(ks)
        elif "relative_position_index" in k or "attn_mask" in k:
            continue
        else:
            nk = k
        sd[nk] = v
    out = {}
    for k, v in sd.items():
        if any(n in k for n in ("relative_position_index", "attn_mask")):
            continue
        if "swinT.patch_embed.proj.weight" in k:
            _, _, H, W = model.swinT.patch_embed.proj.weight.shape
            if v.shape[-2] != H or v.shape[-1] != W:
                v = resample_patch_embed(v, (H, W), interpolation="bicubic", antialias=True, verbose=True)
        if k.endswith("relative_position_bias_table"):
            m = model.get_submodule(k[:-29])
            if v.shape != m.relative_position_bias_table.shape or m.window_size[0] != m.window_size[1]:
                v = resize_rel_pos_bias_table(v, new_window_size=m.window_size,
                                              new_bias_shape=m.relative_position_bias_table.shape)
        out[k] = v
    return out


def load_model(key, device):
    import torch.nn as nn
    ckpt = BENCHMARKING_DIR / REGISTRY[key][1]

    if key == "miphei-vit":
        spec = importlib.util.spec_from_file_location("miphei_ckpt_model", str(ckpt / "model.py"))
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod.MIPHEIViT.from_pretrained_hf(repo_path=str(ckpt)).to(device).eval()

    if key == "pix2pix":
        p2p = _load_file("p2p_mod", "benchmark/models/pix2pix.py")
        model = p2p.UnetGenerator(3, NUM_OUTPUTS, 8, ngf=64,
                                  norm_layer=nn.BatchNorm2d, use_dropout=False)
        sd = torch.load(ckpt / "best.pth", map_location="cpu")
        model.load_state_dict(sd)
        return model.to(device).eval()

    if key == "hemit":
        hm = _load_file("hemit_mod", "src/generators/hemit_models.py")
        model = hm.ResnetGeneratorSwinT(input_nc=3, output_nc=NUM_OUTPUTS,
                                        img_size=[NATIVE_IMG, NATIVE_IMG], patch_size=4,
                                        window_size=8, depths=[2, 2, 6, 2], embed_dim=96)
        sd = torch.load(ckpt / "best.pth", map_location="cpu")
        sd = _adapt_hemit_statedict(sd, model)
        sd = hm.resize_embed_hemit_statedict(sd, model)
        model.load_state_dict(sd)
        return model.to(device).eval()

    if key == "rosie":
        ros = _load_file("rosie_mod", "benchmark/models/rosie.py")
        model = ros.get_model(num_outputs=NUM_OUTPUTS)
        sd = torch.load(ckpt / "rosie.pth", map_location="cpu")
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
        return model.to(device).eval()

    if key == "diffusion":
        # MIPHEI's 1-step HE→MIF Stable-Diffusion pipeline. load_diffusion_pipeline
        # assembles vae+unet+text_encoder+scheduler and returns a callable pipeline
        # whose __call__(rgb, return_torch=True) → (B, 16, H, W).
        dm   = _load_file("diffusionft_mod", "benchmark/models/diffusionft.py")
        pipe = dm.load_diffusion_pipeline(str(ckpt), device)

        class _DiffusionWrap(nn.Module):
            """Adapt the pipeline to the (B,C,256,256) pixel path. Input rgb is in
            [-1,1] (PIX2PIX_MEAN/STD); forced to fp32 since diffusers modules and
            autocast don't mix cleanly."""
            def forward(self, x):
                with torch.autocast("cuda", enabled=False):
                    return pipe(x.float(), return_torch=True)

        return _DiffusionWrap().to(device).eval()

    if key == "miphei-convnext":
        # MIPHEI's smp U-Net with ConvNeXt-Large encoder and 16 attention-gated
        # per-marker heads (UnetMultiHeads). Tanh output → flows through the pixel
        # path. encoder_weights=None: we load the trained weights, no ImageNet pull.
        su  = _load_file("smp_unet_mod", "src/generators/smp_unet.py")
        net = su.UnetMultiHeads(
            encoder_name="tu-convnext_large", encoder_weights=None,
            decoder_use_batchnorm=True, in_channels=3, classes=NUM_OUTPUTS,
            activation=torch.nn.Tanh, dropout=0.1, use_attention=True)
        from safetensors.torch import load_file
        sd = load_file(str(ckpt / "model.safetensors"))
        sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
        missing, unexpected = net.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  [convnext] missing={len(missing)} unexpected={len(unexpected)}")
        return net.to(device).eval()

    raise ValueError(key)


# ── Slide access ───────────────────────────────────────────────────────────--

def open_he_zarr(slide):
    matches = list((TIFF_DIR / slide).glob("*-registered.ome.tif"))
    if not matches:
        raise FileNotFoundError(f"No *-registered.ome.tif in {TIFF_DIR/slide}")
    tif   = tifffile.TiffFile(str(matches[0]))
    store = zarr.LRUStoreCache(tif.aszarr(), max_size=512 * 2**20)
    z     = zarr.open(store, mode="r")
    return z["0"] if isinstance(z, zarr.hierarchy.Group) else z


def read_region(arr, x0, y0, size):
    """(size,size,3) uint8 at level-0 origin (x0,y0), zero-padded out of bounds."""
    if arr.shape[-1] <= 4:
        H, W, chan_last = arr.shape[0], arr.shape[1], True
    else:
        H, W, chan_last = arr.shape[1], arr.shape[2], False
    out = np.zeros((size, size, 3), dtype=np.uint8)
    sx, sy = max(0, x0), max(0, y0)
    ex, ey = min(W, x0 + size), min(H, y0 + size)
    if ex <= sx or ey <= sy:
        return out
    crop = (np.asarray(arr[sy:ey, sx:ex, :3]) if chan_last
            else np.asarray(arr[:3, sy:ey, sx:ex]).transpose(1, 2, 0))
    out[sy - y0:ey - y0, sx - x0:ex - x0] = crop
    return out


# ── Inference: one model over all coords → (N, C, 16, 16) tokens ─────────────--

def infer_model(model, otype, mean, std, arr, coords, psz, device, batch=8):
    ctx_psz = int(round(NATIVE_IMG * NATIVE_MPP / MPP_HE))             # ≈394
    crop_px = int(round(psz * MPP_HE / (ctx_psz * MPP_HE) * NATIVE_IMG))  # ≈224
    off     = (NATIVE_IMG - crop_px) // 2
    tok_px  = crop_px // TOKEN_GRID
    crop_px = tok_px * TOKEN_GRID

    in_size = NATIVE_IMG if otype == "pixel" else 224   # Rosie (convnext) trained at 224
    out = np.empty((len(coords), NUM_OUTPUTS, TOKEN_GRID, TOKEN_GRID), dtype=np.float32)
    buf, idxs = [], []

    def flush():
        if not buf:
            return
        a = np.stack(buf).astype(np.float32) / 255.0
        a = (a - mean) / std
        t = torch.from_numpy(a.transpose(0, 3, 1, 2)).to(device)
        with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda"):
            pred = model(t)
        pred = pred.float()
        if otype == "pixel":
            # Tanh range → p99-log1p [0,1] (affine; irrelevant for Pearson but keeps scale sane)
            pred = ((pred.clamp(-0.9, 0.9) + 0.9) / 1.8)
            pred = pred[:, :, off:off + crop_px, off:off + crop_px]
            B, C = pred.shape[:2]
            tok = pred.reshape(B, C, TOKEN_GRID, tok_px, TOKEN_GRID, tok_px).mean((3, 5))
            out[np.array(idxs)] = tok.cpu().numpy()
        else:  # scalar (B, C) → broadcast over grid
            v = pred.reshape(pred.shape[0], NUM_OUTPUTS, 1, 1).cpu().numpy()
            out[np.array(idxs)] = np.broadcast_to(v, (len(idxs), NUM_OUTPUTS, TOKEN_GRID, TOKEN_GRID))
        buf.clear(); idxs.clear()

    for i, (x, y) in enumerate(coords):
        x, y = int(x), int(y)
        if otype == "pixel":
            cx0 = int(round(x + psz / 2 - ctx_psz / 2))
            cy0 = int(round(y + psz / 2 - ctx_psz / 2))
            region = read_region(arr, cx0, cy0, ctx_psz)
        else:  # scalar models see the patch's own FOV
            region = read_region(arr, x, y, psz)
        region = cv2.resize(region, (in_size, in_size), interpolation=cv2.INTER_LINEAR)
        buf.append(region); idxs.append(i)
        if len(buf) >= batch:
            flush()
        if i % 1000 == 0:
            print(f"      {i}/{len(coords)}", flush=True)
    flush()
    return out


def infer_rosie(model, mean, std, arr, coords, psz, device, batch=8):
    """
    Faithful Rosie inference via its own sliding-window (8×8 cell from 128×128
    context). We feed a context-padded region at ROSIE_MPP so the valid output
    map covers the patch's FOV, then area-resample that map to the 16×16 grid.
    """
    ros = _load_file("rosie_mod", "benchmark/models/rosie.py")
    patch_um = psz * MPP_HE
    in_px = int(round(patch_um / ROSIE_MPP)) + ROSIE_P     # patch FOV + context margin
    in_px = ((in_px + ROSIE_S - 1) // ROSIE_S) * ROSIE_S    # tidy to multiple of stride
    in_l0 = int(round(in_px * ROSIE_MPP / MPP_HE))          # level-0 px to read
    out = np.empty((len(coords), NUM_OUTPUTS, TOKEN_GRID, TOKEN_GRID), dtype=np.float32)
    buf, idxs = [], []

    def flush():
        if not buf:
            return
        a = np.stack(buf).astype(np.float32) / 255.0
        a = (a - mean) / std
        t = torch.from_numpy(a.transpose(0, 3, 1, 2)).to(device)
        g = ros.infer_sliding_window(t, model, P=ROSIE_P, S=ROSIE_S)  # (B,C,n,n) uint8 cpu
        g = g.float().to(device) / 255.0
        tok = torch.nn.functional.interpolate(g, size=(TOKEN_GRID, TOKEN_GRID), mode="area")
        out[np.array(idxs)] = tok.cpu().numpy()
        buf.clear(); idxs.clear()

    for i, (x, y) in enumerate(coords):
        x, y = int(x), int(y)
        cx0 = int(round(x + psz / 2 - in_l0 / 2))
        cy0 = int(round(y + psz / 2 - in_l0 / 2))
        region = read_region(arr, cx0, cy0, in_l0)
        region = cv2.resize(region, (in_px, in_px), interpolation=cv2.INTER_LINEAR)
        buf.append(region); idxs.append(i)
        if len(buf) >= batch:
            flush()
        if i % 200 == 0:
            print(f"      rosie {i}/{len(coords)}", flush=True)
    flush()
    return out


# ── Metric + output helpers ─────────────────────────────────────────────────--

def flat(x):
    """(M, C, G, G) → (M*G*G, C)."""
    M, C, G, _ = x.shape
    return x.transpose(0, 2, 3, 1).reshape(-1, C)


def pearson_per_marker(pred, gt, n_markers):
    pf, gf = flat(pred), flat(gt)
    return [pearsonr(pf[:, c], gf[:, c])[0] for c in range(n_markers)]


class PooledPearson:
    """Streaming per-marker Pearson accumulator, so pooling across slides doesn't
    require holding every slide's tokens in memory at once."""
    def __init__(self, n_markers):
        z = np.zeros(n_markers, dtype=np.float64)
        self.n = 0.0
        self.sx, self.sy = z.copy(), z.copy()
        self.sxx, self.syy, self.sxy = z.copy(), z.copy(), z.copy()

    def update(self, pred, gt):
        pf, gf = flat(pred).astype(np.float64), flat(gt).astype(np.float64)
        self.n   += pf.shape[0]
        self.sx  += pf.sum(0);         self.sy  += gf.sum(0)
        self.sxx += (pf * pf).sum(0);  self.syy += (gf * gf).sum(0)
        self.sxy += (pf * gf).sum(0)

    def pearson(self):
        n = self.n
        cov = n * self.sxy - self.sx * self.sy
        dx  = np.sqrt(np.clip(n * self.sxx - self.sx ** 2, 0, None))
        dy  = np.sqrt(np.clip(n * self.syy - self.sy ** 2, 0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = cov / (dx * dy)
        return list(np.where((dx * dy) > 0, r, np.nan))


def positive_threshold(markers):
    """Per-marker 'positive' threshold for PCC+ = strictly above 0 (GT == 0 are the
    background tokens that clip to the floor; GT > 0 is any present signal). We use 0
    rather than the training token-mean ON PURPOSE: the model trained with the
    `> token-mean` mask, so scoring PCC+ at that exact boundary risks flattering it (and
    is asymmetric vs MIPHEI, which never saw that mask). `> 0` is a GT-only boundary the
    model never trained on, so the ours-vs-competitor comparison is unambiguously fair.
    PCC+ (Pearson on positive tokens only) drops the huge mass of exactly-0 background
    that dominates global PCC for sparse markers — where the model's inevitable
    noise-around-0 sits — and isolates signal fidelity where the marker is present."""
    print("  PCC+ threshold: GT > 0 (non-circular; model trained on >mean, not >0)")
    return np.zeros(len(markers), dtype=np.float64)


def pearson_pos_per_marker(pred, gt, thr, min_tokens=50):
    """Per-marker Pearson on GT-positive tokens only (gt > thr[c]). NaN where a marker has
    too few positive tokens or no spread (the metric is undefined / pure noise there)."""
    pf, gf = flat(pred), flat(gt)
    out = []
    for c in range(pf.shape[1]):
        m = gf[:, c] > thr[c]
        x, y = pf[m, c], gf[m, c]
        if m.sum() < min_tokens or x.std() < 1e-8 or y.std() < 1e-8:
            out.append(np.nan)
        else:
            out.append(pearsonr(x, y)[0])
    return out


class PooledPearsonPos:
    """Streaming per-marker Pearson over each marker's GT-positive tokens (gt > thr[c]).
    The positive set differs per marker, so sums are accumulated per channel."""
    def __init__(self, n_markers, thr, min_tokens=50):
        z = np.zeros(n_markers, dtype=np.float64)
        self.thr = np.asarray(thr, dtype=np.float64)
        self.min_tokens = min_tokens
        self.n = z.copy()
        self.sx, self.sy = z.copy(), z.copy()
        self.sxx, self.syy, self.sxy = z.copy(), z.copy(), z.copy()

    def update(self, pred, gt):
        pf, gf = flat(pred).astype(np.float64), flat(gt).astype(np.float64)
        for c in range(pf.shape[1]):
            m = gf[:, c] > self.thr[c]
            if not m.any():
                continue
            x, y = pf[m, c], gf[m, c]
            self.n[c]   += x.shape[0]
            self.sx[c]  += x.sum();        self.sy[c]  += y.sum()
            self.sxx[c] += (x * x).sum();  self.syy[c] += (y * y).sum()
            self.sxy[c] += (x * y).sum()

    def pearson(self):
        n = self.n
        cov = n * self.sxy - self.sx * self.sy
        dx  = np.sqrt(np.clip(n * self.sxx - self.sx ** 2, 0, None))
        dy  = np.sqrt(np.clip(n * self.syy - self.sy ** 2, 0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = cov / (dx * dy)
        ok = (dx * dy > 0) & (n >= self.min_tokens)
        return list(np.where(ok, r, np.nan))


# ── Patch-bootstrap of per-marker Pearson (CI across all slides' patches) ───────--
# Resampling unit = the PATCH (a model's whole 16×16 token grid). For each patch we
# precompute Pearson's sufficient statistics per marker (n, Σx, Σy, Σx², Σy², Σxy)
# so a bootstrap draw is just a weighted sum of those rows → re-deriving r for the
# resampled token pool. Two stat sets per model: PCC (all tokens) and PCC+ (gt>0).

def patch_stats(pred, gt):
    """pred, gt: (N, C, G, G). Returns (all_stats, pos_stats); each a dict of (N, C)
    float64 arrays {n, sx, sy, sxx, syy, sxy}. Summing rows over any patch set and
    feeding _pearson_from_sums gives that set's per-marker Pearson."""
    N, C = pred.shape[:2]
    p = np.asarray(pred, np.float32).reshape(N, C, -1)        # (N, C, T)
    g = np.asarray(gt,   np.float32).reshape(N, C, -1)
    T = p.shape[2]
    f = np.float64
    alls = dict(n=np.full((N, C), T, f),
                sx=p.sum(2, dtype=f),     sy=g.sum(2, dtype=f),
                sxx=(p * p).sum(2, dtype=f), syy=(g * g).sum(2, dtype=f),
                sxy=(p * g).sum(2, dtype=f))
    m = g > 0.0                                               # PCC+ positive mask (gt>0)
    pos = dict(n=m.sum(2, dtype=f),
               sx=(p * m).sum(2, dtype=f),     sy=(g * m).sum(2, dtype=f),
               sxx=(p * p * m).sum(2, dtype=f), syy=(g * g * m).sum(2, dtype=f),
               sxy=(p * g * m).sum(2, dtype=f))
    return alls, pos


def _pearson_from_sums(n, sx, sy, sxx, syy, sxy, min_n=1):
    cov = n * sxy - sx * sy
    dx  = np.sqrt(np.clip(n * sxx - sx * sx, 0, None))
    dy  = np.sqrt(np.clip(n * syy - sy * sy, 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / (dx * dy)
    return np.where((dx * dy > 0) & (n >= min_n), r, np.nan)


def bootstrap_ci(stats, n_boot, seed, min_n=1):
    """Resample patches with replacement n_boot times → per-marker Pearson distribution.
    Returns dict(point, mean, lo, hi), each (C,). The same seed+patch-count gives the
    SAME resampled patch sets across models, so the CIs are paired/comparable."""
    keys = ("n", "sx", "sy", "sxx", "syy", "sxy")
    Np   = stats["n"].shape[0]
    point = _pearson_from_sums(*[stats[k].sum(0) for k in keys], min_n=min_n)
    rng  = np.random.default_rng(seed)
    rs   = np.empty((n_boot, stats["n"].shape[1]), np.float64)
    for b in range(n_boot):
        cnt = np.bincount(rng.integers(0, Np, Np), minlength=Np).astype(np.float64)
        agg = [(stats[k] * cnt[:, None]).sum(0) for k in keys]
        rs[b] = _pearson_from_sums(*agg, min_n=min_n)
    return dict(point=point, mean=np.nanmean(rs, 0),
                lo=np.nanpercentile(rs, 2.5, 0), hi=np.nanpercentile(rs, 97.5, 0))


def write_bootstrap_outputs(boot, markers, title, out_stem,
                            ylabel="Pearson r (token-level pred vs GT)"):
    """boot: label → dict(point, mean, lo, hi) (each (C,)). Grouped bars (bootstrap mean)
    with asymmetric 95%-CI whiskers; CSV has point/mean/lo/hi per model per marker."""
    import csv
    import matplotlib.patches as mpatches
    labels  = list(boot)
    n       = len(labels); x = np.arange(len(markers)); w = 0.8 / n
    palette = ["steelblue", "tomato", "seagreen", "darkorange", "mediumpurple", "saddlebrown"]
    color   = {l: palette[i % len(palette)] for i, l in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(16, 5))
    for ci in range(len(markers)):
        means = np.array([boot[l]["mean"][ci] for l in labels])
        order = np.argsort(-np.nan_to_num(means, nan=-np.inf))      # best → worst
        for slot, mi in enumerate(order):
            l = labels[mi]
            m, lo, hi = boot[l]["mean"][ci], boot[l]["lo"][ci], boot[l]["hi"][ci]
            yerr = [[max(0.0, m - lo)], [max(0.0, hi - m)]]
            ax.bar(x[ci] + (slot - n / 2 + 0.5) * w, m, w, color=color[l],
                   yerr=yerr, capsize=2, error_kw=dict(lw=0.6, alpha=0.7))
    ax.set_xticks(x); ax.set_xticklabels(markers, rotation=45, ha="right")
    ax.axhline(0, color="black", lw=0.5); ax.set_ylim(-0.2, 1.0)
    ax.set_ylabel(ylabel); ax.set_title(title)
    leg = np.argsort(-np.array([np.nanmean(boot[l]["mean"]) for l in labels]))
    handles = [mpatches.Patch(color=color[labels[i]],
               label=f"{labels[i]} (mean={np.nanmean(boot[labels[i]]['mean']):.3f})") for i in leg]
    ax.legend(handles=handles, fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(f"{out_stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(f"{out_stem}.csv", "w", newline="") as fcsv:
        wr = csv.writer(fcsv)
        hdr = ["marker"]
        for l in labels:
            hdr += [f"{l}_point", f"{l}_mean", f"{l}_lo", f"{l}_hi"]
        wr.writerow(hdr)
        for ci, name in enumerate(markers):
            row = [name]
            for l in labels:
                row += [boot[l]["point"][ci], boot[l]["mean"][ci],
                        boot[l]["lo"][ci], boot[l]["hi"][ci]]
            wr.writerow(row)
    print(f"Saved → {out_stem}.png / .csv")


def print_table(results, markers):
    hdr = f"{'Marker':<14}" + "".join(f"{l[:14]:>16}" for l in results)
    print("\n" + hdr); print("-" * len(hdr))
    for ci, name in enumerate(markers):
        print(f"{name:<14}" + "".join(f"{results[l][ci]:>+16.4f}" for l in results))
    print(f"{'MEAN':<14}" + "".join(f"{np.nanmean(results[l]):>+16.4f}" for l in results))


def mean_per_slide(per_slide, slides, labels):
    """per_slide[slide][label] = [r per marker] → ({label:[mean over slides]}, {label:[std]}).
    Slides are the independent units (one r per slide, then averaged), so this is invariant
    to per-slide scaling and yields a cross-slide std as the error bar. NaN-safe (a marker
    with no positives on a slide is skipped for that slide)."""
    means, stds = {}, {}
    for lab in labels:
        M = np.array([per_slide[s][lab] for s in slides if lab in per_slide[s]], dtype=float)
        means[lab] = np.nanmean(M, axis=0)
        stds[lab]  = np.nanstd(M, axis=0)
    return means, stds


def print_table_pm(means, stds, markers):
    labels = list(means)
    hdr = f"{'Marker':<14}" + "".join(f"{l[:18]:>22}" for l in labels)
    print("\n" + hdr); print("-" * len(hdr))
    for ci, name in enumerate(markers):
        print(f"{name:<14}" + "".join(f"{means[l][ci]:>+10.3f}±{stds[l][ci]:<10.3f}" for l in labels))
    print(f"{'MEAN':<14}" + "".join(f"{np.nanmean(means[l]):>+10.3f}±{np.nanmean(stds[l]):<10.3f}" for l in labels))


def write_mean_std_csv(per_slide, means, stds, markers, labels, slides, path):
    """Self-contained per-slide-mean CSV: for each model and marker, the raw per-slide
    metric value on every slide, then the cross-slide mean and std those produce. So the
    std is auditable from the visible per-slide numbers (cols: {label}_{slide}…, _mean, _std)."""
    import csv
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        hdr = ["marker"]
        for l in labels:
            hdr += [f"{l}_{s}" for s in slides] + [f"{l}_mean", f"{l}_std"]
        wr.writerow(hdr)
        for ci, name in enumerate(markers):
            row = [name]
            for l in labels:
                for s in slides:
                    v = per_slide[s].get(l)
                    row.append(v[ci] if v is not None else "")
                row += [means[l][ci], stds[l][ci]]
            wr.writerow(row)
    print(f"Saved → {path}")


def write_outputs(results, markers, title, out_stem,
                  ylabel="Pearson r (token-level pred vs GT)", errs=None, write_csv=True):
    """results: label → [r per marker]. Writes <out_stem>.png and (when write_csv) a
    means-only <out_stem>.csv. Within each marker, bars are sorted high→low so the ranking
    is readable at a glance; colour still identifies the model (legend ordered by mean r).
    When `errs` (label → [std per marker]) is given, each bar gets a ±std error bar — used
    for the mean-of-per-slide plots (the pooled plots pass errs=None: a single pooled r has
    no cross-slide std). Pass write_csv=False when a richer CSV (e.g. per-slide detail) is
    written separately to the same stem, to avoid clobbering it."""
    import csv
    import matplotlib.patches as mpatches
    labels = list(results)
    mat    = np.array([results[l] for l in labels])          # (n_models, C)
    emat   = np.array([errs[l] for l in labels]) if errs is not None else None
    n      = len(labels)
    x      = np.arange(len(markers)); w = 0.8 / n
    palette = ["steelblue", "tomato", "seagreen", "darkorange", "mediumpurple", "saddlebrown"]
    model_color = {l: palette[i % len(palette)] for i, l in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(16, 5))
    for ci in range(len(markers)):
        order = np.argsort(-np.nan_to_num(mat[:, ci], nan=-np.inf))   # best → worst
        for slot, mi in enumerate(order):
            ax.bar(x[ci] + (slot - n / 2 + 0.5) * w, mat[mi, ci], w,
                   color=model_color[labels[mi]],
                   yerr=(emat[mi, ci] if emat is not None else None),
                   capsize=2, error_kw=dict(lw=0.6, alpha=0.7))
    ax.set_xticks(x); ax.set_xticklabels(markers, rotation=45, ha="right")
    ax.axhline(0, color="black", lw=0.5); ax.set_ylim(-0.2, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    leg_order = np.argsort(-np.nanmean(mat, axis=1))
    handles = [mpatches.Patch(color=model_color[labels[i]],
                              label=f"{labels[i]} (mean={np.nanmean(mat[i]):.3f})") for i in leg_order]
    ax.legend(handles=handles, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{out_stem}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    if not write_csv:
        print(f"Saved → {out_stem}.png")
        return
    with open(f"{out_stem}.csv", "w", newline="") as fcsv:
        wr = csv.writer(fcsv); wr.writerow(["marker"] + labels)
        for ci, name in enumerate(markers):
            wr.writerow([name] + [results[l][ci] for l in labels])
    print(f"Saved → {out_stem}.png / .csv")


# ── Per-slide loading ────────────────────────────────────────────────────────--

def load_slide(slide, max_patches, seed, your_pred_dir=PRED_DIR):
    """GT + your-preds + a zarr handle for one slide (subsampled to max_patches)."""
    with h5py.File(H5_DIR / f"{slide}_patch_dataset.h5", "r") as f:
        coords  = f["coords"][:]
        targets = f["targets"][:]
        psz     = int(f.attrs["patch_size_level0"])
        markers = list(f.attrs["marker_names"])

    your_path = Path(your_pred_dir) / f"{slide}_preds.npy"
    if not your_path.exists():
        sys.exit(f"Missing {your_path}. Run visualize_orion_predictions.py --slides {slide} "
                 f"--out_dir {your_pred_dir} first.")
    your_preds = np.load(your_path)
    assert len(your_preds) == len(coords) == len(targets)

    # Keep FULL arrays; `sel` only subsamples the METRIC. Inference (and thus each
    # model's cache) always runs over all N coords, so caches don't depend on
    # --max_patches/--seed and stay reusable across runs.
    N = len(coords)
    if max_patches and max_patches < N:
        sel = np.sort(np.random.default_rng(seed).choice(N, max_patches, replace=False))
    else:
        sel = np.arange(N)
    print(f"{slide}: scoring {len(sel)}/{N} patches (psz_level0={psz})")
    return dict(coords=coords, gt=targets, your=your_preds,
                psz=psz, markers=markers, arr=open_he_zarr(slide), sel=sel)


# ── Inference timing (end-to-end, whole slide) ───────────────────────────────--
# We time the real wall-clock cost of running a model over an ENTIRE slide — read +
# preprocess + forward + tokenize — i.e. exactly the infer_* call. It's measured once,
# when a model's cache is (re)generated, and saved as a sidecar next to the preds, so
# the timing table still shows up on fully-cached re-runs. The infer_* functions end by
# moving results to CPU (a CUDA sync), so the wall-clock already covers all GPU work.
#   sidecar: preds_cache/<key>/<slide>_time.txt  →  "<seconds> <n_patches>"

def save_slide_time(cache_dir, slide, seconds, n_patches):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    (Path(cache_dir) / f"{slide}_time.txt").write_text(f"{seconds:.4f} {n_patches}\n")


def read_slide_time(cache_dir, slide):
    """→ (seconds, n_patches) or None."""
    fp = Path(cache_dir) / f"{slide}_time.txt"
    if not fp.exists():
        return None
    try:
        s, n = fp.read_text().split()
        return float(s), int(n)
    except ValueError:
        return None


def print_timing_summary(label_dirs, slides):
    """label_dirs: {display_label: cache_dir}. Reads each model's per-slide time
    sidecars, prints a whole-slide inference-time table, and writes results/timing.csv."""
    rows = {}                                   # label → (total_sec, total_patches, n_slides_timed)
    for label, cdir in label_dirs.items():
        tot_s = tot_n = 0.0; n_done = 0
        for s in slides:
            t = read_slide_time(cdir, s)
            if t:
                tot_s += t[0]; tot_n += t[1]; n_done += 1
        if n_done:
            rows[label] = (tot_s, tot_n, n_done)
    if not rows:
        return
    print(f"\n===== INFERENCE TIME — whole-slide, end-to-end ({'+'.join(slides)}) =====")
    print(f"{'Model':<18}{'sec/slide':>12}{'patch/s':>12}{'ms/patch':>12}{'slides':>8}")
    print("-" * 62)
    import csv
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "timing.csv", "w", newline="") as f:
        wr = csv.writer(f); wr.writerow(["model", "sec_per_slide", "patches_per_sec", "ms_per_patch", "n_slides_timed"])
        for label, (tot_s, tot_n, n_done) in sorted(rows.items(), key=lambda kv: kv[1][0] / kv[1][2]):
            sec_slide = tot_s / n_done          # average over slides actually timed
            pps       = tot_n / tot_s
            print(f"{label:<18}{sec_slide:>12.1f}{pps:>12.0f}{1000.0 / pps:>12.3f}{n_done:>8d}")
            wr.writerow([label, f"{sec_slide:.2f}", f"{pps:.1f}", f"{1000.0 / pps:.4f}", n_done])
    print(f"\nSaved → {RESULTS_DIR / 'timing.csv'}")


# ── Main ───────────────────────────────────────────────────────────────────--

def print_boot_table(boot, markers):
    labels = list(boot)
    hdr = f"{'Marker':<14}" + "".join(f"{l[:24]:>26}" for l in labels)
    print("\n" + hdr); print("-" * len(hdr))
    for ci, name in enumerate(markers):
        print(f"{name:<14}" + "".join(
            f"{boot[l]['mean'][ci]:+.3f}[{boot[l]['lo'][ci]:+.2f},{boot[l]['hi'][ci]:+.2f}]".rjust(26)
            for l in labels))
    print(f"{'MEAN':<14}" + "".join(f"{np.nanmean(boot[l]['mean']):+.3f}".rjust(26) for l in labels))


def run_bootstrap(args, slides, data, markers, keys, want_cached, ours_label):
    """Pool every patch over all `slides`, resample patches with replacement n_boot times,
    and report per-marker per-model 95% CI for PCC (all tokens) and PCC+ (gt>0). All preds
    come from cache (no GPU); GT is whatever --h5_dir points at (e.g. cleaned crcv2 targets)."""
    from collections import defaultdict
    MIN_TOKENS = 50
    # (label, source dir of {slide}_preds.npy); ours reuses the in-memory preds.
    sources = [(ours_label, None)]
    sources += [(REGISTRY[k][0], PRED_CACHE / k) for k in keys]
    sources += [(lbl, Path(CACHED_PREDS[lbl]["pred_dir"])) for lbl in want_cached]

    boot_pcc, boot_pccp = {}, {}
    for label, src in sources:
        all_l, pos_l, ok = defaultdict(list), defaultdict(list), True
        for s in slides:
            if src is None:
                preds = data[s]["your"]
            else:
                fp = src / f"{s}_preds.npy"
                if not fp.exists():
                    print(f"  [skip] {label}: missing {fp}"); ok = False; break
                preds = np.load(fp)
            gt = data[s]["gt"]
            assert len(preds) == len(gt), f"{label} {s}: preds {len(preds)} != gt {len(gt)}"
            a, p = patch_stats(preds, gt)
            for k in a:
                all_l[k].append(a[k]); pos_l[k].append(p[k])
            if src is not None:
                del preds
        if not ok:
            continue
        allcat = {k: np.concatenate(v) for k, v in all_l.items()}
        poscat = {k: np.concatenate(v) for k, v in pos_l.items()}
        print(f"  [{label}] {allcat['n'].shape[0]:,} patches pooled → bootstrap {args.n_boot}×")
        boot_pcc[label]  = bootstrap_ci(allcat, args.n_boot, args.seed, min_n=1)
        boot_pccp[label] = bootstrap_ci(poscat, args.n_boot, args.seed, min_n=MIN_TOKENS)

    tag = "+".join(slides)
    print(f"\n===== BOOTSTRAP PCC ({tag}, {args.n_boot}×, 95% CI) =====")
    print_boot_table(boot_pcc, markers)
    write_bootstrap_outputs(boot_pcc, markers,
        f"Bootstrap PCC · {tag} · {args.n_boot}× patch resamples (95% CI)",
        str(RESULTS_DIR / f"bootstrap_{tag}_pcc"), ylabel="Pearson r")
    print(f"\n===== BOOTSTRAP PCC+ (gt>0) ({tag}, {args.n_boot}×, 95% CI) =====")
    print_boot_table(boot_pccp, markers)
    write_bootstrap_outputs(boot_pccp, markers,
        f"Bootstrap PCC+ (gt>0) · {tag} · {args.n_boot}× patch resamples (95% CI)",
        str(RESULTS_DIR / f"bootstrap_{tag}_pccplus"), ylabel="Pearson r⁺ (GT>0 tokens)")


def main():
    # CLI keys for cached (separate-env) models → their CACHED_PREDS display label.
    cached_keys = {"miphei-convnext": "MIPHEI-ConvNeXt"}
    all_keys    = list(REGISTRY) + list(cached_keys)

    ap = argparse.ArgumentParser()
    ap.add_argument("--slides",      default="CRC19",
                    help="comma-separated slide ids, e.g. CRC19,CRC18,CRC17")
    ap.add_argument("--models",      default=",".join(all_keys),
                    help=f"comma-separated model keys (default: all). choices: {', '.join(all_keys)}")
    ap.add_argument("--max_patches", type=int, default=0,
                    help="0 = all patches (default). Only subsamples the METRIC; inference/caching is full-slide.")
    ap.add_argument("--batch",       type=int, default=8)
    ap.add_argument("--seed",        type=int, default=0)
    ap.add_argument("--refresh",     action="store_true",
                    help="ignore + overwrite any cached preds (force re-inference; also re-times each model)")
    ap.add_argument("--your_preds",  default=str(PRED_DIR),
                    help="dir holding the 'ours' {SLIDE}_preds.npy (default: the finetune-4 baseline)")
    ap.add_argument("--your_label",  default=LABEL,
                    help="legend label for the 'ours' bars")
    ap.add_argument("--h5_dir",      default=None,
                    help="override GT h5 dir (e.g. the clean subset); relative paths resolve under REPO")
    ap.add_argument("--pred_cache",  default=None,
                    help="override preds_cache dir (e.g. preds_cache_clean); relative paths resolve under benchmarking/")
    ap.add_argument("--results_dir", default=None,
                    help="override output dir for csvs/plots (e.g. results_clean); relative paths resolve under benchmarking/")
    ap.add_argument("--mode", choices=["slide", "bootstrap"], default="slide",
                    help="slide: per-slide PCC/PCC+ (one image each). "
                         "bootstrap: pool patches over ALL --slides, resample with replacement "
                         "--n_boot times → per-marker per-model 95%% CI (no GPU; needs cached preds)")
    ap.add_argument("--n_boot", type=int, default=100,
                    help="bootstrap resamples (mode=bootstrap)")
    args = ap.parse_args()

    # optional clean-subset overrides: point GT + cached preds at a filtered tree.
    global H5_DIR, PRED_CACHE, RESULTS_DIR
    if args.h5_dir:
        H5_DIR = Path(args.h5_dir) if Path(args.h5_dir).is_absolute() else REPO / args.h5_dir
        print(f"[override] H5_DIR = {H5_DIR}")
    if args.pred_cache:
        PRED_CACHE = (Path(args.pred_cache) if Path(args.pred_cache).is_absolute()
                      else BENCHMARKING_DIR / args.pred_cache)
        CACHED_PREDS["MIPHEI-ConvNeXt"]["pred_dir"] = PRED_CACHE / "miphei-convnext"
        print(f"[override] PRED_CACHE = {PRED_CACHE}")
    if args.results_dir:
        RESULTS_DIR = (Path(args.results_dir) if Path(args.results_dir).is_absolute()
                       else BENCHMARKING_DIR / args.results_dir)
        print(f"[override] RESULTS_DIR = {RESULTS_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = [k.strip() for k in args.models.split(",") if k.strip()]
    keys     = [k for k in requested if k in REGISTRY]                # in-harness models
    want_cached = {cached_keys[k] for k in requested if k in cached_keys}  # separate-env models
    slides   = [s.strip() for s in args.slides.split(",") if s.strip()]

    # load every slide up front (cheap: coords/GT/your-preds + a zarr handle)
    data    = {s: load_slide(s, args.max_patches, args.seed, args.your_preds) for s in slides}
    markers = data[slides[0]]["markers"]
    for s in slides[1:]:
        assert data[s]["markers"] == markers, f"marker mismatch on {s}"

    OURS = args.your_label
    thr  = positive_threshold(markers)         # PCC+ positive cut = GT > 0 (per marker)

    if args.mode == "bootstrap":
        run_bootstrap(args, slides, data, markers, keys, want_cached, OURS)
        return

    results     = {s: {} for s in slides}      # results[slide][label]     = [PCC  per marker]
    results_pos = {s: {} for s in slides}      # results_pos[slide][label] = [PCC+ per marker]
    pooled      = {}                           # label → PooledPearson    (global PCC)
    pooled_pos  = {}                           # label → PooledPearsonPos (PCC+ on GT-positive)

    def record(label, slide, full_preds):
        """Apply the metric subsample `sel` to a full-N (N,C,G,G) pred array, score
        per-marker PCC and PCC+ for this slide, and fold both into the pooled accumulators."""
        sel = data[slide]["sel"]
        p   = np.asarray(full_preds)[sel]
        g   = data[slide]["gt"][sel]
        assert len(p) == len(g), f"{label} {slide}: pred/gt length mismatch ({len(p)} vs {len(g)})"
        results[slide][label]     = pearson_per_marker(p, g, len(markers))
        results_pos[slide][label] = pearson_pos_per_marker(p, g, thr)
        pooled.setdefault(label, PooledPearson(len(markers))).update(p, g)
        pooled_pos.setdefault(label, PooledPearsonPos(len(markers), thr)).update(p, g)

    # ours — preds already on disk (full N), no model to load
    for s in slides:
        record(OURS, s, data[s]["your"])

    # separate-env model(s) (convnext/smp0.4.0): cache generated via conda run.
    for label, cfg in CACHED_PREDS.items():
        if label not in want_cached:
            continue
        pred_dir = Path(cfg["pred_dir"])
        if args.refresh:                                      # force re-generation
            for s in slides:
                fp = pred_dir / f"{s}_preds.npy"
                if fp.exists():
                    fp.unlink()
        available = ensure_cached_preds(label, cfg, slides)   # auto-generates missing preds
        if not available:
            print(f"  [cached] {label}: no preds available in {pred_dir} — skipping")
            continue
        for s in available:
            record(label, s, np.load(pred_dir / f"{s}_preds.npy"))

    # in-harness baselines — load each model lazily (only on a cache miss), run it over
    # every slide, and cache full-slide preds to disk so the next run just loads them.
    for key in keys:
        label = REGISTRY[key][0]
        _, _, mean, std, otype = REGISTRY[key]
        cache_dir = PRED_CACHE / key
        model = None
        try:
            for s in slides:
                d     = data[s]
                cpath = cache_dir / f"{s}_preds.npy"
                full  = None
                if cpath.exists() and not args.refresh:
                    full = np.load(cpath)
                    if full.shape[0] != len(d["coords"]):     # stale cache (coords changed)
                        print(f"  [cache] {label} {s}: stale ({full.shape[0]}≠{len(d['coords'])}) — re-inferring")
                        full = None
                    else:
                        print(f"  [cache] {label} {s}: loaded {full.shape}")
                if full is None:
                    if model is None:                         # pay the load only when needed
                        print(f"\nLoading {label}…", flush=True)
                        model = load_model(key, device)
                    print(f"  running {label} ({otype}) on {s} [{len(d['coords']):,} patches]…", flush=True)
                    t0 = time.perf_counter()                  # whole-slide, end-to-end wall clock
                    if key == "rosie":
                        full = infer_rosie(model, mean, std, d["arr"], d["coords"], d["psz"], device, args.batch)
                    else:
                        full = infer_model(model, otype, mean, std, d["arr"], d["coords"], d["psz"], device, args.batch)
                    dt = time.perf_counter() - t0
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    np.save(cpath, full)
                    save_slide_time(cache_dir, s, dt, len(d["coords"]))
                    print(f"    cached → {cpath}  {full.shape}  ({dt:.1f}s, {len(d['coords'])/dt:.0f} patch/s)")
                record(label, s, full)
        except Exception as e:
            import traceback
            print(f"  [SKIP] {label}: {type(e).__name__}: {e}")
            traceback.print_exc()
        finally:
            if model is not None:
                del model; torch.cuda.empty_cache()

    # per-slide outputs — global PCC (all tokens) and PCC+ (GT-positive tokens only)
    for s in slides:
        n_patches = len(data[s]["sel"])
        print(f"\n===== {s} =====")
        print_table(results[s], markers)
        write_outputs(results[s], markers,
                      f"{s}: token-level benchmark · {n_patches:,} patches ",
                      str(RESULTS_DIR / f"{s}_all_models"))
        print(f"\n----- {s}  PCC+ (GT-positive tokens only) -----")
        print_table(results_pos[s], markers)
        write_outputs(results_pos[s], markers,
                      f"{s}: PCC+ on GT>0 tokens · {n_patches:,} patches",
                      str(RESULTS_DIR / f"{s}_all_models_pccplus"),
                      ylabel="Pearson r⁺ (GT>0 tokens only)")

    # pooled summary across slides (only for >1 slide)
    if len(slides) > 1:
        tag = "+".join(slides)
        pooled_res = {l: acc.pearson() for l, acc in pooled.items()}
        print(f"\n===== POOLED ({tag}) — global PCC =====")
        print_table(pooled_res, markers)
        write_outputs(pooled_res, markers,
                      f"Pooled {tag}: token-level benchmark (Pearson over all tokens)",
                      str(RESULTS_DIR / f"pooled_{tag}_all_models"))

        pooled_pos_res = {l: acc.pearson() for l, acc in pooled_pos.items()}
        print(f"\n===== POOLED ({tag}) — PCC+ (GT-positive tokens only) =====")
        print_table(pooled_pos_res, markers)
        write_outputs(pooled_pos_res, markers,
                      f"Pooled {tag}: PCC+ on GT-positive tokens (Pearson, gt>0)",
                      str(RESULTS_DIR / f"pooled_{tag}_all_models_pccplus"),
                      ylabel="Pearson r⁺ (GT>0 tokens only)")

        # mean-of-per-slide ± std (slides = independent units): the principled aggregate
        # with a cross-slide error bar, invariant to per-slide p99 scaling. Complements
        # pooled-over-tokens (literature-comparable but size-weighted). Report both.
        labels = list(pooled.keys())
        for which, per_slide, sfx, yl in [
                ("global PCC", results,     "",         "Pearson r (mean over slides)"),
                ("PCC+ (gt>0)", results_pos, "_pccplus", "Pearson r⁺ (mean over slides)")]:
            means, stds = mean_per_slide(per_slide, slides, labels)
            print(f"\n===== MEAN-OF-PER-SLIDE ± std ({tag}) — {which} =====")
            print_table_pm(means, stds, markers)
            write_mean_std_csv(per_slide, means, stds, markers, labels, slides,
                               RESULTS_DIR / f"perslidemean_{tag}_all_models{sfx}.csv")
            write_outputs(means, markers,
                          f"Mean-of-per-slide {tag}: {which} (±std across {len(slides)} slides)",
                          str(RESULTS_DIR / f"perslidemean_{tag}_all_models{sfx}"), ylabel=yl,
                          errs=stds, write_csv=False)

    # whole-slide inference-time table (from sidecars saved during inference; persists
    # across cached re-runs). Covers the models we benchmark against, not "ours".
    #label_dirs = {REGISTRY[k][0]: PRED_CACHE / k for k in keys}
    #label_dirs.update({l: Path(CACHED_PREDS[l]["pred_dir"]) for l in want_cached})
    #print_timing_summary(label_dirs, slides)


if __name__ == "__main__":
    main()