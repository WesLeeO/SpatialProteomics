"""
Generate MIPHEI-ConvNeXt token predictions for the benchmark — run in an ISOLATED
env (segmentation_models_pytorch==0.4.0), because that smp version conflicts with
the timm that UNI2 needs in thesis_env.

It reproduces benchmark_all_slides.py's pixel→token pipeline EXACTLY (128 µm context
window → native 256 px → center-crop to the patch FOV → average 14×14 blocks → 16×16
tokens), so the cached numbers are directly comparable to the in-harness pixel models.
Writes preds_cache/miphei-convnext/{slide}_preds.npy of shape (N, 16, 16, 16), full N
in coords order. benchmark_all_slides.py picks them up via CACHED_PREDS["MIPHEI-ConvNeXt"]
(the same preds_cache/<model>/ layout every other benchmarked model uses).

Env setup (one-time)
--------------------
  conda create -y -n smp040 python=3.10
  # IMPORTANT: torch must be the cu128 build — the workstation GPU is Blackwell
  # (RTX PRO 6000), which the default-index torch wheels do NOT support.
  conda run -n smp040 pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  conda run -n smp040 pip install "segmentation_models_pytorch==0.4.0" timm safetensors \
              tifffile imagecodecs zarr h5py opencv-python-headless numpy
  # imagecodecs is required — the ORION OME-TIFFs are JPEG-compressed.

Usage
-----
  python gen_convnext_preds.py --slides CRC19              # or CRC19,CRC18,...

Normally you don't run this directly: benchmark_all_slides.py --models miphei-convnext
auto-invokes it via `conda run -n smp040` for any slides whose preds are missing.
"""
import sys
import time
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import h5py
import cv2
import torch
import tifffile
import zarr
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
H5_DIR    = REPO / "datasets/orion_crc_patch_dataset_benchmark"
TIFF_DIR  = Path("/mnt/ssd1/virtual_proteomics/data/ORION_CRC")
CKPT      = REPO / "checkpoints" / "MIPHEI-convnext" / "model.safetensors"
SMP_UNET  = REPO / "MIPHEI-ViT" / "src" / "generators" / "smp_unet.py"
OUT_DIR   = HERE.parent / "preds_cache" / "miphei-convnext"   # matches benchmark_all_slides.CACHED_PREDS

MPP_HE, NATIVE_IMG, NATIVE_MPP = 0.325, 256, 0.5
TOKEN_GRID, NUM_OUTPUTS = 16, 16
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], np.float32)


def open_he_zarr(slide):
    m = list((TIFF_DIR / slide).glob("*-registered.ome.tif"))
    if not m:
        raise FileNotFoundError(f"No *-registered.ome.tif in {TIFF_DIR/slide}")
    store = zarr.LRUStoreCache(tifffile.TiffFile(str(m[0])).aszarr(), max_size=512 * 2**20)
    z = zarr.open(store, mode="r")
    return z["0"] if isinstance(z, zarr.hierarchy.Group) else z


def read_region(arr, x0, y0, size):
    if arr.shape[-1] <= 4:
        H, W, cl = arr.shape[0], arr.shape[1], True
    else:
        H, W, cl = arr.shape[1], arr.shape[2], False
    out = np.zeros((size, size, 3), np.uint8)
    sx, sy = max(0, x0), max(0, y0)
    ex, ey = min(W, x0 + size), min(H, y0 + size)
    if ex <= sx or ey <= sy:
        return out
    crop = (np.asarray(arr[sy:ey, sx:ex, :3]) if cl
            else np.asarray(arr[:3, sy:ey, sx:ex]).transpose(1, 2, 0))
    out[sy - y0:ey - y0, sx - x0:ex - x0] = crop
    return out


def load_convnext(device):
    spec = importlib.util.spec_from_file_location("smp_unet_mod", str(SMP_UNET))
    su = importlib.util.module_from_spec(spec); spec.loader.exec_module(su)
    net = su.UnetMultiHeads(
        encoder_name="tu-convnext_large", encoder_weights=None,
        decoder_use_batchnorm=True, in_channels=3, classes=NUM_OUTPUTS,
        activation=torch.nn.Tanh, dropout=0.1, use_attention=True)
    sd = load_file(str(CKPT))
    sd = {k[6:] if k.startswith("model.") else k: v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"  [load] missing={len(missing)} unexpected={len(unexpected)}")
    return net.to(device).eval()


def infer_slide(model, arr, coords, psz, device, batch=8):
    """Replicates benchmark_all_slides.infer_model's pixel branch → (N,16,16,16)."""
    ctx_psz = int(round(NATIVE_IMG * NATIVE_MPP / MPP_HE))
    crop_px = int(round(psz * MPP_HE / (ctx_psz * MPP_HE) * NATIVE_IMG))
    off     = (NATIVE_IMG - crop_px) // 2
    tok_px  = crop_px // TOKEN_GRID
    crop_px = tok_px * TOKEN_GRID
    out = np.empty((len(coords), NUM_OUTPUTS, TOKEN_GRID, TOKEN_GRID), np.float32)
    buf, idxs = [], []

    def flush():
        if not buf:
            return
        a = (np.stack(buf).astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        t = torch.from_numpy(a.transpose(0, 3, 1, 2)).to(device)
        with torch.no_grad(), torch.autocast("cuda", enabled=device.type == "cuda"):
            pred = model(t)
        pred = (pred[0] if isinstance(pred, (tuple, list)) else pred).float()
        pred = (pred.clamp(-0.9, 0.9) + 0.9) / 1.8
        pred = pred[:, :, off:off + crop_px, off:off + crop_px]
        B, C = pred.shape[:2]
        tok = pred.reshape(B, C, TOKEN_GRID, tok_px, TOKEN_GRID, tok_px).mean((3, 5))
        out[np.array(idxs)] = tok.cpu().numpy()
        buf.clear(); idxs.clear()

    for i, (x, y) in enumerate(coords):
        x, y = int(x), int(y)
        cx0 = int(round(x + psz / 2 - ctx_psz / 2))
        cy0 = int(round(y + psz / 2 - ctx_psz / 2))
        region = read_region(arr, cx0, cy0, ctx_psz)
        region = cv2.resize(region, (NATIVE_IMG, NATIVE_IMG), interpolation=cv2.INTER_LINEAR)
        buf.append(region); idxs.append(i)
        if len(buf) >= batch:
            flush()
        if i % 1000 == 0:
            print(f"    {i}/{len(coords)}", flush=True)
    flush()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides", required=True, help="comma-separated slide ids")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_convnext(device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for slide in [s.strip() for s in args.slides.split(",") if s.strip()]:
        with h5py.File(H5_DIR / f"{slide}_patch_dataset.h5", "r") as f:
            coords = f["coords"][:]
            psz    = int(f.attrs["patch_size_level0"])
        print(f"{slide}: {len(coords)} patches (psz={psz})", flush=True)
        t0    = time.perf_counter()               # whole-slide, end-to-end (matches benchmark harness)
        preds = infer_slide(model, open_he_zarr(slide), coords, psz, device, args.batch)
        dt    = time.perf_counter() - t0
        np.save(OUT_DIR / f"{slide}_preds.npy", preds)
        # sidecar read by benchmark_all_slides.print_timing_summary (same format)
        (OUT_DIR / f"{slide}_time.txt").write_text(f"{dt:.4f} {len(coords)}\n")
        print(f"  saved → {OUT_DIR / f'{slide}_preds.npy'}  {preds.shape}  "
              f"({dt:.1f}s, {len(coords)/dt:.0f} patch/s)")


if __name__ == "__main__":
    main()