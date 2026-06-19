"""
Matched-resolution fairness test: run MIPHEI-vit on PanNuke ourselves and aggregate
its predictions two ways, so we can compare ours vs MIPHEI with the ONLY difference
being the model.

MIPHEI-vit (H-optimus encoder + ViTMatte decoder, weights in checkpoints/MIPHEI-vit)
predicts pixel-level IF (16 ORION markers) at native 256. We emit two feature tables:

  miphei_pixel : mean over the strict nucleus mask of the PIXEL prediction
                 → validates our inference (should ≈ their parquet, mean-5 ≈ 0.600).
  miphei_token : block-average the pixel prediction into the SAME 16×16 token grid we
                 use, then OUR area-weighted token aggregation per nucleus
                 → MIPHEI at OUR resolution + OUR aggregation (the fair ablation).

Then `eval_cell_auprc.py --tag miphei_pixel|miphei_token` scores each with the same
logreg, so:  ours(token) vs miphei_token isolates pure model quality at matched
resolution;  miphei_pixel vs their parquet checks our MIPHEI inference is faithful.

Example:
  HF_HOME=.../foundation_models HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  python cell_cls/pannuke_cls/run_miphei_matched.py
"""
import os
os.environ.setdefault("HF_HOME", "/home/wesley/spatial_proteomics/foundation_models")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils import (REPO_ROOT, FEAT_DIR, TOKEN_GRID, list_images, slide_name_of,
                   load_image_and_mask)

# MIPHEI's model.py may import from their src package; make it importable + shim the
# smp version gap (CenterBlock removed in smp 0.5.0) just in case.
sys.path.insert(0, str(REPO_ROOT / "MIPHEI-ViT"))
import segmentation_models_pytorch.decoders.unet.decoder as _dec
if not hasattr(_dec, "CenterBlock"):
    _dec.CenterBlock = nn.Identity

# checkpoint dir that ships model.py + config_hf.json + model.safetensors (the
# self-contained HF bundle the user already used for benchmarking inference).
CKPT = REPO_ROOT / "benchmarking/MIPHEI-vit"
MARKERS = ['Hoechst', 'CD31', 'CD45', 'CD68', 'CD4', 'FOXP3', 'CD8a', 'CD45RO',
           'CD20', 'PD-L1', 'CD3e', 'CD163', 'E-cadherin', 'Ki67', 'Pan-CK', 'SMA']
HE_MEAN = np.array([123.675, 116.28, 103.53], np.float32)     # their cfg.data.normalization
HE_STD  = np.array([58.395, 57.12, 57.375], np.float32)
IMG = 256
G   = TOKEN_GRID
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_miphei():
    """Load via the checkpoint's own model.py (correct HF weight mapping) — same path
    the user used in benchmarking/benchmark_all_slides.py."""
    spec = importlib.util.spec_from_file_location("miphei_ckpt_model", str(CKPT / "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    model = mod.MIPHEIViT.from_pretrained_hf(repo_path=str(CKPT))
    print("  loaded MIPHEI-vit via from_pretrained_hf")
    return model.to(DEVICE).eval()


def preprocess(he: np.ndarray) -> np.ndarray:
    """uint8 256×256×3 → (3,256,256) float32, ImageNet/MIPHEI normalisation."""
    return ((he.astype(np.float32) - HE_MEAN) / HE_STD).transpose(2, 0, 1)


def pixel_means(inst: np.ndarray, pred_pix: np.ndarray):
    """Strict nucleus-mask mean of the pixel prediction. pred_pix (C,H,W)."""
    C = pred_pix.shape[0]
    flat = inst.ravel(); keep = flat > 0
    if not keep.any():
        return np.empty(0, np.int64), np.empty((0, C), np.float32)
    labs = flat[keep]
    vals = pred_pix.reshape(C, -1).T[keep]                    # (n, C)
    ulab, inv = np.unique(labs, return_inverse=True)
    sums = np.zeros((len(ulab), C), np.float64); np.add.at(sums, inv, vals)
    cnt = np.bincount(inv, minlength=len(ulab))
    return ulab.astype(np.int64), (sums / cnt[:, None]).astype(np.float32)


def token_means(inst: np.ndarray, pred_pix: np.ndarray, ps0: int = IMG):
    """Block-average pixels → 16×16 tokens, then OUR area-weighted nucleus aggregation."""
    C = pred_pix.shape[0]
    bp = IMG // G
    tok = pred_pix.reshape(C, G, bp, G, bp).mean(axis=(2, 4))  # (C, G, G)
    # reuse the exact area-weighted token aggregation from the build
    from build_cell_token_features import aggregate_image
    labels, mean_feat, _ = aggregate_image(inst, tok, C, ps0=ps0)
    return labels, mean_feat


def main():
    model = load_miphei()
    imgs = list_images()
    print(f"{len(imgs)} images | running MIPHEI-vit @ {IMG}")

    pix_parts, tok_parts = [], []
    B = 16
    for s in range(0, len(imgs), B):
        chunk = imgs[s:s + B]
        batch, masks, names = [], [], []
        for p in chunk:
            he, inst = load_image_and_mask(p)
            batch.append(preprocess(he)); masks.append(inst); names.append(slide_name_of(p))
        x = torch.from_numpy(np.stack(batch)).to(DEVICE)
        with torch.no_grad(), torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
            preds = model(x).float().cpu().numpy()            # (b,16,256,256)
        for inst, name, pr in zip(masks, names, preds):
            lp, mp = pixel_means(inst, pr)
            lt, mt = token_means(inst, pr)
            for labels, feat, store in [(lp, mp, pix_parts), (lt, mt, tok_parts)]:
                if len(labels) == 0:
                    continue
                df = pd.DataFrame({"cell_id": labels, "slide_name": name})
                for j, m in enumerate(MARKERS):
                    df[f"mean_{m}"] = feat[:, j]
                store.append(df)
        if s % (B * 25) == 0:
            print(f"  {s}/{len(imgs)}")

    for parts, tag in [(pix_parts, "miphei_pixel"), (tok_parts, "miphei_token")]:
        out = FEAT_DIR / f"cell_token_features_{tag}.parquet"
        pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
