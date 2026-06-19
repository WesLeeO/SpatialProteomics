"""
Matched-resolution fairness test on Lizard: run MIPHEI-vit ourselves, aggregate its
pixel predictions two ways, so ours vs MIPHEI differs ONLY in the model.

  miphei_pixel : strict nucleus-mask mean of the PIXEL prediction
                 → validates our inference (should ≈ their parquet, mean-6 ≈ 0.517).
  miphei_token : block-average the pixel prediction into OUR 16×16 token grid, then OUR
                 area-weighted nucleus aggregation → MIPHEI at OUR resolution + aggregation.

Then eval_cell_auprc.py --tag miphei_pixel|miphei_token. If miphei_token drops toward
ours, MIPHEI's Lizard edge was its pixel resolution (separating dense small cells); if it
holds, the model is genuinely better here.

Lizard geometry differs from PanNuke: 20x (0.5 µm/px) = our scale, so ps0=224 and images
are tiled into a 224 grid (variable-size). Reuses the lizard build's grid + aggregation.

Example:
  HF_HOME=.../foundation_models HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  python cell_cls/lizard_cls/run_miphei_matched.py
"""
import os
os.environ.setdefault("HF_HOME", "/home/wesley/spatial_proteomics/foundation_models")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
import importlib.util

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from utils import (REPO_ROOT, FEAT_DIR, PATCH_SIZE_LEVEL0, TOKEN_GRID, NTOK,
                   list_images, slide_name_of, load_image_and_inst)

sys.path.insert(0, str(REPO_ROOT / "MIPHEI-ViT"))
import segmentation_models_pytorch.decoders.unet.decoder as _dec
if not hasattr(_dec, "CenterBlock"):
    _dec.CenterBlock = nn.Identity

CKPT = REPO_ROOT / "benchmarking/MIPHEI-vit"
MARKERS = ['Hoechst', 'CD31', 'CD45', 'CD68', 'CD4', 'FOXP3', 'CD8a', 'CD45RO',
           'CD20', 'PD-L1', 'CD3e', 'CD163', 'E-cadherin', 'Ki67', 'Pan-CK', 'SMA']
HE_MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
HE_STD  = np.array([0.229, 0.224, 0.225], np.float32) * 255
G   = TOKEN_GRID
# MIPHEI-vit is locked to 256 input (its config); feed its native 256 tiling — that's
# its exact training FOV/scale @ 0.5 µm/px, and validates against their 0.517. (Our UNI2
# model is fixed at 224 → 7 µm tokens; MIPHEI here gets 8 µm tokens. Both ~cell scale; the
# diagnostic asks whether MIPHEI's pixel edge survives coarsening to ~cell-scale tokens.)
PS0 = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_miphei():
    spec = importlib.util.spec_from_file_location("miphei_ckpt_model", str(CKPT / "model.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    print("  loaded MIPHEI-vit via from_pretrained_hf")
    return mod.MIPHEIViT.from_pretrained_hf(repo_path=str(CKPT)).to(DEVICE).eval()


def preprocess(crop):
    """uint8 (h,w,3) crop (≤224, edge-clamped) → pad white to 224 → (3,224,224)."""
    import cv2
    h, w = crop.shape[:2]
    if (h, w) != (PS0, PS0):
        crop = cv2.copyMakeBorder(crop, 0, PS0 - h, 0, PS0 - w, cv2.BORDER_CONSTANT, value=255)
    return ((crop.astype(np.float32) - HE_MEAN) / HE_STD).transpose(2, 0, 1)


def predict_image(model, he):
    """Tile the whole image into a 224 grid, run MIPHEI, stitch a (16,H,W) pixel map."""
    H, W = he.shape[:2]
    pix = np.zeros((len(MARKERS), H, W), np.float32)
    crops, boxes = [], []
    for y in range(0, H, PS0):
        for x in range(0, W, PS0):
            y1, x1 = min(y + PS0, H), min(x + PS0, W)
            crops.append(preprocess(he[y:y1, x:x1])); boxes.append((x, y, x1, y1))
    x = torch.from_numpy(np.stack(crops)).to(DEVICE)
    with torch.no_grad(), torch.autocast("cuda", enabled=DEVICE.type == "cuda"):
        out = model(x).float().cpu().numpy()                  # (Ntiles,16,224,224)
    for o, (bx, by, bx1, by1) in zip(out, boxes):
        pix[:, by:by1, bx:bx1] = o[:, :by1 - by, :bx1 - bx]   # un-pad
    return pix


def pixel_means(inst, pix):
    C = pix.shape[0]; flat = inst.ravel(); keep = flat > 0
    if not keep.any():
        return np.empty(0, np.int64), np.empty((0, C), np.float32)
    labs = flat[keep]; vals = pix.reshape(C, -1).T[keep]
    ulab, inv = np.unique(labs, return_inverse=True)
    sums = np.zeros((len(ulab), C), np.float64); np.add.at(sums, inv, vals)
    cnt = np.bincount(inv, minlength=len(ulab))
    return ulab.astype(np.int64), (sums / cnt[:, None]).astype(np.float32)


def token_means(inst, pix):
    """Block-average each 224-tile of the pixel map into 16×16 tokens, then OUR
    area-weighted nucleus aggregation over the whole image (same as the lizard build)."""
    C = pix.shape[0]; H, W = inst.shape
    max_lab = int(inst.max())
    wsum = np.zeros((max_lab + 1, C), np.float64); area = np.zeros(max_lab + 1, np.float64)
    bp = PS0 // G
    for y in range(0, H, PS0):
        for x in range(0, W, PS0):
            y1, x1 = min(y + PS0, H), min(x + PS0, W)
            sub = inst[y:y1, x:x1]; ch, cw = sub.shape
            # block-average this tile's pixels → (C,16,16); pad the tile to 224 first
            tile = np.zeros((C, PS0, PS0), np.float32)
            tile[:, :ch, :cw] = pix[:, y:y1, x:x1]
            tok = tile.reshape(C, G, bp, G, bp).mean(axis=(2, 4))   # (C,G,G)
            flat = sub.ravel(); keep = flat > 0
            if not keep.any():
                continue
            rr = (np.arange(ch) * G) // PS0; cc = (np.arange(cw) * G) // PS0
            tids = (rr[:, None] * G + cc[None, :]).ravel()[keep]
            labs = flat[keep].astype(np.int64)
            comb = labs * NTOK + tids
            u, ac = np.unique(comb, return_counts=True)
            wsum_pred = tok.reshape(C, NTOK).T[(u % NTOK).astype(np.intp)] * ac[:, None]
            np.add.at(wsum, u // NTOK, wsum_pred); np.add.at(area, u // NTOK, ac)
    labels = np.nonzero(area > 0)[0]
    return labels.astype(np.int64), (wsum[labels] / area[labels, None]).astype(np.float32)


def main():
    model = load_miphei()
    imgs = list_images()
    print(f"{len(imgs)} images | running MIPHEI-vit @ {PS0} (native 0.5µm/px)")
    pix_parts, tok_parts = [], []
    for i, p in enumerate(imgs):
        he, inst = load_image_and_inst(p)
        pix = predict_image(model, he)
        lp, mp = pixel_means(inst, pix); lt, mt = token_means(inst, pix)
        for labels, feat, store in [(lp, mp, pix_parts), (lt, mt, tok_parts)]:
            if len(labels) == 0:
                continue
            df = pd.DataFrame({"cell_id": labels, "slide_name": slide_name_of(p)})
            for j, m in enumerate(MARKERS):
                df[f"mean_{m}"] = feat[:, j]
            store.append(df)
        if i % 40 == 0:
            print(f"  {i}/{len(imgs)}")
    for parts, tag in [(pix_parts, "miphei_pixel"), (tok_parts, "miphei_token")]:
        out = FEAT_DIR / f"cell_token_features_{tag}.parquet"
        pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
