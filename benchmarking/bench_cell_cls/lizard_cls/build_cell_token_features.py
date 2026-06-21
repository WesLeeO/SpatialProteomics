"""
Per-cell token statistics from token-level predictions on Lizard.

Mirror of cell_cls/pathocell_cls/build_cell_token_features.py. The ORION 16-marker
model runs zero-shot on each Lizard H&E image; for each nucleus (from the .mat
inst_map) we record per marker:

  mean_<marker>  area-weighted mean predicted intensity over the tokens the cell
                 footprint overlaps (weight = footprint area per token).

The nucleus label IS MIPHEI's cell_id, so MIPHEI's GT one-hot + 20/80 `split` are
merged on by (slide_name, cell_id). Output one parquet:

  cell_id, slide_name, area_px, mean_<marker> ×16, <celltype> one-hot ×6, split

Geometry: Lizard is 20x (0.5 µm/px) = our training scale, so PATCH_SIZE_LEVEL0=224 and
a native 224 crop IS the model input (no rescale). Images are variable-size, so each is
tiled into a non-overlapping 224 grid (edges padded white). A nucleus straddling a tile
boundary gets area-weighted contributions from both tiles. token = floor(offset·G/224).

Example
-------
  HF_HOME=.../foundation_models HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  python cell_cls/lizard_cls/build_cell_token_features.py --tag bg0.2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    DEFAULT_GT_PARQUET, DEFAULT_PRED_DIR, PATCH_SIZE_LEVEL0, TOKEN_GRID, NTOK,
    NUCLEI_CLASSES, FEAT_DIR, list_images, slide_name_of, load_image_and_inst,
    load_model, normalise_patch,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(model, patches: np.ndarray, batch_size: int) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(patches)),
                        batch_size=batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for (batch,) in loader:
            with torch.autocast("cuda", enabled=torch.cuda.is_available()):
                pred, _ = model(batch.to(DEVICE))
            out.append(pred.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def aggregate_image(he: np.ndarray, inst: np.ndarray, model, C: int, batch_size: int,
                    ps0: int = PATCH_SIZE_LEVEL0, G: int = TOKEN_GRID):
    """Tile the whole image into a 224 grid, predict, area-weighted aggregate per nucleus."""
    H, W = inst.shape
    patches, boxes = [], []
    for y in range(0, H, ps0):
        for x in range(0, W, ps0):
            y1, x1 = min(y + ps0, H), min(x + ps0, W)
            patches.append(normalise_patch(he[y:y1, x:x1]))
            boxes.append((x, y, x1, y1))
    preds = run_inference(model, np.stack(patches), batch_size)    # (Ntiles, C, G, G)

    max_lab   = int(inst.max())
    wsum      = np.zeros((max_lab + 1, C), np.float64)
    area_cell = np.zeros(max_lab + 1, np.float64)
    for ti, (x, y, x1, y1) in enumerate(boxes):
        sub = inst[y:y1, x:x1]
        ch, cw = sub.shape
        flat = sub.ravel(); keep = flat > 0
        if not keep.any():
            continue
        rr = (np.arange(ch) * G) // ps0
        cc = (np.arange(cw) * G) // ps0
        tok = (rr[:, None] * G + cc[None, :]).ravel()[keep]
        labs = flat[keep].astype(np.int64)
        comb = labs * NTOK + tok
        u_comb, area_ct = np.unique(comb, return_counts=True)
        cell_of = u_comb // NTOK
        tok_of  = (u_comb % NTOK).astype(np.intp)
        wpred = preds[ti].reshape(C, NTOK).T[tok_of] * area_ct[:, None]
        np.add.at(wsum, cell_of, wpred)
        np.add.at(area_cell, cell_of, area_ct)

    labels = np.nonzero(area_cell > 0)[0]
    mean_feat = wsum[labels] / area_cell[labels, None]
    return labels.astype(np.int64), mean_feat.astype(np.float32), area_cell[labels].astype(np.int64)


def build(pred_dir: str, out_path: str, gt_parquet: str, batch_size: int,
          limit: int | None = None) -> pd.DataFrame:
    model, marker_names = load_model(pred_dir)
    C = len(marker_names)
    gt = pd.read_parquet(gt_parquet,
                         columns=["cell_id", "slide_name"] + NUCLEI_CLASSES + ["split"])
    gt["slide_name"] = gt["slide_name"].astype(str)
    gt["cell_id"]    = gt["cell_id"].astype(np.int64)

    imgs = list_images()
    if limit:
        imgs = imgs[:limit]
    print(f"{len(imgs)} images | ps0={PATCH_SIZE_LEVEL0} (native, no resize) | markers={C}")

    parts = []
    for i, p in enumerate(imgs):
        he, inst = load_image_and_inst(p)
        labels, mean_feat, area = aggregate_image(he, inst, model, C, batch_size)
        if len(labels) == 0:
            continue
        df = pd.DataFrame({"cell_id": labels, "slide_name": slide_name_of(p), "area_px": area})
        for j, m in enumerate(marker_names):
            df[f"mean_{m}"] = mean_feat[:, j]
        parts.append(df)
        if i % 50 == 0:
            print(f"  [{i}/{len(imgs)}] {slide_name_of(p):<14} {len(df)} cells")

    feats = pd.concat(parts, ignore_index=True)
    merged = feats.merge(gt, on=["slide_name", "cell_id"], how="inner")
    print(f"\nmerged {len(merged)} cells with GT "
          f"({len(feats)-len(merged)} of {len(feats)} ours had no MIPHEI label)")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f"saved {len(merged)} cells x {C} markers -> {out_path}")
    print(merged["split"].value_counts().to_string())
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", default=str(DEFAULT_PRED_DIR))
    ap.add_argument("--gt_parquet", default=str(DEFAULT_GT_PARQUET))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    tag = args.tag or Path(args.pred_dir).name.replace("training_outputs/outputs_orion_token_UNI2_", "")
    out = args.out or str(FEAT_DIR / f"cell_token_features_{tag}.parquet")
    build(args.pred_dir, out, args.gt_parquet, args.batch_size, args.limit)


if __name__ == "__main__":
    main()
