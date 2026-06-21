"""
Per-cell token statistics from token-level predictions on PanNuke.

Mirror of cell_cls/pathocell_cls/build_cell_token_features.py. The ORION 16-marker
model is run zero-shot on each 256×256 PanNuke H&E image (one padded patch → 224),
and for each nucleus in the instance mask we record, per marker:

  mean_<marker>  area-weighted mean predicted intensity over the tokens the cell
                 footprint overlaps (weight = footprint area per token).

The nucleus label IS MIPHEI's cell_id, so MIPHEI's GT cell-type one-hot + 20/80
`split` are merged straight on by (slide_name, cell_id). Output one parquet:

  cell_id, slide_name, area_px, mean_<marker> ×16, <celltype> one-hot ×5, split

Geometry (--ps0): PanNuke is 256 px @ 40x (0.25 µm/px) = 64 µm.
  ps0=256 (DEFAULT, canonical): resize the whole 256 image to 224 (≈native 40x) — this
    matches MIPHEI's no-resize condition, so it's the apples-to-apples setting (→ 0.654).
  ps0=448: pad (white) the 256 image up to 448 then resize 224, keeping the trained
    0.5 µm/px scale (→ 0.641).
  Token id per pixel = floor(offset · G / ps0), so a nucleus pixel maps to the token the
  model saw; both choices beat MIPHEI (the win is robust to this knob).

Example
-------
  HF_HOME=.../foundation_models HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  python cell_cls/pannuke_cls/build_cell_token_features.py \
      --pred_dir outputs_orion_token_UNI2_baseline_bg0.2 --tag bg0.2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from utils import (
    DEFAULT_GT_PARQUET, DEFAULT_PRED_DIR, PATCH_SIZE_LEVEL0, TOKEN_GRID, NTOK,
    NUCLEI_CLASSES, FEAT_DIR, list_images, slide_name_of, load_image_and_mask,
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


def aggregate_image(inst: np.ndarray, pred: np.ndarray, C: int,
                    ps0: int = PATCH_SIZE_LEVEL0, G: int = TOKEN_GRID):
    """
    Area-weighted token aggregation per nucleus for one image.
    Returns (labels (n,), mean_feat (n, C), area (n,)).
    """
    H, W = inst.shape
    flat = inst.ravel()
    keep = flat > 0
    if not keep.any():
        return np.empty(0, np.int64), np.empty((0, C), np.float32), np.empty(0, np.int64)
    # token id per pixel: divide by FULL padded patch size (image sits in top-left)
    rr = (np.arange(H) * G) // ps0
    cc = (np.arange(W) * G) // ps0
    tok = (rr[:, None] * G + cc[None, :]).ravel()[keep]
    labs = flat[keep].astype(np.int64)

    comb = labs * NTOK + tok
    u_comb, area_ct = np.unique(comb, return_counts=True)
    cell_of = u_comb // NTOK
    tok_of  = (u_comb % NTOK).astype(np.intp)

    pred_tok = pred.reshape(C, NTOK).T                    # (256, C)
    wpred = pred_tok[tok_of] * area_ct[:, None]

    labels = np.unique(cell_of)
    lab2row = {int(l): i for i, l in enumerate(labels)}
    rows = np.array([lab2row[int(c)] for c in cell_of])
    wsum = np.zeros((len(labels), C), np.float64)
    area = np.zeros(len(labels), np.float64)
    np.add.at(wsum, rows, wpred)
    np.add.at(area, rows, area_ct)
    return labels.astype(np.int64), (wsum / area[:, None]).astype(np.float32), area.astype(np.int64)


NATIVE_PS0 = 256   # PanNuke image size: resize the whole 256 image to 224 (≈native 40x,
                   # MIPHEI's condition). This is the CANONICAL default (gives 0.654).


def build(pred_dir: str, out_path: str, gt_parquet: str, batch_size: int,
          limit: int | None = None, ps0: int = NATIVE_PS0) -> pd.DataFrame:
    model, marker_names = load_model(pred_dir)
    C = len(marker_names)

    gt = pd.read_parquet(gt_parquet,
                         columns=["cell_id", "slide_name"] + NUCLEI_CLASSES + ["split"])
    gt["slide_name"] = gt["slide_name"].astype(str)
    gt["cell_id"]    = gt["cell_id"].astype(np.int64)

    imgs = list_images()
    if limit:
        imgs = imgs[:limit]
    print(f"{len(imgs)} images | ps0={ps0} ({'native resize→224' if ps0==NATIVE_PS0 else 'pad→scale-matched'}) | markers={C}")

    # batch images (one patch each) through the model, then aggregate
    parts = []
    B = batch_size
    for s in range(0, len(imgs), B):
        chunk = imgs[s:s + B]
        masks, names = [], []
        patches = []
        for p in chunk:
            he, inst = load_image_and_mask(p)
            patches.append(normalise_patch(he, ps0=ps0))
            masks.append(inst); names.append(slide_name_of(p))
        preds = run_inference(model, np.stack(patches), B)         # (b, C, G, G)
        for inst, name, pr in zip(masks, names, preds):
            labels, mean_feat, area = aggregate_image(inst, pr, C, ps0=ps0)
            if len(labels) == 0:
                continue
            df = pd.DataFrame({"cell_id": labels, "slide_name": name, "area_px": area})
            for j, m in enumerate(marker_names):
                df[f"mean_{m}"] = mean_feat[:, j]
            parts.append(df)
        if s % (B * 20) == 0:
            print(f"  {s}/{len(imgs)} images")

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
    ap.add_argument("--tag", default=None, help="model tag for output name")
    ap.add_argument("--out", default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="cap #images (quick test)")
    ap.add_argument("--ps0", type=int, default=NATIVE_PS0,
                    help="native patch size. 256 (DEFAULT, canonical) = resize whole 256 image "
                         "to 224 (~native 40x, matches MIPHEI's condition → 0.654). "
                         "448 = pad to 0.5µm/px scale-matched (→ 0.641).")
    args = ap.parse_args()

    tag = args.tag or Path(args.pred_dir).name.replace("training_outputs/outputs_orion_token_UNI2_", "")
    out = args.out or str(FEAT_DIR / f"cell_token_features_{tag}.parquet")
    build(args.pred_dir, out, args.gt_parquet, args.batch_size, args.limit, args.ps0)


if __name__ == "__main__":
    main()
