"""
Per-cell token statistics from token-level predictions on PathoCell.

Mirror of orion_cell_cls/build_cell_token_features.py, but for the external
CRC-CODEX (PathoCell) cohort and for cell-TYPE classification rather than
per-marker positivity.

For one core we tile the H&E into PATCH_SIZE_LEVEL0 (297 px) crops, read each
crop's 16x16 token prediction grid, and for each nucleus in `gt_inst` record,
per marker:

  mean_<marker>  area-weighted mean predicted intensity over the tokens the cell
                 footprint overlaps, weighting each token by the footprint area
                 inside it (token analog of MIPHEI's regionprops mean-intensity-
                 per-nucleus).

plus area_px (footprint pixels) and the source core (`tile`) as the bootstrap
unit. The nucleus label IS MIPHEI's cell_id, so MIPHEI's GT cell-type one-hot +
train/test `split` are merged straight on by (slide_name, cell_id) —
model-independent GT, exactly as in the ORION folder.

Tiling is non-overlapping and covers the whole core, so a nucleus straddling a
tile boundary just gets area-weighted contributions from both tiles. No
re-segmentation (cells come from MIPHEI's gt_inst), so the comparison is faithful.

Example
-------
  python cell_cls/pathocell_cls/build_cell_token_features.py \
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
    NUCLEI_CLASSES, FEAT_DIR, list_cores, slide_name_of, load_hdf_core,
    load_model, normalise_patch,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(model, patches: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """(N,3,224,224) float32 -> (N, C, G, G) token predictions."""
    loader = DataLoader(TensorDataset(torch.from_numpy(patches)),
                        batch_size=batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for (batch,) in loader:
            with torch.autocast("cuda", enabled=torch.cuda.is_available()):
                pred, _ = model(batch.to(DEVICE))
            out.append(pred.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def core_boxes(H: int, W: int, ps0: int = PATCH_SIZE_LEVEL0):
    """Non-overlapping tile grid over a core, row-major. Returns [(x,y,x1,y1), ...]."""
    return [(x, y, min(x + ps0, W), min(y + ps0, H))
            for y in range(0, H, ps0) for x in range(0, W, ps0)]


def accumulate_tokens(inst: np.ndarray, preds: np.ndarray, boxes: list, C: int,
                      G: int = TOKEN_GRID):
    """Area-weighted token aggregation per nucleus, given per-tile token grids.

    `preds` is (Ntiles, C, G, G) aligned to `boxes`. Works for ANY model's token
    grids (ours or a re-tokenised competitor), so the aggregation is identical and
    only the prediction values differ. Returns (labels, mean_feat (n,C), area (n,))."""
    max_lab   = int(inst.max())
    wsum      = np.zeros((max_lab + 1, C), np.float64)
    area_cell = np.zeros(max_lab + 1, np.float64)

    for ti, (x, y, x1, y1) in enumerate(boxes):
        sub  = inst[y:y1, x:x1]                               # (ch, cw)
        ch, cw = sub.shape
        flat = sub.ravel()
        keep = flat > 0
        if not keep.any():
            continue
        # token id per pixel: floor(r*G/ch), floor(c*G/cw)  (crop is resized to 224
        # then block-averaged into G×G, so the mapping is proportional to crop dims)
        rr  = (np.arange(ch) * G) // ch                      # (ch,) in [0, G)
        cc  = (np.arange(cw) * G) // cw                      # (cw,)
        tok = (rr[:, None] * G + cc[None, :]).ravel()[keep]
        labs = flat[keep].astype(np.int64)

        # one np.unique over (cell, token) pairs -> footprint area per pair
        comb = labs * NTOK + tok
        u_comb, area_ct = np.unique(comb, return_counts=True)
        cell_of = (u_comb // NTOK)
        tok_of  = (u_comb %  NTOK).astype(np.intp)

        pred_tok = preds[ti].reshape(C, NTOK).T              # (256, C): row t = token t
        wpred    = pred_tok[tok_of] * area_ct[:, None]
        np.add.at(wsum, cell_of, wpred)
        np.add.at(area_cell, cell_of, area_ct)

    labels = np.nonzero(area_cell > 0)[0]
    mean_feat = wsum[labels] / area_cell[labels, None]
    return labels, mean_feat.astype(np.float32), area_cell[labels].astype(np.int64)


def aggregate_core(he: np.ndarray, inst: np.ndarray, model, C: int,
                   batch_size: int, ps0: int = PATCH_SIZE_LEVEL0,
                   G: int = TOKEN_GRID):
    """Run OUR model over a core's tile grid → token grids → per-nucleus aggregation."""
    H, W = inst.shape
    boxes   = core_boxes(H, W, ps0)
    patches = np.stack([normalise_patch(he[y:y1, x:x1]) for (x, y, x1, y1) in boxes], axis=0)
    preds   = run_inference(model, patches, batch_size)      # (Ntiles, C, G, G)
    return accumulate_tokens(inst, preds, boxes, C, G)


def build(pred_dir: str, out_path: str, gt_parquet: str, batch_size: int,
          cores: list[str] | None = None) -> pd.DataFrame:
    model, marker_names = load_model(pred_dir)
    C = len(marker_names)

    gt = pd.read_parquet(gt_parquet,
                         columns=["cell_id", "slide_name"] + NUCLEI_CLASSES + ["split"])
    gt["slide_name"] = gt["slide_name"].astype(str)
    gt["cell_id"]    = gt["cell_id"].astype(np.int64)

    cores = cores or list_cores()
    print(f"{len(cores)} cores | crop={PATCH_SIZE_LEVEL0}px | markers={C}")

    parts = []
    for ci, core in enumerate(cores):
        he, inst = load_hdf_core(core)
        labels, mean_feat, area = aggregate_core(he, inst, model, C, batch_size)
        df = pd.DataFrame({"label": labels.astype(np.int64),       # == MIPHEI cell_id
                           "slide_name": slide_name_of(core),
                           "tile": core,                           # bootstrap unit (core)
                           "area_px": area})
        for j, m in enumerate(marker_names):
            df[f"mean_{m}"] = mean_feat[:, j]
        parts.append(df)
        print(f"  [{ci+1}/{len(cores)}] {core:<12} {len(df):>5} cells")

    feats = pd.concat(parts, ignore_index=True)
    # join GT celltype one-hot + split on (slide_name, cell_id) — keep only scored cells
    merged = feats.merge(gt, left_on=["slide_name", "label"],
                         right_on=["slide_name", "cell_id"], how="inner")
    dropped = len(feats) - len(merged)
    print(f"\nmerged {len(merged)} cells with GT "
          f"({dropped} of {len(feats)} ours had no MIPHEI label)")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)
    print(f"saved {len(merged)} cells x {C} markers -> {out_path}")
    print(merged["split"].value_counts().to_string())
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", default=str(DEFAULT_PRED_DIR))
    ap.add_argument("--gt_parquet", default=str(DEFAULT_GT_PARQUET),
                    help="source of GT celltype one-hot + split (model-independent)")
    ap.add_argument("--tag", default=None,
                    help="model tag for the output name (default: from --pred_dir)")
    ap.add_argument("--out", default=None,
                    help="output parquet (default cell_token_features_<tag>.parquet)")
    ap.add_argument("--cores", default="all",
                    help="'all' or comma-separated core stems, e.g. reg001_A,reg001_B")
    ap.add_argument("--batch_size", type=int, default=256,
                    help="tiles per forward pass (lower for a busy/small GPU)")
    args = ap.parse_args()

    tag = args.tag or Path(args.pred_dir).name.replace("training_outputs/outputs_orion_token_UNI2_", "")
    out = args.out or str(FEAT_DIR / f"cell_token_features_{tag}.parquet")
    cores = None if args.cores == "all" else [c.strip() for c in args.cores.split(",")]
    build(args.pred_dir, out, args.gt_parquet, args.batch_size, cores)


if __name__ == "__main__":
    main()