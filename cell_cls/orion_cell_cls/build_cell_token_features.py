"""
Per-cell token statistics from token-level ORION predictions.

For one slide we walk every benchmark patch, read its 16x16 token prediction grid
and the native-20x nuclei mask crop, and for each cell (attributed to the single
patch that contains its centroid) record, per marker:

  mean_<marker>  area-weighted mean predicted intensity over the tokens the cell
                 footprint overlaps, weighting each token by the footprint area
                 inside it (token analog of MIPHEI's regionprops mean-intensity-
                 per-nucleus)

plus area_px (footprint pixels in the patch crop) and n_tokens (tokens touched).
GT positivity (`<marker>_pos`, MIPHEI's Zenodo GMM gating on the real IF) is
merged in as gt_<marker>_pos.

This is the feature table for the deterministic rule in eval_deterministic.py:
    a cell is predicted <marker>+ iff mean_<marker> > threshold.

Token aggregation note: our model emits a coarse 16x16 grid, not pixel IF, so
"intensity over the cell" is summarised from the tokens the cell overlaps. A
cell is bound to exactly one patch (the one holding its centroid), so no
cross-patch merging is needed.

Example
-------
  python orion_cell_cls/build_cell_token_features.py \
      --sample CRC30 \
      --pred_dir outputs_orion_token_UNI2_baseline_unfreeze4_2loss_lbg8 \
      --out orion_cell_cls/cell_token_features_CRC30.parquet
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils import (
    sample_rows, csv_marker_col, index_nuclei_tiles, load_patch_labels,
    token_ids, load_slide_predictions, TILE_DIR, NTOK, DEFAULT_PRED_DIR,
)


def build(sample: str, pred_dir: str, out_path: str, max_patches: int | None = None) -> pd.DataFrame:
    coords, ps0, marker_names, preds = load_slide_predictions(sample, pred_dir)
    C = preds.shape[1]

    row   = sample_rows(sample)
    cells = pd.read_csv(TILE_DIR / row.nuclei_csv_path)
    cx, cy = cells.x.values, cells.y.values
    max_lab = int(cells.label.max())
    # centroid indexed by label, to pick each cell's best (most-centered) framing
    cent_x = np.zeros(max_lab + 1, np.float64); cent_x[cells.label.values] = cx
    cent_y = np.zeros(max_lab + 1, np.float64); cent_y[cells.label.values] = cy

    base       = Path(row.nuclei_slide_path).stem
    tile_index = index_nuclei_tiles(base)
    print(f"[{sample}] {len(coords)} patches  {len(cells)} cells  {len(tile_index)} mask tiles")

    # Benchmark patches OVERLAP (side=ps0 but grid stride<ps0), so a centroid can
    # fall in up to 4 patches. Keep, per cell, only the framing where it is most
    # centered (smallest centroid→patch-centre distance) → one token-overlap set.
    best_dist  = np.full(max_lab + 1, np.inf, np.float64)
    best_mean  = np.zeros((max_lab + 1, C), np.float32)
    best_area  = np.zeros(max_lab + 1, np.int64)
    best_ntok  = np.zeros(max_lab + 1, np.int32)
    best_patch = np.full(max_lab + 1, -1, np.int32)   # winning patch idx → tile id for bootstrap

    n = len(coords) if max_patches is None else min(max_patches, len(coords))
    for i in range(n):
        if i % 2000 == 0:
            print(f"  {i}/{n}")
        px, py = int(coords[i, 0]), int(coords[i, 1])
        lab = load_patch_labels(tile_index, px, py, ps0)            # (mh, mw) int
        mh, mw = lab.shape
        if mh == 0 or mw == 0:
            continue

        # cells whose centroid is inside this patch — single-attribution
        sel = (cx >= px) & (cx < px + ps0) & (cy >= py) & (cy < py + ps0)
        if not sel.any():
            continue
        fov = np.zeros(max_lab + 1, dtype=bool)
        fov[cells.label.values[sel]] = True

        flat = lab.ravel()
        keep = (flat > 0) & fov[np.clip(flat, 0, max_lab)]
        if not keep.any():
            continue
        labs_px = flat[keep]                                        # global cell labels
        toks_px = token_ids(mh, mw)[keep]                           # 0..255

        # Group pixels into (cell, token) buckets with their pixel-area, all in C
        # via one np.unique. Remap big global labels → compact local ids 0..ncell-1
        # (labs_px == u_cell[inv]) so downstream arrays are size ncell, not max_lab.
        u_cell, inv = np.unique(labs_px, return_inverse=True)       # u_cell: global labels present
        ncell = len(u_cell)
        # encode each pixel's (local_cell, token) pair as one int: cell*256 + token
        # (bijective since token ∈ [0, NTOK)); int64 guards against overflow.
        """
        comb = inv * 256 + toks_px
        comb = [0*256+10, 0*256+10, 0*256+11, 1*256+11, 1*256+11, 1*256+12]
            = [   10,       10,       11,       267,      267,      268   ]
        Each (cell, token) pair now has a unique number (token < 256 guarantees no collision). Two pixels in the same cell+token get the same number (the two 10s, the two 267s).
        """
        comb = inv.astype(np.int64) * NTOK + toks_px
        # unique pairs + how many pixels each has = the cell's footprint AREA in that token
        u_comb, area_ct = np.unique(comb, return_counts=True)
        cell_of = (u_comb // NTOK).astype(np.intp)                  # decode → local cell idx
        tok_of  = (u_comb %  NTOK).astype(np.intp)                  # decode → token id 0..255

        # token's predicted marker vector; reshape is row-major to match token_ids()
        pred_tok = preds[i].reshape(C, NTOK).T                      # (256, C): row t = token t
        predvec  = pred_tok[tok_of]                                 # (n_pairs, C): pred at each pair's token

        """
        token 10 → [0.8, 0.1, 0.0]
        token 11 → [0.2, 0.6, 0.0]
        token 12 → [0.0, 0.9, 0.1]
        Then predvec = pred_tok[[10,11,11,12]]:
        predvec = [[0.8,0.1,0.0],   # pair0
        [0.2,0.6,0.0],   # pair1
        [0.2,0.6,0.0],   # pair2
        [0.0,0.9,0.1]]   # pair3
        """

        # area-weighted mean over overlapped tokens
        # cell_of is unique cells ids in the patch, area_ct is numebr of pixels inside each unique (token, cell) -> total pixels per cell
        area_cell = np.bincount(cell_of, weights=area_ct, minlength=ncell)
        wsum = np.empty((ncell, C), dtype=np.float64)
        wpred = predvec * area_ct[:, None] # multiplied by number of pixels in pair (token, cell) + predictions for all pairs
        for c in range(C):
            wsum[:, c] = np.bincount(cell_of, weights=wpred[:, c], minlength=ncell)
        mean_feat = wsum / area_cell[:, None]

        n_tok = np.bincount(cell_of, minlength=ncell)               # tokens touched (pairs/cell)

        # keep this framing only for cells more centred here than seen before
        pcx, pcy = px + ps0 / 2.0, py + ps0 / 2.0
        dist = (cent_x[u_cell] - pcx) ** 2 + (cent_y[u_cell] - pcy) ** 2
        better = dist < best_dist[u_cell]
        lab_b = u_cell[better]
        best_dist[lab_b]  = dist[better]
        best_mean[lab_b]  = mean_feat[better].astype(np.float32)
        best_area[lab_b]  = area_cell[better].astype(np.int64)
        best_ntok[lab_b]  = n_tok[better].astype(np.int32)
        best_patch[lab_b] = i

    labels = np.nonzero(np.isfinite(best_dist))[0]
    if labels.size == 0:
        raise RuntimeError("no cells accumulated — check pred cache / patch coords")

    df = pd.DataFrame({"label": labels,
                       "tile": best_patch[labels],     # source patch idx (tile bootstrap unit)
                       "area_px": best_area[labels],
                       "n_tokens": best_ntok[labels]})
    for j, m in enumerate(marker_names):
        df[f"mean_{m}"] = best_mean[labels, j]

    # centroids + GT positivity
    df = df.merge(cells[["label", "x", "y"]], on="label", how="left")
    gt_n = 0
    for m in marker_names:
        try:
            col = csv_marker_col(cells, m)                          # e.g. E-Cadherin -> E-cadherin_pos
        except KeyError:
            continue                                               # Hoechst etc. have no _pos
        lut = pd.Series(cells[col].values.astype(np.int8), index=cells.label.values)
        df[f"gt_{m}_pos"] = df.label.map(lut).astype("Int8")
        gt_n += 1
    df["slide"] = sample

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(df)} cells x {len(marker_names)} markers ({gt_n} with GT) -> {out_path}")
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="CRC30")
    ap.add_argument("--pred_dir", default=str(DEFAULT_PRED_DIR))
    ap.add_argument("--out", default=None,
                    help="output parquet (default orion_cell_cls/cell_token_features_<sample>.parquet)")
    ap.add_argument("--max_patches", type=int, default=None,
                    help="cap patches for a quick test run")
    args = ap.parse_args()
    out = args.out or f"orion_cell_cls/cell_token_features_{args.sample}.parquet"
    build(args.sample, args.pred_dir, out, args.max_patches)


if __name__ == "__main__":
    main()