"""
Per-cell token statistics from HEMIT token-level predictions (v4 — clean native geometry).

Single coordinate space (mirrors cell_cls/pathocell_cls). The dataset/pred h5 stores
patch top-lefts in NATIVE 1024 space with a non-overlapping 448-px grid; the nuclei
mask + csv are native 1024 too, so the cell↔token map is one floor-division — no
40x→20x ÷2, no 512 intermediate, no overlap, no "most-centered framing" dedup.

Geometry recap (build_patch_dataset_hemit_token.py v4):
  source tile : 1024² @ 40x (0.25 µm/px)   ← H&E + IF + nuclei mask, all native
  patch       : 448 native px → resize 224  (112 µm FOV @ 20x); top-lefts {0,448,896}
  edge patch  : padded to 448 before resize, so token = floor(offset · G / 448)
                — divide by the FULL patch size, never the clamped crop size.

A nucleus straddling a 448 boundary just gets area-weighted contributions from both
patches (its pixels are partitioned across them) — still a correct mean over the
whole footprint, accumulated into global per-label sums.

Per cell, per marker:
  mean_<marker>  nucleus-area-weighted mean predicted intensity over the tokens the
                 footprint overlaps (weight = #nucleus pixels in each token).
                 With --centroid_token: just the single token under the centroid.
GT positivity (Pan-CK_pos, CD3_pos) is read straight from the per-tile csv.

Predictions source:
  --pred_h5 PATH   h5 with /coords (N,2 NATIVE), /sources (N,), /preds (N,C,G,G)
                   + attrs marker_names, patch_size_level0, token_grid.
  --from_targets   self-test: use the dataset h5's /targets as a perfect predictor
                   (validates the nuclei↔token mapping before a model exists).

Example
-------
  # geometry self-test on GT targets:
  python cell_cls/hemit_cell_cls/build_cell_token_features.py \
      --split test --from_targets hemit_patch_dataset/test.h5 \
      --out cell_cls/hemit_cell_cls/cell_token_features_test.parquet
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tifffile

NUCLEI_DIR = Path("/mnt/ssd/virtual_proteomics/data/HEMIT_nuclei_analysis")
GT_MARKERS = ["Pan-CK", "CD3"]            # markers with a _pos column in the csv


def load_pred_source(args):
    """Return coords(N,2 NATIVE int), sources(N,), preds(N,C,G,G float32), names, geom."""
    path = args.from_targets or args.pred_h5
    key = "targets" if args.from_targets else "preds"
    with h5py.File(path, "r") as f:
        coords = f["coords"][:].astype(np.int64)
        sources = f["sources"][:].astype(str)
        preds = f[key][:].astype(np.float32)
        marker_names = [m.decode() if isinstance(m, bytes) else str(m)
                        for m in f.attrs["marker_names"]]
        geom = {"ps0": int(f.attrs["patch_size_level0"]),
                "token_grid": int(f.attrs["token_grid"])}
    return coords, sources, preds, marker_names, geom


def nuclei_paths(split: str, tile_name: str):
    """mask .tif + csv for a source tile (tile_name = the input .tif filename)."""
    stem = Path(tile_name).stem
    return (NUCLEI_DIR / split / "mask" / f"{stem}.tif",
            NUCLEI_DIR / split / "csv" / f"{stem}.csv")


def build(args) -> pd.DataFrame:
    coords, sources, preds, marker_names, geom = load_pred_source(args)
    C  = preds.shape[1]
    G  = geom["token_grid"]
    ps0 = geom["ps0"]                                            # 448 native px per patch

    rows = []
    tiles = np.unique(sources)
    print(f"[{args.split}] {len(sources)} patches over {len(tiles)} tiles | "
          f"ps0={ps0} native px | token≈{ps0//G} native px | centroid_token={args.centroid_token}")

    for ti, tile in enumerate(tiles):
        if ti % 200 == 0:
            print(f"  {ti}/{len(tiles)}")
        mask_p, csv_p = nuclei_paths(args.split, tile)
        if not mask_p.exists() or not csv_p.exists():
            continue
        cells = pd.read_csv(csv_p)
        if len(cells) == 0:
            continue
        mask = tifffile.imread(str(mask_p))                      # (1024,1024) uint, NATIVE
        H, W = mask.shape
        max_lab = int(mask.max())
        if max_lab == 0:
            continue

        # csv centroids: skimage (row, col) named X=row, Y=col.
        # image horizontal (col, matches patch x) = Y_centroid; vertical (row) = X_centroid.
        cell_cx = cells["Y_centroid"].values.astype(np.float64)  # NATIVE, horizontal (col)
        cell_cy = cells["X_centroid"].values.astype(np.float64)  # NATIVE, vertical (row)
        labels = cells["label"].values.astype(np.int64)
        nlab = len(labels)
        lab2row = np.full(max_lab + 1, -1, np.int64)
        lab2row[labels] = np.arange(nlab)

        # this tile's patches (NATIVE top-lefts) + their token-pred grids
        idx = np.where(sources == tile)[0]
        patch_xy = coords[idx]                                   # (P,2) NATIVE
        patch_pred = preds[idx]                                  # (P,C,G,G)

        if args.centroid_token:
            # one token under each centroid: locate its patch, then floor(offset·G/ps0)
            best = np.zeros((nlab, C), np.float32)
            got  = np.zeros(nlab, bool)
            for p in range(len(idx)):
                X, Y = int(patch_xy[p, 0]), int(patch_xy[p, 1])
                pred_tok = patch_pred[p].reshape(C, G * G).T
                inside = (cell_cx >= X) & (cell_cx < X + ps0) & \
                         (cell_cy >= Y) & (cell_cy < Y + ps0) & ~got
                if not inside.any():
                    continue
                tj = ((cell_cx[inside] - X) * G // ps0).astype(np.int64)
                ti_ = ((cell_cy[inside] - Y) * G // ps0).astype(np.int64)
                best[inside] = pred_tok[ti_ * G + tj]
                got[inside] = True
            keep = got
            best_mean = best
        else:
            # area-weighted: accumulate over ALL patches into global per-label sums
            wsum = np.zeros((nlab, C), np.float64)
            area = np.zeros(nlab, np.float64)
            for p in range(len(idx)):
                X, Y = int(patch_xy[p, 0]), int(patch_xy[p, 1])
                sub = mask[Y:min(Y + ps0, H), X:min(X + ps0, W)]   # native crop
                ch, cw = sub.shape
                flat = sub.ravel()
                sel = flat > 0
                if not sel.any():
                    continue
                # token id per pixel: floor(offset · G / ps0) — divide by FULL patch
                rr = (np.arange(ch) * G) // ps0
                cc = (np.arange(cw) * G) // ps0
                tok = (rr[:, None] * G + cc[None, :]).ravel()[sel]
                row = lab2row[flat[sel]]
                valid = row >= 0                                  # mask labels absent from csv
                if not valid.any():
                    continue
                row, tok = row[valid], tok[valid]
                comb = row.astype(np.int64) * (G * G) + tok
                u, area_ct = np.unique(comb, return_counts=True)
                r_of = (u // (G * G)).astype(np.intp)
                t_of = (u %  (G * G)).astype(np.intp)
                predvec = patch_pred[p].reshape(C, G * G).T[t_of]  # (n_pairs, C)
                np.add.at(area, r_of, area_ct)
                np.add.at(wsum, r_of, predvec * area_ct[:, None])
            keep = area > 0
            best_mean = np.zeros((nlab, C), np.float32)
            best_mean[keep] = (wsum[keep] / area[keep, None]).astype(np.float32)

        if not keep.any():
            continue
        df = pd.DataFrame({"label": labels[keep]})
        for j, m in enumerate(marker_names):
            df[f"mean_{m}"] = best_mean[keep, j]
        df["X_centroid"] = cell_cx[keep]
        df["Y_centroid"] = cell_cy[keep]
        for m in GT_MARKERS:
            col = f"{m}_pos"
            if col in cells.columns:
                df[f"gt_{m}_pos"] = cells[col].values[keep].astype(np.int8)
        df["image_name"] = Path(tile).stem
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"\nSaved {len(out)} cells x {len(marker_names)} markers -> {args.out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pred_h5", help="predictions h5 (/preds aligned to /coords,/sources)")
    src.add_argument("--from_targets", help="dataset h5; use /targets as a perfect predictor")
    ap.add_argument("--centroid_token", action="store_true",
                    help="single token under the centroid (csv-only flavor / ablation)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    args.out = args.out or f"cell_cls/hemit_cell_cls/cell_token_features_{args.split}.parquet"
    build(args)


if __name__ == "__main__":
    main()