"""
Build the ORION_CRCv2 token h5 by reusing the existing benchmark patch coordinates
and resampling the IF target from the MIPHEI 20x tile dataset.

Why this exists
---------------
The benchmark coords (level-0, from TRIDENT on the original continuous slide) hug the
tissue border cleanly. The benchmark IF target had artifacts; the MIPHEI 20x tiles are
already-cleaned (AF-corrected + 8-bit). The benchmark level-0 space and the MIPHEI tile
space are the SAME registered slide (verified: HE cross-correlation NCC>0.95 at coord*SCALE),
so we can keep each benchmark patch and just resample its IF from the 20x reconstruction.

Per slide
---------
1. stitch only the 16-marker IF canvas (transient disk memmap) from the 20x tiles
   + a coverage mask of which pixels are backed by a real tile.
2. load benchmark coords (level-0), map to 20x px = round(coord * 333/512), and DROP
   patches whose 20x window leaves the canvas or lacks IF coverage (MIPHEI-removed tile).
3. token target = mean(IF_uint8/255) over each 14px UNI2 token cell -> (N, C, 16, 16).
4. save <DATASET_DIR>/<CRC>_patch_dataset.h5 mirroring the benchmark h5 (level-0 `coords`
   + `patch_size_level0`), so it is drop-in for visualize_orion_predictions.py.

NO H&E reconstruction, NO TRIDENT: H&E for training/visualisation still comes from the
original slide via the level-0 coords (unchanged benchmark pipeline). Just run it:

    python build_orion_tile20x_dataset.py            # all slides
    python build_orion_tile20x_dataset.py --slides CRC02   # one slide (testing)
"""

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tifffile

from artifact_filter_orion import apply_lineage_gate, apply_shape_gate, apply_block_gate

# ── Paths / constants ───────────────────────────────────────────────────────────
TILE_ROOT = Path("/mnt/ssd/virtual_proteomics/data/ORIONCRC_dataset_tile_20x")
SPLIT_CSVS = ["train_dataframe.csv", "val_dataframe.csv", "test_dataframe.csv"]
SLIDE_CSV = "slide_dataframe.csv"
BENCHMARK_DIR = Path("datasets/orion_crc_patch_dataset_benchmark")
DATASET_DIR = Path("datasets/orion_crcv2_patch_dataset")
TMP_DIR = Path("/mnt/ssd/virtual_proteomics/data/ORION_CRCv2/_tmp")

TILE_PX = 333
SCALE = TILE_PX / 512        # level-0 px -> 20x saved px
MPP_20X = 0.5
TOKEN_GRID = 16
MIN_COVERAGE = 1.0           # require FULL IF coverage (no zero-filled pixels in kept patches)

# 16 benchmark markers (drop PD-1=13) -> (tile channel index, benchmark name)
MARKERS = [
    (0,  "Hoechst"), (1, "CD31"), (2, "CD45"), (3, "CD68"), (4, "CD4"),
    (5,  "FOXP3"), (6, "CD8a"), (7, "CD45RO"), (8, "CD20"), (9, "PD-L1"),
    (10, "CD3e"), (11, "CD163"), (12, "E-Cadherin"), (14, "Ki-67"),
    (15, "Pan-CK"), (16, "SMA"),
]
TILE_CHANNELS = [c for c, _ in MARKERS]
MARKER_NAMES = [n for _, n in MARKERS]

_COORD_RE = re.compile(r"_(\d+)_(\d+)_\d+_\d+_\d+\.(?:jpeg|tiff)$")


def load_tile_table() -> pd.DataFrame:
    df = pd.concat([pd.read_csv(TILE_ROOT / c) for c in SPLIT_CSVS], ignore_index=True)
    sd = pd.read_csv(TILE_ROOT / SLIDE_CSV)[["in_slide_name", "orion_slide_id"]]
    return df.merge(sd, on="in_slide_name", how="left")


def parse_xy(path: str) -> tuple[int, int]:
    m = _COORD_RE.search(path)
    return int(m.group(1)), int(m.group(2))


# ── IF canvas ────────────────────────────────────────────────────────────────────

def build_if_canvas(crc: str, rows: pd.DataFrame):
    """Stitch the 16-marker IF canvas + coverage mask from the 20x tiles (transient)."""
    placements, W, H = [], 0, 0
    for if_rel in rows.target_path:
        x0, y0 = parse_xy(if_rel)
        px, py = round(x0 * SCALE), round(y0 * SCALE)
        placements.append((px, py, if_rel))
        W, H = max(W, px + TILE_PX), max(H, py + TILE_PX)
    print(f"  [{crc}] {len(placements)} tiles -> IF canvas {W}x{H} "
          f"({len(TILE_CHANNELS)*W*H/1e9:.1f} GB scratch)")

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    if_mm = np.memmap(TMP_DIR / f"{crc}_if.dat", np.uint8, "w+",
                      shape=(len(TILE_CHANNELS), H, W))
    cov_mm = np.memmap(TMP_DIR / f"{crc}_cov.dat", np.uint8, "w+", shape=(H, W))
    for i, (px, py, if_rel) in enumerate(placements):
        if i % 1000 == 0:
            print(f"    stitching IF tile {i}/{len(placements)}", flush=True)
        iff = tifffile.imread(TILE_ROOT / if_rel)            # (333,333,17)
        h, w = iff.shape[:2]
        if_mm[:, py:py+h, px:px+w] = iff[:, :, TILE_CHANNELS].transpose(2, 0, 1)
        cov_mm[py:py+h, px:px+w] = 1
    if_mm.flush(); cov_mm.flush()
    return if_mm, cov_mm, W, H


def cleanup(crc: str, mms) -> None:
    for mm in mms:
        mm._mmap.close()
    for suf in ("if", "cov"):
        (TMP_DIR / f"{crc}_{suf}.dat").unlink(missing_ok=True)


# ── Per-slide ────────────────────────────────────────────────────────────────────

def process_slide(crc: str, rows: pd.DataFrame, clean: bool = True) -> None:
    print(f"\n{'='*60}\n  {crc}\n{'='*60}")
    if_mm, cov_mm, W, H = build_if_canvas(crc, rows)
    try:
        # benchmark coords (level-0) for this slide
        with h5py.File(BENCHMARK_DIR / f"{crc}_patch_dataset.h5", "r") as h:
            c0 = h["coords"][:].astype(np.int64)
            ps0 = int(h.attrs.get("patch_size_level0", 345))
        ps20 = TOKEN_GRID * round(ps0 * SCALE / TOKEN_GRID)     # 224, divisible by grid
        tok_px = ps20 // TOKEN_GRID
        c20 = np.round(c0 * SCALE).astype(np.int64)

        # keep in-canvas patches with enough IF coverage (drop MIPHEI-gap patches)
        keep = np.zeros(len(c0), bool)
        for i, (px, py) in enumerate(c20):
            if (0 <= px and 0 <= py and px + ps20 <= W and py + ps20 <= H
                    and cov_mm[py:py+ps20, px:px+ps20].mean() >= MIN_COVERAGE):
                keep[i] = True
        kept = np.where(keep)[0]
        print(f"  [{crc}] kept {len(kept)}/{len(c0)} patches "
              f"(dropped {len(c0)-len(kept)}: off-canvas / IF gap)")

        # token targets = mean(IF/255) per 14px cell
        C = len(MARKER_NAMES)
        targets = np.zeros((len(kept), C, TOKEN_GRID, TOKEN_GRID), np.float32)
        for j, i in enumerate(kept):
            if j % 5000 == 0:
                print(f"    tokens {j}/{len(kept)}", flush=True)
            px, py = c20[i]
            ifp = if_mm[:, py:py+ps20, px:px+ps20].astype(np.float32) / 255.0
            targets[j] = (ifp.reshape(C, TOKEN_GRID, tok_px, TOKEN_GRID, tok_px)
                              .mean(axis=(2, 4)))

        # IF artifact filtration (see artifact_filter.py), in place on `targets`:
        #   lineage gate -- child marker zeroed where its parent lineage (Hoechst->CD45
        #                   ->immune) is absent at that token;
        #   shape gate   -- PD-L1 uniform circular discs (imaging artifacts) removed;
        #   block gate   -- corrupted-tile blocks co-located in PD-L1 + CD31 removed.
        clean_tag = "off"
        if clean:
            lin = apply_lineage_gate(targets, MARKER_NAMES)
            shp = apply_shape_gate(targets, c0[kept], ps0, MARKER_NAMES)
            blk = apply_block_gate(targets, c0[kept], ps0, MARKER_NAMES)
            print(f"  [{crc}] artifact filter: lineage zeroed {sum(lin.values())} tokens, "
                  f"shape (PD-L1) {sum(shp.values())}, block (tile) {sum(blk.values())}")
            clean_tag = "lineage+shape+block"

        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        out = DATASET_DIR / f"{crc}_patch_dataset.h5"
        with h5py.File(out, "w") as f:
            f.create_dataset("coords", data=c0[kept], compression="gzip")       # level-0
            f.create_dataset("coords_20x", data=c20[kept], compression="gzip")
            f.create_dataset("benchmark_index", data=kept.astype(np.int64),
                             compression="gzip")
            f.create_dataset("targets", data=targets, compression="gzip",
                             chunks=(min(256, max(1, len(targets))), C, TOKEN_GRID, TOKEN_GRID))
            f.attrs["sample"] = crc
            f.attrs["marker_names"] = MARKER_NAMES
            f.attrs["patch_size_level0"] = ps0          # for visualize_orion_predictions
            f.attrs["patch_size"] = ps20
            f.attrs["token_grid"] = TOKEN_GRID
            f.attrs["normalisation"] = "uint8_div255"
            f.attrs["mpp"] = MPP_20X
            f.attrs["coord_space"] = "level0"
            f.attrs["coords_source"] = "benchmark_tile20x"
            f.attrs["min_coverage"] = MIN_COVERAGE
            f.attrs["artifact_filter"] = clean_tag
        print(f"  [{crc}] saved {out}  targets {targets.shape} mean={targets.mean():.4f}")
    finally:
        cleanup(crc, (if_mm, cov_mm))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slides", default="all",
                    help="'all' (default) or comma-separated CRC ids")
    ap.add_argument("--no-clean", action="store_true",
                    help="skip the IF artifact filter (lineage + PD-L1 shape gates)")
    args = ap.parse_args()

    df = load_tile_table()
    all_crcs = sorted(df.orion_slide_id.dropna().unique(), key=lambda s: (len(s), s))
    crcs = all_crcs if args.slides == "all" else [s.strip() for s in args.slides.split(",")]
    print(f"Slides ({len(crcs)}): {crcs}\nDataset -> {DATASET_DIR}")

    for crc in crcs:
        rows = df[df.orion_slide_id == crc]
        if rows.empty:
            print(f"  WARNING: no tiles for {crc}, skipping"); continue
        if not (BENCHMARK_DIR / f"{crc}_patch_dataset.h5").exists():
            print(f"  WARNING: no benchmark h5 for {crc}, skipping"); continue
        try:
            process_slide(crc, rows, clean=not args.no_clean)
        except Exception as e:
            print(f"  ERROR {crc}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()