"""
Visual check of the HEMIT Option-B patches with their nuclei mask in cyan.

For a source tile it reproduces the v4 builder geometry — the non-overlapping 3×3
grid of 448-native-px patches over the 1024² tile (top-lefts {0,448,896}; edge
patches padded white) — and overlays the instance-mask GT-class fill + boundaries on
each native patch, the 112 µm FOV the model sees. Everything stays in native 1024
space (no 40x→20x downscale), so the overlay is pixel-exact with the aggregation.

  python cell_cls/hemit_cell_cls/visualize_patch_overlay.py --split test          # random tile
  python cell_cls/hemit_cell_cls/visualize_patch_overlay.py --split test --n 4     # 4 random tiles
  python cell_cls/hemit_cell_cls/visualize_patch_overlay.py --tile '[10382,50252]_patch_0_0'
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.segmentation import find_boundaries

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from build_patch_dataset_hemit_token import (
    HEMIT_DATA_DIR, PATCH_LEVEL0, TILE_COORDS)

NUCLEI_DIR = Path("/mnt/ssd/virtual_proteomics/data/HEMIT_nuclei_analysis")
CYAN = np.array([0, 255, 255], np.uint8)


# GT class → interior fill colour (RGB). Class = (Pan-CK_pos, CD3_pos).
CLASS_COLOR = {
    (True, False):  (220, 30, 30),    # Pan-CK+ only  → red
    (False, True):  (40, 200, 40),    # CD3+ only     → green
    (True, True):   (245, 215, 0),    # double +      → yellow
    (False, False): (130, 130, 130),  # double −      → gray
}
CLASS_LABEL = {
    (True, False): "Pan-CK+", (False, True): "CD3+",
    (True, True): "Pan-CK+ & CD3+", (False, False): "neg",
}
ALPHA = 0.55


def build_color_lut(cells: pd.DataFrame, max_lab: int) -> np.ndarray:
    """label -> fill colour (max_lab+1, 3) uint8; 0 = no fill (label absent from csv)."""
    lut = np.zeros((max_lab + 1, 3), np.uint8)
    pk = cells["Pan-CK_pos"].values.astype(bool)
    cd = cells["CD3_pos"].values.astype(bool)
    labs = cells["label"].values.astype(np.int64)
    for cls, col in CLASS_COLOR.items():
        sel = (pk == cls[0]) & (cd == cls[1])
        lut[labs[sel]] = col
    return lut


def overlay_tile(split: str, stem: str, out_dir: Path) -> Path:
    he = tifffile.imread(str(HEMIT_DATA_DIR / split / "input" / f"{stem}.tif"))
    if he.ndim == 2:
        he = np.stack([he] * 3, -1)
    he = he[..., :3].astype(np.uint8)
    mask = tifffile.imread(str(NUCLEI_DIR / split / "mask" / f"{stem}.tif"))
    cells = pd.read_csv(NUCLEI_DIR / split / "csv" / f"{stem}.csv")

    # HEMIT csv uses (row, col) named X=row, Y=col → horizontal=Y_centroid, vertical=X_centroid.
    # v4 geometry is all NATIVE 1024 space (no 40x→20x ÷2): centroids stay native.
    cx = cells["Y_centroid"].values   # column (horizontal), NATIVE
    cy = cells["X_centroid"].values   # row (vertical), NATIVE

    lut = build_color_lut(cells, int(mask.max()))
    fill = lut[mask]                                    # (1024,1024,3); 0 where unfilled
    filled = fill.any(axis=-1)                          # pixels that get a class colour
    over = he.astype(np.float32).copy()
    over[filled] = (1 - ALPHA) * he[filled] + ALPHA * fill[filled]
    over[find_boundaries(mask, mode="inner")] = 0       # thin black nucleus outlines
    over = over.astype(np.uint8)

    ps = PATCH_LEVEL0
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, (x, y) in zip(axes.ravel(), TILE_COORDS):
        # native 448 crop (the model's FOV); pad edge crops to ps so every panel matches
        crop = over[y:y + ps, x:x + ps]
        ch, cw = crop.shape[:2]
        if (ch, cw) != (ps, ps):
            crop = np.pad(crop, ((0, ps - ch), (0, ps - cw), (0, 0)),
                          constant_values=255)
        ax.imshow(crop)
        # black dot at every nucleus centroid inside this native patch
        sel = (cx >= x) & (cx < x + ps) & (cy >= y) & (cy < y + ps)
        ax.scatter(cx[sel] - x, cy[sel] - y, s=4, c="black")
        ax.set_title(f"({x},{y})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # class counts (csv) + legend
    pk, cd = cells["Pan-CK_pos"].values.astype(bool), cells["CD3_pos"].values.astype(bool)
    handles = [mpatches.Patch(color=np.array(CLASS_COLOR[c]) / 255,
                              label=f"{CLASS_LABEL[c]} ({int(((pk==c[0])&(cd==c[1])).sum())})")
               for c in [(True, False), (False, True), (True, True), (False, False)]]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10, framealpha=0.9)
    fig.suptitle(f"{split}/{stem}  —  9× 448px-native (112µm @20x) patches | nucleus interior = GT class",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = out_dir / f"overlay_{split}_{stem}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--tile", default=None, help="tile stem (default: random)")
    ap.add_argument("--n", type=int, default=1, help="number of random tiles (ignored if --tile)")
    ap.add_argument("--out_dir", default=str(Path(__file__).resolve().parent / "overlays"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.tile:
        stems = [args.tile]
    else:
        inputs = sorted((HEMIT_DATA_DIR / args.split / "input").glob("*.tif"))
        rng = np.random.default_rng(args.seed)
        stems = [inputs[i].stem for i in rng.choice(len(inputs), size=min(args.n, len(inputs)), replace=False)]

    for stem in stems:
        print("saved ->", overlay_tile(args.split, stem, out_dir))


if __name__ == "__main__":
    main()