"""One Lizard 224 tile AS THE MODEL SEES IT + its nuclei mask + 16x16 grid.

Lizard is already 20x (0.5 µm/px = our training scale), so build_cell_token_features
tiles each image into a NATIVE 224 grid with NO resize — a 224 crop IS the model
input. We take one such tile at (--px,--py), edge-pad to 224 exactly like
normalise_patch, and draw the 16x16 token grid (token = 224/16 = 14px). The
gt_inst (.mat inst_map) is shown at native scale — no rescaling at all.

  python cell_cls/lizard_cls/visualize_patch_overlay.py                       # random image, centre tile
  python cell_cls/lizard_cls/visualize_patch_overlay.py --img consep_1 --px 0 --py 0
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from utils import (MODEL_INPUT, TOKEN_GRID, list_images, slide_name_of,
                   load_image_and_inst, FEAT_DIR)


def pad_to(arr, s, value):
    h, w = arr.shape[:2]
    if (h, w) == (s, s):
        return arr
    return cv2.copyMakeBorder(arr, 0, max(0, s - h), 0, max(0, s - w),
                              cv2.BORDER_CONSTANT, value=value)[:s, :s]


def add_token_grid(ax, m=MODEL_INPUT, g=TOKEN_GRID):
    step = m / g
    for k in range(1, g):
        ax.axhline(k * step, color="yellow", lw=0.5, alpha=0.7)
        ax.axvline(k * step, color="yellow", lw=0.5, alpha=0.7)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--img", default=None, help="image stem (default: random)")
    ap.add_argument("--px", type=int, default=None, help="tile top-left x (default: centre)")
    ap.add_argument("--py", type=int, default=None, help="tile top-left y (default: centre)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    imgs = list_images()
    if args.img:
        path = next(p for p in imgs if slide_name_of(p) == args.img)
    else:
        path = imgs[np.random.default_rng(args.seed).integers(len(imgs))]

    he, inst = load_image_and_inst(path)
    H, W = inst.shape
    s = MODEL_INPUT
    px = args.px if args.px is not None else max(0, (W - s) // 2)
    py = args.py if args.py is not None else max(0, (H - s) // 2)
    px, py = max(0, min(px, max(0, W - s))), max(0, min(py, max(0, H - s)))

    # native 224 crop, edge-padded (white H&E / 0 mask), NO resize
    he_c   = pad_to(he[py:py + s, px:px + s], s, 255)
    inst_c = pad_to(inst[py:py + s, px:px + s].astype(np.int32), s, 0)
    n = len(np.unique(inst_c)) - (1 if 0 in inst_c else 0)

    over = he_c.copy()
    over[find_boundaries(inst_c, mode="inner")] = [255, 0, 0]
    rng = np.random.default_rng(0)
    lut = rng.random((int(inst_c.max()) + 1, 3)); lut[0] = 0
    mask_rgb = lut[inst_c]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
    ax[0].imshow(he_c);     ax[0].set_title("H&E 224 (native, no resize)")
    ax[1].imshow(over);     ax[1].set_title(f"gt_inst boundaries + 16x16 grid — {n} nuclei")
    ax[2].imshow(mask_rgb); ax[2].set_title("gt_inst (random colours) + grid")
    add_token_grid(ax[1]); add_token_grid(ax[2])
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Lizard {slide_name_of(path)}  tile ({px},{py})  {s}x{s}px native  "
                 f"(token = {MODEL_INPUT//TOKEN_GRID}px)", fontsize=12)
    fig.tight_layout()
    out = args.out or str(FEAT_DIR / f"patch_overlay_{slide_name_of(path)}_{px}_{py}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"{n} nuclei -> {out}")


if __name__ == "__main__":
    main()