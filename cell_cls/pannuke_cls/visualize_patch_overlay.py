"""One PanNuke image AS THE MODEL SEES IT (224x224) + its nuclei mask + 16x16 grid.

PanNuke is a single 256px @40x image. Mirroring build_cell_token_features: with
ps0=256 (canonical) the whole 256 image is resized to 224; with ps0=448 it is
padded white up to 448 (image in top-left) then resized to 224. We resize the
gt_inst mask the same way (nearest, labels preserved) and draw the 16x16 token
grid (token = 224/16 = 14px), so you see the actual HDF nuclei in the model frame.

  python cell_cls/pannuke_cls/visualize_patch_overlay.py                 # random image
  python cell_cls/pannuke_cls/visualize_patch_overlay.py --img img_Breast_1_00000
  python cell_cls/pannuke_cls/visualize_patch_overlay.py --ps0 448
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from utils import (MODEL_INPUT, TOKEN_GRID, PATCH_SIZE_LEVEL0, list_images,
                   slide_name_of, load_image_and_mask, inst_path_of, FEAT_DIR)

NATIVE_PS0 = 256


def pad_to(arr, ps0, value):
    H, W = arr.shape[:2]
    if (H, W) == (ps0, ps0):
        return arr
    return cv2.copyMakeBorder(arr, 0, max(0, ps0 - H), 0, max(0, ps0 - W),
                              cv2.BORDER_CONSTANT, value=value)[:ps0, :ps0]


def add_token_grid(ax, m=MODEL_INPUT, g=TOKEN_GRID):
    step = m / g
    for k in range(1, g):
        ax.axhline(k * step, color="yellow", lw=0.5, alpha=0.7)
        ax.axvline(k * step, color="yellow", lw=0.5, alpha=0.7)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--img", default=None, help="image stem (default: random)")
    ap.add_argument("--ps0", type=int, default=NATIVE_PS0,
                    help="256 (canonical: resize whole image) or 448 (pad to scale)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    imgs = list_images()
    if args.img:
        path = next(p for p in imgs if slide_name_of(p) == args.img)
    else:
        path = imgs[np.random.default_rng(args.seed).integers(len(imgs))]

    he, inst = load_image_and_mask(path)
    he_p   = pad_to(he, args.ps0, 255)
    inst_p = pad_to(inst.astype(np.int32), args.ps0, 0)

    he_r   = cv2.resize(he_p, (MODEL_INPUT, MODEL_INPUT), interpolation=cv2.INTER_AREA)
    inst_r = cv2.resize(inst_p, (MODEL_INPUT, MODEL_INPUT), interpolation=cv2.INTER_NEAREST)
    n = len(np.unique(inst_r)) - (1 if 0 in inst_r else 0)

    over = he_r.copy()
    over[find_boundaries(inst_r, mode="inner")] = [255, 0, 0]
    rng = np.random.default_rng(0)
    lut = rng.random((int(inst_r.max()) + 1, 3)); lut[0] = 0
    mask_rgb = lut[inst_r]

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.4))
    ax[0].imshow(he_r);     ax[0].set_title("H&E 224 (as model sees it)")
    ax[1].imshow(over);     ax[1].set_title(f"gt_inst boundaries + 16x16 grid — {n} nuclei")
    ax[2].imshow(mask_rgb); ax[2].set_title("gt_inst (random colours) + grid")
    add_token_grid(ax[1]); add_token_grid(ax[2])
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"PanNuke {slide_name_of(path)}  ps0={args.ps0}->{MODEL_INPUT}px  "
                 f"(token = {MODEL_INPUT//TOKEN_GRID}px)", fontsize=12)
    fig.tight_layout()
    out = args.out or str(FEAT_DIR / f"patch_overlay_{slide_name_of(path)}_ps{args.ps0}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"{n} nuclei -> {out}")


if __name__ == "__main__":
    main()