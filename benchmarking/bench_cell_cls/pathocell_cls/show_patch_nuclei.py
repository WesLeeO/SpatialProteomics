"""Dead-simple check: one PathoCell patch AS THE MODEL SEES IT (224x224) + its
nuclei mask + the 16x16 token grid.

Loads a core, takes a single PATCH_SIZE_LEVEL0 (297px) crop at (--px,--py), then
resizes BOTH the H&E (bilinear, exactly like normalise_patch) and the gt_inst
mask (nearest, to keep integer labels) to MODEL_INPUT (224). Shows three panels:

  H&E 224 | gt_inst boundaries on H&E + 16x16 grid | gt_inst (random colours) + grid

So you see the actual nuclei from the HDF in the same 224x224 / 16-token frame the
token model operates in (each token = 224/16 = 14px). No GT class, no aggregation.

  python cell_cls/pathocell_cls/show_patch_nuclei.py --core reg036_B --px 1188 --py 594
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries

from utils import PATCH_SIZE_LEVEL0, MODEL_INPUT, TOKEN_GRID, load_hdf_core, FEAT_DIR


def add_token_grid(ax, m=MODEL_INPUT, g=TOKEN_GRID):
    step = m / g                                       # 14 px per token
    for k in range(1, g):
        ax.axhline(k * step, color="yellow", lw=0.5, alpha=0.7)
        ax.axvline(k * step, color="yellow", lw=0.5, alpha=0.7)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--core", default="reg036_B")
    ap.add_argument("--px", type=int, default=None, help="patch top-left x (default: centre)")
    ap.add_argument("--py", type=int, default=None, help="patch top-left y (default: centre)")
    ap.add_argument("--size", type=int, default=PATCH_SIZE_LEVEL0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    he, inst = load_hdf_core(args.core)
    H, W = inst.shape
    s = args.size
    px = args.px if args.px is not None else (W - s) // 2
    py = args.py if args.py is not None else (H - s) // 2
    px, py = max(0, min(px, W - s)), max(0, min(py, H - s))

    he_c   = he[py:py + s, px:px + s]
    inst_c = inst[py:py + s, px:px + s]

    # resize to the model's 224x224 frame: H&E bilinear (as normalise_patch),
    # mask nearest so labels are preserved exactly.
    he_r   = cv2.resize(he_c, (MODEL_INPUT, MODEL_INPUT), interpolation=cv2.INTER_LINEAR)
    inst_r = cv2.resize(inst_c.astype(np.int32), (MODEL_INPUT, MODEL_INPUT),
                        interpolation=cv2.INTER_NEAREST)
    n = len(np.unique(inst_r)) - (1 if 0 in inst_r else 0)

    over = he_r.copy()
    over[find_boundaries(inst_r, mode="inner")] = [255, 0, 0]   # red outlines

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
    fig.suptitle(f"{args.core}  patch ({px},{py})  {s}px -> {MODEL_INPUT}px  "
                 f"(token = {MODEL_INPUT//TOKEN_GRID}px)", fontsize=12)
    fig.tight_layout()
    out = args.out or str(FEAT_DIR / f"patch_nuclei_{args.core}_{px}_{py}.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"{n} nuclei in patch -> {out}")


if __name__ == "__main__":
    main()