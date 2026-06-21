#!/usr/bin/env python3
"""
Visualise one random HNSCC source patch: H&E + mIF channels + 16×16 token grids.

Layout:
  Rows    : 4 sub-patches (TL, TR, BL, BR)
  Columns : H&E | for each marker: mIF image | token heatmap
  Each marker uses its own colour for both the image and the token heatmap.
"""

import argparse
import numpy as np
import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from PIL import Image

from build_patch_dataset_hnscc import BOUNDS, DEFAULT_BOUNDS

H5_PATH  = Path('datasets/hnscc_patch_dataset/hnscc_patch_dataset.h5')
MIF_DIR  = Path('/mnt/ssd1/virtual_proteomics/data/HNSCC/mIF_Data')
MIHC_DIR = Path('/mnt/ssd1/virtual_proteomics/data/HNSCC/mIHC_Data')
OUT_DIR  = Path('visualization_out/hnscc/tokens')

SUB_SIZE   = 256
PATCH_SIZE = 224
TOKEN_GRID = 16
SUB_COORDS = [(0, 0), (256, 0), (0, 256), (256, 256)]
SUB_LABELS = ['TL', 'TR', 'BL', 'BR']

CMAPS = {
    'CD3':   LinearSegmentedColormap.from_list('cd3',   ['black', 'limegreen']),
    'CD8':   LinearSegmentedColormap.from_list('cd8',   ['black', 'cyan']),
    'FoxP3': LinearSegmentedColormap.from_list('foxp3', ['black', 'yellow']),
    'PanCK': LinearSegmentedColormap.from_list('pck',   ['black', 'magenta']),
    'DAPI':  LinearSegmentedColormap.from_list('dapi',  ['black', 'cornflowerblue']),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid',  default=None,
                        help='Source patch ID (random if omitted)')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    with h5py.File(H5_PATH) as f:
        markers   = list(f.attrs['marker_names'])
        patch_ids = f['patch_ids'][:]
        targets   = f['targets'][:]   # (N, C, G, G)

    source_ids = sorted(set(p.decode() for p in patch_ids))
    rng = np.random.default_rng(args.seed)
    pid = args.pid or rng.choice(source_ids)
    print(f'Patch: {pid}')

    # targets for this source: (4, C, G, G)
    mask = patch_ids == pid.encode()
    tgts = targets[mask]   # rows are in SUB_COORDS order from build script

    # ── Load + normalise images ───────────────────────────────────────────────
    he_full = np.array(Image.open(MIHC_DIR / f'{pid}_Hematoxylin.png'))  # (512,512,3)

    bounds = BOUNDS.get(pid, DEFAULT_BOUNDS)
    mif_norm = {}
    for m in markers:
        arr    = np.array(Image.open(MIF_DIR / f'{pid}_{m}.png')).astype(np.float32)
        nz     = arr[arr > 0]
        p99    = max(float(np.percentile(nz, 99)) if len(nz) else 1.0, 1.0)
        normed = np.clip(arr / p99, 0.0, 1.0)
        normed[normed < bounds.get(m, DEFAULT_BOUNDS[m])] = 0.0
        mif_norm[m] = normed   # (512, 512) float32

    # ── Figure: rows=sub-patches, cols=H&E + (img,tok) per marker ────────────
    C    = len(markers)
    nrow = len(SUB_COORDS)
    ncol = 1 + C * 2   # H&E | img tok | img tok | ...

    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(ncol * 1.7, nrow * 1.9),
                             gridspec_kw={'wspace': 0.04, 'hspace': 0.25})

    # Column headers
    axes[0, 0].set_title('H&E', fontsize=8, fontweight='bold')
    for ci, m in enumerate(markers):
        axes[0, 1 + ci * 2    ].set_title(m,       fontsize=8, fontweight='bold',
                                           color=CMAPS[m](0.85))
        axes[0, 1 + ci * 2 + 1].set_title('tokens', fontsize=7,
                                           color=CMAPS[m](0.85))

    for si, (x, y) in enumerate(SUB_COORDS):
        # Row label
        axes[si, 0].set_ylabel(SUB_LABELS[si], fontsize=8, rotation=0,
                               labelpad=20, va='center')

        # H&E crop → resize to 224 for consistent display size
        he_crop = he_full[y:y + SUB_SIZE, x:x + SUB_SIZE]
        he_224  = cv2.resize(he_crop, (PATCH_SIZE, PATCH_SIZE),
                             interpolation=cv2.INTER_LINEAR)
        axes[si, 0].imshow(he_224)
        axes[si, 0].axis('off')

        for ci, m in enumerate(markers):
            cmap = CMAPS[m]
            img_ax = axes[si, 1 + ci * 2]
            tok_ax = axes[si, 1 + ci * 2 + 1]

            # mIF crop resized to 224
            crop  = mif_norm[m][y:y + SUB_SIZE, x:x + SUB_SIZE]
            sized = cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE),
                               interpolation=cv2.INTER_LINEAR)
            img_ax.imshow(sized, cmap=cmap, vmin=0, vmax=1)
            img_ax.axis('off')

            # token heatmap — same colormap as image
            tok = tgts[si, ci]   # (G, G)
            tok_ax.imshow(tok, cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
            tok_ax.axis('off')
            tok_ax.set_xlabel(f'{tok.mean():.3f}', fontsize=6, labelpad=1)

    fig.suptitle(f'HNSCC — {pid}', fontsize=10, fontweight='bold', y=1.002)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f'{pid}.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()