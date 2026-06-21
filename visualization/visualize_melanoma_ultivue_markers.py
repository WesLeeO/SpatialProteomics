#!/usr/bin/env python3
"""Per-slide thumbnail: H&E + 8 immuno8 markers (DAPI/DAPI2 skipped).

Normalization (no AF channel) — triangle-based, after MIPHEI's AF-less recipe
(MIPHEI-ViT/datasets/preprocessing/pathocell_preprocess.py::compute_if_percentiles)
but with a PER-SLIDE background (technical) and a GLOBAL ceiling (partly biological):

    display = clip((raw - bg[slide]) / (max - bg[slide]), 0, 1)

  • bg[slide]  = peak + frac·(triangle - peak) for the marker on that slide (per-slide
    background; the raw triangle is harsh on haze markers, so the per-marker frac in the
    BG_FRAC dict softens it toward the histogram peak/mode).
  • max        = p{FG_PCT} of foreground pixels (> each slide's triangle), pooled across
    all slides → one ceiling per marker, so a given intensity means the same everywhere.
  Computed once by compute_global_norm() and cached.
"""

import sys
sys.path.insert(0, '/home/wesley/spatial_proteomics')

import xml.etree.ElementTree as ET
import numpy as np
import zarr
import tifffile
import openslide
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

import json
import argparse
from build_patch_dataset_melanoma_ultivue import (
    sample_paths, IF_CHANNELS, _load_channel)

LEVEL      = 6                 # display pyramid level (smaller = higher-res/bigger images);
                               # overridden by --level. Normalization is unaffected — min/max
                               # are always computed from the LEVEL-0 histograms.
DOWNSAMPLE = 2 ** LEVEL
SAVE_DPI   = 120               # savefig dpi (overridden by --dpi); the other size lever
OUT_DIR    = Path('visualization_out/melanoma_ultivue/all_markers')
NORM_CACHE = OUT_DIR / 'norm_stats.json'   # per-marker (min, max), pooled across slides

# Background floor is PER SLIDE; the foreground ceiling is GLOBAL = p{FG_PCT} of
# foreground pixels pooled across all slides.
FG_PCT  = 99.0   # foreground ceiling percentile (pooled across slides)
# The triangle anchors its line on the brightest pixel, so a few outlier pixels can shove
# the threshold up 3-4x on some slides. Capping the upper anchor at a percentile fixes that
# — BUT the full slide×marker sweep (melanoma_hicap_investigation/) shows the cap MUST be
# PER MARKER, and per-slide is a trap:
#   • PD-1 / CD8a have a wide flat plateau of clean slides + ONE pathological outlier slide
#     (literal pixel outliers: MAKYGIW PD-1 403, CD8a 1885; MISYPUP CD8a 1316 — all ~3-4x
#     cohort). A single cap snaps that outlier onto the plateau and is a no-op on every
#     other slide. → cap is safe and ideal.
#   • Every other marker has NO plateau: the triangle apex is brittle, so lowering the cap
#     makes each slide fall off a CLIFF at its own scattered percentile. There is no cap
#     that fixes the (mild, ≤1.8x) spread without collapsing some clean slide — e.g. FoXP3
#     MAJOFIJ 749→175 at 99.999 was the green flood. → raw (no cap) is the only safe floor;
#     the per-marker BG_FRAC is the real aggressiveness knob, not the cap.
# So: a per-marker cap dict (None = raw/uncapped, the stable default). Per-(marker,slide)
# is unnecessary — the only markers a cap helps are fixed by one shared value.
TRIANGLE_HI_PCT_DEFAULT = None
TRIANGLE_HI_PCT = {
    'PD-1':  99.999,   
    'CD8a':  99.999,   
}
# The per-slide floor blends the histogram peak (mode) and that slide's triangle
# threshold:  bg = peak + frac * (triangle - peak).  frac=1 → raw triangle (harsh on
# haze markers, keeps only ~2-9% of FoXP3/CD68/CD3); frac=0 → histogram peak (removes
# ~nothing).  PER MARKER, because the right aggressiveness differs by marker.
BG_FRAC_DEFAULT = 0.1
BG_FRAC = {
    'SOX10': 0.1,
    'PD-L1': 0.1,
    'CD68':  0.1,
    'CD4':   0.8,
    'CD3':   0.7, 
    'PD-1':  3.0,
    'CD8a':  3.0,
    'FoXP3': 0.7,
}

SKIP_CHANNELS = {'DAPI', 'DAPI2'}

# Background is removed by the triangle-based (min, max) from compute_global_norm — the
# cases that used to need hand-tuned `lo` (FoXP3 flooding the tumor with no AF channel;
# MISYPUP being globally dim) are handled by the data-driven floor/ceiling per marker.
MARKERS = ['SOX10', 'PD-L1', 'CD68', 'CD4', 'CD3', 'PD-1', 'CD8a', 'FoXP3']

SAMPLES = [
    'MACEGEJ', 'MAHEFOG', 'MAJOFIJ', 'MAKYGIW', 'MANOFYB',
    'MELIPIT', 'MIDEKOG', 'MIDOBOL', 'MISYPUP', 'MUDUKEF',
]

CHANNEL_CMAPS = {
    'PD-L1': LinearSegmentedColormap.from_list('pdl1',  ['black', 'gold']),
    'CD68':  LinearSegmentedColormap.from_list('cd68',  ['black', 'tomato']),
    'CD8a':  LinearSegmentedColormap.from_list('cd8a',  ['black', 'cyan']),
    'PD-1':  LinearSegmentedColormap.from_list('pd1',   ['black', 'orange']),
    'FoXP3': LinearSegmentedColormap.from_list('foxp3', ['black', 'lime']),
    'SOX10': LinearSegmentedColormap.from_list('sox10', ['black', 'magenta']),
    'CD3':   LinearSegmentedColormap.from_list('cd3',   ['black', 'limegreen']),
    'CD4':   LinearSegmentedColormap.from_list('cd4',   ['black', 'mediumpurple']),
}


_NBINS            = 1 << 16   # uint16 intensity range
STATS_TILE_STRIDE = 4         # read every Nth level-0 tile (in x & y) for the stats:
                              # full level-0 intensities, ~stride² less I/O, spatially
                              # representative. 1 = read every tile (exact, slow).


def _channel_hist_subsampled(tif_path, name: str, stride: int = STATS_TILE_STRIDE):
    """Pooled uint16 histogram of nonzero level-0 pixels from a STRIDED subset of
    tiles (corrupt tiles skipped). Reads ~1/stride² of the slice → much faster than
    decoding the whole channel, while keeping true level-0 intensities."""
    h = np.zeros(_NBINS, dtype=np.int64)
    n_corrupt = n_read = 0
    with tifffile.TiffFile(str(tif_path)) as t:
        z = zarr.open(t.aszarr(level=0), mode='r')
        H, W = z.shape[0], z.shape[1]
        ch, cw = z.chunks[0], z.chunks[1]
        for iy in range(0, H, ch * stride):
            for ix in range(0, W, cw * stride):
                try:
                    tile = z[iy:min(iy + ch, H), ix:min(ix + cw, W)]
                except Exception:
                    n_corrupt += 1
                    continue
                vals = tile[tile > 0].astype(np.int64)
                if vals.size:
                    np.clip(vals, 0, _NBINS - 1, out=vals)
                    h += np.bincount(vals, minlength=_NBINS)
                n_read += 1
    return h, n_read, n_corrupt


def _triangle_threshold_from_hist(h: np.ndarray, hi_pct: float = TRIANGLE_HI_PCT_DEFAULT) -> int:
    """Triangle-method threshold from a 1-D intensity histogram (skimage's algorithm,
    run on the pooled histogram so it stays corrupt-safe and memory-light). Splits the
    background mode from the foreground tail — the no-AF stand-in for AF subtraction
    (this is what MIPHEI's pathocell_preprocess uses for AF-less CODEX).

    hi_pct caps the upper anchor at that percentile of the pixels instead of the absolute
    brightest pixel, so a few outliers can't stretch the line (stable across slides).
    None → original behaviour (anchor on the last nonzero bin)."""
    h = h.astype(np.float64)
    nz = np.nonzero(h)[0]
    if nz.size < 2:
        return int(nz[0]) if nz.size else 0
    lo, hi = int(nz[0]), int(nz[-1])
    peak = int(np.argmax(h))
    if hi_pct is not None:
        hi_cap = int(np.searchsorted(np.cumsum(h), hi_pct / 100.0 * h.sum()))
        hi = min(hi, max(hi_cap, peak + 1))     # don't let outlier pixels stretch the line
    flip = (peak - lo) > (hi - peak)            # keep the long tail to the right of the peak
    if flip:
        h = h[::-1].copy()
        peak, lo, hi = len(h) - 1 - peak, len(h) - 1 - hi, len(h) - 1 - lo
    xs = np.arange(peak, hi + 1)
    ys = h[peak:hi + 1]
    denom = np.hypot(hi - peak, h[hi] - h[peak]) or 1.0
    d = np.abs((h[hi] - h[peak]) * xs - (hi - peak) * ys
               + hi * h[peak] - h[hi] * peak) / denom # distance from bin points after peak to peak-high line
    thr = int(xs[np.argmax(d)])
    return len(h) - 1 - thr if flip else thr


def _pct_from_hist(h: np.ndarray, lo_bin: int, hi_bin: int, pct: float) -> int:
    seg = h[lo_bin:hi_bin + 1]
    tot = int(seg.sum())
    if tot == 0:
        return lo_bin
    return lo_bin + int(np.searchsorted(np.cumsum(seg), pct / 100.0 * tot))


def compute_global_norm(refresh: bool = False) -> dict[str, dict]:
    """Per-marker normalization: PER-SLIDE background + GLOBAL ceiling.

      bg[slide] = peak + frac[marker]·(triangle - peak) for the marker on that slide — a
                  softened triangle floor (frac=1 → raw triangle; 0 → histogram peak),
                  frac chosen PER MARKER (BG_FRAC dict). Background is technical → per slide.
      max       = p{FG_PCT} of foreground pixels (> each slide's floor), pooled across
                  all slides (one ceiling per marker → cross-slide intensities comparable).

    Applied per slide: clip((raw - bg[slide]) / (max - bg[slide]), 0, 1).

    Returns {marker: {'max': float, 'bg': {sample: float}}}; cached to NORM_CACHE.
    """
    # resolved per-marker fracs and hi_caps actually in use (default-filled) — both are
    # part of the cache check, so changing any marker's frac OR cap invalidates the cache.
    fracs  = {name: BG_FRAC.get(name, BG_FRAC_DEFAULT)
              for name, _ in IF_CHANNELS if name not in SKIP_CHANNELS}
    hicaps = {name: TRIANGLE_HI_PCT.get(name, TRIANGLE_HI_PCT_DEFAULT)
              for name, _ in IF_CHANNELS if name not in SKIP_CHANNELS}
    key   = f'NORM@fg{FG_PCT:g}_permarkerbg_permarkercap'
    cache = {}
    if NORM_CACHE.exists():
        with open(NORM_CACHE) as f:
            cache = json.load(f)
    entry = cache.get(key)
    if (not refresh and entry and entry.get('samples') == SAMPLES
            and entry.get('bg_frac') == fracs and entry.get('hi_cap') == hicaps):
        print(f'  Loaded cached norm stats from {NORM_CACHE}')
        return entry['markers']

    print(f'  Computing norm stats (per-slide bg = peak+frac·(triangle-peak), per-marker '
          f'frac, global p{FG_PCT} ceiling, {len(SAMPLES)} slides, '
          f'tile stride={STATS_TILE_STRIDE})…')
    fg_hist: dict[str, np.ndarray]      = {}   # pooled foreground (> per-slide floor)
    bg: dict[str, dict[str, float]]     = {}   # per-(marker, slide) background floor
    for sample in SAMPLES:
        print(f'    {sample}…')
        _, _, _, tif_paths = sample_paths(sample)
        for (name, _), tif_path in zip(IF_CHANNELS, tif_paths):
            if name in SKIP_CHANNELS:
                continue
            h, _, _ = _channel_hist_subsampled(tif_path, name)
            peak = int(np.argmax(h))                       # background mode
            tri  = _triangle_threshold_from_hist(h, hi_pct=hicaps[name])  # triangle bg/fg split
            thr  = int(round(peak + fracs[name] * (tri - peak)))   # softened floor
            bg.setdefault(name, {})[sample] = float(thr)
            fg_hist.setdefault(name, np.zeros(_NBINS, dtype=np.int64))[thr + 1:] += h[thr + 1:]

    markers: dict[str, dict] = {}
    for name in fg_hist:
        vmax = (float(_pct_from_hist(fg_hist[name], 0, _NBINS - 1, FG_PCT))
                if int(fg_hist[name].sum()) > 0 else 1.0)
        markers[name] = {'max': vmax, 'bg': bg[name]}
        bgs = bg[name]
        print(f'    {name:<8} frac={fracs[name]:<4g} max(p{FG_PCT})={vmax:.0f}  '
              f'bg per slide: {min(bgs.values()):.0f}–{max(bgs.values()):.0f}')

    cache[key] = {'samples': SAMPLES, 'fg_pct': FG_PCT, 'bg_frac': fracs,
                  'hi_cap': hicaps, 'markers': markers}
    NORM_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(NORM_CACHE, 'w') as f:
        json.dump(cache, f, indent=2)
    print(f'  Saved norm stats → {NORM_CACHE}')
    return markers


def get_exclusion_mask(h, w, f_annotations, downsample_factor):
    mask = np.zeros((h, w), dtype=np.uint8)
    tree = ET.parse(str(f_annotations))
    for annotation in tree.getroot().findall('Annotation'):
        for region in annotation.findall('Regions/Region'):
            if int(region.get('NegativeROA', 0)) != 1:
                continue
            pts = [
                [int(float(v.attrib['X']) / downsample_factor),
                 int(float(v.attrib['Y']) / downsample_factor)]
                for v in region.findall('Vertices/V')
            ]
            if pts:
                cv2.drawContours(mask, [np.array(pts, dtype=np.int32)], 0, 255, -1)
    return mask.astype(bool)



def process_channel(raw_uint16, Q_inv, he_h, crop_w, vmin: float, vmax: float,
                    log: bool = False):
    """Warp IF channel and apply the MIPHEI no-AF normalization:

      d = clip((warped - vmin) / (vmax - vmin), 0, 1)

    vmin/vmax are the per-marker background floor / signal ceiling from
    compute_global_norm. Returns uint8 [0, 255].
    """
    warped = cv2.warpPerspective(
        raw_uint16.astype(np.float32), Q_inv, (crop_w, he_h), borderValue=0)
    d = np.clip((warped - vmin) / max(vmax - vmin, 1.0), 0.0, 1.0)
    if log:
        d = np.log1p(d)            # [0,1] → [0,1], boosts low signal
    return (d * 255).astype(np.uint8)


def visualize_sample(sample: str, norm: dict[str, dict], log: bool = False):
    he_path, npz_path, _, tif_paths = sample_paths(sample)

    # H&E thumbnail
    slide    = openslide.OpenSlide(str(he_path))
    he_level = slide.get_best_level_for_downsample(DOWNSAMPLE)
    he_dim   = slide.level_dimensions[he_level]
    he_img   = np.array(slide.read_region((0, 0), he_level, he_dim).convert('RGB'))
    slide.close()
    he_h, he_w = he_img.shape[:2]

    # Alignment
    npz      = np.load(str(npz_path))
    Q_he2if  = npz['transformation_matrix_Q']
    reg_ds   = int(npz['downsample'][0])
    range_HE = npz['range_HE'] * reg_ds
    S_down   = np.array([[1/DOWNSAMPLE, 0, 0], [0, 1/DOWNSAMPLE, 0], [0, 0, 1]])
    S_up     = np.array([[DOWNSAMPLE,   0, 0], [0, DOWNSAMPLE,   0], [0, 0, 1]])
    Q_inv    = np.linalg.inv(S_down @ Q_he2if @ S_up)
    x0       = int(range_HE[0] // DOWNSAMPLE)
    x1       = int(range_HE[1] // DOWNSAMPLE)
    crop_w   = x1 - x0

    # Exclusion mask
    ann_dir   = Path(f'/mnt/ssd1/virtual_proteomics/data/melanoma2/{sample}/exclusion_mask')
    ann_files = list(ann_dir.glob('*.annotations'))
    excl = None
    if ann_files:
        excl = get_exclusion_mask(he_h, he_w, ann_files[0], DOWNSAMPLE)
        excl = excl[:, x0:x1]

    he_img = he_img[:, x0:x1, :]

    channels = []
    for (name, _), tif_path in zip(IF_CHANNELS, tif_paths):
        if name in SKIP_CHANNELS:
            continue
        with tifffile.TiffFile(str(tif_path)) as t:
            z   = zarr.open(t.aszarr(), mode='r')
            raw = z[str(LEVEL)][:]
        vmin = norm[name]['bg'][sample]
        vmax = norm[name]['max']
        d    = process_channel(raw, Q_inv, he_h, crop_w, vmin, vmax, log=log).astype(np.float32) / 255.0
        if excl is not None:
            d[excl] = 0
        print(f'  {name:6s}  min={vmin:.0f} max={vmax:.0f}  nonzero={100*(d>0).mean():.2f}%')
        channels.append((name, d))

    ncols, nrows = 5, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 5))

    panels = [('H&E', he_img, None)] + [
        (name, img, CHANNEL_CMAPS[name]) for name, img in channels
    ]
    for idx, (title, img, cmap) in enumerate(panels):
        ax = axes[idx // ncols, idx % ncols]
        if cmap is None:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f'{title}', fontsize=11, fontweight='bold')
        ax.axis('off')

    for idx in range(len(panels), nrows * ncols):
        axes[idx // ncols, idx % ncols].axis('off')

    plt.suptitle(
        f'{sample} — immuno8 panel   (per-slide triangle bg, global p{FG_PCT} ceiling'
        f'{", log" if log else ""})',
        fontsize=13,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)

    out = OUT_DIR / f'{sample}_all_markers.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
    plt.close()
    print(f'  Saved -> {out}')


SWEEP_LO_VALUES = list(range(0, 251, 10))   # 50, 60, …, 250  (21 values)
SWEEP_COLS      = 11                          # H&E + 10 per row


def visualize_threshold_sweep(sample: str, norm: dict[str, dict],
                               lo_values: list[int] = SWEEP_LO_VALUES,
                               log: bool = False) -> None:
    """One PNG per marker per slide: H&E reference + the triangle-normalized channel at
    each extra lo cutoff in 0-255 display space (diagnostic; bg is already auto-removed)."""
    he_path, npz_path, _, tif_paths = sample_paths(sample)

    npz      = np.load(str(npz_path))
    Q_he2if  = npz['transformation_matrix_Q']
    reg_ds   = int(npz['downsample'][0])
    range_HE = npz['range_HE'] * reg_ds
    S_down   = np.array([[1/DOWNSAMPLE, 0, 0], [0, 1/DOWNSAMPLE, 0], [0, 0, 1]])
    S_up     = np.array([[DOWNSAMPLE,   0, 0], [0, DOWNSAMPLE,   0], [0, 0, 1]])
    Q_inv    = np.linalg.inv(S_down @ Q_he2if @ S_up)
    x0       = int(range_HE[0] // DOWNSAMPLE)
    x1       = int(range_HE[1] // DOWNSAMPLE)
    crop_w   = x1 - x0

    # H&E thumbnail cropped to registered region
    slide    = openslide.OpenSlide(str(he_path))
    he_level = slide.get_best_level_for_downsample(DOWNSAMPLE)
    he_img   = np.array(slide.read_region((0, 0), he_level,
                                          slide.level_dimensions[he_level]).convert('RGB'))
    slide.close()
    he_h    = he_img.shape[0]
    he_crop = he_img[:, x0:x1, :]

    out_dir  = OUT_DIR / 'sweeps' / sample
    out_dir.mkdir(parents=True, exist_ok=True)

    # H&E occupies slot 0; lo sweep fills the rest
    n_panels = 1 + len(lo_values)
    n_cols   = SWEEP_COLS
    n_rows   = (n_panels + n_cols - 1) // n_cols

    for (name, _), tif_path in zip(IF_CHANNELS, tif_paths):
        if name in SKIP_CHANNELS:
            continue

        with tifffile.TiffFile(str(tif_path)) as t:
            z   = zarr.open(t.aszarr(), mode='r')
            raw = z[str(LEVEL)][:]
        vmin, vmax = norm[name]['bg'][sample], norm[name]['max']
        normed = process_channel(raw, Q_inv, he_h, crop_w,
                                 vmin, vmax, log=log).astype(np.float32) / 255.0
        cmap   = CHANNEL_CMAPS.get(name, 'gray')

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(n_cols * 3, n_rows * 3),
                                 squeeze=False)

        axes[0, 0].imshow(he_crop)
        axes[0, 0].set_title('H&E', fontsize=9, fontweight='bold')
        axes[0, 0].axis('off')

        for i, lo in enumerate(lo_values):
            row, col = divmod(i + 1, n_cols)
            display  = normed.copy()
            display[display < lo / 255.0] = 0.0          # lo is a cutoff in 0-255 display space
            axes[row, col].imshow(display, cmap=cmap, vmin=0, vmax=1,
                                  interpolation='nearest')
            axes[row, col].set_title(f'lo={lo}', fontsize=9)
            axes[row, col].axis('off')

        for i in range(n_panels, n_rows * n_cols):
            axes[divmod(i, n_cols)].axis('off')

        plt.suptitle(f'{sample} — {name}  (min={vmin:.0f} max={vmax:.0f}; extra lo = 0-255 cutoff)', fontsize=11)
        plt.tight_layout()
        out = out_dir / f'{name}_sweep.png'
        plt.savefig(out, dpi=SAVE_DPI, bbox_inches='tight')
        plt.close()
        print(f'  Saved → {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', default='all',
                        help="'all' or comma-separated e.g. MACEGEJ,MAHEFOG")
    parser.add_argument('--sweep', action='store_true',
                        help='Produce threshold sweep grids instead of final viz')
    parser.add_argument('--linear', action='store_true',
                        help='Disable the log1p display curve (use linear bg/fg clip)')
    parser.add_argument('--level', type=int, default=LEVEL,
                        help=f'Display pyramid level (default {LEVEL}); smaller = '
                             f'higher-res/bigger images. Normalization is unchanged.')
    parser.add_argument('--dpi', type=int, default=SAVE_DPI,
                        help=f'savefig dpi (default {SAVE_DPI}).')
    parser.add_argument('--refresh-norm', action='store_true',
                        help='Recompute the per-marker (min, max) and overwrite the cache '
                             f'({NORM_CACHE.name}) instead of loading it.')
    args = parser.parse_args()

    # Display resolution knobs (normalization is LEVEL-0 based, so it is unaffected).
    LEVEL      = args.level
    DOWNSAMPLE = 2 ** LEVEL
    SAVE_DPI   = args.dpi

    samples = (SAMPLES if args.samples.lower() == 'all'
               else [s.strip() for s in args.samples.split(',')])
    log = False

    # Normalization: clip((raw - min)/(max - min)), per-marker (min, max) from the triangle
    # fg/bg split pooled across all slides (MIPHEI no-AF recipe; cached in NORM_CACHE).
    norm = compute_global_norm(refresh=args.refresh_norm)

    for sample in samples:
        print(f'\n=== {sample} ===')
        try:
            if args.sweep:
                visualize_threshold_sweep(sample, norm, log=log)
            else:
                visualize_sample(sample, norm, log=log)
        except Exception as e:
            import traceback
            print(f'  ERROR: {e}')
            traceback.print_exc()