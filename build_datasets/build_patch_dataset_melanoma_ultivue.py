"""
Build patch dataset for melanoma2 (HistoPlexer-Ultivue, Scene-2 tumor tissue).

Pipeline (per sample)
---------------------
1. TRIDENT  : tissue segmentation + patch coord extraction on H&E (.ndpi)
2. Load     : all 10 IF channels at full resolution (uint8 = uint16 // 256)
3. Warp     : per patch, map 4 corners through Q_matrix → IF bbox →
              getPerspectiveTransform → warpPerspective to OUT_PX×OUT_PX
4. Save HDF5: /he (N,OUT_PX,OUT_PX,3)  /if (N,C,OUT_PX,OUT_PX)  /coords (N,2)

Coordinate note
---------------
Q_matrix (NPZ) maps  CROPPED_HE_fullres → IF_fullres  at ds=1.
TRIDENT coords are in full-HE space → subtract range_HE[0] before applying Q_matrix.
Rows (y) are the same in full-HE and cropped-HE.
"""

import csv
import glob
import argparse
import subprocess
import numpy as np
import tifffile
import zarr
import openslide
import cv2
import h5py
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path('/mnt/ssd/virtual_proteomics/data/melanoma2')
JOB_DIR     = Path('datasets/melanoma_ultivue_trident_output')
OUTPUT_DIR  = Path('datasets/melanoma_ultivue_patch_dataset')
TRIDENT_SCRIPT = Path('TRIDENT/run_batch_of_slides.py')

SCENE   = 2
MAG     = 20      # target magnification for TRIDENT
PS      = 224     # TRIDENT patch_size
OUT_PX  = 224     # output pixels per patch (HE + IF)
MPP_HE  = 0.23   # H&E µm/px

# Channel order: (name, glob_pattern_in_scene_dir)
IF_CHANNELS = [
    ('DAPI',   '*_Rd1_-DAPI.tif'),
    ('PD-L1',  '*_Rd1_Cy5_*.tif'),
    ('CD68',   '*_Rd1_Cy7_*.tif'),
    ('CD8a',   '*_Rd1_FITC_*.tif'),
    ('PD-1',   '*_Rd1_TRITC_*.tif'),
    ('DAPI2',  '*_Rd2*-DAPI2.tif'),
    ('FoXP3',  '*_Rd2*Cy5_*.tif'),
    ('SOX10',  '*_Rd2*Cy7_*.tif'),
    ('CD3',    '*_Rd2*FITC_*.tif'),
    ('CD4',    '*_Rd2*TRITC_*.tif'),
]
CH_NAMES = [n for n, _ in IF_CHANNELS]

SKIP_CHANNELS   = {'DAPI', 'DAPI2'}
TARGET_CHANNELS = [n for n in CH_NAMES if n not in SKIP_CHANNELS]  # 8 markers

# Per-(sample, marker) background cutoff `lo` in clip(x/p99_nonzero, 0, 1)*255 space.
# SINGLE SOURCE OF TRUTH — visualize_melanoma_ultivue_markers.py imports this dict, so the
# build dataset and the QC viz use identical cutoffs. Default 80 everywhere, 200 for FoXP3
# (floods without AF); tune any single panel by editing its number.
SAMPLE_BG: dict[str, dict[str, float]] = {
    'MACEGEJ': {'SOX10': 80, 'PD-L1': 80, 'CD68': 80, 'CD4': 80, 'CD3': 80, 'PD-1': 80, 'CD8a': 80, 'FoXP3': 200},
    'MAHEFOG': {'SOX10': 80, 'PD-L1': 80, 'CD68': 80, 'CD4': 80, 'CD3': 80, 'PD-1': 80, 'CD8a': 80, 'FoXP3': 200},
    'MAJOFIJ': {'SOX10': 80, 'PD-L1': 80, 'CD68': 80, 'CD4': 80, 'CD3': 80, 'PD-1': 80, 'CD8a': 80, 'FoXP3': 200},
    'MAKYGIW': {'SOX10': 80, 'PD-L1': 80, 'CD68': 80, 'CD4': 80, 'CD3': 80, 'PD-1': 80, 'CD8a': 80, 'FoXP3': 200},
    'MANOFYB': {'SOX10': 80, 'PD-L1': 80, 'CD68': 80, 'CD4': 80, 'CD3': 80, 'PD-1': 80, 'CD8a': 80, 'FoXP3': 200},
    'MELIPIT': {'SOX10': 80, 'PD-L1': 80, 'CD68': 80, 'CD4': 80, 'CD3': 80, 'PD-1': 80, 'CD8a': 80, 'FoXP3': 200},
    'MIDEKOG': {'SOX10': 80, 'PD-L1': 80, 'CD68': 80, 'CD4': 80, 'CD3': 80, 'PD-1': 80, 'CD8a': 80, 'FoXP3': 200},
    'MIDOBOL': {'SOX10': 80, 'PD-L1': 80, 'CD68': 80, 'CD4': 80, 'CD3': 80, 'PD-1': 80, 'CD8a': 80, 'FoXP3': 200},
    'MISYPUP': {'SOX10': 80, 'PD-L1': 80, 'CD68': 80, 'CD4': 80, 'CD3': 80, 'PD-1': 80, 'CD8a': 80, 'FoXP3': 200},
    'MUDUKEF': {'SOX10': 80, 'PD-L1': 80, 'CD68': 80, 'CD4': 80, 'CD3': 80, 'PD-1': 80, 'CD8a': 80, 'FoXP3': 200},
}

# Global p99 ceilings: pooled across ALL slides by the viz script and cached here. The
# build loads the SAME ceilings so the dataset matches the QC sweeps exactly. Run the viz
# (visualize_melanoma_ultivue_markers.py) once to populate this before building.
FG_PERCENTILE = 99.0
FG_CACHE      = Path('visualization_out/melanoma_ultivue/all_markers/fg_p99.json')


def load_global_p99(percentile: float = FG_PERCENTILE) -> dict[str, float]:
    """Per-marker GLOBAL p{percentile} ceiling pooled across all slides, from FG_CACHE
    (written by the viz script). Raises if absent — run the viz first to compute it."""
    key = f'GLOBAL@p{percentile:g}'
    if not FG_CACHE.exists():
        raise FileNotFoundError(
            f'{FG_CACHE} not found — run visualize_melanoma_ultivue_markers.py first '
            f'to compute the global p99 ceilings.')
    with open(FG_CACHE) as f:
        data = json.load(f)
    if key not in data:
        raise KeyError(f"'{key}' not in {FG_CACHE} — run the viz to compute global fg.")
    return data[key]['fg']


DST_PTS = np.float32([[0, 0], [OUT_PX, 0], [0, OUT_PX], [OUT_PX, OUT_PX]])


# ── Path helpers ──────────────────────────────────────────────────────────────

def sample_paths(sample: str) -> tuple[Path, Path, Path, list[Path]]:
    """Return (he_path, npz_path, scene_dir, [tif_path x C])."""
    d = BASE_DIR / sample
    he_matches  = list((d / 'HE').glob('*.ndpi'))
    npz_matches = list((d / 'alignment_immuno8_HE').glob(f'{sample}-Scene-{SCENE}.npz'))
    if len(he_matches) != 1:
        raise FileNotFoundError(f'{sample}: expected 1 HE, found {he_matches}')
    if len(npz_matches) != 1:
        raise FileNotFoundError(f'{sample}: expected 1 NPZ, found {npz_matches}')
    scene_dir = d / 'immuno8_panel' / f'{sample}-Scene-{SCENE}-stacked'
    tif_paths = []
    for name, pattern in IF_CHANNELS:
        hits = list(scene_dir.glob(pattern))
        if not hits:
            raise FileNotFoundError(f'{sample}: channel {name} not found in {scene_dir}')
        tif_paths.append(hits[0])
    return he_matches[0], npz_matches[0], scene_dir, tif_paths


def get_samples() -> list[str]:
    return sorted(d.name for d in BASE_DIR.iterdir()
                  if d.is_dir() and not d.name.startswith('.'))


# ── TRIDENT ───────────────────────────────────────────────────────────────────

def make_wsi_csv(he_slide: Path, job_dir: Path) -> Path:
    csv_path = job_dir / 'wsi_list.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['wsi', 'mpp'])
        writer.writerow([he_slide.name, MPP_HE])
    return csv_path


def run_trident(he_slide: Path, job_dir: Path, args: argparse.Namespace) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    wsi_csv  = make_wsi_csv(he_slide, job_dir)
    base_cmd = [
        'python', str(TRIDENT_SCRIPT),
        '--wsi_dir',   str(he_slide.parent),
        '--job_dir',   str(job_dir),
        '--gpu',       str(args.gpu),
        '--segmenter', args.segmenter,
        '--seg_conf_thresh', str(args.seg_thresh),
        '--mag',        str(args.mag),
        '--patch_size', str(args.patch_size),
        '--overlap',    str(args.overlap),
        '--min_tissue_proportion', str(args.min_tissue),
        '--wsi_ext',    '.ndpi',
        '--custom_list_of_wsis', str(wsi_csv),
    ]
    print('[TRIDENT] Segmenting…')
    subprocess.run(base_cmd + ['--task', 'seg'], check=True)
    print('[TRIDENT] Extracting coords…')
    subprocess.run(base_cmd + ['--task', 'coords'], check=True)
    h5_files = list(job_dir.rglob('*_patches.h5'))
    if not h5_files:
        raise FileNotFoundError(f'No coords H5 found under {job_dir}')
    return h5_files[0]


def load_trident_coords(h5_path: Path) -> tuple[np.ndarray, int, float]:
    with h5py.File(h5_path, 'r') as f:
        key        = 'coords' if 'coords' in f else list(f.keys())[0]
        coords     = f[key][:]
        patch_size = int(f[key].attrs.get('patch_size', PS))
        target_mag = float(f[key].attrs.get('target_magnification', MAG))
    print(f'  {len(coords)} patches  patch_size={patch_size}  @ {target_mag}x')
    return coords, patch_size, target_mag


# ── IF loading ────────────────────────────────────────────────────────────────

def _load_channel(
    path: Path, name: str, target_hw: tuple[int, int] | None,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Load one IF channel (uint16) at full resolution.

    Returns (arr, corrupt_bboxes) where corrupt_bboxes is a list of
    (x0, y0, x1, y1) tile regions that failed to decode (zeroed in arr).
    Level 0 is read tile-by-tile via zarr; falls back to level 1+ only if
    the zarr store itself can't be opened.
    """
    with tifffile.TiffFile(str(path)) as t:
        levels   = t.series[0].levels
        n_levels = len(levels)

        # ── tile-by-tile level-0 read ─────────────────────────────────────────
        try:
            store = t.aszarr(level=0)
            z     = zarr.open(store, mode='r')
            H, W  = z.shape[0], z.shape[1]
            arr   = np.zeros((H, W), dtype=z.dtype)
            ch, cw = z.chunks[0], z.chunks[1]
            corrupt_bboxes: list[tuple[int, int, int, int]] = []
            for iy in range(0, H, ch):
                for ix in range(0, W, cw):
                    sl = (slice(iy, min(iy + ch, H)), slice(ix, min(ix + cw, W)))
                    try:
                        arr[sl] = z[sl]
                    except Exception:
                        corrupt_bboxes.append((ix, iy, min(ix + cw, W), min(iy + ch, H)))
            if corrupt_bboxes:
                print(f'  {name}: lv0 {arr.shape}  ({len(corrupt_bboxes)} corrupt tiles → 0)')
            else:
                print(f'  {name}: {arr.shape}')
            if target_hw is not None and arr.shape[:2] != target_hw:
                arr = cv2.resize(arr, (target_hw[1], target_hw[0]),
                                 interpolation=cv2.INTER_LINEAR)
            return arr, corrupt_bboxes
        except Exception as e:
            print(f'  {name}: zarr lv0 failed ({e}), falling back to lower levels…')

        # ── whole-level fallback ──────────────────────────────────────────────
        for lv in range(1, n_levels):
            try:
                arr = levels[lv].asarray()
                if target_hw is not None and arr.shape[:2] != target_hw:
                    arr = cv2.resize(arr, (target_hw[1], target_hw[0]),
                                     interpolation=cv2.INTER_LINEAR)
                    print(f'  {name}: lv{lv} upsampled → {arr.shape}')
                else:
                    print(f'  {name}: lv{lv} {arr.shape}')
                return arr, []   # no tile-level tracking for fallback levels
            except Exception:
                print(f'  {name}: lv{lv} corrupt, trying next…')

    raise RuntimeError(f'All pyramid levels corrupt for {name} at {path}')


def load_if_channels(
    tif_paths: list[Path],
) -> tuple[list[np.ndarray], tuple[int, int], list[tuple[int, int, int, int]]]:
    """Load all IF channels at full resolution as uint16.

    Returns (arrays, (H, W), corrupt_union) where corrupt_union is the union
    of corrupt tile bboxes across all channels.
    """
    arrays        = []
    target_hw     = None
    corrupt_union: list[tuple[int, int, int, int]] = []
    for path, (name, _) in zip(tif_paths, IF_CHANNELS):
        arr, corrupt = _load_channel(path, name, target_hw)
        if target_hw is None:
            target_hw = arr.shape[:2]
        corrupt_union.extend(corrupt)
        arrays.append(arr)          # keep native uint16
    H, W = target_hw
    return arrays, (H, W), corrupt_union


# ── Per-patch warping ─────────────────────────────────────────────────────────

def warp_patch(
    if_arr: np.ndarray,
    Q_he2if: np.ndarray,
    cx_crop: int, cy: int,
    ps_he: int,
    H_if: int, W_if: int,
) -> np.ndarray | None:
    """
    Map 4 HE patch corners (in cropped-HE coords) through Q_he2if to IF space,
    then warpPerspective to OUT_PX×OUT_PX.  Returns uint8 or None if OOB.

    cx_crop = cx_full - range_HE[0]   (crops the left blank strip)
    cy      = same in full-HE and cropped-HE
    """
    corners = np.array([
        [cx_crop,          cy,          1],
        [cx_crop + ps_he,  cy,          1],
        [cx_crop,          cy + ps_he,  1],
        [cx_crop + ps_he,  cy + ps_he,  1],
    ], dtype=np.float64)

    pts_h  = (Q_he2if @ corners.T).T        # homogeneous IF coords
    pts_if = pts_h[:, :2] / pts_h[:, 2:3]  # (4, 2)

    ix0 = int(pts_if[:, 0].min()) 
    iy0 = int(pts_if[:, 1].min()) 
    ix1 = int(pts_if[:, 0].max()) 
    iy1 = int(pts_if[:, 1].max()) 

    if ix0 < 0 or iy0 < 0 or ix1 > W_if or iy1 > H_if:
        return None

    crop    = if_arr[iy0:iy1, ix0:ix1]
    src_pts = np.float32([[pts_if[j, 0] - ix0, pts_if[j, 1] - iy0] for j in range(4)])
    M       = cv2.getPerspectiveTransform(src_pts, DST_PTS)
    return cv2.warpPerspective(crop, M, (OUT_PX, OUT_PX),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT)


# ── Corrupt-tile helpers ──────────────────────────────────────────────────────

def _if_corners(Q_he2if: np.ndarray, cx_crop: int, cy: int, ps_he: int) -> np.ndarray:
    """Return the 4 patch corners (TL,TR,BL,BR) mapped to IF pixel space. (4,2)"""
    corners = np.array([
        [cx_crop,          cy,          1],
        [cx_crop + ps_he,  cy,          1],
        [cx_crop,          cy + ps_he,  1],
        [cx_crop + ps_he,  cy + ps_he,  1],
    ], dtype=np.float64)
    pts_h = (Q_he2if @ corners.T).T
    return pts_h[:, :2] / pts_h[:, 2:3]


def _hits_corrupt(
    pts_if: np.ndarray,
    corrupt_union: list[tuple[int, int, int, int]],
) -> bool:
    """Return True if the IF patch bounding box overlaps any corrupt tile."""
    if not corrupt_union:
        return False
    px0, py0 = pts_if[:, 0].min(), pts_if[:, 1].min()
    px1, py1 = pts_if[:, 0].max(), pts_if[:, 1].max()
    for cx0, cy0, cx1, cy1 in corrupt_union:
        if px1 > cx0 and px0 < cx1 and py1 > cy0 and py0 < cy1:
            return True
    return False


# ── Slide-level IF normalization ──────────────────────────────────────────────

def normalize_slide_channel(
    raw_uint16: np.ndarray,
    lo: int | None,
    p99: float | None = None,
) -> tuple[np.ndarray, float]:
    """clip(x / p99, 0, 1) → float32 [0, 1], then zero pixels below `lo`.

    `p99` is the GLOBAL pooled ceiling (load_global_p99) so every slide shares one ceiling
    per marker — identical to the viz sweeps. If None, falls back to this slide's own p99.
    `lo` is the SAMPLE_BG cutoff in [0, 255] display space (applied AFTER the p99 clip).
    Returns (normed, p99).
    """
    if p99 is None:
        nz  = raw_uint16[raw_uint16 > 0]
        p99 = float(np.percentile(nz, 99)) if len(nz) else 1.0
    normed = np.clip(raw_uint16.astype(np.float32) / max(p99, 1.0), 0.0, 1.0)
    if lo is not None:
        normed[normed < lo / 255.0] = 0.0
    return normed, p99


def save_p99s(p99s: dict[str, float], out_path: Path) -> None:
    with open(out_path, 'w') as f:
        for name, val in p99s.items():
            f.write(f'{name}\t{val:.4f}\n')


def load_p99s(path: Path) -> dict[str, float]:
    p99s = {}
    with open(path) as f:
        for line in f:
            name, val = line.strip().split('\t')
            p99s[name] = float(val)
    return p99s


def pif_channels(
    ch_arrays: list[np.ndarray],
    sample: str,
) -> tuple[list[np.ndarray], list[str], dict[str, float]]:
    """Normalize all non-DAPI channels at slide level.

    Returns (norm_arrays, names, p99s) for TARGET_CHANNELS only (DAPI/DAPI2 excluded).
    """
    sample_bg  = SAMPLE_BG.get(sample, {})
    global_p99 = load_global_p99()                 # one ceiling per marker, all slides
    norm_arrays, names, p99s = [], [], {}
    for arr, (name, _) in zip(ch_arrays, IF_CHANNELS):
        if name in SKIP_CHANNELS:
            continue
        lo = sample_bg.get(name)
        print(f'  {name:<8} normalizing…  lo={lo if lo is not None else "none"}  '
              f'p99(global)={global_p99.get(name)}')
        normed, p99 = normalize_slide_channel(arr, lo, p99=global_p99.get(name))
        norm_arrays.append(normed)
        names.append(name)
        p99s[name] = p99
    return norm_arrays, names, p99s


# ── Token-grid targets ────────────────────────────────────────────────────────

def compute_token_targets(
    norm_arrays: list[np.ndarray],
    Q_he2if: np.ndarray,
    valid_idx: list[int],
    coords: np.ndarray,
    c0_he: int,
    ps_he: int,
    H_if: int, W_if: int,
    token_grid: int = 16,
) -> np.ndarray:
    """Compute (N, C, G, G) mean-expression token targets from pre-normalized slides.

    norm_arrays are already float32 [0, 1] (HistoPlexer pipeline applied at slide level).
    Token values are the spatial max over each (token_px × token_px) cell.
    """
    N        = len(valid_idx)
    C        = len(norm_arrays)
    G        = token_grid
    token_px = OUT_PX // G
    targets  = np.zeros((N, C, G, G), dtype=np.float32)

    for out_i, coord_i in enumerate(valid_idx):
        if (out_i + 1) % 500 == 0:
            print(f'  [{out_i + 1}/{N}] token targets…', flush=True)

        cx, cy  = int(coords[coord_i, 0]), int(coords[coord_i, 1])
        cx_crop = cx - c0_he

        patches = []
        for arr in norm_arrays:
            w = warp_patch(arr, Q_he2if, cx_crop, cy, ps_he, H_if, W_if)
            patches.append(w if w is not None
                           else np.zeros((OUT_PX, OUT_PX), dtype=np.float32))

        if_patch = np.stack(patches)                                       # (C, P, P)
        targets[out_i] = (
            if_patch.reshape(C, G, token_px, G, token_px).mean(axis=(2, 4))
        )

    return targets


# ── HDF5 save ─────────────────────────────────────────────────────────────────

def save_dataset(
    out_path: Path,
    coords: np.ndarray,
    targets: np.ndarray,
    ch_names: list[str],
    sample: str,
    target_mag: float,
    ps_he: int,
    mi: float,
    token_grid: int,
) -> None:
    N, C, G, _ = targets.shape
    with h5py.File(str(out_path), 'w') as f:
        f.create_dataset('coords',  data=coords,
                         dtype='int32', compression='gzip')
        f.create_dataset('targets', data=targets, compression='gzip',
                         chunks=(min(256, N), C, G, G))
        f.attrs['sample']        = sample
        f.attrs['scene']         = SCENE
        f.attrs['marker_names']  = ';'.join(ch_names)
        f.attrs['norm']          = 'histoplexer_slidelevel'
        f.attrs['target_mag']    = target_mag
        f.attrs['ps_he']         = ps_he
        f.attrs['mpp_he']        = MPP_HE
        f.attrs['out_px']        = OUT_PX
        f.attrs['token_grid']    = token_grid
        f.attrs['mutual_info']   = mi
    mb = out_path.stat().st_size / 1e6
    print(f'  Saved {N} patches → {out_path}  ({mb:.0f} MB)')
    print(f'  /coords {coords.shape}  /targets {targets.shape}  mean={targets.mean():.4f}')
    print(f'  markers: {ch_names}')


# ── Visualization ─────────────────────────────────────────────────────────────

def visualize_patches_full(
    sample: str,
    n: int = 6,
    channels: list[str] | None = None,
    seed: int = 42,
    out_png: Path | None = None,
) -> None:
    """Show n random patches at full resolution: H&E alongside each IF channel."""
    if channels is None:
        channels = ['CD8a', 'SOX10', 'CD3', 'FoXP3']  # fallback if called directly

    he_path, npz_path, _, tif_paths = sample_paths(sample)

    npz    = np.load(str(npz_path))
    Q_he2if = npz['transformation_matrix_Q']
    range_HE = npz['range_HE'] * npz['downsample'][0]
    c0_he  = int(range_HE[0])

    h5_path = Path(OUTPUT_DIR) / sample / f'{sample}_patches.h5'
    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]
        raw    = f.attrs.get('marker_names', '')
        stored = (raw.decode() if isinstance(raw, bytes) else raw).split(';')
        ps_he  = int(f.attrs.get('ps_he', OUT_PX))

    channels = [nm for nm in channels if nm in stored]
    ch_indices = [stored.index(nm) for nm in channels]
    ch_tifs    = [tif_paths[CH_NAMES.index(nm)] for nm in channels]

    # Load and normalize full slide channels
    print(f'  Loading {len(channels)} channels for visualization…')
    global_p99 = load_global_p99()
    norm_dict: dict[str, np.ndarray] = {}
    H_if = W_if = None
    for nm, tif_p in zip(channels, ch_tifs):
        arr, _ = _load_channel(tif_p, nm, None)
        if H_if is None:
            H_if, W_if = arr.shape[:2]
        lo = SAMPLE_BG.get(sample, {}).get(nm)
        norm_dict[nm] = normalize_slide_channel(arr, lo, p99=global_p99.get(nm))

    rng = np.random.default_rng(seed)
    sel = rng.choice(len(coords), min(n, len(coords)), replace=False)

    slide  = openslide.OpenSlide(str(he_path))
    n_cols = 1 + len(channels)
    fig, axes = plt.subplots(n, n_cols, figsize=(3 * n_cols, 3 * n))
    if n == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    for row, idx in enumerate(sel):
        cx, cy   = int(coords[idx, 0]), int(coords[idx, 1])
        cx_crop  = cx - c0_he

        # H&E patch
        he = np.array(slide.read_region((cx, cy), 0, (ps_he, ps_he)).convert('RGB'))
        axes[row, 0].imshow(he)
        axes[row, 0].set_title('H&E', fontsize=8)
        axes[row, 0].axis('off')

        for col, nm in enumerate(channels):
            w = warp_patch(norm_dict[nm], Q_he2if, cx_crop, cy, ps_he, H_if, W_if)
            normed = w if w is not None else np.zeros((OUT_PX, OUT_PX), dtype=np.float32)
            axes[row, col + 1].imshow(normed, cmap='gray', vmin=0, vmax=1)
            axes[row, col + 1].set_title(nm, fontsize=8)
            axes[row, col + 1].axis('off')

    plt.suptitle(f'{sample}  —  full-resolution patches (HistoPlexer slide-level norm)', fontsize=11)
    plt.tight_layout()
    if out_png is None:
        out_png = Path(OUTPUT_DIR) / sample / f'{sample}_full_viz.png'
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_png}')


def verify_alignment(
    sample: str,
    n: int = 5,
    seed: int = 42,
    out_png: Path | None = None,
    thumb_level: int = 6,
) -> None:
    """Draw patch bboxes on H&E and IF thumbnails to verify coordinate mapping."""
    he_path, npz_path, _, tif_paths = sample_paths(sample)

    npz      = np.load(str(npz_path))
    Q_he2if  = npz['transformation_matrix_Q']
    range_HE = npz['range_HE'] * npz['downsample'][0]
    c0_he    = int(range_HE[0])

    h5_path = Path(OUTPUT_DIR) / sample / f'{sample}_patches.h5'
    with h5py.File(h5_path, 'r') as f:
        coords = f['coords'][:]
        ps_he  = int(f.attrs.get('ps_he', OUT_PX))

    rng = np.random.default_rng(seed)
    sel = rng.choice(len(coords), min(n, len(coords)), replace=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(sel)))

    # ── H&E thumbnail ────────────────────────────────────────────────────────
    slide  = openslide.OpenSlide(str(he_path))
    ds     = slide.level_downsamples[thumb_level]
    thumb_he = np.array(slide.read_region((0, 0), thumb_level,
                                          slide.level_dimensions[thumb_level]).convert('RGB'))

    # ── IF thumbnail (DAPI, level 6) ─────────────────────────────────────────
    dapi_tif = tif_paths[0]      # DAPI is channel 0
    with tifffile.TiffFile(str(dapi_tif)) as t:
        levels = t.series[0].levels
        lv     = min(thumb_level, len(levels) - 1)
        thumb_if = levels[lv].asarray()
    # normalise for display
    p = np.percentile(thumb_if[thumb_if > 0], 99) if thumb_if.max() > 0 else 1
    thumb_if_disp = np.clip(thumb_if.astype(np.float32) / max(p, 1), 0, 1)

    # IF thumbnail downsample relative to level 0
    H_if_lv0 = levels[0].shape[0]
    ds_if    = H_if_lv0 / thumb_if.shape[0]

    fig, (ax_he, ax_if) = plt.subplots(1, 2, figsize=(14, 7))
    ax_he.imshow(thumb_he);  ax_he.set_title('H&E', fontsize=10);  ax_he.axis('off')
    ax_if.imshow(thumb_if_disp, cmap='gray'); ax_if.set_title('IF DAPI', fontsize=10); ax_if.axis('off')

    from matplotlib.patches import Polygon as MplPolygon, Circle

    for i, idx in enumerate(sel):
        cx, cy  = int(coords[idx, 0]), int(coords[idx, 1])
        cx_crop = cx - c0_he
        col = colors[i]

        # H&E: box + center dot
        he_box = np.array([
            [cx,          cy         ],
            [cx + ps_he,  cy         ],
            [cx + ps_he,  cy + ps_he ],
            [cx,          cy + ps_he ],
        ]) / ds
        ax_he.add_patch(MplPolygon(he_box, closed=True, fill=True,
                                   facecolor=(*col[:3], 0.25), edgecolor=col, linewidth=2))
        ctr_he = he_box.mean(axis=0)
        ax_he.plot(*ctr_he, 'o', color=col, markersize=6)
        ax_he.text(ctr_he[0], ctr_he[1], str(i), color='white',
                   fontsize=7, ha='center', va='center', fontweight='bold')

        # IF box: map 4 corners through Q_he2if then scale to thumbnail
        corners_h = np.array([
            [cx_crop,         cy,          1],
            [cx_crop + ps_he, cy,          1],
            [cx_crop + ps_he, cy + ps_he,  1],
            [cx_crop,         cy + ps_he,  1],
        ], dtype=np.float64)
        pts_h  = (Q_he2if @ corners_h.T).T
        pts_if = (pts_h[:, :2] / pts_h[:, 2:3]) / ds_if
        ax_if.add_patch(MplPolygon(pts_if, closed=True, fill=True,
                                   facecolor=(*col[:3], 0.25), edgecolor=col, linewidth=2))
        ctr_if = pts_if.mean(axis=0)
        ax_if.plot(*ctr_if, 'o', color=col, markersize=6)
        ax_if.text(ctr_if[0], ctr_if[1], str(i), color='white',
                   fontsize=7, ha='center', va='center', fontweight='bold')

    plt.suptitle(f'{sample}  —  alignment check', fontsize=11)
    plt.tight_layout()
    if out_png is None:
        out_png = Path(OUTPUT_DIR) / sample / f'{sample}_alignment_check.png'
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_png}')


def visualize_patches(
    h5_path: Path,
    out_png: Path,
    n: int = 8,
    channels: list[str] | None = None,
    seed: int = 42,
) -> None:
    """Show n random token-grid targets from a saved HDF5 as an inferno heatmap grid."""
    if channels is None:
        channels = ['DAPI', 'CD8a', 'CD68', 'CD4']

    with h5py.File(h5_path, 'r') as f:
        targets = f['targets'][:]          # (N, C, G, G)
        raw     = f.attrs.get('marker_names', '')
        stored  = (raw.decode() if isinstance(raw, bytes) else raw).split(';')

    ch_indices = [stored.index(nm) for nm in channels if nm in stored]
    channels   = [nm for nm in channels if nm in stored]

    N   = len(targets)
    G   = targets.shape[2]
    rng = np.random.default_rng(seed)
    sel = rng.choice(N, min(n, N), replace=False)

    fig, axes = plt.subplots(len(sel), len(channels),
                             figsize=(3 * len(channels), 3 * len(sel)))
    if len(sel) == 1:
        axes = axes[None, :]
    if len(channels) == 1:
        axes = axes[:, None]

    for row, idx in enumerate(sel):
        for col, (name, ci) in enumerate(zip(channels, ch_indices)):
            axes[row, col].imshow(targets[idx, ci], cmap='inferno',
                                  vmin=0, vmax=1, interpolation='nearest')
            axes[row, col].set_title(name, fontsize=8)
            axes[row, col].axis('off')

    sample = h5_path.stem.split('_patches')[0]
    plt.suptitle(f'{sample}  —  token targets (HistoPlexer norm, {G}×{G})', fontsize=11)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_png}')


# ── Per-sample pipeline ───────────────────────────────────────────────────────

def process_sample(sample: str, args: argparse.Namespace) -> None:
    print(f'\n{"="*60}\n  {sample}\n{"="*60}')

    he_path, npz_path, _, tif_paths = sample_paths(sample)

    npz      = np.load(npz_path)
    Q_matrix = npz['transformation_matrix_Q']
    reg_ds   = int(npz['downsample'][0])
    range_HE = npz['range_HE'] * reg_ds
    mi       = float(npz['mutual_info'])
    print(f'  reg_ds={reg_ds}  MI={mi:.4f}  range_HE={range_HE}')

    Q_he2if = Q_matrix
    c0_he   = int(range_HE[0])
    c1_he   = int(range_HE[1])

    # 1. TRIDENT
    job_dir = Path(args.job_dir) / sample
    if args.skip_trident:
        h5_files = list(job_dir.rglob('*_patches.h5'))
        if not h5_files:
            raise FileNotFoundError(f'--skip_trident: no H5 under {job_dir}')
        coords_h5 = h5_files[0]
        print(f'[TRIDENT] reusing {coords_h5}')
    else:
        coords_h5 = run_trident(he_path, job_dir, args)

    coords, patch_size, target_mag = load_trident_coords(coords_h5)
    coords = coords[np.lexsort((coords[:, 0], coords[:, 1]))]
    ps_he  = round(patch_size * (10.0 / target_mag) / MPP_HE)
    print(f'  ps_he={ps_he}px')

    # 2. Load all IF channels + collect corrupt tile bboxes
    print('  Loading IF channels at full resolution…')
    ch_arrays, (H_if, W_if), corrupt_union = load_if_channels(tif_paths)
    if corrupt_union:
        print(f'  corrupt tiles (union across channels): {len(corrupt_union)}')

    # 3. Filter: IF bbox in-bounds AND doesn't overlap a corrupt tile
    valid_idx = []
    n_corrupt  = 0
    for i, (cx, cy) in enumerate(coords):
        cx      = int(cx)
        if cx < c0_he or cx + ps_he > c1_he:
            continue                          # outside registered HE crop
        cx_crop = cx - c0_he
        pts_if  = _if_corners(Q_he2if, cx_crop, int(cy), ps_he)
        ix0 = int(pts_if[:, 0].min()) - 4
        iy0 = int(pts_if[:, 1].min()) - 4
        ix1 = int(pts_if[:, 0].max()) + 4
        iy1 = int(pts_if[:, 1].max()) + 4
        if ix0 < 0 or iy0 < 0 or ix1 > W_if or iy1 > H_if:
            continue
        if _hits_corrupt(pts_if, corrupt_union):
            n_corrupt += 1
            continue
        valid_idx.append(i)

    N = len(valid_idx)
    print(f'  Valid patches: {N} / {len(coords)}'
          f'  (OOB dropped, corrupt dropped: {n_corrupt})')
    if N == 0:
        print('  Skipping — no valid patches.')
        return

    # 4. Slide-level normalization (HistoPlexer pipeline, DAPI/DAPI2 excluded)
    print('  Normalizing IF channels…')
    norm_arrays, ch_names, p99s = normalize_if_channels(ch_arrays, sample)

    # 5. Token-grid targets
    print(f'  Computing {args.token_grid}×{args.token_grid} targets…')
    targets = compute_token_targets(
        norm_arrays, Q_he2if, valid_idx, coords, c0_he,
        ps_he, H_if, W_if, token_grid=args.token_grid,
    )

    # 6. Save
    out_dir = Path(args.output_dir) / sample
    out_dir.mkdir(parents=True, exist_ok=True)
    out_h5  = out_dir / f'{sample}_patches.h5'
    save_p99s(p99s, out_dir / f'{sample}_p99s.txt')
    save_dataset(
        out_h5,
        coords[valid_idx].astype(np.int32),
        targets, ch_names, sample,
        target_mag, ps_he, mi, args.token_grid,
    )

    # 7. Optional visualization
    if args.visualize:
        vis_png = out_dir / f'{sample}_patches_preview.png'
        visualize_patches(out_h5, vis_png, n=args.vis_n,
                          channels=args.vis_channels)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build melanoma2 patch dataset (H&E + 10-plex IF)'
    )
    parser.add_argument('--samples',      default='all',
                        help="'all' or comma-separated e.g. MACEGEJ,MAHEFOG")
    parser.add_argument('--skip_trident', action='store_true',
                        help='Reuse existing TRIDENT coords')
    parser.add_argument('--patch_size',   type=int,   default=224)
    parser.add_argument('--mag',          type=float, default=20)
    parser.add_argument('--overlap',      type=int,   default=0)
    parser.add_argument('--min_tissue',   type=float, default=0.25)
    parser.add_argument('--segmenter',    default='hest',
                        choices=['hest', 'grandqc', 'otsu'])
    parser.add_argument('--seg_thresh',   type=float, default=0.5)
    parser.add_argument('--gpu',             type=int,   default=0)
    parser.add_argument('--job_dir',         default=str(JOB_DIR))
    parser.add_argument('--output_dir',      default=str(OUTPUT_DIR))
    parser.add_argument('--token_grid',      type=int,   default=16,
                        help='Token grid size G; targets shape (N,C,G,G). '
                             '16 matches UNI2 patch tokens.')
    # visualization
    parser.add_argument('--visualize',       action='store_true',
                        help='Save token-target preview PNG after each sample')
    parser.add_argument('--vis_n',           type=int,   default=8)
    parser.add_argument('--vis_channels',    nargs='+',
                        default=['CD8a', 'SOX10', 'CD3', 'FoXP3'])
    parser.add_argument('--vis_only',        type=str,   default=None,
                        help='Path to existing H5; render token-target preview and exit')
    parser.add_argument('--vis_full',        type=str,   default=None,
                        help='Sample name; render full-resolution H&E+IF preview and exit')
    parser.add_argument('--verify_align',   type=str,   default=None,
                        help='Sample name; draw patch bboxes on H&E and IF thumbnails and exit')
    args = parser.parse_args()

    if args.vis_only:
        h5 = Path(args.vis_only)
        visualize_patches(h5, h5.parent / f'{h5.stem}_preview.png',
                          n=args.vis_n, channels=args.vis_channels)
        return

    if args.vis_full:
        visualize_patches_full(args.vis_full, n=args.vis_n, channels=args.vis_channels)
        return

    if args.verify_align:
        verify_alignment(args.verify_align, n=args.vis_n)
        return

    samples = (get_samples() if args.samples.lower() == 'all'
               else [s.strip() for s in args.samples.split(',')])
    print(f'Samples: {samples}')

    for s in samples:
        try:
            process_sample(s, args)
        except Exception as e:
            import traceback
            print(f'ERROR on {s}: {e}')
            traceback.print_exc()


if __name__ == '__main__':
    main()

"""

MACEGEJ': {'SOX10': [100, 255], 'CD3': [248, 255], 'CD8a': [245, 255], 'HLA-DR': [235, 255]}, 
'MELIPIT': {'SOX10': [100, 255], 'CD3': [235, 255], 'CD8a': [222, 255], 'HLA-DR': [245, 255]},
'MIDEKOG': {'SOX10': [120, 255], 'CD3': [245, 255], 'CD8a': [238, 255], 'HLA-DR': [235, 255]},
'MANOFYB': {'SOX10': [120, 255], 'CD3': [210, 255], 'CD8a': [249, 255], 'HLA-DR': [180, 255]}  

"""