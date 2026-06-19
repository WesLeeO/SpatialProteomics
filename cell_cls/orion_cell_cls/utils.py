"""
Shared helpers for orion_cell_cls.

Holds everything used by more than one script in this folder:
  * paths / grid + mask geometry constants
  * the cell <-> token <-> mask machinery (label crops, token ids, marker lookups)
  * the cached-prediction loader
  * deterministic-rule metrics (confusion, best-F1 threshold sweep)

Importing this module also puts the project root on sys.path, so sibling
root-level modules (e.g. build_patch_dataset_orion_crc_reg, imported by the
overlay) resolve no matter which directory a script is launched from.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

# project root = parent of orion_cell_cls/  → make root modules importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Paths ────────────────────────────────────────────────────────────────────
BENCH_DIR  = ROOT / "orion_crc_patch_dataset_benchmark"
TILE_DIR   = Path("/mnt/ssd/virtual_proteomics/data/ORIONCRC_dataset_tile_20x")
SLIDE_DF   = TILE_DIR / "slide_dataframe.csv"
NUCLEI_DIR = TILE_DIR / "nuclei"
DEFAULT_PRED_DIR = ROOT / "outputs_orion_token_UNI2_baseline_unfreeze4_2loss_lbg8"

# ── Grid / mask geometry ───────────────────────────────────────────────────────
TILE_L0   = 512   # level-0 pixels covered by one mask tile
TILE_MASK = 333   # stored mask-tile side at 20× (≈ 512 × 0.325/0.5)
MPP_20X   = 0.5   # µm/px of the stored masks (20×)
GRID      = 16    # token grid (UNI2: 224/16 = 14 px per token)
NTOK      = GRID * GRID


# ── Lookups ────────────────────────────────────────────────────────────────────

def sample_rows(sample: str) -> pd.Series:
    df  = pd.read_csv(SLIDE_DF)
    row = df[df.orion_slide_id == sample]
    if len(row) == 0:
        raise KeyError(f"{sample} not found in {SLIDE_DF}")
    return row.iloc[0]


def csv_marker_col(cells: pd.DataFrame, marker: str) -> str:
    """Resolve benchmark marker name → CSV `<marker>_pos` column (E-Cadherin→E-cadherin_pos)."""
    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s.lower())
    want = norm(marker)
    for col in cells.columns:
        if col.endswith("_pos") and norm(col[:-4]) == want:
            return col
    raise KeyError(f"no _pos column for '{marker}' "
                   f"(have {[c for c in cells.columns if c.endswith('_pos')]})")


def index_nuclei_tiles(base: str) -> dict[tuple[int, int], Path]:
    """Map (X, Y) level-0 tile origin → mask tile path for one slide."""
    idx = {}
    for p in NUCLEI_DIR.glob(f"{base}_*_0_512_512.tiff"):
        m = re.search(r"_(\d+)_(\d+)_0_512_512\.tiff$", p.name)
        if m:
            idx[(int(m.group(1)), int(m.group(2)))] = p
    return idx


def load_patch_labels(tile_index: dict, px: int, py: int, ps0: int) -> np.ndarray:
    """Crop the instance-label raster for patch [px:px+ps0, py:py+ps0] at native 20×.

    The mask is NEVER resampled — we work in the masks' own 20× pixel grid
    (a tile at level-0 X = k·512 lives at mask-px k·333) and integer-slice the
    overlapping native tiles into the crop. Returns a (~224, ~224) int32 raster.
    """
    scale = TILE_MASK / TILE_L0                       # mask-px per level-0 px
    m0, m1 = round(px * scale),         round((px + ps0) * scale)
    n0, n1 = round(py * scale),         round((py + ps0) * scale)
    out = np.zeros((n1 - n0, m1 - m0), dtype=np.int32)

    for tx in range((px // TILE_L0) * TILE_L0, px + ps0, TILE_L0):
        for ty in range((py // TILE_L0) * TILE_L0, py + ps0, TILE_L0):
            path = tile_index.get((tx, ty))
            if path is None:
                continue
            tile = tifffile.imread(path)              # native 20× raster (~333²)
            th, tw = tile.shape
            xm, ym = round(tx * scale), round(ty * scale)   # tile origin in mask-px
            mx0, mx1 = max(xm, m0), min(xm + tw, m1)
            my0, my1 = max(ym, n0), min(ym + th, n1)
            if mx1 <= mx0 or my1 <= my0:
                continue
            out[my0 - n0:my1 - n0, mx0 - m0:mx1 - m0] = \
                tile[my0 - ym:my1 - ym, mx0 - xm:mx1 - xm]
    return out


_TOK_CACHE: dict[tuple[int, int], np.ndarray] = {}

def token_ids(mh: int, mw: int, grid: int = GRID) -> np.ndarray:
    """Map every pixel of an (mh, mw) crop to its token index in a grid×grid lattice.

    The model predicts a grid×grid (16×16) token map over the SAME field of view as
    the (mh, mw) mask crop, so each token covers an (mh/grid) × (mw/grid) block of
    pixels. This returns, per pixel (row-major), the flat token id

        token_id = token_row * grid + token_col   ∈ [0, grid²)

    Row-major on purpose: it matches a (C, grid, grid) prediction reshaped to
    (grid², C), so `preds[i].reshape(C, grid*grid).T[token_id]` is that pixel's
    predicted vector — and it lines up with `lab.ravel()` pixel-for-pixel.

    Cached per (mh, mw): almost every patch is ~224×224, so the lattice is built once.
    """
    key = (mh, mw)
    t = _TOK_CACHE.get(key)
    if t is None:
        # token-ROW for each pixel row: floor(row / mh * grid) == (row*grid)//mh.
        # np.minimum clips the last row (mh-1) to grid-1 so it doesn't spill to grid.
        ty = np.minimum((np.arange(mh) * grid) // mh, grid - 1)   # (mh,)  in 0..grid-1
        # token-COLUMN for each pixel column, same idea
        tx = np.minimum((np.arange(mw) * grid) // mw, grid - 1)   # (mw,)  in 0..grid-1
        # broadcast to the 2D map token_id[r, c] = ty[r]*grid + tx[c], then flatten
        # row-major (C-order) to match lab.ravel() and the reshaped prediction grid
        t = (ty[:, None] * grid + tx[None, :]).ravel()            # (mh*mw,)
        _TOK_CACHE[key] = t
    return t


# ── Predictions ─────────────────────────────────────────────────────────────────

def load_slide_predictions(sample: str, pred_dir):
    """Return (coords, ps0, marker_names, preds) for one slide.

    coords/ps0/marker_names come from the benchmark h5; preds is the cached
    (N, C, GRID, GRID) array in `pred_dir`, row-aligned to coords.
    """
    import h5py
    with h5py.File(BENCH_DIR / f"{sample}_patch_dataset.h5", "r") as f:
        coords       = f["coords"][:]
        ps0          = int(f.attrs["patch_size_level0"])
        marker_names = [str(m) for m in f.attrs["marker_names"]]
    preds = np.load(Path(pred_dir) / f"{sample}_preds.npy")
    assert len(preds) == len(coords), (
        f"{sample}: preds {len(preds)} != coords {len(coords)} — stale cache in {pred_dir}")
    assert preds.shape[1] == len(marker_names), (
        f"{preds.shape[1]} pred channels vs {len(marker_names)} marker names")
    return coords, ps0, marker_names, preds


# ── Metrics ──────────────────────────────────────────────────────────────────────

def confusion(pred: np.ndarray, y: np.ndarray) -> dict:
    """TP/FP/FN/TN + precision/recall/specificity/F1 for boolean arrays."""
    tp = int(np.count_nonzero(pred & y))
    fp = int(np.count_nonzero(pred & ~y))
    fn = int(np.count_nonzero(~pred & y))
    tn = int(np.count_nonzero(~pred & ~y))
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec  = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    return dict(TP=tp, FP=fp, FN=fn, TN=tn,
                precision=prec, recall=rec, specificity=spec, f1=f1)


