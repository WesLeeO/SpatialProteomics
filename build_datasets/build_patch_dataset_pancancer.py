"""
Build patch dataset for CODEX TMA data (CRC_TMA_A/B, Multi-tumor, Tonsil).

Dataset hierarchy
-----------------
PKG - CRC_FFPE-CODEX_CellNeighs_v1/
  {TMA}_HandE/
    reg{NNN}_X{XX}_Y{YY}.tif   shape (2, Z, 4, H, W) uint16
      T=0: actual H&E scan, T=1: empty
      Z:   z-stack (17 slices for CRC/Multi-tumor, 11 for Tonsil) — fluorescence
           microscopy captures multiple focal depths; only one Z is sharpest
      C=0,1,2: scanner RGB channels (uint16)
  {TMA}_hyperstacks/
    channelNames.txt            one name per line, groups of 4 per CODEX round:
                                  line 4k+0: Hoechst (skip)
                                  line 4k+1: Marker A
                                  line 4k+2: Marker B
                                  line 4k+3: Marker C
    bestFocus/
      reg{NNN}_X{XX}_Y{YY}_Z{ZZ}.tif   (T_rounds, 4, H, W) uint16
        The _Z{ZZ} suffix encodes which Z slice had sharpest focus — the
        bestFocus file already contains only that single plane, hence no Z dim.
    reg{NNN}_X{XX}_Y{YY}.tif           (T_rounds, Z, 4, H, W) full z-stack

Resolution: 0.377 µm/px (20x).  To match the project standard of 224px @ 0.5 µm/px
(same physical FOV as Orion), the native crop is:
  patch_size_level0 = round(224 × 0.5 / 0.377) = 297 px

Pipeline (per core)
-------------------
1. Parse channelNames.txt → protein channel list (skip Hoechst/blank/empty)
2. Determine best-Z from bestFocus filename
3. Export H&E as uint8 RGB TIFF → run TRIDENT for tissue segmentation + coords
4. Compute p99s per protein channel across tissue patches
5. Compute (C, G, G) mean-expression token-grid targets per patch
6. Save HDF5 matching orion_crc_patch_dataset_reg format

HDF5 layout
-----------
  /coords   (N, 2)       int64   — (x, y) top-left in native core pixel space
  /p99s     (C,)         float32 — foreground p99 for log1p normalisation
  /targets  (N, C, G, G) float32 — normalised mean expression per token cell
  attrs: sample, marker_names, patch_size, patch_size_level0, token_grid, tma, mpp
"""

import csv
import re
import subprocess
import argparse
import numpy as np
import cv2
import h5py
import tifffile
from PIL import Image
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_ROOT      = Path("/mnt/ssd/virtual_proteomics/data/PathoCell/PKG - CRC_FFPE-CODEX_CellNeighs_v1")
OUTPUT_DIR     = Path("datasets/pancancer_patch_dataset")
JOB_DIR        = Path("datasets/pancancer_trident_output")
TRIDENT_SCRIPT = Path("TRIDENT/run_batch_of_slides.py")

TMAS = {
    "CRC_TMA_A":   "CRC_TMA_A",
    "CRC_TMA_B":   "CRC_TMA_B",
    "Multi-tumor": "Multi-tumor_TMA",
    "Tonsil":      "Tonsil",
}

# TO TEST
# p95 
MPP        = 0.377   # µm/px at 20x (from Experiment.json per_pixel_XY_resolution)
MPP_TARGET = 0.5     # µm/px target (project standard: 224px covers 112 µm)

_SKIP_RE = re.compile(r"^(hoechst|hochst|blank|empty|draq|hande)", re.IGNORECASE)


# ── Channel parsing ────────────────────────────────────────────────────────────

def parse_channel_names(txt_path: Path) -> list[tuple[int, int, str]]:
    """
    Parse channelNames.txt → [(round_idx, ch_within_round, clean_name), ...]
    Includes the Hoechst from round 0 (nuclear stain, cycle 1) as 'Hoechst'.
    Skips all subsequent Hoechst repeats, blank, empty, DRAQ5, and HandE lines.
    """
    channels = []
    hoechst_kept = False
    with open(txt_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    for flat_idx, raw_name in enumerate(lines):
        round_idx = flat_idx // 4
        ch_idx    = flat_idx % 4
        if ch_idx == 0:
            if not hoechst_kept:
                channels.append((round_idx, ch_idx, "Hoechst"))
                hoechst_kept = True
            continue
        if _SKIP_RE.match(raw_name):
            continue
        clean = raw_name.split(" - ")[0].strip()
        channels.append((round_idx, ch_idx, clean))
    return channels


# ── Core discovery ─────────────────────────────────────────────────────────────

def discover_cores(tma_key: str) -> list[str]:
    """Return sorted core IDs (e.g. 'reg001_X01_Y01') found in bestFocus/."""
    bf_dir  = DATA_ROOT / f"{TMAS[tma_key]}_hyperstacks" / "bestFocus"
    pattern = re.compile(r"^(reg\d+_X\d+_Y\d+)_Z\d+\.tif$")
    return sorted({
        m.group(1)
        for f in bf_dir.iterdir()
        if (m := pattern.match(f.name))
    })


def best_focus_z(tma_key: str, core_id: str) -> tuple[Path, int]:
    """
    Locate the bestFocus hyperstack and parse its Z index from the filename.
    Falls back to the full z-stack at Z=8 if the bestFocus file is missing.
    """
    bf_dir  = DATA_ROOT / f"{TMAS[tma_key]}_hyperstacks" / "bestFocus"
    pattern = re.compile(rf"^{re.escape(core_id)}_Z(\d+)\.tif$")
    for f in bf_dir.iterdir():
        m = pattern.match(f.name)
        if m and f.stat().st_size > 1_000_000:
            return f, int(m.group(1))
    full = DATA_ROOT / f"{TMAS[tma_key]}_hyperstacks" / f"{core_id}.tif"
    return full, 8


# ── Image loading ──────────────────────────────────────────────────────────────

def load_hyperstack(hs_path: Path, z_idx: int) -> np.ndarray:
    """
    Return protein hyperstack as (T_rounds, 4, H, W) uint16.
    Handles both bestFocus files (already Z-selected, ndim=4) and full
    z-stacks (ndim=5), selecting z_idx from the latter.
    """
    arr = tifffile.imread(str(hs_path))
    if arr.ndim == 4:
        return arr
    if arr.ndim == 5:
        return arr[:, z_idx, :, :, :]
    raise ValueError(f"Unexpected hyperstack ndim={arr.ndim} shape={arr.shape}")


def export_he_rgb(he_tif: Path, z_idx: int, out_tif: Path) -> None:
    """
    Extract the brightfield H&E from (2, Z, 4, H, W) uint16 → tiled uint8 RGB TIFF.

    T=1, C=1:4 is the brightfield H&E (purple/pink on white background).
    Per-channel p1/p99 stretch is used instead of a fixed >>8 shift so the
    full [0, 255] range is always used regardless of scanner exposure.
    """
    arr   = tifffile.imread(str(he_tif))          # (2, Z, 4, H, W)
    rgb16 = arr[1, z_idx, 1:4].astype(np.float32) # (3, H, W)

    rgb8 = np.empty_like(rgb16, dtype=np.uint8)
    for c in range(3):
        p99 = np.percentile(rgb16[c], 99)
        rgb8[c] = np.clip(rgb16[c] / p99 * 255, 0, 255).astype(np.uint8)

    rgb_hw3 = rgb8.transpose(1, 2, 0)             # (H, W, 3)
    tifffile.imwrite(str(out_tif), rgb_hw3, photometric='rgb', tile=(256, 256))


# ── TRIDENT helpers ────────────────────────────────────────────────────────────

def make_wsi_csv(he_tif: Path, job_dir: Path) -> Path:
    csv_path = job_dir / "wsi_list.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wsi", "mpp"])
        writer.writerow([he_tif.name, MPP])
    return csv_path


def run_trident(he_tif: Path, job_dir: Path, args: argparse.Namespace) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    wsi_csv  = make_wsi_csv(he_tif, job_dir)
    base_cmd = [
        "python", str(TRIDENT_SCRIPT),
        "--wsi_dir",               str(he_tif.parent),
        "--job_dir",               str(job_dir),
        "--gpu",                   str(args.gpu),
        "--segmenter",             args.segmenter,
        "--seg_conf_thresh",       str(args.seg_thresh),
        "--mag",                   str(args.mag),
        "--patch_size",            str(args.patch_size),
        "--overlap",               str(args.overlap),
        "--min_tissue_proportion", str(args.min_tissue),
        "--wsi_ext",               ".tif",
        "--reader_type",           "image",
        "--custom_list_of_wsis",   str(wsi_csv),
    ]
    subprocess.run(base_cmd + ["--task", "seg"],    check=True)
    subprocess.run(base_cmd + ["--task", "coords"], check=True)
    h5_files = list(job_dir.rglob("*_patches.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No coords H5 under {job_dir}")
    return h5_files[0]


def load_trident_coords(h5_path: Path) -> tuple[np.ndarray, int, float]:
    with h5py.File(h5_path, "r") as f:
        key        = "coords" if "coords" in f else list(f.keys())[0]
        coords     = f[key][:]
        patch_size = int(f[key].attrs.get("patch_size", 224))
        target_mag = float(f[key].attrs.get("target_magnification", 20.0))
    print(f"  {len(coords)} patches  patch_size={patch_size} @ {target_mag}x")
    return coords, patch_size, target_mag


# ── Normalisation ──────────────────────────────────────────────────────────────

def compute_global_p99s(
    tma_key: str,
    cores: list[str],
    channels: list[tuple[int, int, str]],
) -> np.ndarray:
    """
    Per-channel p99 of foreground (>0) pixels pooled across ALL cores in one TMA.
    Loads each core's bestFocus hyperstack in turn and accumulates a uint16
    histogram so memory stays bounded.
    """
    C     = len(channels)
    hists = [np.zeros(65536, dtype=np.float64) for _ in range(C)]

    for ki, core_id in enumerate(cores):
        hs_path, z_idx = best_focus_z(tma_key, core_id)
        if not hs_path.exists():
            print(f"  [p99 pass {ki+1}/{len(cores)}] {core_id} — hyperstack missing, skipping")
            continue
        print(f"  [p99 pass {ki+1}/{len(cores)}] {core_id}", flush=True)
        hs = load_hyperstack(hs_path, z_idx)   # (T_rounds, 4, H, W) uint16
        for ci, (t, c, _) in enumerate(channels):
            fg = hs[t, c].ravel()
            fg = fg[fg > 0]
            if len(fg):
                h, _ = np.histogram(fg, bins=65536, range=(0, 65536))
                hists[ci] += h

    p99s = np.ones(C, dtype=np.float32)
    print(f"\n  Global p99s for {tma_key}:")
    for ci, (_, _, name) in enumerate(channels):
        total = hists[ci].sum()
        if total > 0:
            cdf      = np.cumsum(hists[ci] / total)
            p99_bin  = max(int(np.searchsorted(cdf, 0.99, side="right")), 1)
            p99s[ci] = float(p99_bin)
        print(f"    {name:<20}  p99={p99s[ci]:.1f}")
    return p99s


def compute_token_targets(
    hs: np.ndarray,
    channels: list[tuple[int, int, str]],
    coords: np.ndarray,
    patch_size_level0: int,
    p99s: np.ndarray,
    token_grid: int = 16,
) -> np.ndarray:
    """
    (N, C, G, G) mean-expression token grid.  Each patch is cropped at native
    resolution (patch_size_level0 × patch_size_level0), resized to 224×224,
    then block-averaged into a token_grid × token_grid spatial grid.
    Normalisation: clip(x / p99, 0, 1).
    """
    N        = len(coords)
    C        = len(channels)
    H, W     = hs.shape[2], hs.shape[3]
    token_px = 224 // token_grid
    targets  = np.zeros((N, C, token_grid, token_grid), dtype=np.float32)

    for i, (x, y) in enumerate(coords):
        if i % 200 == 0:
            print(f"    [{i}/{N}] computing targets…", flush=True)
        x, y = int(x), int(y)
        h_sl = slice(y, min(y + patch_size_level0, H))
        w_sl = slice(x, min(x + patch_size_level0, W))

        sigs = np.stack([
            hs[t, c, h_sl, w_sl].astype(np.float32) for t, c, _ in channels
        ], axis=0)   # (C, H_p, W_p)

        if sigs.shape[1] < token_grid or sigs.shape[2] < token_grid:
            continue

        normed = np.clip(sigs / p99s[:, None, None], 0.0, 1.0)

        resized = cv2.resize(
            normed.transpose(1, 2, 0), (224, 224),
            interpolation=cv2.INTER_LINEAR,
        )   # (224, 224, C)

        targets[i] = (
            resized
            .reshape(token_grid, token_px, token_grid, token_px, C)
            .mean(axis=(1, 3))
            .transpose(2, 0, 1)
        )   # (C, G, G)

    return targets


# ── HDF5 save ──────────────────────────────────────────────────────────────────

def save_dataset(
    out_path: Path,
    coords: np.ndarray,
    p99s: np.ndarray,
    targets: np.ndarray,
    marker_names: list[str],
    patch_size: int,
    patch_size_level0: int,
    tma: str,
    sample: str,
    token_grid: int = 16,
) -> None:
    N, C, G, _ = targets.shape
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("coords",  data=coords,  compression="gzip")
        f.create_dataset("p99s",    data=p99s)
        f.create_dataset("targets", data=targets, compression="gzip",
                         chunks=(min(256, N), C, G, G))
        f.attrs["sample"]            = sample
        f.attrs["tma"]               = tma
        f.attrs["marker_names"]      = marker_names
        f.attrs["patch_size"]        = patch_size
        f.attrs["patch_size_level0"] = patch_size_level0
        f.attrs["token_grid"]        = token_grid
        f.attrs["mpp"]               = MPP
        f.attrs["normalisation"]     = "clip(x/global_p99, 0, 1)"
    mb = out_path.stat().st_size / 1e6
    print(f"\n  Saved → {out_path}  ({mb:.1f} MB)")
    print(f"    /coords   {coords.shape}")
    print(f"    /targets  {targets.shape}  mean={targets.mean():.4f}")


# ── Per-core pipeline ──────────────────────────────────────────────────────────

def process_core(
    tma_key: str,
    core_id: str,
    channels: list[tuple[int, int, str]],
    global_p99s: np.ndarray,
    args: argparse.Namespace,
) -> None:
    out_path = Path(args.output_dir) / tma_key / f"{core_id}_patch_dataset.h5"
    if out_path.exists() and not args.overwrite:
        print(f"  skipping {core_id} (already exists)")
        return

    hs_path, z_idx = best_focus_z(tma_key, core_id)
    he_tif  = DATA_ROOT / f"{TMAS[tma_key]}_HandE" / f"{core_id}.tif"
    job_dir = Path(args.job_dir) / tma_key / core_id

    if not hs_path.exists():
        print(f"  SKIP {core_id}: hyperstack not found")
        return
    if not he_tif.exists():
        print(f"  SKIP {core_id}: H&E not found")
        return

    print(f"\n{'='*60}\n  {tma_key}  /  {core_id}  (Z={z_idx})\n{'='*60}")

    # 1. Export H&E as tiled uint8 RGB TIFF for TRIDENT (ImageWSI/PIL backend)
    job_dir.mkdir(parents=True, exist_ok=True)
    he_rgb_tif = job_dir / f"{core_id}.tif"
    if not he_rgb_tif.exists():
        print("  Exporting H&E RGB…")
        export_he_rgb(he_tif, z_idx, he_rgb_tif)

    # 2. TRIDENT — tissue segmentation + patch coords
    if args.skip_trident:
        h5_files = list(job_dir.rglob("*_patches.h5"))
        if not h5_files:
            raise FileNotFoundError(f"--skip_trident: no coords H5 under {job_dir}")
        coords_h5 = h5_files[0]
        print(f"[TRIDENT] reusing {coords_h5}")
    else:
        coords_h5 = run_trident(he_rgb_tif, job_dir, args)

    coords, patch_size, target_mag = load_trident_coords(coords_h5)
    coords = coords[np.lexsort((coords[:, 0], coords[:, 1]))]

    # Native crop size covering the same physical FOV as patch_size @ MPP_TARGET
    patch_size_level0 = round(patch_size * MPP_TARGET / MPP)
    print(f"  patch_size_level0 = {patch_size_level0} px  "
          f"({patch_size_level0 * MPP:.1f} µm at {MPP} µm/px)")

    # 3. Load protein hyperstack at best-focus Z
    hs = load_hyperstack(hs_path, z_idx)   # (T_rounds, 4, H, W) uint16

    # 4. Token-grid targets (global_p99s shared across all cores in this TMA)
    print(f"\n  Computing {args.token_grid}×{args.token_grid} token targets ({len(coords)} patches)…")
    targets = compute_token_targets(
        hs, channels, coords, patch_size_level0, global_p99s, args.token_grid,
    )

    # 5. Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(
        out_path, coords, global_p99s, targets,
        marker_names=[name for *_, name in channels],
        patch_size=patch_size,
        patch_size_level0=patch_size_level0,
        tma=tma_key,
        sample=core_id,
        token_grid=args.token_grid,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build CODEX TMA patch dataset (reg format)"
    )
    parser.add_argument("--tma",         default="all",
                        help=f"'all' or one of: {list(TMAS)}")
    parser.add_argument("--cores",       default="all",
                        help="'all' or comma-separated core IDs e.g. reg001_X01_Y01")
    parser.add_argument("--skip_trident", action="store_true")
    parser.add_argument("--patch_size",  type=int,   default=224)
    parser.add_argument("--mag",         type=float, default=20)
    parser.add_argument("--overlap",     type=int,   default=0)
    parser.add_argument("--min_tissue",  type=float, default=0.1)
    parser.add_argument("--segmenter",   default="hest",
                        choices=["hest", "grandqc", "otsu"])
    parser.add_argument("--seg_thresh",  type=float, default=0.5)
    parser.add_argument("--gpu",         type=int,   default=1)
    parser.add_argument("--token_grid",  type=int,   default=16)
    parser.add_argument("--output_dir",  default=str(OUTPUT_DIR))
    parser.add_argument("--job_dir",     default=str(JOB_DIR))
    parser.add_argument("--overwrite",   action="store_true")
    args = parser.parse_args()

    tma_keys = list(TMAS) if args.tma == "all" else [args.tma]

    for tma_key in tma_keys:
        channels_txt = DATA_ROOT / f"{TMAS[tma_key]}_hyperstacks" / "channelNames.txt"
        channels     = parse_channel_names(channels_txt)
        print(f"\n{'#'*60}")
        print(f"  TMA: {tma_key}  ({len(channels)} protein channels)")
        print(f"  Markers: {[n for *_, n in channels]}")
        print(f"{'#'*60}")

        cores = (discover_cores(tma_key) if args.cores == "all"
                 else [c.strip() for c in args.cores.split(",")])
        print(f"  {len(cores)} cores")

        # Pass 1: global p99s pooled across all cores in this TMA
        print(f"\n{'='*60}")
        print(f"  Pass 1: computing global p99s across {len(cores)} cores")
        print(f"{'='*60}")
        global_p99s = compute_global_p99s(tma_key, cores, channels)

        # Pass 2: per-core patch extraction + token targets
        print(f"\n{'='*60}")
        print(f"  Pass 2: extracting patches")
        print(f"{'='*60}")
        for core_id in cores:
            try:
                process_core(tma_key, core_id, channels, global_p99s, args)
            except Exception as e:
                print(f"  ERROR {core_id}: {e}")


if __name__ == "__main__":
    main()