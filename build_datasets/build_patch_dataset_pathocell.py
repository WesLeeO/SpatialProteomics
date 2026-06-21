"""
PathoCell HDF patch dataset builder  (v2 — token-grid targets)

Data layout:
    <DATA_DIR>/
        reg001_A.hdf, reg001_B.hdf, ...  (109 files, 67 unique regions)
        IHC_channels.txt   — "1; CD44 - stroma" per line (1-indexed, 58 channels)

HDF contents per file:
    img : (3, H, W)  uint16 — H&E
    ifl : (58, H, W) uint16 — IF protein channels

Skipped channels:
    DRAQ5 (index 57, 0-based) — nuclear stain, not a protein marker

Geometry:
    Native MPP : 0.377 µm/px (CODEX)
    Target mag : 20x  →  target MPP = 0.5 µm/px
    Native crop: round(224 × 0.5 / 0.377) = 297 px per side
    Token grid : 16×16  (UNI2 aligned — 224 / 16 = 14 px per token)

IF normalisation:
    Per-slide p99 from non-zero pixels  →  clip(log1p(x / p99), 0, 1)

HDF5 layout (one file per slide, consumed by a PathoCell dataset class)
------------------------------------------------------------------------
  /coords   (N, 2)       int32   — (x, y) top-left in tile pixel space
  /targets  (N, C, G, G) float32 — normalised mean expression per token cell
  /p99s     (C,)         float32 — per-slide foreground p99 per channel
  /sources  (N,)         bytes   — HDF stem repeated for each patch
  attrs: marker_names, patch_size, patch_size_level0, token_grid,
         native_mpp, target_mpp, n_patches, normalisation

Run:
    python build_patch_dataset_pathocell.py
    python build_patch_dataset_pathocell.py --files reg001_A,reg002_B
    python build_patch_dataset_pathocell.py --skip_trident  # reuse existing coords
"""

import sys
import csv
import argparse
import subprocess
import numpy as np
import cv2
import h5py
import tifffile
from pathlib import Path

DATA_DIR       = Path("/mnt/ssd1/virtual_proteomics/data/pathocell/pathocell/pathocell_hdf")
TRIDENT_SCRIPT = Path("TRIDENT/run_batch_of_slides.py")
OUTPUT_DIR     = Path("datasets/pathocell_patch_dataset")

NATIVE_MPP = 0.377           # µm/px (CODEX)
TOKEN_GRID = 16              # UNI2 token grid side
TOKEN_PX   = 224 // TOKEN_GRID  # = 14 px per token after resize

SKIP_CHANNEL_INDICES = {57}  # DRAQ5 — nuclear stain


# ── channel parsing ───────────────────────────────────────────────────────────

def parse_ihc_channels(txt_path: Path) -> list[tuple[int, str]]:
    """
    Parse IHC_channels.txt  →  list of (channel_idx_0based, marker_name).
    Format per line:  "1; CD44 - stroma"
    """
    channels = []
    for line in txt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        idx_str, rest = line.split(";", 1)
        idx  = int(idx_str.strip()) - 1          # 1-based → 0-based
        name = rest.strip().split(" - ")[0].strip()
        if idx in SKIP_CHANNEL_INDICES:
            continue
        channels.append((idx, name))
    return channels


# ── H&E normalisation ─────────────────────────────────────────────────────────

def he_uint16_to_uint8(img: np.ndarray) -> np.ndarray:
    """(3, H, W) uint16  →  (H, W, 3) uint8 via per-channel p99 stretch."""
    rgb = img.astype(np.float32)
    for c in range(3):
        p99 = np.percentile(rgb[c], 99)
        if p99 > 0:
            rgb[c] = np.clip(rgb[c] / p99, 0.0, 1.0) * 255.0
    return rgb.transpose(1, 2, 0).astype(np.uint8)   # (H, W, 3)


# ── TRIDENT tissue segmentation ───────────────────────────────────────────────

def run_trident(
    he_rgb: np.ndarray,
    stem: str,
    job_dir: Path,
    mag: float,
    patch_size: int,
    overlap: int,
    min_tissue: float,
    segmenter: str,
    seg_thresh: float,
    gpu: int,
) -> np.ndarray:
    """Write H&E as tiled TIFF, run TRIDENT seg+coords, return (N, 2) coords."""
    tmp_dir = job_dir / "tmp_he"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_tif = tmp_dir / f"{stem}.tif"
    tifffile.imwrite(str(tmp_tif), he_rgb, tile=(256, 256), photometric="rgb")

    csv_path = job_dir / "wsi_list.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wsi", "mpp"])
        writer.writerow([tmp_tif.name, NATIVE_MPP])

    base_cmd = [
        sys.executable, str(TRIDENT_SCRIPT),
        "--wsi_dir",   str(tmp_dir),
        "--job_dir",   str(job_dir),
        "--gpu",       str(gpu),
        "--segmenter", segmenter,
        "--seg_conf_thresh", str(seg_thresh),
        "--mag",        str(mag),
        "--patch_size", str(patch_size),
        "--overlap",    str(overlap),
        "--min_tissue_proportion", str(min_tissue),
        "--wsi_ext",    ".tif",
        "--custom_list_of_wsis", str(csv_path),
    ]
    subprocess.run(base_cmd + ["--task", "seg"],    check=True)
    subprocess.run(base_cmd + ["--task", "coords"], check=True)

    h5_files = list(job_dir.rglob(f"{stem}_patches.h5"))
    if not h5_files:
        h5_files = list(job_dir.rglob("*_patches.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No coords HDF5 found under {job_dir} for {stem}")

    with h5py.File(h5_files[0], "r") as f:
        key = "coords" if "coords" in f else list(f.keys())[0]
        return f[key][:]


# ── p99 computation ───────────────────────────────────────────────────────────

def compute_slide_p99s(
    ifl: np.ndarray,                   # (58, H, W) uint16 — full slide
    channels: list[tuple[int, str]],
) -> np.ndarray:
    """Per-channel p99 from non-zero pixels across the full slide."""
    C   = len(channels)
    p99 = np.ones(C, dtype=np.float32)
    for ci, (ch_idx, name) in enumerate(channels):
        px      = ifl[ch_idx].ravel().astype(np.float32)
        nonzero = px[px > 0]
        val     = float(np.percentile(nonzero, 99)) if len(nonzero) else 1.0
        p99[ci] = val
        print(f"    {name:<20s}  p99={val:.1f}")
    return p99


# ── token-grid target extraction ──────────────────────────────────────────────

def compute_token_grid_targets(
    ifl: np.ndarray,                   # (58, H, W) uint16
    channels: list[tuple[int, str]],
    coords: np.ndarray,                # (N, 2) int  — (x, y) in tile space
    patch_size_level0: int,
    p99s: np.ndarray,                  # (C,) float32
    token_grid: int = TOKEN_GRID,
) -> np.ndarray:
    """
    For each patch:
      1. Crop patch_size_level0 × patch_size_level0 from ifl
      2. Normalise: clip(log1p(crop / p99), 0, 1)
      3. Resize (224, 224) via INTER_LINEAR
      4. Block-mean into (token_grid, token_grid) cells

    Returns (N, C, token_grid, token_grid) float32.
    """
    _, H, W = ifl.shape
    N   = len(coords)
    C   = len(channels)
    ch_indices = [ch_idx for ch_idx, _ in channels]
    p99s_bcst  = p99s[:, None, None]               # (C, 1, 1) for broadcasting

    targets = np.zeros((N, C, token_grid, token_grid), dtype=np.float32)

    for i, (px, py) in enumerate(coords):
        if i % 200 == 0:
            print(f"    [{i}/{N}] computing token targets…", flush=True)
        x0, x1 = int(px), min(int(px) + patch_size_level0, W)
        y0, y1 = int(py), min(int(py) + patch_size_level0, H)

        crop = ifl[np.ix_(ch_indices,
                           np.arange(y0, y1),
                           np.arange(x0, x1))].astype(np.float32)  # (C, H_p, W_p)

        if crop.shape[1] < token_grid or crop.shape[2] < token_grid:
            continue

        normed = np.clip(np.log1p(crop / p99s_bcst), 0.0, 1.0)   # (C, H_p, W_p)

        resized = cv2.resize(
            normed.transpose(1, 2, 0),                             # (H_p, W_p, C)
            (224, 224),
            interpolation=cv2.INTER_LINEAR,
        )                                                           # (224, 224, C)

        targets[i] = (
            resized
            .reshape(token_grid, TOKEN_PX, token_grid, TOKEN_PX, C)
            .mean(axis=(1, 3))
            .transpose(2, 0, 1)
        )                                                           # (C, G, G)

    return targets


# ── HDF5 save ─────────────────────────────────────────────────────────────────

def save_dataset(
    out_path: Path,
    stem: str,
    coords: np.ndarray,
    targets: np.ndarray,
    p99s: np.ndarray,
    marker_names: list[str],
    patch_size: int,
    patch_size_level0: int,
    token_grid: int,
    target_mpp: float,
) -> None:
    N, C, G, _ = targets.shape
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("coords",  data=coords.astype(np.int32),     compression="gzip")
        f.create_dataset("targets", data=targets,                      compression="gzip",
                         chunks=(min(256, N), C, G, G))
        f.create_dataset("p99s",    data=p99s)
        f.create_dataset("sources",
                         data=np.array([stem] * N, dtype=f"S{len(stem)}"),
                         compression="gzip")
        f.attrs["marker_names"]      = marker_names
        f.attrs["patch_size"]        = patch_size
        f.attrs["patch_size_level0"] = patch_size_level0
        f.attrs["token_grid"]        = token_grid
        f.attrs["native_mpp"]        = NATIVE_MPP
        f.attrs["target_mpp"]        = target_mpp
        f.attrs["n_patches"]         = N
        f.attrs["normalisation"]     = "per-slide p99 (nonzero px): clip(log1p(x/p99),0,1)"

    mb = out_path.stat().st_size / 1e6
    print(f"  Saved → {out_path}  ({mb:.1f} MB)")
    print(f"    /coords   {coords.shape}")
    print(f"    /targets  {targets.shape}  mean={targets.mean():.4f}")


# ── per-slide pipeline ────────────────────────────────────────────────────────

def process_slide(
    hdf_path: Path,
    channels: list[tuple[int, str]],
    args: argparse.Namespace,
    patch_size_level0: int,
    target_mpp: float,
) -> None:
    stem    = hdf_path.stem
    out_path = OUTPUT_DIR / f"{stem}_patch_dataset.h5"
    job_dir  = Path(args.job_dir) / stem

    print(f"\n{'='*60}\n  {stem}\n{'='*60}")

    with h5py.File(hdf_path, "r") as f:
        img = f["img"][:]   # (3, H, W) uint16
        ifl = f["ifl"][:]   # (58, H, W) uint16

    he_rgb = he_uint16_to_uint8(img)   # (H, W, 3) uint8

    # 1. TRIDENT coords
    if args.skip_trident:
        h5_files = list(job_dir.rglob("*_patches.h5"))
        if not h5_files:
            raise FileNotFoundError(f"--skip_trident: no coords h5 under {job_dir}")
        with h5py.File(h5_files[0], "r") as f:
            key    = "coords" if "coords" in f else list(f.keys())[0]
            coords = f[key][:]
        print(f"  Reusing {len(coords)} coords from {h5_files[0]}")
    else:
        job_dir.mkdir(parents=True, exist_ok=True)
        coords = run_trident(
            he_rgb, stem, job_dir,
            args.mag, args.patch_size, args.overlap, args.min_tissue,
            args.segmenter, args.seg_thresh, args.gpu,
        )

    print(f"  {len(coords)} patches")
    if len(coords) == 0:
        print("  No patches — skipping.")
        return

    # Sort by (row, col) for better zarr/HDF cache locality
    coords = coords[np.lexsort((coords[:, 0], coords[:, 1]))]

    # 2. Per-slide p99
    print(f"  Computing per-slide p99s for {len(channels)} channels…")
    p99s = compute_slide_p99s(ifl, channels)

    # 3. Token-grid targets
    print(f"  Computing {TOKEN_GRID}×{TOKEN_GRID} token targets ({len(coords)} patches)…")
    targets = compute_token_grid_targets(
        ifl, channels, coords, patch_size_level0, p99s, TOKEN_GRID,
    )

    # 4. Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    marker_names = [name for _, name in channels]
    save_dataset(
        out_path, stem, coords, targets, p99s,
        marker_names, args.patch_size, patch_size_level0,
        TOKEN_GRID, target_mpp,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PathoCell HDF patch dataset builder (token-grid targets)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--files",        default=None,
                        help="Comma-separated HDF stems (default: all). E.g. reg001_A,reg002_B")
    parser.add_argument("--mag",          type=float, default=20,
                        help="Target magnification (default: 20 → 0.5 mpp)")
    parser.add_argument("--patch_size",   type=int,   default=224)
    parser.add_argument("--overlap",      type=int,   default=0)
    parser.add_argument("--min_tissue",   type=float, default=0.25)
    parser.add_argument("--segmenter",    default="hest",
                        choices=["hest", "grandqc", "otsu"])
    parser.add_argument("--seg_thresh",   type=float, default=0.5)
    parser.add_argument("--gpu",          type=int,   default=0)
    parser.add_argument("--skip_trident", action="store_true",
                        help="Reuse existing TRIDENT coords")
    parser.add_argument("--job_dir",      default=str(OUTPUT_DIR / "trident_output"))
    args = parser.parse_args()

    ch_txt      = DATA_DIR / "IHC_channels.txt"
    channels    = parse_ihc_channels(ch_txt)
    target_mpp  = 10.0 / args.mag
    patch_size_level0 = round(args.patch_size * target_mpp / NATIVE_MPP)

    print("=" * 60)
    print("  PathoCell patch dataset builder  (token-grid v2)")
    print(f"  data dir      : {DATA_DIR}")
    print(f"  output dir    : {OUTPUT_DIR}")
    print(f"  magnification : {args.mag}x  ({target_mpp:.4f} mpp)")
    print(f"  native mpp    : {NATIVE_MPP} µm/px")
    print(f"  native crop   : {patch_size_level0} px  →  224 px model input")
    print(f"  token grid    : {TOKEN_GRID}×{TOKEN_GRID}  ({TOKEN_PX}px per token)")
    print(f"  channels      : {len(channels)} (DRAQ5 excluded)")
    print(f"  normalisation : per-slide p99 → clip(log1p(x/p99), 0, 1)")
    print("=" * 60)

    all_hdf = sorted(DATA_DIR.glob("*.hdf"))
    if args.files:
        stems   = {s.strip() for s in args.files.split(",")}
        all_hdf = [f for f in all_hdf if f.stem in stems]
    print(f"\n  {len(all_hdf)} HDF files to process")

    for hdf_path in all_hdf:
        try:
            process_slide(hdf_path, channels, args, patch_size_level0, target_mpp)
        except Exception as e:
            import traceback
            print(f"  ERROR [{hdf_path.stem}]: {e}")
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
