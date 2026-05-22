"""
Build NSCLC Charité patch dataset for token-level regression training.

Dataset structure
-----------------
  /mnt/ssd1/virtual_proteomics/data/nsclc_charite/extracted/spot_center_crops/
    {uuid}_he.jpg          H&E center crop (4096×4096, RGB uint8)
    {uuid}_{marker}.jpg    IF marker crop   (4096×4096, L uint8)

  77 spots have H&E; each has 12 IF markers (some spots missing 1 marker).
  H&E and IF are pre-aligned — same pixel coordinate space, no registration needed.

Markers (canonical order)
--------------------------
  CD163, CD20, CD3, CD4, CD56, CD68, CD8, CK, FoxP3, Granzyme_B, PD-1, PD-L1

Pipeline (per spot)
--------------------
1. p99 pass  : per-channel 99th-percentile of foreground pixels across all spots.
2. TRIDENT   : tissue segmentation + patch coords on H&E (reader_type=image).
3. targets   : (N, C, G, G) token-grid mean expression (G=16 for UNI2).
4. HDF5      : same layout as other patch datasets in this project.

Output
------
  nsclc_charite_patch_dataset.h5

Notes
-----
- MPP is not encoded in the JPEG files. Pass --mpp to match the scanner resolution
  (default 0.5 µm/px = 20×). If unknown, use 0.5 and verify patch coverage visually.
- Missing markers are filled with zeros (not excluded) so all patches have shape
  (N, C, 16, 16) with C=12 always.
- Normalisation: log1p(x / global_p99) clipped to [0, 1], matching ORION convention.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import csv
import argparse
import subprocess
import numpy as np
import cv2
import h5py
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# ── Paths ─────────────────────────────────────────────────────────────────────
CROPS_DIR      = Path("/mnt/ssd1/virtual_proteomics/data/nsclc_charite/extracted/spot_center_crops")
TRIDENT_SCRIPT = Path("TRIDENT/run_batch_of_slides.py")
JOB_DIR        = Path("nsclc_charite_trident_output")
OUTPUT_PATH    = Path("nsclc_charite_patch_dataset.h5")

MARKERS = [
    "CD163", "CD20", "CD3", "CD4", "CD56", "CD68",
    "CD8", "CK", "FoxP3", "Granzyme_B", "PD-1", "PD-L1",
]
TOKEN_GRID = 16   # UNI2: 224/14 = 16 tokens per side


# ── Spot discovery ─────────────────────────────────────────────────────────────

def discover_spots() -> list[str]:
    """Return sorted list of spot UUIDs that have an H&E JPEG."""
    spots = sorted(
        p.stem.replace("_he", "")
        for p in CROPS_DIR.glob("*_he.jpg")
    )
    print(f"Found {len(spots)} spots with H&E")
    return spots


# ── Pass 1: global p99 per channel ────────────────────────────────────────────

def compute_global_p99s(spots: list[str], markers: list[str]) -> np.ndarray:
    """Pool non-background pixels across all spots → one p99 per marker."""
    pixel_pools: list[list[np.ndarray]] = [[] for _ in markers]

    for si, spot in enumerate(spots):
        print(f"  [p99 pass {si+1}/{len(spots)}] {spot}", flush=True)
        for mi, marker in enumerate(markers):
            path = CROPS_DIR / f"{spot}_{marker}.jpg"
            if not path.exists():
                continue
            img = np.array(Image.open(path)).astype(np.float32)
            fg = img[img > 5]
            if len(fg):
                pixel_pools[mi].append(fg)

    global_p99s = np.ones(len(markers), dtype=np.float32)
    print("\nGlobal p99s:")
    for mi, marker in enumerate(markers):
        if pixel_pools[mi]:
            all_vals = np.concatenate(pixel_pools[mi])
            global_p99s[mi] = float(np.percentile(all_vals, 99))
        print(f"  {marker}: {global_p99s[mi]:.1f}  ({sum(len(a) for a in pixel_pools[mi]):,} px)")

    return global_p99s


# ── TRIDENT ────────────────────────────────────────────────────────────────────

def make_wsi_csv(he_path: Path, job_dir: Path, mpp: float) -> Path:
    csv_path = job_dir / "wsi_list.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wsi", "mpp"])
        writer.writerow([he_path.name, mpp])
    return csv_path


def run_trident(he_path: Path, job_dir: Path, mpp: float, mag: float,
                patch_size: int, overlap: int, min_tissue: float,
                segmenter: str, seg_conf: float, gpu: int) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    wsi_csv = make_wsi_csv(he_path, job_dir, mpp)

    base_cmd = [
        sys.executable, str(TRIDENT_SCRIPT),
        "--wsi_dir",    str(he_path.parent),
        "--job_dir",    str(job_dir),
        "--gpu",        str(gpu),
        "--segmenter",  segmenter,
        "--seg_conf_thresh", str(seg_conf),
        "--mag",        str(mag),
        "--patch_size", str(patch_size),
        "--overlap",    str(overlap),
        "--min_tissue_proportion", str(min_tissue),
        "--custom_list_of_wsis",  str(wsi_csv),
        "--reader_type", "image",
    ]
    subprocess.run(base_cmd + ["--task", "seg"],    check=True)
    subprocess.run(base_cmd + ["--task", "coords"], check=True)

    h5_files = list(job_dir.rglob("*_patches.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No coords HDF5 found under {job_dir}")
    return h5_files[0]


def load_trident_coords(coords_h5: Path):
    with h5py.File(coords_h5, "r") as f:
        key        = "coords" if "coords" in f else list(f.keys())[0]
        coords     = f[key][:]
        patch_size = int(f[key].attrs.get("patch_size", 224))
        mag        = float(f[key].attrs.get("target_magnification", 20.0))
    print(f"  {len(coords)} patches  (patch_size={patch_size}, mag={mag}x)")
    return coords, patch_size, mag


# ── Pass 2: token-grid targets ────────────────────────────────────────────────

def load_spot_markers(spot: str, markers: list[str]) -> np.ndarray:
    """Load all IF marker images for one spot → (C, H, W) float32."""
    imgs = []
    shape = None
    for marker in markers:
        path = CROPS_DIR / f"{spot}_{marker}.jpg"
        if path.exists():
            img = np.array(Image.open(path)).astype(np.float32)
            shape = img.shape[:2]
            imgs.append(img)
        else:
            print(f"    [{marker}] missing — filling zeros")
            imgs.append(None)

    if shape is None:
        raise RuntimeError(f"No IF marker images found for spot {spot}")

    channels = [
        img if img is not None else np.zeros(shape, dtype=np.float32)
        for img in imgs
    ]
    return np.stack(channels, axis=0)   # (C, H, W)


def compute_token_grid_targets(
    coords: np.ndarray,
    proteins: np.ndarray,
    patch_size_level0: int,
    global_p99s: np.ndarray,
    token_grid: int = TOKEN_GRID,
) -> tuple[np.ndarray, np.ndarray]:
    """
    coords            — (N, 2) patch top-left corners in image pixel space
    proteins          — (C, H, W) raw float32 IF intensity
    patch_size_level0 — patch footprint in pixels (same resolution as coords)
    global_p99s       — (C,) dataset-wide p99 per channel

    Returns (sorted_coords, targets) with targets shaped (N, C, token_grid, token_grid).
    Normalisation: log1p(x / p99) clipped to [0, 1].
    """
    C, H, W  = proteins.shape
    N        = len(coords)
    pw       = patch_size_level0
    token_px = 224 // token_grid   # = 14 for token_grid=16

    p99s_arr = np.maximum(global_p99s, 1.0).astype(np.float32)

    sort_idx      = np.lexsort((coords[:, 0], coords[:, 1]))
    sorted_coords = coords[sort_idx]

    valid_coords  = []
    valid_targets = []

    for i, (px, py) in enumerate(sorted_coords):
        if i % 200 == 0:
            print(f"    [{i}/{N}] computing token targets…", flush=True)
        x0, y0 = int(px), int(py)
        x1, y1 = x0 + pw, y0 + pw
        if x1 > W or y1 > H:
            continue

        region = proteins[:, y0:y1, x0:x1]   # (C, pw, pw)

        resized = cv2.resize(
            region.transpose(1, 2, 0), (224, 224),
            interpolation=cv2.INTER_LINEAR,
        )   # (224, 224, C)

        normed = np.clip(
            np.log1p(resized / p99s_arr[np.newaxis, np.newaxis, :]),
            0.0, 1.0,
        )   # (224, 224, C)

        target = (
            normed
            .reshape(token_grid, token_px, token_grid, token_px, C)
            .mean(axis=(1, 3))
            .transpose(2, 0, 1)
        )   # (C, token_grid, token_grid)

        valid_coords.append([x0, y0])
        valid_targets.append(target)

    if not valid_coords:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, C, token_grid, token_grid), dtype=np.float32)

    return np.array(valid_coords, dtype=np.int64), np.stack(valid_targets, axis=0)


# ── HDF5 save ─────────────────────────────────────────────────────────────────

def save_dataset(
    out_path: Path,
    all_coords: list[np.ndarray],
    all_targets: list[np.ndarray],
    all_spot_ids: list[str],
    global_p99s: np.ndarray,
    mag: float,
    patch_size: int,
    patch_size_level0: int,
    mpp: float,
    markers: list[str],
    token_grid: int = TOKEN_GRID,
):
    coords   = np.concatenate(all_coords,  axis=0)
    targets  = np.concatenate(all_targets, axis=0)
    spot_ids = np.array(all_spot_ids, dtype=h5py.string_dtype())

    N, C, G, _ = targets.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("coords",   data=coords,      compression="gzip")
        f.create_dataset("targets",  data=targets,     compression="gzip",
                         chunks=(min(256, N), C, G, G))
        f.create_dataset("spot_ids", data=spot_ids,    compression="gzip")
        f.create_dataset("p99s",     data=global_p99s, compression="gzip")

        f.attrs["dataset"]           = "nsclc_charite"
        f.attrs["marker_names"]      = markers
        f.attrs["patch_size"]        = patch_size
        f.attrs["patch_size_level0"] = patch_size_level0
        f.attrs["token_grid"]        = token_grid
        f.attrs["mpp"]               = mpp
        f.attrs["magnification"]     = mag
        f.attrs["n_patches"]         = N
        f.attrs["normalisation"]     = "log1p(x/global_p99) clip[0,1]"

    mb = out_path.stat().st_size / 1e6
    print(f"\nSaved → {out_path}  ({mb:.2f} MB)")
    print(f"  /coords   {coords.shape}")
    print(f"  /targets  {targets.shape}  mean={targets.mean():.4f}")
    print(f"  /p99s     {global_p99s.shape}  (global per-channel)")
    print(f"  patch_size={patch_size}px @ {mag}x  |  level-0={patch_size_level0}px @ {mpp}µm/px")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mpp",          type=float, default=0.25,
                        help="Scanner resolution in µm/px (Zeiss AxioScan Z1 20x = 0.25)")
    parser.add_argument("--mag",          type=float, default=20.0,
                        help="Target magnification for TRIDENT (default 20x)")
    parser.add_argument("--patch_size",   type=int,   default=224)
    parser.add_argument("--overlap",      type=int,   default=0)
    parser.add_argument("--min_tissue",   type=float, default=0.1)
    parser.add_argument("--segmenter",    type=str,   default="hest",
                        choices=["hest", "grandqc", "otsu"])
    parser.add_argument("--seg_conf",     type=float, default=0.5)
    parser.add_argument("--gpu",          type=int,   default=0)
    parser.add_argument("--output",       type=str,   default=str(OUTPUT_PATH))
    parser.add_argument("--job_dir",      type=str,   default=str(JOB_DIR))
    parser.add_argument("--skip_trident", action="store_true",
                        help="Re-use existing TRIDENT coords without re-running")
    parser.add_argument("--spots",        type=str,   default=None,
                        help="Comma-separated subset of spot UUIDs to process")
    args = parser.parse_args()

    target_mpp        = 10.0 / args.mag
    patch_size_level0 = round(args.patch_size * target_mpp / args.mpp)

    print("=" * 60)
    print("  NSCLC Charité patch dataset builder")
    print(f"  mpp={args.mpp}µm/px  mag={args.mag}x  patch_size={args.patch_size}px")
    print(f"  patch_size_level0={patch_size_level0}px")
    print("=" * 60)

    spots = discover_spots()
    if args.spots:
        requested = set(args.spots.split(","))
        spots = [s for s in spots if s in requested]
        print(f"Filtered to {len(spots)} requested spots")

    # Pass 1: global p99 per channel
    print("\n" + "=" * 60)
    print("  Pass 1: computing global p99s")
    print("=" * 60)
    global_p99s = compute_global_p99s(spots, MARKERS)

    # Pass 2: TRIDENT + token targets
    print("\n" + "=" * 60)
    print("  Pass 2: extracting patches")
    print("=" * 60)

    all_coords   = []
    all_targets  = []
    all_spot_ids = []

    for si, spot in enumerate(spots):
        he_path = CROPS_DIR / f"{spot}_he.jpg"
        print(f"\n[{si+1}/{len(spots)}] {spot}")

        spot_job_dir = Path(args.job_dir) / spot

        if args.skip_trident:
            h5_files = list(spot_job_dir.rglob("*_patches.h5"))
            if not h5_files:
                print(f"  No cached TRIDENT coords — skipping")
                continue
            coords_h5 = h5_files[0]
        else:
            try:
                coords_h5 = run_trident(
                    he_path, spot_job_dir, args.mpp, args.mag,
                    args.patch_size, args.overlap, args.min_tissue,
                    args.segmenter, args.seg_conf, args.gpu,
                )
            except Exception as e:
                print(f"  TRIDENT failed: {e} — skipping")
                continue

        coords, _, _ = load_trident_coords(coords_h5)
        if len(coords) == 0:
            print(f"  No tissue patches — skipping")
            continue

        proteins = load_spot_markers(spot, MARKERS)
        sorted_coords, targets = compute_token_grid_targets(
            coords, proteins, patch_size_level0, global_p99s,
        )

        all_coords.append(sorted_coords)
        all_targets.append(targets)
        all_spot_ids.extend([spot] * len(sorted_coords))

        print(f"  → {len(sorted_coords)} patches  targets mean={targets.mean():.4f}")

    if not all_coords:
        print("No patches extracted.")
        return

    save_dataset(
        Path(args.output), all_coords, all_targets, all_spot_ids,
        global_p99s, args.mag, args.patch_size, patch_size_level0,
        args.mpp, MARKERS,
    )
    print(f"\nTotal patches: {sum(len(c) for c in all_coords)}")


if __name__ == "__main__":
    main()