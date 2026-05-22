import sys
import csv
import subprocess
import argparse
import numpy as np
import cv2
import h5py
import imageio.v3 as iio
from pathlib import Path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

DATASET        = "immunoatlas_NOLN210920"
ROOT           = Path(f"/mnt/ssd1/virtual_proteomics/data/{DATASET}")
MANIFEST       = ROOT / "manifest.tsv"
CORE_PNG_DIR   = Path("immunoatlas_70_png")   # 70 per-core H&E PNGs at HE_MPP
TRIDENT_SCRIPT = Path("TRIDENT/run_batch_of_slides.py")
OUTPUT_PATH    = Path(f"{DATASET}_patch_dataset.h5")
JOB_DIR        = Path(f"{DATASET}_trident_output")

WEBP_MPP = 0.377   # µm/px of the WebP marker images (confirmed)
HE_MPP   = WEBP_MPP * 2   # 0.754 µm/px — H&E core PNGs are half the WebP resolution
TOKEN_GRID = 16            # UNI2 token grid (224/14 = 16 tokens/side)

EXCLUDE_CHANNELS = {"DRAQ5", "composite"}

IMMUNOATLAS_PROTEIN_COLS = [
    "Hoechst",
    "CD164", "FOXP3", "GATA3", "MUC-1", "p53", "Vimentin", "T-bet", "Cytokeratin",
    "PD-L1", "Ki-67", "CD15", "CD30", "CD2", "GranzymeB", "CD5", "MMP-9", "CD4",
    "LAG-3", "CD25", "CD56", "CD20", "PD-1", "CD11c", "CD162", "CD16", "CD11b",
    "CD194", "IDO-1", "EGFR", "VISTA", "HLA-DR", "ICOS", "BCL-2", "CD3", "CD69",
    "CD8", "CD7", "CD45RA", "CD45", "CD1a", "CD57", "B-catenin", "CD45RO", "CD71",
    "CD34", "CD68", "CD38", "Collagen-IV", "CD31", "Podoplanin", "CD138", "CD163",
    "Mast-cell-tryptase", "MMP-12",
]


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------

def parse_manifest():
    core_channels = {}
    with open(MANIFEST) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            core  = row["core_name"]
            cidx  = row["channel_index"]
            cname = row["channel_name"]
            fpath = ROOT / row["filename"]
            if cname in EXCLUDE_CHANNELS or cidx == "-":
                continue
            if core not in core_channels:
                core_channels[core] = {}
            core_channels[core][cname] = fpath
    return core_channels


def make_wsi_csv(core_png, job_dir):
    csv_path = Path(job_dir) / "wsi_list.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wsi", "mpp"])
        writer.writerow([core_png.name, HE_MPP])
    print(f"[TRIDENT] WSI: {core_png.name}  (mpp={HE_MPP})")
    return csv_path


def run_trident(core_png, job_dir, mag, patch_size, overlap,
                min_tissue_proportion, segmenter, seg_conf_thresh, gpu):
    Path(job_dir).mkdir(parents=True, exist_ok=True)
    wsi_csv = make_wsi_csv(core_png, job_dir)

    base_cmd = [
        sys.executable, str(TRIDENT_SCRIPT),
        "--wsi_dir",    str(core_png.parent),
        "--job_dir",    str(job_dir),
        "--gpu",        str(gpu),
        "--segmenter",  segmenter,
        "--seg_conf_thresh", str(seg_conf_thresh),
        "--mag",        str(mag),
        "--patch_size", str(patch_size),
        "--overlap",    str(overlap),
        "--min_tissue_proportion", str(min_tissue_proportion),
        "--custom_list_of_wsis", str(wsi_csv),
        "--reader_type", "image",
    ]

    subprocess.run(base_cmd + ["--task", "seg"],    check=True)
    subprocess.run(base_cmd + ["--task", "coords"], check=True)

    h5_files = list(Path(job_dir).rglob("*_patches.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No coords HDF5 found under {job_dir}")
    return h5_files[0]


def load_trident_coords(coords_h5_path):
    with h5py.File(coords_h5_path, "r") as f:
        key        = "coords" if "coords" in f else list(f.keys())[0]
        coords     = f[key][:]
        patch_size = int(f[key].attrs.get("patch_size", 224))
        mag        = float(f[key].attrs.get("target_magnification", 40.0))
    print(f"  {len(coords)} patches  (patch_size={patch_size}, mag={mag}x)")
    return coords, patch_size, mag


def compute_global_p95s(core_names, core_channels, protein_cols):
    """Pool non-zero pixels across all TMA cores to compute one p95 per channel."""
    pixel_pools = [[] for _ in protein_cols]

    for ci, core_name in enumerate(core_names):
        print(f"  [p95 pass {ci+1}/{len(core_names)}] {core_name}", flush=True)
        ch_map = core_channels[core_name]
        for mi, cname in enumerate(protein_cols):
            path = ch_map.get(cname)
            if path and path.exists():
                img = iio.imread(str(path)).astype(np.float32)
                if img.ndim == 3:
                    img = img[:, :, 0]
                nz = img[img > 0]
                if len(nz):
                    pixel_pools[mi].append(nz)

    global_p95s = np.ones(len(protein_cols), dtype=np.float32)
    print("\nGlobal p95s:")
    for mi, cname in enumerate(protein_cols):
        if pixel_pools[mi]:
            all_vals = np.concatenate(pixel_pools[mi])
            global_p95s[mi] = float(np.percentile(all_vals, 95))
        print(f"  {cname}: {global_p95s[mi]:.1f}  ({sum(len(a) for a in pixel_pools[mi]):,} px)")

    return global_p95s


def load_core_proteins(core_name, core_channels, protein_cols):
    """Load raw (un-normalised) WebP marker images for one core."""
    ch_map = core_channels[core_name]
    channels  = []
    img_shape = None

    for i, cname in enumerate(protein_cols):
        path = ch_map.get(cname)
        if path and path.exists():
            img = iio.imread(str(path)).astype(np.float32)
            if img.ndim == 3:
                img = img[:, :, 0]   # WebP RGB-encoded grayscale: R=B=true value, G has ±1 rounding artifact
            img_shape = img.shape
            channels.append(img)
        else:
            print(f"    [{cname}] missing — filling zeros")
            channels.append(None)

    for i, ch in enumerate(channels):
        if ch is None:
            channels[i] = np.zeros(img_shape or (1, 1), dtype=np.float32)

    proteins = np.stack(channels, axis=0)   # (C, H, W)  raw counts
    print(f"  Protein array: {proteins.shape}")
    return proteins


WEBP_SCALE = 2  # protein webps are 2x the H&E PNG resolution


def compute_token_grid_targets(patch_coords, proteins, patch_size_level0,
                               global_p95s, token_grid=TOKEN_GRID):
    """
    patch_coords      — (N, 2) in H&E core PNG pixel space (TRIDENT output)
    proteins          — (C, H, W) raw float32 in WebP pixel space
    patch_size_level0 — patch size in H&E core PNG pixels
    global_p95s       — (C,) dataset-wide p95 per channel

    Returns (sorted_coords, targets) shaped (N, C, token_grid, token_grid).

    Pipeline per patch:
      1. Convert H&E coord → WebP coord (×WEBP_SCALE)
      2. Extract raw patch region
      3. cv2.resize to exactly 224×224
      4. Normalise: log1p(x / global_p95) clipped to [0, 1]
      5. Reshape → block-mean → (C, token_grid, token_grid)
    """
    C, H, W   = proteins.shape
    N         = len(patch_coords)
    pw        = patch_size_level0 * WEBP_SCALE   # patch footprint in WebP pixels
    token_px  = 224 // token_grid                # = 14 for TOKEN_GRID=16

    p95s_arr = np.maximum(global_p95s, 1.0).astype(np.float32)   # (C,)

    sort_idx      = np.lexsort((patch_coords[:, 0], patch_coords[:, 1]))
    sorted_coords = patch_coords[sort_idx]

    valid_coords  = []
    valid_targets = []

    for i, (px, py) in enumerate(sorted_coords):
        if i % 200 == 0:
            print(f"    [{i}/{N}] computing token targets…", flush=True)
        x0 = int(px * WEBP_SCALE)
        y0 = int(py * WEBP_SCALE)
        if x0 + pw > W or y0 + pw > H:
            continue   # partial patch — full square doesn't fit
        x1 = x0 + pw
        y1 = y0 + pw

        region = proteins[:, y0:y1, x0:x1]   # (C, H_p, W_p)  raw

        # resize (C, H_p, W_p) → (224, 224, C)
        resized = cv2.resize(
            region.transpose(1, 2, 0), (224, 224),
            interpolation=cv2.INTER_LINEAR,
        )  # (224, 224, C)

        # normalise with global p95s
        normed = np.clip(np.log1p(resized / p95s_arr[np.newaxis, np.newaxis, :]),
                         0.0, 1.0)   # (224, 224, C)

        target = (
            normed
            .reshape(token_grid, token_px, token_grid, token_px, C)
            .mean(axis=(1, 3))
            .transpose(2, 0, 1)
        )  # (C, token_grid, token_grid)

        valid_coords.append([int(px), int(py)])
        valid_targets.append(target)

    if not valid_coords:
        return np.empty((0, 2), dtype=np.int64), np.empty((0, C, token_grid, token_grid), dtype=np.float32)

    return np.array(valid_coords, dtype=np.int64), np.stack(valid_targets, axis=0)


def save_dataset(out_path, all_coords, all_targets, all_core_ids,
                 global_p95s, mag, patch_size, patch_size_level0, protein_cols,
                 token_grid=TOKEN_GRID):
    coords   = np.concatenate(all_coords,  axis=0)
    targets  = np.concatenate(all_targets, axis=0)
    core_ids = np.array(all_core_ids, dtype=h5py.string_dtype())

    N, C, G, _ = targets.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("coords",   data=coords,      compression="gzip")
        f.create_dataset("targets",  data=targets,     compression="gzip",
                         chunks=(min(256, N), C, G, G))
        f.create_dataset("core_ids", data=core_ids,    compression="gzip")
        f.create_dataset("p95s",     data=global_p95s, compression="gzip")

        f.attrs["dataset"]           = DATASET
        f.attrs["marker_names"]      = protein_cols
        f.attrs["patch_size"]        = patch_size
        f.attrs["patch_size_level0"] = patch_size_level0
        f.attrs["token_grid"]        = token_grid
        f.attrs["webp_scale"]        = WEBP_SCALE
        f.attrs["webp_mpp"]          = WEBP_MPP
        f.attrs["he_mpp"]            = HE_MPP
        f.attrs["magnification"]     = mag
        f.attrs["n_patches"]         = len(coords)
        f.attrs["normalisation"]     = "log1p(x/global_p95) clip[0,1]"

    mb = out_path.stat().st_size / 1e6
    print(f"\nSaved → {out_path}  ({mb:.2f} MB)")
    print(f"  /coords   {coords.shape}")
    print(f"  /targets  {targets.shape}  mean={targets.mean():.4f}")
    print(f"  /p95s     {global_p95s.shape}  (global per-channel)")
    print(f"  patch_size={patch_size}px @ {mag}x  |  level-0={patch_size_level0}px (HE@{HE_MPP}µm)  |  webp_crop={patch_size_level0*WEBP_SCALE}px (IF@{WEBP_MPP}µm)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_patch_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_dir",    type=str,   default=str(JOB_DIR))
    parser.add_argument("--mag",        type=float, default=20)
    parser.add_argument("--patch_size", type=int,   default=224)
    parser.add_argument("--overlap",    type=int,   default=0)
    parser.add_argument("--min_tissue_proportion", type=float, default=0.1)
    parser.add_argument("--segmenter",  type=str,   default="hest",
                        choices=["hest", "grandqc", "otsu"])
    parser.add_argument("--seg_conf_thresh", type=float, default=0.5)
    parser.add_argument("--gpu",        type=int,   default=0)
    parser.add_argument("--output",     type=str,   default=str(OUTPUT_PATH))
    parser.add_argument("--channels",   type=str,   default=None,
                        help="Comma-separated subset, e.g. 'CD3,CD8'")
    args = parser.parse_args()

    if args.channels:
        protein_cols = [c.strip() for c in args.channels.split(",")]
    else:
        protein_cols = IMMUNOATLAS_PROTEIN_COLS

    # MPP-based patch size computation (mirrors singular genomics)
    target_mpp        = 10.0 / args.mag
    patch_size_level0 = round(args.patch_size * target_mpp / HE_MPP)

    print("=" * 60)
    print(f"  TRIDENT patch dataset builder -- {DATASET}")
    print(f"  mag={args.mag}x  patch_size={args.patch_size}px  HE_MPP={HE_MPP}µm/px")
    print(f"  patch_size_level0={patch_size_level0}px (HE)  webp_crop={args.patch_size * WEBP_SCALE}px (WEBP_SCALE={WEBP_SCALE})")
    print(f"  channels ({len(protein_cols)}): {protein_cols}")
    print("=" * 60)

    core_channels = parse_manifest()
    core_names    = sorted(core_channels.keys())

    # Pass 1: compute dataset-wide p95 per channel
    print("\n" + "=" * 60)
    print("  Pass 1: computing global p95s across all cores")
    print("=" * 60)
    global_p95s = compute_global_p95s(core_names, core_channels, protein_cols)

    # Pass 2: extract patches and normalise with global p95s
    print("\n" + "=" * 60)
    print("  Pass 2: extracting patches")
    print("=" * 60)

    all_coords   = []
    all_targets  = []
    all_core_ids = []

    for core_idx, core_name in enumerate(core_names):
        core_png = CORE_PNG_DIR / f"{core_name}.png"
        if not core_png.exists():
            print(f"  [{core_name}] PNG not found -- skipping")
            continue

        print(f"\n[{core_idx+1}/{len(core_names)}] {core_name}")
        core_job_dir = Path(args.job_dir) / core_name

        try:
            coords_h5 = run_trident(
                core_png, core_job_dir, args.mag, args.patch_size,
                args.overlap, args.min_tissue_proportion,
                args.segmenter, args.seg_conf_thresh, args.gpu,
            )
        except Exception as e:
            print(f"  TRIDENT failed: {e} -- skipping")
            continue

        coords, _, _ = load_trident_coords(coords_h5)
        proteins = load_core_proteins(core_name, core_channels, protein_cols)
        sorted_coords, targets = compute_token_grid_targets(
            coords, proteins, patch_size_level0, global_p95s, token_grid=TOKEN_GRID,
        )

        all_coords.append(sorted_coords)
        all_targets.append(targets)
        all_core_ids.extend([core_name] * len(sorted_coords))

    if not all_coords:
        print("No patches extracted.")
        return

    save_dataset(
        Path(args.output), all_coords, all_targets, all_core_ids, global_p95s,
        args.mag, args.patch_size, patch_size_level0, protein_cols,
        token_grid=TOKEN_GRID,
    )
    print(f"\nTotal patches: {sum(len(c) for c in all_coords)}")


if __name__ == "__main__":
    build_patch_dataset()

# Subset for SG-compatible panel:
# 'aSMA,CD3,CD4,CD8,CD11c,CD20,CD31,CD45RA,CD68,HLA-DR,Ki-67,Cytokeratin,PD-1,PD-L1'


"""


0. Hoechst
  1. CD164
  2. FOXP3
  3. GATA3
  4. MUC-1
  5. p53
  6. Vimentin
  7. T-bet
  8. Cytokeratin
  9. PD-L1
  10. Ki-67
  11. CD15
  12. CD30
  13. CD2
  14. GranzymeB
  15. CD5
  16. MMP-9
  17. CD4
  18. LAG-3
  19. CD25
  20. CD56
  21. CD20
  22. PD-1
  23. CD11c
  24. CD162
  25. CD16
  26. CD11b
  27. CD194
  28. IDO-1
  29. EGFR
  30. VISTA
  31. HLA-DR
  32. ICOS
  33. BCL-2
  34. CD3
  35. CD69
  36. CD8
  37. CD7
  38. CD45RA
  39. CD45
  40. CD1a
  41. CD57
  42. B-catenin
  43. CD45RO
  44. CD71
  45. CD34
  46. CD68
  47. CD38
  48. Collagen-IV
  49. CD31
  50. Podoplanin
  51. CD138
  52. CD163
  53. Mast-cell-tryptase
  54. MMP-12

"""
