import sys
import csv
import subprocess
import argparse
import numpy as np
import cv2
import h5py
import imageio.v3 as iio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


from PIL import Image
Image.MAX_IMAGE_PIXELS = None


ALL_DISEASES = [
    "breast_cancer", "breast_normal",
    "kidney_cancer", "kidney_normal",
    "lung_cancer",   "lung_normal",
    "tonsil_rep1",
]

SG_DATA_ROOT    = Path("/mnt/ssd1/virtual_proteomics/data/singular_genomics")
TRIDENT_SCRIPT  = Path("TRIDENT/run_batch_of_slides.py")
MMP = 0.3125  # µm/px

# Full union of all markers across all tissue types.
# Markers absent in a given tissue are filled with zeros (JP2 missing → zeros).
SINGULAR_GENOMICS_BIOMARKERS = ['aSMA', 'ATPase', 'CD3', 'CD4', 'CD8', 'CD11c', 'CD20', 'CD31', 'CD45', 'CD45RA', 'CD68', 'FOXP3', 'HLA-DR', 'Isotype', 'KI67', 'PanCK', 'PD1', 'PDL1']

def make_wsi_csv(wsi_dir, mpp, job_dir):
    """
    Writes a CSV with a single wsi,mpp row for the HE slide (*_HE.ome.tiff).
    Required when slides lack embedded MPP metadata (TRIDENT --custom_list_of_wsis).
    """
    wsi_dir = Path(wsi_dir)
    he_slides = sorted(wsi_dir.glob("*_HE.ome.tiff"))
    if not he_slides:
        raise FileNotFoundError(f"No *_HE.ome.tiff found in {wsi_dir}")
    he_slide = he_slides[0]
    csv_path = Path(job_dir) / "wsi_list.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wsi", "mpp"])
        writer.writerow([he_slide.name, mpp])
    print(f"[TRIDENT] Wrote WSI list: {he_slide.name} (mpp={mpp}) → {csv_path}")
    return csv_path, he_slide.stem.replace(".ome", "")


def run_trident(he_dir, job_dir, mag, patch_size, overlap,
                min_tissue_proportion, segmenter, seg_conf_thresh, gpu):
    """
    Calls run_batch_of_slides.py for seg then coords tasks.
    Returns path to the coords HDF5.
    """
    Path(job_dir).mkdir(parents=True, exist_ok=True)

    base_cmd = [
        sys.executable, str(TRIDENT_SCRIPT),
        "--wsi_dir",  str(he_dir),
        "--job_dir",  str(job_dir),
        "--gpu",      str(gpu),
        "--segmenter", segmenter,
        "--seg_conf_thresh", str(seg_conf_thresh),
        "--mag",        str(mag),
        "--patch_size", str(patch_size),
        "--overlap",    str(overlap),
        "--min_tissue_proportion", str(min_tissue_proportion),
    ]

    wsi_csv, slide_stem = make_wsi_csv(he_dir, MMP, job_dir)
    base_cmd += ["--custom_list_of_wsis", str(wsi_csv)]

    print("\n[TRIDENT] Running segmentation...")
    subprocess.run(base_cmd + ["--task", "seg"], check=True)

    print("\n[TRIDENT] Extracting patch coordinates...")
    subprocess.run(base_cmd + ["--task", "coords"], check=True)
    h5_files = list(Path(job_dir).rglob("*_patches.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No coords HDF5 found under {job_dir}")
    print(f"[TRIDENT] Coords: {h5_files[0]}")
    return h5_files[0]

def load_trident_coords(coords_h5_path):
    with h5py.File(coords_h5_path, "r") as f:
        print(f"  H5 keys: {list(f.keys())}")
        key        = "coords" if "coords" in f else list(f.keys())[0]
        coords     = f[key][:]
        patch_size = int(f[key].attrs.get("patch_size", 224))
        mag        = float(f[key].attrs.get("target_magnification", 40.0))
    print(f"Loaded {len(coords)} patch coords (patch_size={patch_size}, mag={mag}x)")
    return coords, patch_size, mag


def load_protein_channels(protein_dir, max_workers=32):
    """
    Load all protein JP2s into a (C, H, W) float32 array (raw, unnormalised).
    Computes per-channel foreground p99 for use during target normalisation.
    Returns (proteins_raw, valid_mask, p99s).
    """
    def load_one(col):
        try:
            return col, iio.imread(str(protein_dir / f"{col}.jp2")).astype(np.float32)
        except Exception as e:
            print(f"  {col}  [missing] {e} — filling with zeros")
            return col, None

    print(f"  Loading {len(SINGULAR_GENOMICS_BIOMARKERS)} JP2 channels ({max_workers} threads)...")
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        results = dict(ex.map(load_one, SINGULAR_GENOMICS_BIOMARKERS))

    img_shape = next(v for v in results.values() if v is not None).shape
    channels   = []
    valid_mask = []
    p99s       = []
    for i, col in enumerate(SINGULAR_GENOMICS_BIOMARKERS):
        img     = results[col]
        present = img is not None
        if present:
            fg  = img[img > 0]
            p99 = float(np.percentile(fg, 99)) if len(fg) > 0 else 1.0
            channels.append(img)
        else:
            p99 = 1.0
            channels.append(np.zeros(img_shape, dtype=np.float32))
        valid_mask.append(present)
        p99s.append(p99)
        print(f"  [{i+1:2d}/{len(SINGULAR_GENOMICS_BIOMARKERS)}] {col}  p99={p99:.1f}{'  [MISSING]' if not present else ''}")

    proteins   = np.stack(channels, axis=0)          # (C, H, W)
    valid_mask = np.array(valid_mask, dtype=bool)
    p99s       = np.array(p99s, dtype=np.float32)
    print(f"Protein array: {proteins.shape}  dtype={proteins.dtype}  present: {valid_mask.sum()}/{len(valid_mask)}")
    return proteins, valid_mask, p99s


def compute_token_grid_targets(
    proteins_raw: np.ndarray,
    p99s: np.ndarray,
    coords: np.ndarray,
    patch_size_level0: int,
    token_grid: int = 16,
) -> np.ndarray:
    """
    Compute (token_grid, token_grid) mean-expression grid for every patch.
    No AF correction — SG channels are already per-biomarker JP2s.

    Mirrors the ORION pipeline:
      level-0 crop → log1p/p99 normalise → cv2.resize(224,224) →
      reshape (G, token_px, G, token_px, C).mean(axes 1,3) → (C, G, G)

    With patch_size=224 and token_grid=16: token_px = 224//16 = 14 (UNI2 aligned).

    Returns (N, C, token_grid, token_grid) float32.
    """
    N = len(coords)
    C, H_arr, W_arr = proteins_raw.shape
    token_px = 224 // token_grid          # = 14 for G=16
    targets  = np.zeros((N, C, token_grid, token_grid), dtype=np.float32)
    p99s_arr = p99s[:, None, None]        # (C, 1, 1) for broadcasting

    for i, (px, py) in enumerate(coords):
        if i % 200 == 0:
            print(f"    [{i}/{N}] computing token targets…", flush=True)
        px, py = int(px), int(py)
        x1 = min(px + patch_size_level0, W_arr)
        y1 = min(py + patch_size_level0, H_arr)

        sigs = proteins_raw[:, py:y1, px:x1]  # (C, H_p, W_p)

        if sigs.shape[1] < token_grid or sigs.shape[2] < token_grid:
            continue

        normed = np.clip(np.log1p(sigs / p99s_arr), 0.0, 1.0)  # (C, H_p, W_p)

        resized = cv2.resize(
            normed.transpose(1, 2, 0), (224, 224), interpolation=cv2.INTER_LINEAR
        )  # (224, 224, C)

        targets[i] = (
            resized
            .reshape(token_grid, token_px, token_grid, token_px, C)
            .mean(axis=(1, 3))
            .transpose(2, 0, 1)
        )  # (C, G, G)

    return targets


def save_dataset(out_path, coords, targets, valid_mask, p99s,
                 mag, patch_size, patch_size_level0, token_grid=16):
    """
    HDF5 layout:
        /coords        (N, 2)       int64   — (x, y) top-left pixel in H&E level-0 space
        /targets       (N, C, G, G) float32 — mean normalised expression per token cell
        /p99s          (C,)         float32 — per-channel foreground 99th percentile
        /valid_markers (C,)         bool    — True where the JP2 was actually present
        attrs: marker_names, patch_size, patch_size_level0, token_grid, magnification, mpp, n_patches
    """
    N, C, G, _ = targets.shape
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("coords",        data=coords,   compression="gzip")
        f.create_dataset("targets",       data=targets,  compression="gzip",
                         chunks=(min(256, N), C, G, G))
        f.create_dataset("p99s",          data=p99s)
        f.create_dataset("valid_markers", data=valid_mask)
        f.attrs["marker_names"]      = SINGULAR_GENOMICS_BIOMARKERS
        f.attrs["patch_size"]        = patch_size
        f.attrs["patch_size_level0"] = patch_size_level0
        f.attrs["token_grid"]        = token_grid
        f.attrs["magnification"]     = mag
        f.attrs["mpp"]               = MMP
        f.attrs["n_patches"]         = len(coords)
        f.attrs["normalisation"]     = "log1p(x/p99) -> clip[0,1]"

    mb = out_path.stat().st_size / 1e6
    print(f"\nSaved → {out_path}  ({mb:.2f} MB)")
    print(f"  /coords   {coords.shape}")
    print(f"  /targets  {targets.shape}  ({token_grid}×{token_grid} token grid, {C} proteins)")
    print(f"  patch_size={patch_size} @ {mag}x  (level-0 crop: {patch_size_level0}px, mpp={MMP}µm/px)")


def process_disease(disease: str, args) -> None:
    root        = SG_DATA_ROOT / disease
    he_dir      = root / "g4x_viewer"
    protein_dir = root / "protein"
    job_dir     = Path(f"singular_genomics/{disease}_trident_output")
    out_path    = Path(f"singular_genomics/{disease}_patch_dataset.h5")

    print("\n" + "=" * 60)
    print(f"  Disease: {disease}")
    print(f"  HE dir:  {he_dir}")
    print(f"  Output:  {out_path}")
    print("=" * 60)

    coords_h5 = run_trident(
        he_dir, str(job_dir), args.mag, args.patch_size,
        args.overlap, args.min_tissue_proportion,
        args.segmenter, args.seg_conf_thresh, args.gpu,
    )

    patch_coords, patch_size, mag = load_trident_coords(coords_h5)
    # Sort by (row, col) so consecutive patches share memory locality in proteins array
    patch_coords = patch_coords[np.lexsort((patch_coords[:, 0], patch_coords[:, 1]))]

    target_mpp        = 10.0 / mag
    patch_size_level0 = round(patch_size * target_mpp / MMP)
    print(f"\n  mpp={MMP}µm/px  target_mpp={target_mpp:.4f}µm/px  patch_size_level0={patch_size_level0}px")

    print("\nLoading protein channels (raw)...")
    proteins, valid_mask, p99s = load_protein_channels(protein_dir)

    print(f"\nComputing {args.token_grid}×{args.token_grid} token targets ({len(patch_coords)} patches)...")
    targets = compute_token_grid_targets(
        proteins, p99s, patch_coords, patch_size_level0, args.token_grid
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(out_path, patch_coords, targets, valid_mask, p99s,
                 mag, patch_size, patch_size_level0, args.token_grid)


def build_patch_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disease",    type=str,   default="breast_normal",
                        choices=ALL_DISEASES,
                        help=f"Disease to process. One of: {ALL_DISEASES}")
    parser.add_argument("--all",        action="store_true",
                        help=f"Process all diseases sequentially: {ALL_DISEASES}")
    parser.add_argument("--mag",        type=float, default=20)
    parser.add_argument("--patch_size", type=int,   default=224)
    parser.add_argument("--overlap",    type=int,   default=0)
    parser.add_argument("--min_tissue_proportion", type=float, default=0.25)
    parser.add_argument("--segmenter",  type=str,   default="hest",
                        choices=["hest", "grandqc", "otsu"])
    parser.add_argument("--seg_conf_thresh", type=float, default=0.5)
    parser.add_argument("--gpu",        type=int,   default=0)
    parser.add_argument("--token_grid", type=int,   default=16,
                        help="Spatial grid size (must match FM patch tokens: UNI2=16)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Singular Genomics patch dataset builder")
    print(f"  mag={args.mag}x  patch_size={args.patch_size}px  token_grid={args.token_grid}×{args.token_grid}")
    print(f"  normalisation: log1p(x/p99) → clip[0,1]")
    print("=" * 60)

    diseases = ALL_DISEASES if args.all else [args.disease]
    for disease in diseases:
        process_disease(disease, args)


if __name__ == "__main__":
    build_patch_dataset()