"""
Build ORION-CRC patch dataset for LDM training.

Pipeline (per slide)
--------------------
1. TRIDENT  : tissue segmentation + patch coordinate extraction on H&E
2. p99s     : per-channel 99th-percentile of AF-corrected foreground pixels
              (used for log1p normalisation in dataset_orion_ldm.__getitem__)
3. pixel_stds: per-channel pixel std of LDM-normalised values across tissue
              patches (used for 1/σ loss weights)
4. Save HDF5: minimal file consumed by OrionLDMDataset

HDF5 layout
-----------
  /coords      (N, 2) int64   — (x, y) top-left in shared H&E/IF level-0 space
  /p99s        (C,)   float32 — AF-corrected foreground p99 per channel
  /pixel_stds  (C,)   float32 — pixel std of LDM-normalised values per channel
  attrs: sample, marker_names, patch_size, patch_size_level0

H&E (*-registered.ome.tif) and IF (*-zlib.ome.tiff) share the same pixel
coordinate space (both MPP=0.325, pre-registered) — no Valis warp needed.
"""

import csv
import json
import subprocess
import argparse
import numpy as np
import h5py
import tifffile
import zarr
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ORION_DATA_DIR = Path("/mnt/ssd1/virtual_proteomics/data/ORION_CRC")
TRIDENT_SCRIPT = Path("TRIDENT/run_batch_of_slides.py")
LAMBDA_JSON    = Path("MIPHEI-ViT/preprocessings/mif_cleaning/lambda_settings/orion.json")
JOB_DIR        = Path("orion_trident_output")
OUTPUT_DIR     = Path("orion_crc_patch_dataset")

ALL_CRC_SAMPLES = (
    [f"CRC{i:02d}" for i in range(1, 33)] +
    ["CRC33_01", "CRC33_02"] +
    [f"CRC{i:02d}" for i in range(34, 41)]
)

ORION_AF_RAW_CH = 1   # ch1 = AF1 (autofluorescence 488 nm)
MPP_HE = 0.325        # µm/px at level-0 (same for both H&E and IF)

# IF channel table: (raw_ch_index, marker_name, orion_json_key)
# ch0=Hoechst, ch1=AF1, ch2=CD31, ch3=CD45, ch4=CD68, ch5=Blank,
# ch6=CD4, ch7=FOXP3, ch8=CD8a, ch9=CD45RO, ch10=CD20, ch11=PD-L1,
# ch12=CD3e, ch13=CD163, ch14=E-Cadherin, ch15=PD-1, ch16=Ki-67,
# ch17=Pan-CK, ch18=SMA
ORION_CRC_CHANNELS = [
    (0,  "Hoechst",    "0"),
    (2,  "CD31",       "2"),
    (3,  "CD45",       "3"),
    (4,  "CD68",       "4"),
    (6,  "CD4",        "6"),
    (7,  "FOXP3",      "7"),
    (8,  "CD8a",       "8"),
    (9,  "CD45RO",     "9"),
    (10, "CD20",       "10"),
    (11, "PD-L1",      "11"),
    (12, "CD3e",       "12"),
    (13, "CD163",      "13"),
    (14, "E-Cadherin", "14"),
    (16, "Ki-67",      "16"),
    (17, "Pan-CK",     "17"),
    (18, "SMA",        "18"),
]
ORION_PD1 = (15, "PD-1", "15")   # excluded by default (poor signal quality)


# ── Path helpers ──────────────────────────────────────────────────────────────

def crc_paths(sample: str) -> tuple[Path, Path]:
    """Return (he_path, if_path) for a CRC sample."""
    slide_dir  = ORION_DATA_DIR / sample
    he_matches = list(slide_dir.glob("*-registered.ome.tif"))
    if_matches = list(slide_dir.glob("*-zlib.ome.tiff"))
    if len(he_matches) != 1:
        raise FileNotFoundError(f"{sample}: expected 1 H&E, found {he_matches}")
    if len(if_matches) != 1:
        raise FileNotFoundError(f"{sample}: expected 1 IF, found {if_matches}")
    return he_matches[0], if_matches[0]


# ── Channel params ────────────────────────────────────────────────────────────

def load_lambda_settings(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def build_channel_params(
    channels: list, settings: dict
) -> list[tuple[int, str, float, float]]:
    """Return [(raw_ch, marker_name, lambda, bias), ...] for each channel."""
    return [
        (raw_ch, name, float(settings[key]["lambda"]), float(settings[key]["bias"]))
        for raw_ch, name, key in channels
    ]


# ── TRIDENT helpers ───────────────────────────────────────────────────────────

def make_wsi_csv(he_slide: Path, job_dir: Path) -> Path:
    csv_path = job_dir / "wsi_list.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wsi", "mpp"])
        writer.writerow([he_slide.name, MPP_HE])
    return csv_path


def run_trident(he_slide: Path, job_dir: Path, mag: float, patch_size: int,
                overlap: int, min_tissue: float,
                segmenter: str, seg_thresh: float, gpu: int) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    wsi_csv = make_wsi_csv(he_slide, job_dir)
    base_cmd = [
        "python", str(TRIDENT_SCRIPT),
        "--wsi_dir",   str(he_slide.parent),
        "--job_dir",   str(job_dir),
        "--gpu",       str(gpu),
        "--segmenter", segmenter,
        "--seg_conf_thresh", str(seg_thresh),
        "--mag",        str(mag),
        "--patch_size", str(patch_size),
        "--overlap",    str(overlap),
        "--min_tissue_proportion", str(min_tissue),
        "--wsi_ext",    ".tif",
        "--custom_list_of_wsis", str(wsi_csv),
    ]
    print("[TRIDENT] Segmenting…")
    subprocess.run(base_cmd + ["--task", "seg"], check=True)
    print("[TRIDENT] Extracting coords…")
    subprocess.run(base_cmd + ["--task", "coords"], check=True)
    h5_files = list(job_dir.rglob("*_patches.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No coords HDF5 found under {job_dir}")
    return h5_files[0]


def load_trident_coords(h5_path: Path) -> tuple[np.ndarray, int, float]:
    with h5py.File(h5_path, "r") as f:
        key        = "coords" if "coords" in f else list(f.keys())[0]
        coords     = f[key][:]
        patch_size = int(f[key].attrs.get("patch_size", 224))
        target_mag = float(f[key].attrs.get("target_magnification", 40.0))
    print(f"  {len(coords)} patches  patch_size={patch_size} @ {target_mag}x")
    return coords, patch_size, target_mag


# ── Zarr / patch I/O ──────────────────────────────────────────────────────────

def open_zarr_level0(path: Path):
    """
    Open OME-TIFF as a full-resolution zarr array.
    Returns (arr, c_ax, h_ax, w_ax).
    """
    tif   = tifffile.TiffFile(str(path))
    store = tif.aszarr()
    z     = zarr.open(store, mode="r")
    arr   = z["0"] if isinstance(z, zarr.hierarchy.Group) else z
    ndim  = len(arr.shape)

    if ndim == 3:
        if arr.shape[2] <= 4: 
            return arr, 2, 0, 1
        else:
            return arr, 0, 1, 2
    else:
        raise ValueError(f"Unexpected OME-TIFF shape {arr.shape}")


def read_patch(arr, c_ax, h_ax, w_ax,
               ch: int, px: int, py: int,
               patch_size: int, H: int, W: int) -> np.ndarray:
    """Read one (patch_size × patch_size) region for channel ch. Returns float32."""
    idx       = [0] * arr.ndim
    idx[c_ax] = ch
    idx[h_ax] = slice(int(py), min(int(py) + patch_size, H))
    idx[w_ax] = slice(int(px), min(int(px) + patch_size, W))
    return arr[tuple(idx)].astype(np.float32)


# ── Normalisation ─────────────────────────────────────────────────────────────

def af_subtract(signal: np.ndarray, af: np.ndarray,
                lam: float, bias: float) -> np.ndarray:
    """AF correction: out = max(signal − λ·AF + bias, 0)."""
    return np.maximum(signal - lam * af + bias, 0.0)


def normalize_patch(signal: np.ndarray, af: np.ndarray,
                    lam: float, bias: float,
                    p99_val: float) -> np.ndarray:
    """
    MIPHEI normalisation → float32 in [0, 1] 

      1. AF subtraction : max(signal − λ·AF + bias, 0)
      2. log1p / clip   : clip(log1p(x / p99_val), 0, 1)
    """
    x = af_subtract(signal, af, lam, bias)
    x = np.clip(np.log1p(x / p99_val), 0.0, 1.0)
    return x.astype(np.float32)



# ── p99 computation ───────────────────────────────────────────────────────────

def compute_slide_p99s(
    arr, c_ax, h_ax, w_ax,
    channel_params: list,
    bio_coords: np.ndarray,
    patch_size: int,
    af_raw_ch: int = ORION_AF_RAW_CH,
    max_patches: int = 2000,
) -> list[float]:
    """
    Compute per-channel p99 of AF-corrected foreground pixels for one slide.

    Samples up to max_patches tissue patches, builds a uint16 histogram of
    foreground (non-zero) pixels per channel, returns the 99th-percentile bin.
    """
    C = len(channel_params)
    H, W = arr.shape[h_ax], arr.shape[w_ax]
    N = len(bio_coords)

    if N > max_patches:
        rng    = np.random.default_rng(0)
        idx    = np.sort(rng.choice(N, max_patches, replace=False))
        coords = bio_coords[idx]
        print(f"    [p99] sampling {max_patches}/{N} patches", flush=True)
    else:
        coords = bio_coords

    hists = [np.zeros(65536, dtype=np.float64) for _ in range(C)]

    for px, py in coords:
        px, py = int(px), int(py)
        af = read_patch(arr, c_ax, h_ax, w_ax, af_raw_ch, px, py, patch_size, H, W)
        for ci, (raw_ch, _, lam, bias) in enumerate(channel_params):
            sig       = read_patch(arr, c_ax, h_ax, w_ax, raw_ch, px, py, patch_size, H, W)
            corrected = af_subtract(sig, af, lam, bias)
            u16       = np.uint16(np.minimum(corrected, 65535)).ravel()
            fg        = u16[u16 > 0]
            if len(fg):
                h, _ = np.histogram(fg, bins=65536, range=(0, 65536))
                hists[ci] += h.astype(np.float64)

    p99s = []
    for ci, (_, name, _, _) in enumerate(channel_params):
        total = hists[ci].sum()
        if total == 0:
            p99s.append(1.0)
            print(f"    {name:<14}  EMPTY → p99=1.0")
            continue
        cdf     = np.cumsum(hists[ci] / total)
        p99_bin = max(int(np.searchsorted(cdf, 0.99, side="right")), 1)
        p99s.append(float(p99_bin))
        print(f"    {name:<14}  p99={p99_bin:.1f}")

    return p99s


# ── Pixel std computation ─────────────────────────────────────────────────────

def compute_pixel_stds(
    arr, c_ax, h_ax, w_ax,
    channel_params: list[tuple[int, str, float, float]],
    bio_coords: np.ndarray,
    patch_size: int,
    p99s: list[float],
    af_raw_ch: int = ORION_AF_RAW_CH,
    max_patches: int = 2000,
) -> np.ndarray:
    """
    Compute per-channel pixel std of LDM-normalised values across tissue patches.
    Uses one-pass E[X²] − E[X]² with float64 accumulators.

    Returns (C,) float32 array of pixel standard deviations.

    For a marker where fraction f of pixels express (value → +1) and the rest
    are background (value → −1):
        σ = 2√(f(1−f))
    Dense markers (f≈0.30) → σ≈0.92; sparse markers (f≈0.01) → σ≈0.20.
    The resulting 1/σ weight ratio is ~4–5× — stable, no loss explosion.
    """
    C = len(channel_params)
    H, W = arr.shape[h_ax], arr.shape[w_ax]
    N = len(bio_coords)

    if N > max_patches:
        rng    = np.random.default_rng(0)
        idx    = np.sort(rng.choice(N, max_patches, replace=False))
        coords = bio_coords[idx]
        print(f"    [pixel_stds] sampling {max_patches}/{N} patches", flush=True)
    else:
        coords = bio_coords

    sum_x  = np.zeros(C, dtype=np.float64)
    sum_x2 = np.zeros(C, dtype=np.float64)
    n_pix  = 0

    for px, py in coords:
        px, py = int(px), int(py)
        af = read_patch(arr, c_ax, h_ax, w_ax, af_raw_ch, px, py, patch_size, H, W)
        for ci, (raw_ch, _, lam, bias) in enumerate(channel_params):
            sig    = read_patch(arr, c_ax, h_ax, w_ax, raw_ch, px, py, patch_size, H, W)
            normed = normalize_patch(sig, af, lam, bias, p99s[ci]).ravel()
            flat   = normed.astype(np.float64)
            sum_x[ci]  += flat.sum()
            sum_x2[ci] += (flat ** 2).sum()
        n_pix += af.size

    if n_pix == 0:
        return np.ones(C, dtype=np.float32)

    means = sum_x / n_pix
    stds  = np.sqrt(np.maximum(sum_x2 / n_pix - means ** 2, 0.0)).astype(np.float32)

    for (_, name, _, _), s in zip(channel_params, stds):
        print(f"    {name:<14}  σ={s:.4f}")

    return stds


def load_display_p99s(sample: str,
                      blend_channels: list = ORION_CRC_CHANNELS,
                      p99s_txt: Path = OUTPUT_DIR / 'p99s_slide.txt') -> dict:
    """
    Load slide-level p99s from the pre-computed p99s.txt file produced during
    training.  Returns {ch_idx: p99_value} for each channel in blend_channels.
    """
    # Parse the flat text file into {sample_name: {channel_name: p99}}
    all_p99s: dict[str, dict[str, float]] = {}
    current: str | None = None
    with open(p99s_txt) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if not line.startswith(" "):
                current = line.strip()
                all_p99s[current] = {}
            else:
                parts = line.split()
                if len(parts) == 2 and current is not None:
                    all_p99s[current][parts[0]] = float(parts[1])

    if sample not in all_p99s:
        raise KeyError(f"Sample '{sample}' not found in {p99s_txt}")

    sample_p99s = all_p99s[sample]
    p99s = []

    for ch_idx, name, _key in blend_channels:
        if name not in sample_p99s:
            raise KeyError(f"Channel '{name}' not found in {p99s_txt} for sample '{sample}'")
        p99s.append(sample_p99s[name])
        print(f"  [display p99] ch{ch_idx:02d}  {name:<10}  p99={p99s[-1]:.1f}")
    return p99s

# ── HDF5 save ─────────────────────────────────────────────────────────────────

def save_dataset(
    out_path: Path,
    coords: np.ndarray,
    p99s: list[float],
    pixel_stds: np.ndarray,
    marker_names: list[str],
    patch_size: int,
    patch_size_level0: int,
    sample: str,
) -> None:
    """
    Save the minimal HDF5 consumed by OrionLDMDataset.

    /coords      (N, 2) int64   — (x, y) top-left in shared H&E/IF level-0 space
    /p99s        (C,)   float32 — AF-corrected foreground p99, for normalize_patch
    /pixel_stds  (C,)   float32 — pixel std of LDM-normalised values, for weights
    attrs: sample, marker_names, patch_size, patch_size_level0
    """
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("coords",      data=coords,                  compression="gzip")
        f.create_dataset("p99s",        data=np.array(p99s,    dtype=np.float32))
        f.create_dataset("pixel_stds",  data=np.array(pixel_stds, dtype=np.float32))
        f.attrs["sample"]             = sample
        f.attrs["marker_names"]       = marker_names
        f.attrs["patch_size"]         = patch_size
        f.attrs["patch_size_level0"]  = patch_size_level0

    mb = out_path.stat().st_size / 1e6
    print(f"\n  Saved → {out_path}  ({mb:.1f} MB)")
    print(f"    /coords     {coords.shape}")
    print(f"    /p99s       {np.array(p99s)}")
    print(f"    /pixel_stds {pixel_stds}")


# ── Per-sample pipeline ───────────────────────────────────────────────────────

def process_sample(
    sample: str,
    args: argparse.Namespace,
    channel_params: list,
) -> None:
    print(f"\n{'='*60}\n  {sample}\n{'='*60}")

    he_path, if_path = crc_paths(sample)
    job_dir          = Path(args.job_dir) / sample
    out_path         = Path(args.output_dir) / f"{sample}_patch_dataset.h5"

    # 1. TRIDENT — tissue segmentation + patch coordinates on H&E
    if args.skip_trident:
        h5_files = list(job_dir.rglob("*_patches.h5"))
        if not h5_files:
            raise FileNotFoundError(f"--skip_trident: no coords h5 under {job_dir}")
        coords_h5 = h5_files[0]
        print(f"[TRIDENT] reusing {coords_h5}")
    else:
        coords_h5 = run_trident(
            he_path, job_dir,
            args.mag, args.patch_size, args.overlap, args.min_tissue,
            args.segmenter, args.seg_thresh, args.gpu,
        )

    coords, patch_size, target_mag = load_trident_coords(coords_h5)
    # Level-0 crop size: scale from target mag resolution to level-0 resolution
    patch_size_level0 = round(patch_size * (10.0 / target_mag) / MPP_HE)

    # 2. Open IF zarr (H&E and IF share the same coordinate space — no warp)
    arr, c_ax, h_ax, w_ax = open_zarr_level0(if_path)

    # 3. p99s — needed for normalize_patch in __getitem__

    print(f"\n  Computing p99s ({min(args.max_patches, len(coords))} patches)…")

    try:
        p99s = load_display_p99s(sample)
    except:
        p99s = compute_slide_p99s(
            arr, c_ax, h_ax, w_ax,
            channel_params, coords, patch_size_level0,
            af_raw_ch=ORION_AF_RAW_CH,
            max_patches=args.max_patches,
        )


    # 4. Pixel stds — used for 1/σ loss weights (must run after p99s)
    print(f"\n  Computing pixel stds ({min(args.max_patches, len(coords))} patches)…")
    pixel_stds = compute_pixel_stds(
        arr, c_ax, h_ax, w_ax,
        channel_params, coords, patch_size_level0,
        p99s=p99s,
        af_raw_ch=ORION_AF_RAW_CH,
        max_patches=args.max_patches,
    )

    # 5. Save HDF5
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    marker_names = [p[1] for p in channel_params]
    save_dataset(
        out_path, coords, p99s, pixel_stds,
        marker_names, patch_size, patch_size_level0, sample,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ORION-CRC patch dataset builder for LDM training"
    )
    parser.add_argument("--samples",      default="all",
                        help="'all' or comma-separated e.g. CRC01,CRC02")
    parser.add_argument("--include_pd1",  action="store_true",
                        help="Include PD-1 channel (excluded by default)")
    parser.add_argument("--skip_trident", action="store_true",
                        help="Reuse existing TRIDENT coords")
    parser.add_argument("--patch_size",   type=int,   default=224)
    parser.add_argument("--mag",          type=float, default=20)
    parser.add_argument("--overlap",      type=int,   default=0)
    parser.add_argument("--min_tissue",   type=float, default=0.0)
    parser.add_argument("--segmenter",    default="hest",
                        choices=["hest", "grandqc", "otsu"])
    parser.add_argument("--seg_thresh",   type=float, default=0.5)
    parser.add_argument("--gpu",          type=int,   default=1)
    parser.add_argument("--job_dir",      default=str(JOB_DIR))
    parser.add_argument("--output_dir",   default=str(OUTPUT_DIR))
    parser.add_argument("--lambda_json",  default=str(LAMBDA_JSON))
    parser.add_argument("--max_patches",  type=int,   default=2000,
                        help="Patches sampled for p99 and pixel_std estimation")
    args = parser.parse_args()

    channels = list(ORION_CRC_CHANNELS)
    if args.include_pd1:
        channels.append(ORION_PD1)

    settings       = load_lambda_settings(Path(args.lambda_json))
    channel_params = build_channel_params(channels, settings)

    samples = (ALL_CRC_SAMPLES if args.samples.lower() == "all"
               else [s.strip() for s in args.samples.split(",")])

    print(f"Samples: {samples}")
    print(f"Channels: {[p[1] for p in channel_params]}")

    for sample in samples:
        try:
            process_sample(sample, args, channel_params)
        except Exception as e:
            print(f"  ERROR {sample}: {type(e)}")


if __name__ == "__main__":
    main()