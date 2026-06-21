"""
Build ORION-CRC patch dataset for token-level regression training.

Pipeline (per slide)
--------------------
1. TRIDENT  : tissue segmentation + patch coordinate extraction on H&E
2. p99s     : per-channel 99th-percentile of AF-corrected foreground pixels
3. targets  : per-patch (C, G, G) mean-expression grid where G=token_grid (16),
              aligned with UNI2 patch tokens (patch_size=14, 224/14=16 tokens/side)
4. Save HDF5: consumed by OrionSpatialDataset

HDF5 layout
-----------
  /coords   (N, 2)       int64   — (x, y) top-left in shared H&E/IF level-0 space
  /p99s     (C,)         float32 — AF-corrected foreground p99 per channel
  /targets  (N, C, G, G) float32 — normalised mean expression per token cell
  attrs: sample, marker_names, patch_size, patch_size_level0, token_grid

H&E (*-registered.ome.tif) and IF (*-zlib.ome.tiff) share the same pixel
coordinate space (both MPP=0.325, pre-registered) — no Valis warp needed.
"""

import csv
import json
import subprocess
import argparse
import numpy as np
import cv2
import h5py
import tifffile
import zarr
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ORION_DATA_DIR = Path("/mnt/ssd/virtual_proteomics/data/ORION_CRC")
TRIDENT_SCRIPT = Path("TRIDENT/run_batch_of_slides.py")
LAMBDA_JSON    = Path("MIPHEI-ViT/preprocessings/mif_cleaning/lambda_settings/orion.json")
JOB_DIR        = Path("datasets/orion_trident_output_reg")
OUTPUT_DIR  = Path("datasets/orion_crc_patch_dataset_benchmark")

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

# MIPHEI split — used when --p99_slides=train_only
MIPHEI_TEST_SLIDES = ["CRC11", "CRC02"]
MIPHEI_VAL_SLIDES  = ["CRC19", "CRC30"]
MIPHEI_TRAIN_SLIDES = [s for s in ALL_CRC_SAMPLES
                        if s not in MIPHEI_TEST_SLIDES + MIPHEI_VAL_SLIDES]


def open_zarr_level0(path: Path, lru_bytes: int = 2 * 2**30):
    """
    Open OME-TIFF as a full-resolution zarr array with an LRU chunk cache.
    Returns (arr, c_ax, h_ax, w_ax).
    """
    return _open_zarr(path, level=0, lru_bytes=lru_bytes)


def _open_zarr(path: Path, level: int = 0, lru_bytes: int = 512 * 2**20):
    tif   = tifffile.TiffFile(str(path))
    store = zarr.LRUStoreCache(tif.aszarr(), max_size=lru_bytes)
    z     = zarr.open(store, mode="r")
    if isinstance(z, zarr.hierarchy.Group):
        key = str(level) if str(level) in z else "0"
        arr = z[key]
    else:
        arr = z
    if len(arr.shape) == 3:
        if arr.shape[2] <= 4:
            return arr, 2, 0, 1
        else:
            return arr, 0, 1, 2
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

def compute_global_p99s(
    samples: list[str],
    channel_params: list,
    af_raw_ch: int = ORION_AF_RAW_CH,
    level: int = 1,
) -> list[float]:
    """
    Compute global per-channel p99/p99.9 from non-zero foreground pixels
    pooled across slides, reading the full slide at the given pyramid level.
    """
    C       = len(channel_params)
    buffers = [[] for _ in range(C)]

    for si, sample in enumerate(samples):
        print(f"  [{si+1}/{len(samples)}] {sample}…", flush=True)
        try:
            _, if_path = crc_paths(sample)
        except Exception as e:
            print(f"    WARNING: skipping {sample} ({e})"); continue

        arr, c_ax, h_ax, w_ax = _open_zarr(if_path, level=level)
        idx_af       = [slice(None)] * arr.ndim
        idx_af[c_ax] = af_raw_ch
        af_ds        = arr[tuple(idx_af)].astype(np.float32)
        for ci, (raw_ch, name, lam, bias) in enumerate(channel_params):
            idx       = [slice(None)] * arr.ndim
            idx[c_ax] = raw_ch
            sig       = arr[tuple(idx)].astype(np.float32)
            corrected = np.maximum(sig - lam * af_ds + bias, 0.0)
            nz        = corrected.ravel(); nz = nz[nz > 0]
            if len(nz):
                buffers[ci].append(nz)
        del af_ds

    print("\n  Global p99 / p99.9:")
    p99s, p999s = [], []
    for ci, (_, name, _, _) in enumerate(channel_params):
        if not buffers[ci]:
            p99s.append(1.0); p999s.append(1.0)
            print(f"    {name:<14}  EMPTY → p99=1.0  p999=1.0")
        else:
            all_vals = np.concatenate(buffers[ci])
            p99_val  = float(np.percentile(all_vals, 99))
            p999_val = float(np.percentile(all_vals, 99.9))
            p99s.append(p99_val); p999s.append(p999_val)
            print(f"    {name:<14}  p99={p99_val:.1f}  p99.9={p999_val:.1f}")

    return p99s, p999s


def save_global_p99s(p99s: list[float], p999s: list[float],
                     channel_params: list, out_path: Path) -> None:
    import json
    data = {name: {"p99": p99, "p999": p999}
            for (_, name, _, _), p99, p999 in zip(channel_params, p99s, p999s)}
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Global p99/p99.9 saved → {out_path}")


def load_global_p99s(path: Path, channel_params: list) -> tuple[list[float], list[float]]:
    import json
    with open(path) as f:
        data = json.load(f)
    p99s  = [data[name]["p99"]  for _, name, _, _ in channel_params]
    p999s = [data[name]["p999"] for _, name, _, _ in channel_params]
    return p99s, p999s


def compute_token_grid_targets(
    arr, c_ax, h_ax, w_ax,
    channel_params: list,
    coords: np.ndarray,
    patch_size_level0: int,
    p99s: list[float],
    af_raw_ch: int = ORION_AF_RAW_CH,
    token_grid: int = 16,
    normalisation: str = "log1p",
) -> np.ndarray:
    """
    Compute (token_grid, token_grid) mean-expression grid for every patch.

    Each cell (i, j) is the mean normalised IF intensity within the pixel region
    that maps to UNI2 patch token (i, j).  With patch_size=14 on 224×224, UNI2
    produces 224/14 = 16 tokens per side, so token_grid=16 by default.

    Returns (N, C, token_grid, token_grid) float32.
    """
    N        = len(coords)
    C        = len(channel_params)
    H_arr    = arr.shape[h_ax]
    W_arr    = arr.shape[w_ax]
    token_px = 224 // token_grid          # = 14 for UNI2
    targets  = np.zeros((N, C, token_grid, token_grid), dtype=np.float32)

    # Precompute per-channel constants as (C,1,1) arrays for broadcasting
    raw_chs  = [rc for rc, *_ in channel_params]
    lams     = np.array([lam  for _, _, lam, _    in channel_params], np.float32)[:, None, None]
    biases   = np.array([bias for _, _, _,   bias in channel_params], np.float32)[:, None, None]
    p99s_arr = np.array(p99s, np.float32)[:, None, None]

    for i, (px, py) in enumerate(coords):
        if i % 200 == 0:
            print(f"    [{i}/{N}] computing token targets…", flush=True)
        px, py = int(px), int(py)

        h_sl = slice(py, min(py + patch_size_level0, H_arr))
        w_sl = slice(px, min(px + patch_size_level0, W_arr))

        # single zarr read for all channels
        idx       = [slice(None)] * arr.ndim
        idx[h_ax] = h_sl
        idx[w_ax] = w_sl
        raw = arr[tuple(idx)].astype(np.float32)   # (C_total, H_p, W_p) or (H_p, W_p, C_total)

        if c_ax == 0:
            af   = raw[af_raw_ch]          # (H_p, W_p)
            sigs = raw[raw_chs]            # (C, H_p, W_p)
        else:
            af   = raw[:, :, af_raw_ch]
            sigs = raw[:, :, raw_chs].transpose(2, 0, 1)

        if af.shape[0] < token_grid or af.shape[1] < token_grid:
            continue

        # vectorised AF-correction + normalisation over all C channels
        corrected = np.maximum(sigs - lams * af[None] + biases, 0.0) / p99s_arr
        if normalisation == "log1p":
            normed = np.clip(np.log1p(corrected), 0.0, 1.0)
        elif normalisation == "arcsinh":
            normed = np.clip(np.arcsinh(corrected), 0.0, 1.0)
        else:
            normed = np.clip(corrected, 0.0, 1.0)
        # (C, H_p, W_p)

        # single resize call on (H_p, W_p, C) image
        resized = cv2.resize(
            normed.transpose(1, 2, 0), (224, 224), interpolation=cv2.INTER_LINEAR
        )  # (224, 224, C)

        # block mean aligned to UNI2 token grid → (C, 16, 16)
        targets[i] = (
            resized
            .reshape(token_grid, token_px, token_grid, token_px, C)
            .mean(axis=(1, 3))
            .transpose(2, 0, 1)
        )

    return targets


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

# ── p99 persistence ──────────────────────────────────────────────────────────

def save_p99s_txt(
    sample: str,
    p99s: list[float],
    channel_params: list,
    p99s_txt: Path = OUTPUT_DIR / 'p99s_slide.txt',
) -> None:
    """Append sample p99s to p99s_slide.txt in load_display_p99s format."""
    p99s_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(p99s_txt, "a") as fh:
        fh.write(f"{sample}\n")
        for (_, name, *_), val in zip(channel_params, p99s):
            fh.write(f"  {name} {val}\n")
    print(f"  p99s appended → {p99s_txt}")


# ── HDF5 save ─────────────────────────────────────────────────────────────────

def save_dataset(
    out_path: Path,
    coords: np.ndarray,
    p99s: list[float],
    p999s: list[float],
    targets: np.ndarray,
    marker_names: list[str],
    patch_size: int,
    patch_size_level0: int,
    sample: str,
    token_grid: int = 16,
    normalisation: str = "log1p",
) -> None:
    N, C, G, _ = targets.shape
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("coords",  data=coords, compression="gzip")
        f.create_dataset("p99s",    data=np.array(p99s,  dtype=np.float32))
        f.create_dataset("p999s",   data=np.array(p999s, dtype=np.float32))
        f.create_dataset("targets", data=targets, compression="gzip",
                         chunks=(min(256, N), C, G, G))
        f.attrs["sample"]            = sample
        f.attrs["marker_names"]      = marker_names
        f.attrs["patch_size"]        = patch_size
        f.attrs["patch_size_level0"] = patch_size_level0
        f.attrs["token_grid"]        = token_grid
        f.attrs["normalisation"]     = normalisation

    mb = out_path.stat().st_size / 1e6
    print(f"\n  Saved → {out_path}  ({mb:.1f} MB)")
    print(f"    /coords   {coords.shape}")
    print(f"    /targets  {targets.shape}  mean={targets.mean():.4f}")


# ── Per-sample pipeline ───────────────────────────────────────────────────────

def process_sample(
    sample: str,
    args: argparse.Namespace,
    channel_params: list,
    global_p99s: list[float],
    global_p999s: list[float],
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
    # Sort by (row, col) so consecutive patches share zarr chunks → better cache hit rate
    coords = coords[np.lexsort((coords[:, 0], coords[:, 1]))]
    # Level-0 crop size: scale from target mag resolution to level-0 resolution
    patch_size_level0 = round(patch_size * (10.0 / target_mag) / MPP_HE)

    # 2. Open IF zarr (H&E and IF share the same coordinate space — no warp)
    arr, c_ax, h_ax, w_ax = open_zarr_level0(if_path)

    # 3. Token-grid targets — (N, C, G, G) mean expression per token cell
    print(f"\n  Computing {args.token_grid}×{args.token_grid} token targets ({len(coords)} patches)…")
    targets = compute_token_grid_targets(
        arr, c_ax, h_ax, w_ax,
        channel_params, coords, patch_size_level0,
        p99s=global_p99s,
        af_raw_ch=ORION_AF_RAW_CH,
        token_grid=args.token_grid,
        normalisation=args.normalisation,
    )

    # 4. Save HDF5
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    marker_names = [p[1] for p in channel_params]
    save_dataset(
        out_path, coords, global_p99s, global_p999s, targets,
        marker_names, patch_size, patch_size_level0, sample,
        token_grid=args.token_grid,
        normalisation=args.normalisation,
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
    parser.add_argument("--min_tissue",   type=float, default=0.25)
    parser.add_argument("--segmenter",    default="hest",
                        choices=["hest", "grandqc", "otsu"])
    parser.add_argument("--seg_thresh",   type=float, default=0.5) # 0.3 for CRC16
    parser.add_argument("--gpu",          type=int,   default=1)
    parser.add_argument("--job_dir",      default=str(JOB_DIR))
    parser.add_argument("--output_dir",   default=str(OUTPUT_DIR))
    parser.add_argument("--lambda_json",  default=str(LAMBDA_JSON))
    parser.add_argument("--token_grid",   type=int,   default=16,
                        help="Spatial grid size (must match FM patch tokens: UNI2=16)")
    parser.add_argument("--normalisation", default="linear", choices=["log1p", "arcsinh", "linear"],
                        help="Target normalisation: log1p, arcsinh, or linear clip(x/p99, 0, 1)")
    parser.add_argument("--p99_level",  type=int, default=1,
                        help="Pyramid level for p99 (1=4x down, default). Full slide read.")
    parser.add_argument("--p99_slides", default="all",
                        choices=["all", "train_only"],
                        help="Slides used for p99: 'all' (default) or "
                             "'train_only' (MIPHEI split: exclude CRC11,CRC02,CRC19,CRC30).")
    args = parser.parse_args()

    channels = list(ORION_CRC_CHANNELS)
    if args.include_pd1:
        channels.append(ORION_PD1)

    settings       = load_lambda_settings(Path(args.lambda_json))
    channel_params = build_channel_params(channels, settings)

    samples = (ALL_CRC_SAMPLES if args.samples.lower() == "all"
               else [s.strip() for s in args.samples.split(",")])

    print(f"Samples ({len(samples)}): {samples}")
    print(f"Channels: {[p[1] for p in channel_params]}")
    print(f"Normalisation: {args.normalisation}  Output: {args.output_dir}")

    # ── Pass 1: global p99s ───────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    p99s_path = Path(args.output_dir) / "global_p99s.json"

    p99_samples = (MIPHEI_TRAIN_SLIDES if args.p99_slides == "train_only"
                   else samples)
    # Only keep p99_samples that are in the requested samples list
    p99_samples = [s for s in p99_samples if s in samples]

    print(f"\np99 config: level={args.p99_level}  slides={args.p99_slides} ({len(p99_samples)} slides)")

    if p99s_path.exists():
        print(f"Loading cached global p99s from {p99s_path}")
        global_p99s, global_p999s = load_global_p99s(p99s_path, channel_params)
        for (_, name, _, _), v99, v999 in zip(channel_params, global_p99s, global_p999s):
            print(f"    {name:<14}  p99={v99:.1f}  p99.9={v999:.1f}")
    else:
        print(f"\n── Pass 1: computing global p99/p99.9 ──")
        global_p99s, global_p999s = compute_global_p99s(
            p99_samples, channel_params,
            level=args.p99_level,
        )
        save_global_p99s(global_p99s, global_p999s, channel_params, p99s_path)

    # ── Pass 2: build per-slide token targets ─────────────────────────────────
    print(f"\n── Pass 2: building patch datasets ──")
    for sample in samples:
        try:
            process_sample(sample, args, channel_params, global_p99s, global_p999s)
        except Exception as e:
            print(f"  ERROR {sample}: {type(e)}")


if __name__ == "__main__":
    main()