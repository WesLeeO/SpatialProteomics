"""
Build melanoma CyCIF patch dataset for token-level regression training.

Dataset structure
-----------------
  /mnt/ssd1/virtual_proteomics/data/melanoma/
    {P}-{B}-{S}.ome.tif           IF multiplex (C, H, W) uint16
    {P}-{B}-{S}-features.zip      single CSV with single-cell features
    {P}-{B}-0-HE-ROI{N}.ome.tif  H&E serial section (3, H, W)

  where P=patient (MEL01–MEL13), B=biopsy index, S=section index, N=ROI number.

  Most patients have one biopsy with one IF section and two H&E ROIs.
  MEL01 has multiple biopsies and multiple IF sections per biopsy; each pairing
  with one H&E ROI (ROI1 only for MEL01).

Channel metadata
----------------
  Derived directly from the CSV header inside each features zip — no external
  metadata file needed.  Column order in the CSV matches TIF channel order (ch0
  = first data column, ch1 = second, ...).

  Included channels : first DNA_* channel + all protein markers
  Skipped           : bg* (background/AF reference), repeat DNA channels,
                      morphological columns (X_centroid, Area, ...)

  bg* channels are not subtracted by default (wavelength-filter mapping is not
  encoded in the CSV).  Pass --lam > 0 together with --bg_strategy to enable
  nearest-bg subtraction.

Pipeline (per H&E ROI × IF pair)
---------------------------------
1. Registration : Valis aligns H&E ROI to IF (both are serial sections of the
                  same tissue; dimensions always differ).
2. TRIDENT      : tissue segmentation + patch coords on H&E ROI.
3. p99s         : per-channel 99th-percentile of foreground pixels.
4. targets      : (N, C, G, G) token-grid mean expression (G=16 for UNI2).
5. HDF5         : same layout as OrionSpatialDataset / crc_atlas_patch_dataset.

Output
------
  melanoma_patch_dataset/{sample}_patch_dataset.h5
  melanoma_patch_dataset/p99s_slide.txt
"""

"""
 H&E → OD (optical density): converts the RGB image to a grayscale stain-density map, which enhances tissue structure contrast for feature matching
  - IF crop → ChannelGetter(channel=dna_ch, adaptive_eq=True): extracts channel channels[0][0] — i.e. the first channel in the filtered channel list, which is always DNA_1
   (the Hoechst/nuclear stain), then applies adaptive histogram equalisation
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import csv as _csv
import subprocess
import xml.etree.ElementTree as ET
import zipfile
import numpy as np
import cv2
import h5py
import tifffile
import zarr
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR       = Path("/mnt/ssd1/virtual_proteomics/data/melanoma")
TRIDENT_SCRIPT = Path("TRIDENT/run_batch_of_slides.py")
JOB_DIR        = Path("datasets/melanoma_trident_output")
OUTPUT_DIR     = Path("datasets/melanoma_patch_dataset")
VALIS_DIR      = Path("datasets/melanoma_valis")

# Per-sample IF split axis.  Default is "h" (top/bottom halves, ROI1=top).
# Use "w" for left/right split with ROI1=right half (e.g. MEL03 where the
# tissue sections are arranged side-by-side rather than stacked vertically).
IF_SPLIT_AXIS: dict[str, str] = {
    "MEL02-1-1": "h",
    "MEL03-1-1": "w",
    "MEL04-1-1": "w",
    "MEL05-1-1": "h",
    "MEL07-1-1": "w_ltr",
    "MEL08-1-1": "h",
    "MEL09-1-1": "h",
    "MEL10-1-1": "h",
    "MEL11-1-1": "w_ltr",
    "MEL12-1-1": "h",
    "MEL13-2-1": "w_ltr"
}

# Additional x-axis crop for the VALIS IF crop (fractions of full IF width).
# Applied on top of the primary height/width ROI split to remove background.
# Key = if_stem.  Value = (x0_frac, x1_frac).
IF_VALIS_X_CROP: dict[str, tuple[float, float]] = {
    "MEL05-1-1-ROI1": (0.0, 0.5),   # tissue in left half; strip right background
}

# Columns in the CSV that are morphological metadata, not IF channels
_MORPHO_COLS = frozenset({
    "CellID",
    "X_centroid", "Y_centroid", "column_centroid", "row_centroid",
    "Area", "MajorAxisLength", "MinorAxisLength",
    "Eccentricity", "Solidity", "Extent", "Orientation",
})

# ── Name normalisation ────────────────────────────────────────────────────────
# Melanoma CSV names → canonical Orion names.
_CANONICAL: dict[str, str] = {
    "pdl1":   "PD-L1",
    "cd3d":   "CD3",
    "cd3e":   "CD3",
    "pan-ck": "Pan-CK",
    "panck":  "Pan-CK",
    "s100a":  "S100A",
    "ecad":   "E-Cadherin",
    "ki67":   "Ki-67",
    "asma":   "SMA",
}

def canonical_name(name: str) -> str:
    return _CANONICAL.get(name.lower(), name)


# ── Marker sets ───────────────────────────────────────────────────────────────
MARKER_SETS: dict[str, frozenset[str]] = {
    "orion_crc": frozenset({
        "DNA_1",        # first Hoechst cycle
        "CD31",
        "CD45",
        "CD68",
        "CD4",
        "FOXP3",
        "CD8a",
        "CD45RO",
        "CD20",
        "PD-L1",        # PDL1 in melanoma CSV → canonical
        "CD3",          # CD3d / CD3e in melanoma CSV → canonical
        "CD163",
        "E-Cadherin",   # ECAD in melanoma CSV → canonical
        "Ki-67",        # KI67 in melanoma CSV → canonical
        "Pan-CK",       # pan-CK in melanoma CSV → canonical
        "SMA",          # aSMA in melanoma CSV → canonical
    }),
}


def filter_channels(
    channels: list[tuple[int, str, int | None]],
    marker_set: str | None,
) -> list[tuple[int, str, int | None]]:
    """Return only channels whose canonical name is in the requested marker set."""
    if marker_set is None:
        return channels
    if marker_set not in MARKER_SETS:
        raise ValueError(f"Unknown marker set '{marker_set}'. "
                         f"Available: {list(MARKER_SETS)}")
    keep = MARKER_SETS[marker_set]
    filtered = [ch for ch in channels if ch[1] in keep]
    missing  = keep - {ch[1] for ch in filtered}
    if missing:
        print(f"  [marker_set={marker_set}] not found in this sample: {missing}")
    return filtered


# ── Channel parsing from CSV ──────────────────────────────────────────────────

def parse_csv_channels(zip_path: Path) -> list[tuple[int, str, int | None]]:
    """
    Parse channel list from the features zip CSV header.

    Returns [(ch_idx, name, af_ch)] where:
      ch_idx  = 0-indexed TIF channel (== position among data columns)
      name    = marker name, compartment suffix stripped
      af_ch   = data_col index of the matching AF (bg) channel, or None

    Included : first DNA_* channel, all protein markers
    Skipped  : bg* channels, repeat DNA channels (DNA_2 onward),
               morphological columns

    AF mapping: bg2a/bg3a/bg4a are measured once in cycle 1 at wavelengths
    2/3/4 (488/555/647 nm).  Within each subsequent cycle, protein channels
    occupy wavelength slots 2, 3, 4, ... in order.  Some later cycles have 4
    protein slots (a 5th wavelength with no bg reference → af_ch=None).
    Grouping by cycle boundaries (DNA channels) and counting slots within each
    cycle is robust to variable cycle lengths; ch_idx % 4 is NOT reliable.
    """
    inner = zip_path.stem.replace("-features", "") + ".csv"
    with zipfile.ZipFile(str(zip_path)) as zf:
        with zf.open(inner) as f:
            header = f.readline().decode().strip().split(",")

    # Data columns = everything that is not a morphological attribute.
    # ch_idx = position in this ordered list (matches TIF axis 0).
    data_cols = [col for col in header if col not in _MORPHO_COLS]

    # bg channels: bgNa → wavelength N → ch_idx in data_cols
    bg_ch: dict[int, int] = {}
    for i, col in enumerate(data_cols):
        name = col.replace("_cellRingMask", "").replace("_cellMask", "")
        if name.lower().startswith("bg") and len(name) >= 3 and name[2].isdigit():
            bg_ch[int(name[2])] = i

    # Group into cycles: each DNA channel starts a new cycle.
    cycles: list[list[tuple[int, str]]] = []
    current: list[tuple[int, str]] = []
    for i, col in enumerate(data_cols):
        name = col.replace("_cellRingMask", "").replace("_cellMask", "")
        if name.upper().startswith("DNA"):
            if current:
                cycles.append(current)
            current = [(i, name)]
        else:
            current.append((i, name))
    if current:
        cycles.append(current)

    # Within each cycle assign wavelength slots (2, 3, 4, …).
    # bg channels also consume a slot but are skipped as targets.
    channels: list[tuple[int, str, int | None]] = []
    dna_seen = False
    for cycle in cycles:
        slot = 2
        for ch_idx, name in cycle:
            if name.upper().startswith("DNA"):
                if not dna_seen:
                    dna_seen = True
                    channels.append((ch_idx, "DNA_1", None))
            elif name.lower().startswith("bg"):
                slot += 1   # bg occupies its wavelength slot; not a target
            else:
                af_ch = bg_ch.get(slot)   # None if no bg at this wavelength
                channels.append((ch_idx, canonical_name(name), af_ch))
                slot += 1

    if bg_ch:
        n_mapped = sum(1 for _, _, af in channels if af is not None)
        print(f"  [af] bg at wl={sorted(bg_ch)}: "
              f"{n_mapped}/{len(channels)} markers have AF reference")
    else:
        print(f"  [af] no bg channels — AF subtraction not available")

    return channels


def build_af_arrays(
    channels: list[tuple[int, str, int | None]],
    af_params_path: str | None,
    slide: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-channel lambda and bias arrays from an AF params JSON.

    JSON format (per-slide):
      {
        "MEL02-1-1-ROI2": {
          "DNA_1":  {"lambda": 0.0, "bias": 0},
          "PD-L1":  {"lambda": 0.5, "bias": 750},
          ...
        },
        "MEL08-1-1-ROI1": { ... }
      }

    If slide is provided and found in the JSON, that entry is used.
    Falls back to a top-level "markers" key for legacy flat format.
    Markers with af_ch=None always get lambda=0 regardless of the JSON.
    """
    import json
    params: dict = {}
    if af_params_path:
        with open(af_params_path) as f:
            params = json.load(f)

    if slide and slide in params:
        marker_cfg = params[slide]
        slide_tag  = f"  slide={slide!r}"
    else:
        marker_cfg = params.get("markers", {})
        slide_tag  = f"  slide={slide!r} (not found — no subtraction)" if slide else ""

    lams   = np.zeros(len(channels), dtype=np.float32)
    biases = np.zeros(len(channels), dtype=np.float32)
    for ci, (_, name, af_ch) in enumerate(channels):
        if af_ch is None:
            continue
        cfg = marker_cfg.get(name, {})
        lams[ci]   = float(cfg.get("lambda", 0.0))
        biases[ci] = float(cfg.get("bias",   0.0))

    if af_params_path:
        print(f"  [af_params] {af_params_path}{slide_tag}")
        for ci, (_, name, af_ch) in enumerate(channels):
            if af_ch is not None:
                print(f"    {name:<15s}  af_ch={af_ch}  λ={lams[ci]:.3f}  bias={biases[ci]:.1f}")
            else:
                print(f"    {name:<15s}  no AF ref")
    else:
        print("  [af_params] none — no AF subtraction")

    return lams, biases


def describe_csv(zip_path: Path) -> None:
    """Print a summary of the CSV channel layout (for inspection)."""
    inner = zip_path.stem.replace("-features", "") + ".csv"
    with zipfile.ZipFile(str(zip_path)) as zf:
        with zf.open(inner) as f:
            header = f.readline().decode().strip().split(",")
    data_cols = [col for col in header if col not in _MORPHO_COLS]
    print(f"  CSV channels ({len(data_cols)} total):")
    for i, col in enumerate(data_cols):
        tag = ""
        name = col.replace("_cellRingMask", "").replace("_cellMask", "")
        if name.lower().startswith("bg"):
            tag = " [bg/AF]"
        elif name.upper().startswith("DNA"):
            tag = " [DNA]"
        print(f"    ch{i:02d}  {name}{tag}")


# ── Sample discovery ──────────────────────────────────────────────────────────

def discover_samples(data_dir: Path) -> list[dict]:
    """
    Auto-discover all processable (IF tif, features zip, H&E ROI) triplets.

    Naming convention:
      IF  : {P}-{B}-{S}.ome.tif          (S > 0, no 'HE' in name)
      zip : {P}-{B}-{S}-features.zip
      H&E : {P}-{B}-0-HE-ROI{N}.ome.tif

    One entry per (IF, zip, H&E ROI) combination.  Each entry is a dict with
    keys: sample, he_path, if_path, zip_path.
    """
    zips: dict[str, Path] = {}
    for p in data_dir.glob("*-features.zip"):
        stem = p.name.replace("-features.zip", "")
        zips[stem] = p

    ifs: dict[str, Path] = {}
    for p in data_dir.glob("*.ome.tif"):
        if "HE" not in p.name:
            stem = p.name.replace(".ome.tif", "")
            ifs[stem] = p

    # he_rois: "{P}-{B}" → sorted list of ROI paths
    he_rois: dict[str, list[Path]] = {}
    for p in sorted(data_dir.glob("*-HE-ROI*.ome.tif")):
        prefix = p.name.split("-0-HE-")[0]     # e.g. "MEL08-1"
        he_rois.setdefault(prefix, []).append(p)

    samples = []
    for stem in sorted(zips):
        zip_path = zips[stem]
        if stem not in ifs:
            print(f"  [discover] skip {stem}: no matching IF tif")
            continue
        if_path = ifs[stem]

        # {P}-{B} prefix: drop last "-{S}" segment
        prefix = "-".join(stem.split("-")[:-1])
        rois = he_rois.get(prefix, [])

        if not rois:
            print(f"  [discover] skip {stem}: no H&E ROIs with prefix {prefix}")
            continue

        for roi_path in sorted(rois):
            roi_n = roi_path.name.split("ROI")[1].replace(".ome.tif", "")
            sample_name = f"{stem}-ROI{roi_n}"
            samples.append({
                "sample":   sample_name,
                "he_path":  roi_path,
                "if_path":  if_path,
                "zip_path": zip_path,
                "n_rois":   len(rois),
                "roi_n":    int(roi_n),
            })
    return samples


# ── Registration helpers ──────────────────────────────────────────────────────

class DirectSlide:
    """Drop-in for Valis Slide when images share coordinate space (identity)."""
    def warp_xy_from_to(self, xy, to_slide_obj, slide_level: int = 0, **kw) -> np.ndarray:
        return np.asarray(xy, dtype=float)


def he_rotation_k(he_path: Path, if_path: Path) -> int:
    """Return the np.rot90 k needed to match H&E orientation to IF.

    Compares portrait/landscape orientation.  If they disagree, a 90° rotation
    (k=1) is needed.  Returns 0 (no rotation) or 1 (90° CCW).
    """
    with tifffile.TiffFile(str(he_path)) as t:
        he_h, he_w = t.series[0].shape[-2:]
    with tifffile.TiffFile(str(if_path)) as t:
        if_h, if_w = t.series[0].shape[-2:]
    he_portrait = he_h > he_w
    if_portrait  = if_h > if_w
    return 3 #if he_portrait != if_portrait else 0


def prerotate_he(he_path: Path, out_path: Path, k: int) -> None:
    """Write a rot90(k)-rotated copy of the H&E as a tiled interleaved RGB TIFF.

    Tiled interleaved (H, W, C) is the format that both pyvips (Valis) and
    OpenSlide (TRIDENT) can open without issues.  No pyramid — conversion will
    be slower for Valis but the file is guaranteed readable.
    Skipped if out_path already exists.
    """
    if out_path.exists():
        print(f"  [prerotate] {out_path.name} already exists — skipping")
        return
    with tifffile.TiffFile(str(he_path)) as tif:
        img = tif.series[0].asarray()                          # (3, H, W) uint8
    rotated     = np.rot90(img, k=k, axes=(-2, -1))            # (3, H', W') zero-copy
    rotated_hwc = np.ascontiguousarray(rotated.transpose(1, 2, 0))  # (H', W', 3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), rotated_hwc,
                     photometric="rgb", compression="lzw",
                     tile=(256, 256), bigtiff=True)
    print(f"  [prerotate] wrote {out_path.name}  {img.shape} → {rotated_hwc.shape}")


def write_if_valis_crop(if_path: Path, roi_n: int, n_rois: int,
                        dna_ch: int, out_path: Path,
                        split_axis: str = "h",
                        x_frac: tuple[float, float] | None = None) -> tuple[int, int]:
    """Write a single-channel (DNA) crop of one ROI for Valis registration.

    split_axis="h": split along height — ROI1=top, ROI2=bottom (default).
    split_axis="w": split along width  — ROI1=right, ROI2=left.

    Returns (x_offset, y_offset) in full-IF pixel space for this ROI crop.
    """
    arr, c_ax, h_ax, w_ax = open_zarr_level0(if_path)
    idx = [slice(None)] * arr.ndim
    idx[c_ax] = dna_ch

    if split_axis == "h":
        H       = arr.shape[h_ax]
        h_slice = H // n_rois
        y0      = (roi_n - 1) * h_slice
        y1      = y0 + h_slice if roi_n < n_rois else H
        idx[h_ax] = slice(y0, y1)
        x_offset, y_offset = 0, y0
        dim_str = f"y={y0}:{y1}"
    else:   # "w" — ROI1=rightmost / "w_ltr" — ROI1=leftmost
        W       = arr.shape[w_ax]
        w_slice = W // n_rois
        chunk   = (roi_n - 1) if split_axis == "w_ltr" else (n_rois - roi_n)
        x0      = chunk * w_slice
        x1      = x0 + w_slice if chunk < n_rois - 1 else W
        idx[w_ax] = slice(x0, x1)
        x_offset, y_offset = x0, 0
        dim_str = f"x={x0}:{x1}"

    # Optional additional x-crop (applied on top of primary ROI split).
    # Only meaningful when split_axis="h"; for "w" it would conflict.
    if x_frac is not None:
        W_arr = arr.shape[w_ax]
        xc0 = int(x_frac[0] * W_arr)
        xc1 = int(x_frac[1] * W_arr)
        idx[w_ax] = slice(xc0, xc1)
        x_offset += xc0
        dim_str += f" x={xc0}:{xc1}"

    if not out_path.exists():
        data = np.array(arr[tuple(idx)])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(out_path), data, photometric="minisblack",
                         tile=(256, 256), compression="lzw", bigtiff=True)
        print(f"  [if_crop] wrote {out_path.name}  1ch {dim_str} "
              f"shape={data.shape}")
    else:
        print(f"  [if_crop] {out_path.name} already exists — skipping")

    return x_offset, y_offset


def _valis_resolution_params(he_path: Path,
                             rigid_target_um: float = 10.0,
                             non_rigid_target_um: float = 10.0,
                             mpp: float | None = None) -> dict:
    """Compute Valis resolution kwargs so both steps hit a fixed physical resolution.

    rigid_target_um    : µm/px for feature detection + rigid alignment.
                         10 µm/px = cell clusters visible; safe for LightGlue.
    non_rigid_target_um: µm/px for the B-spline warp grid.
                         5 µm/px ≈ sub-cellular, enough to correct tissue
                         deformation between serial sections.

    Both use int(max_dim * mpp / target_um) so every slide regardless of size
    gets the same physical resolution at each registration step.

    mpp: pass explicitly when he_path is a rotated copy that lacks OME metadata.
    """
    with tifffile.TiffFile(str(he_path)) as tif:
        s = tif.series[0].shape
        if mpp is None:
            mpp = 0.325
            if tif.ome_metadata:
                root = ET.fromstring(tif.ome_metadata)
                for px in root.iter():
                    if px.tag.endswith("Pixels"):
                        val = px.get("PhysicalSizeX")
                        if val:
                            mpp = float(val)
                        break
    h, w          = (s[0], s[1]) if s[-1] <= 4 else (s[-2], s[-1])
    max_dim       = max(h, w)
    rigid_dim     = max(512,  int(max_dim * mpp / rigid_target_um))
    non_rigid_dim = max(2048, int(max_dim * mpp / non_rigid_target_um))
    params = dict(
        max_processed_image_dim_px=rigid_dim,
        max_image_dim_px=max(1024, rigid_dim),
        max_non_rigid_registration_dim_px=non_rigid_dim,
    )
    print(f"  [valis_res] H&E {h}×{w}  mpp={mpp:.4f}  →  "
          f"rigid={rigid_dim} ({rigid_target_um}µm/px)  "
          f"non_rigid={non_rigid_dim} ({non_rigid_target_um}µm/px)")
    return params


def run_valis(he_path: Path, if_path: Path, valis_dir: Path,
              dna_ch: int = 0, mpp: float | None = None,
              micro_rigid: bool = True) -> None:
    from valis import registration as valis_reg
    from valis.preprocessing import OD, ChannelGetter

    valis_dir.mkdir(parents=True, exist_ok=True)
    processor_dict = {
        he_path.name: OD,
        if_path.name: [ChannelGetter, {"channel": dna_ch, "adaptive_eq": True}],
    }
    #res_params = _valis_resolution_params(he_path, mpp=mpp)
    #target_mpp = 5 # microns / pixel
    micro_kwargs = {}
    if micro_rigid:
        from valis.micro_rigid_registrar import MicroRigidRegistrar
        micro_kwargs = dict(
            micro_rigid_registrar_cls=MicroRigidRegistrar,
            micro_rigid_registrar_params={"scale": 0.0625, "tile_wh": 256, "roi": "mask"},
        )
        # 0.0625 # 256 

    # _match_tile is a closure → cannot be pickled by loky → workers hang at 0%.
    # Fix: thread-based pqdm (shared memory, no pickling needed).
    # Cap at 5 (→ n_jobs=4): uncapped Python threads × libvips sub-threads
    # exhausts OS thread limit and deadlocks.
    import valis.micro_rigid_registrar as _mrr
    from pqdm.threads import pqdm as _pqdm_threads
    _mrr.pqdm = _pqdm_threads
    from valis import valtils as _valtils  
    _valtils.get_ncpus_available = lambda: 8  # → n_jobs = 4 threads
    
    print("[Valis] Registering…")
    registrar = valis_reg.Valis(
        str(valis_dir), str(valis_dir.parent),
        img_list=[str(he_path), str(if_path)],
        reference_img_f=str(he_path),
        align_to_reference=True,
        #**res_params,
        **micro_kwargs,
    )
    registrar.register(processor_dict=processor_dict)
    print(f"[Valis] Done → {valis_dir}")


def load_slides(valis_dir: Path, he_name: str, if_name: str):
    from valis import registration as valis_reg

    pickles = list(valis_dir.rglob("*.pickle"))
    if not pickles:
        raise FileNotFoundError(f"No Valis pickle under {valis_dir}")
    registrar = valis_reg.load_registrar(str(pickles[0]))
    he_slide = if_slide = None
    for slide in registrar.slide_dict.values():
        name = Path(slide.src_f).name
        if name == he_name:
            he_slide = slide
        elif name == if_name:
            if_slide = slide
    if he_slide is None:
        raise KeyError(f"{he_name} not found in registrar")
    if if_slide is None:
        raise KeyError(f"{if_name} not found in registrar")
    return he_slide, if_slide


# ── OME-TIFF helpers ──────────────────────────────────────────────────────────

def get_mpp(tif_path: Path) -> float:
    """Read µm/px from OME-TIFF metadata; fall back to 0.325."""
    with tifffile.TiffFile(str(tif_path)) as tif:
        if tif.ome_metadata:
            root = ET.fromstring(tif.ome_metadata)
            for px in root.iter():
                if px.tag.endswith("Pixels"):
                    val = px.get("PhysicalSizeX")
                    if val:
                        return float(val)
    print("  [MPP] no OME PhysicalSizeX — using fallback 0.325 µm/px")
    return 0.325


def open_zarr_level0(path: Path, lru_bytes: int = 4 * 2**30):
    """Open OME-TIFF as zarr.  Returns (arr, c_ax, h_ax, w_ax)."""
    tif   = tifffile.TiffFile(str(path))
    store = zarr.LRUStoreCache(tif.aszarr(), max_size=lru_bytes)
    z     = zarr.open(store, mode="r")
    arr   = z["0"] if isinstance(z, zarr.hierarchy.Group) else z
    if arr.ndim == 3:
        if arr.shape[2] <= 4:        # (H, W, C) — RGB H&E
            return arr, 2, 0, 1
        return arr, 0, 1, 2          # (C, H, W) — multiplex IF
    raise ValueError(f"Unexpected zarr shape {arr.shape} for {path.name}")


# ── TRIDENT helpers ───────────────────────────────────────────────────────────

def run_trident(he_path: Path, job_dir: Path, mpp: float,
                mag: float, patch_size: int, overlap: int,
                min_tissue: float, segmenter: str,
                seg_thresh: float, gpu: int) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    csv_path = job_dir / "wsi_list.csv"
    with open(csv_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["wsi", "mpp"])
        w.writerow([he_path.name, mpp])

    # CUDA_VISIBLE_DEVICES remaps physical GPUs, so the subprocess sees only
    # the selected GPU as device 0 — pass 0 regardless of the physical index.
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", str(gpu))
    trident_env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible}
    base = [
        "python", str(TRIDENT_SCRIPT),
        "--wsi_dir",    str(he_path.parent),
        "--job_dir",    str(job_dir),
        "--gpu",        "0",
        "--segmenter",  segmenter,
        "--seg_conf_thresh", str(seg_thresh),
        "--mag",        str(mag),
        "--patch_size", str(patch_size),
        "--overlap",    str(overlap),
        "--min_tissue_proportion", str(min_tissue),
        "--wsi_ext",    ".tif",
        "--custom_list_of_wsis", str(csv_path),
    ]
    print("[TRIDENT] Segmenting…")
    subprocess.run(base + ["--task", "seg"],    check=True, env=trident_env)
    print("[TRIDENT] Extracting coords…")
    subprocess.run(base + ["--task", "coords"], check=True, env=trident_env)

    h5_files = list(job_dir.rglob("*_patches.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No coords HDF5 under {job_dir}")
    return h5_files[0]


def load_trident_coords(h5_path: Path) -> tuple[np.ndarray, int, float]:
    with h5py.File(h5_path, "r") as f:
        key        = "coords" if "coords" in f else list(f.keys())[0]
        coords     = f[key][:]
        patch_size = int(f[key].attrs.get("patch_size", 224))
        target_mag = float(f[key].attrs.get("target_magnification", 20.0))
    print(f"  {len(coords)} patches  patch_size={patch_size} @ {target_mag}×")
    return coords, patch_size, target_mag


# ── Coordinate mapping ────────────────────────────────────────────────────────

def _precompute_if_bboxes(
    he_slide, if_slide,
    coords: np.ndarray,
    ps_he: int,
    H_if: int, W_if: int,
    x_offset: int = 0,
    y_offset: int = 0,
) -> tuple[list[tuple[int, int, int, int] | None], np.ndarray]:
    """Returns (bboxes, mapped_corners).

    mapped_corners: (N, 4, 2) — the 4 patch corners (TL, TR, BL, BR) mapped to
    VALIS IF coordinate space (offsets NOT yet added).  Used by the token-grid
    step for perspective warping; not needed by the p99 step.
    """
    tl = coords.astype(float)
    all_corners = np.vstack([
        tl,
        tl + [ps_he, 0],
        tl + [0,     ps_he],
        tl + [ps_he, ps_he],
    ])  # (4N, 2): TL×N, TR×N, BL×N, BR×N
    mapped = he_slide.warp_xy_from_to(all_corners, if_slide)  # (4N, 2)
    N = len(coords)
    mapped = mapped.reshape(4, N, 2).transpose(1, 0, 2)       # (N, 4, 2)

    bboxes: list[tuple[int, int, int, int] | None] = []
    for i in range(N):
        corners = mapped[i]
        if np.any(np.isnan(corners)):
            bboxes.append(None)
            continue
        x0 = max(int(np.floor(corners[:, 0].min())) + x_offset, 0)
        y0 = max(int(np.floor(corners[:, 1].min())) + y_offset, 0)
        x1 = min(int(np.ceil(corners[:, 0].max())) + x_offset, W_if)
        y1 = min(int(np.ceil(corners[:, 1].max())) + y_offset, H_if)
        bboxes.append(None if x1 <= x0 or y1 <= y0 else (x0, y0, x1, y1))
    return bboxes, mapped


def read_if_region(arr, c_ax: int, h_ax: int, w_ax: int,
                   x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Read all channels for [y0:y1, x0:x1].  Returns (C, H, W) float32."""
    idx       = [slice(None)] * arr.ndim
    idx[h_ax] = slice(y0, y1)
    idx[w_ax] = slice(x0, x1)
    raw = arr[tuple(idx)].astype(np.float32)
    return raw if c_ax == 0 else raw.transpose(2, 0, 1)

"""
One detail worth noting: I used INTER_NEAREST for the p99 warp instead of INTER_LINEAR. For histogram-based p99 estimation you want to preserve the original
uint16-equivalent values without interpolation blending — INTER_NEAREST avoids the bilinear averaging that would shift intensity values and distort the histogram. The
token-grid targets keep INTER_LINEAR because they need smooth spatial averaging.
"""
# ── Alignment scoring ─────────────────────────────────────────────────────────

def extract_hematoxylin(rgb_float: np.ndarray) -> np.ndarray:
    """Extract nuclear (hematoxylin) signal from H&E using OD channel difference.

    Hematoxylin absorbs red/orange light → low R, relatively high B.
    Eosin (pink) and background (white) both have R ≥ B → clipped to 0.

      hema = max(log(B/R), 0) = max(OD_R - OD_B, 0)

    No per-patch fitting, no SVD instability.

    rgb_float : (H, W, 3) float32 in [0, 1]
    Returns   : (H, W) float32, non-negative
    """
    rgb = np.clip(rgb_float, 1/255, 1.0)
    return np.maximum(np.log(rgb[:, :, 2] / rgb[:, :, 0]), 0.0).astype(np.float32)


def compute_alignment_scores(
    he_arr, h_ax_he: int, w_ax_he: int,
    if_arr, c_ax: int, h_ax: int, w_ax: int,
    dna_raw_ch: int,
    coords: np.ndarray,
    ps_he: int,
    he_slide, if_slide,
    H_if: int, W_if: int,
    x_offset: int = 0,
    y_offset: int = 0,
    debug_dir: Path | None = None,
    n_debug: int = 20,
    debug_min_ssim: float = 0.0,
) -> np.ndarray:
    """SSIM between H&E hematoxylin and IF DNA channel at 224×224 per patch.

    Both channels are min-max normalised to [0,1] before SSIM so that global
    intensity differences don't affect the structural similarity score.
    Returns (N,) float32; patches with no valid IF bbox get score -1.

    If debug_dir is set, saves a montage of the first n_debug patches that
    pass debug_min_ssim, showing H&E RGB | hematoxylin | IF DNA for each.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from skimage.metrics import structural_similarity as ssim_fn

    bboxes, mapped_corners = _precompute_if_bboxes(
        he_slide, if_slide, coords, ps_he, H_if, W_if,
        x_offset=x_offset, y_offset=y_offset,
    )
    dst_pts = np.float32([[0, 0], [224, 0], [0, 224], [224, 224]])
    scores  = np.full(len(coords), -1.0, dtype=np.float32)

    debug_triplets: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

    for i, (px, py) in enumerate(coords):
        if i % 200 == 0:
            print(f"    [align {i}/{len(coords)}]", flush=True)

        bb = bboxes[i]
        if bb is None:
            continue

        # ── H&E patch → hematoxylin ───────────────────────────────────────────
        x0_he, y0_he = int(px), int(py)
        x1_he, y1_he = x0_he + ps_he, y0_he + ps_he
        if h_ax_he == 0:   # (H, W, C)
            he_patch = np.array(he_arr[y0_he:y1_he, x0_he:x1_he, :])
            he_rgb   = he_patch.astype(np.float32) / 255.0
        else:               # (C, H, W)
            he_patch = np.array(he_arr[:, y0_he:y1_he, x0_he:x1_he])
            he_rgb   = he_patch.transpose(1, 2, 0).astype(np.float32) / 255.0

        if he_rgb.shape[0] < 4 or he_rgb.shape[1] < 4:
            continue

        hema       = extract_hematoxylin(he_rgb)
        hema_224   = cv2.resize(hema,   (224, 224), interpolation=cv2.INTER_LINEAR)
        he_rgb_224 = cv2.resize(he_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        hv = hema_224.max()
        if hv > 1e-6:
            hema_224 /= hv
        else:
            continue   # blank H&E patch

        # ── IF DNA patch ──────────────────────────────────────────────────────
        x0, y0, x1, y1 = bb
        region = read_if_region(if_arr, c_ax, h_ax, w_ax, x0, y0, x1, y1)
        if region.shape[1] < 4 or region.shape[2] < 4:
            continue

        dna = region[dna_raw_ch].astype(np.float32)
        vc  = mapped_corners[i]
        src_pts = np.float32([
            [vc[j, 0] + x_offset - x0, vc[j, 1] + y_offset - y0]
            for j in range(4)
        ])
        M       = cv2.getPerspectiveTransform(src_pts, dst_pts)
        dna_224 = cv2.warpPerspective(dna, M, (224, 224),
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT)
        dv = dna_224.max()
        if dv > 1e-6:
            dna_224 /= dv
        else:
            continue   # blank IF patch

        score = ssim_fn(hema_224, dna_224, data_range=1.0)
        scores[i] = score

        if debug_dir is not None and len(debug_triplets) < n_debug and score >= debug_min_ssim:
            debug_triplets.append((he_rgb_224.copy(), hema_224.copy(), dna_224.copy(), score))

    # ── Save debug montage ────────────────────────────────────────────────────
    if debug_dir is not None and debug_triplets:
        debug_dir = Path(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        n   = len(debug_triplets)
        fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
        if n == 1:
            axes = axes[np.newaxis, :]
        col_titles = ["H&E (RGB)", "Hematoxylin (log B/R)", "IF DNA (Hoechst)"]
        for col, title in enumerate(col_titles):
            axes[0, col].set_title(title, fontsize=10, fontweight="bold")
        for row, (rgb, hema, dna, sc) in enumerate(debug_triplets):
            kept  = sc >= debug_min_ssim
            color = "green" if kept else "red"
            axes[row, 0].imshow(np.clip(rgb, 0, 1))
            axes[row, 1].imshow(hema, cmap="gray", vmin=0, vmax=1)
            axes[row, 2].imshow(dna,  cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_ylabel(f"SSIM={sc:.3f}", fontsize=8, rotation=90,
                                    labelpad=4, color=color)
            for col in range(3):
                axes[row, col].axis("off")
        plt.suptitle(f"Alignment debug — first {n} patches  "
                     f"(green=kept SSIM≥{debug_min_ssim:.2f}, red=dropped)", fontsize=11)
        plt.tight_layout()
        out_path = debug_dir / "alignment_debug.png"
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"  [align_debug] saved montage ({n} patches) → {out_path}")

    return scores


# ── p99 computation ───────────────────────────────────────────────────────────

def compute_slide_p99s(
    if_arr, c_ax: int, h_ax: int, w_ax: int,
    channels: list[tuple[int, str, int | None]],
    coords: np.ndarray,
    ps_he: int,
    he_slide, if_slide,
    H_if: int, W_if: int,
    lams: np.ndarray,
    biases: np.ndarray,
    max_patches: int = 2000,
    x_offset: int = 0,
    y_offset: int = 0,
) -> tuple[list[float], list[float]]:

    bboxes, mapped_corners = _precompute_if_bboxes(
        he_slide, if_slide, coords, ps_he, H_if, W_if,
        x_offset=x_offset, y_offset=y_offset,
    )
    order  = sorted(
        (i for i, bb in enumerate(bboxes) if bb is not None),
        key=lambda i: (bboxes[i][1], bboxes[i][0]),
    )
    print(f"    [p99] valid bboxes: {len(order)}/{len(bboxes)}", flush=True)

    dst_pts = np.float32([[0, 0], [224, 0], [0, 224], [224, 224]])

    C     = len(channels)
    hists = [np.zeros(65536, dtype=np.float64) for _ in range(C)]

    for i in order:

        if i % 500 == 1:
            print(f'{i} coords processed', flush=True)

        x0, y0, x1, y1 = bboxes[i]
        region = read_if_region(if_arr, c_ax, h_ax, w_ax, x0, y0, x1, y1)

        """
        1. read_if_region reads the AABB — a rectangle big enough to contain all 4 rotated corners. No pixels from the true patch are missed.
        2. src_pts are the 4 rotated corners expressed in local AABB pixel coordinates — they form a non-rectangular (rotated) quadrilateral inside the rectangle we just read.
        3. cv2.getPerspectiveTransform(src_pts, dst_pts) takes any 4 non-collinear source points (they do not need to be a rectangle) and computes the 3×3 matrix that maps them
        to the 4 corners of the 224×224 square.
        4. cv2.warpPerspective then for each output pixel (in the 224×224 square) inverts M to find exactly where to sample in the AABB image — following the shape of the
        rotated quadrilateral, not the full rectangle.
        """

        vc = mapped_corners[i]
        src_pts = np.float32([
            [vc[j, 0] + x_offset - x0, vc[j, 1] + y_offset - y0]
            for j in range(4)
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        region_warped = cv2.warpPerspective(
            region.transpose(1, 2, 0), M, (224, 224),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
        ).transpose(2, 0, 1)  # (C, 224, 224)

        for ci, (raw_ch, _, af_ch) in enumerate(channels):
            sig = region_warped[raw_ch]
            if af_ch is not None and lams[ci] > 0:
                corrected = np.maximum(sig - lams[ci] * region_warped[af_ch] - biases[ci], 0.0)
            else:
                corrected = np.maximum(sig, 0.0)
            u16 = np.uint16(np.minimum(corrected, 65535)).ravel()
            fg  = u16[u16 > 0]
            if len(fg):
                h, _ = np.histogram(fg, bins=65536, range=(0, 65536))
                hists[ci] += h.astype(np.float64)

    p99s = []
    p10s = []
    for ci, (_, name, _) in enumerate(channels):
        total = hists[ci].sum()
        if total == 0:
            p99s.append(1.0)
            p10s.append(0.0)
            print(f"    {name:<20}  EMPTY → p99=1.0  p10=0.0")
            continue
        cdf     = np.cumsum(hists[ci] / total)
        p99_bin = max(int(np.searchsorted(cdf, 0.999, side="right")), 1)
        p10_bin = int(np.searchsorted(cdf, 0.10,  side="right"))
        p99s.append(float(p99_bin))
        p10s.append(float(p10_bin))
        print(f"    {name:<20}  p99={p99_bin:.1f}  p10={p10_bin:.1f}")

    return p99s, p10s


# ── Token-grid targets ────────────────────────────────────────────────────────

def compute_token_grid_targets(
    if_arr, c_ax: int, h_ax: int, w_ax: int,
    channels: list[tuple[int, str, int | None]],
    coords: np.ndarray,
    ps_he: int,
    p99s: list[float],
    p10s: list[float],
    he_slide, if_slide,
    H_if: int, W_if: int,
    lams: np.ndarray,
    biases: np.ndarray,
    token_grid: int = 16,
    x_offset: int = 0,
    y_offset: int = 0,
) -> np.ndarray:
    N       = len(coords)
    C       = len(channels)
    targets = np.zeros((N, C, token_grid, token_grid), dtype=np.float32)
    token_px = 224 // token_grid

    raw_chs  = [rc for rc, _, _  in channels]
    af_chs   = [af for _,  _, af in channels]
    p99s_arr = np.maximum(np.array(p99s, dtype=np.float32)[:, None, None], 1.0)

    bboxes, mapped_corners = _precompute_if_bboxes(
        he_slide, if_slide, coords, ps_he, H_if, W_if,
        x_offset=x_offset, y_offset=y_offset,
    )
    order  = sorted(
        (i for i, bb in enumerate(bboxes) if bb is not None),
        key=lambda i: (bboxes[i][1], bboxes[i][0]),
    )

    # Destination corners for the perspective warp: canonical 224×224 square.
    # Order must match _precompute_if_bboxes: TL, TR, BL, BR.
    dst_pts = np.float32([[0, 0], [224, 0], [0, 224], [224, 224]])

    for done, orig_i in enumerate(order):
        if done % 200 == 0:
            print(f"    [{done}/{len(order)}] computing token targets…", flush=True)

        x0, y0, x1, y1 = bboxes[orig_i]
        region = read_if_region(if_arr, c_ax, h_ax, w_ax, x0, y0, x1, y1)

        if region.shape[1] < 4 or region.shape[2] < 4:
            continue

        sigs = np.stack([
            np.maximum(region[rc] - lam * region[af] - bias, 0.0)
            if (af is not None and lam > 0)
            else np.maximum(region[rc], 0.0)
            for rc, af, lam, bias in zip(raw_chs, af_chs, lams, biases)
        ])  # (C, H_aabb, W_aabb)

        normed = np.clip(np.log1p(sigs / p99s_arr), 0.0, 1.0)

        # Convert VALIS corners to local AABB pixel coordinates.
        # mapped_corners[:,1] is in VALIS IF space; y_offset shifts to full-IF
        # space, then subtracting y0 (which already includes y_offset) gives
        # the coordinate within the extracted AABB.
        vc = mapped_corners[orig_i]  # (4, 2): TL, TR, BL, BR in VALIS coords
        src_pts = np.float32([
            [vc[j, 0] + x_offset - x0, vc[j, 1] + y_offset - y0]
            for j in range(4)
        ])

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            normed.transpose(1, 2, 0), M, (224, 224),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )  # (224, 224, C)

        targets[orig_i] = (
            warped
            .reshape(token_grid, token_px, token_grid, token_px, C)
            .mean(axis=(1, 3))
            .transpose(2, 0, 1)
        )

    return targets


# ── p99 persistence ───────────────────────────────────────────────────────────

def save_p99s_txt(sample: str, p99s: list[float], p10s: list[float],
                  channels: list[tuple[int, str, int | None]],
                  p99s_txt: Path) -> None:
    p99s_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(p99s_txt, "a") as fh:
        fh.write(f"{sample}\n")
        for (_, name, _), p99, p10 in zip(channels, p99s, p10s):
            fh.write(f"  {name} {p99} {p10}\n")
    print(f"  p99s → {p99s_txt}")


def load_p99s_txt(sample: str, p99s_txt: Path,
                  channels: list[tuple[int, str, int | None]]) -> tuple[list[float], list[float]]:
    all_data: dict[str, dict[str, tuple[float, float]]] = {}
    current: str | None = None
    with open(p99s_txt) as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if not line.startswith(" "):
                current = line.strip()
                all_data[current] = {}
            else:
                parts = line.split()
                if len(parts) >= 2 and current:
                    p99 = float(parts[1])
                    p10 = float(parts[2]) if len(parts) >= 3 else 0.0
                    all_data[current][parts[0]] = (p99, p10)
    sample_data = all_data[sample]
    p99s = [sample_data[name][0] for _, name, _ in channels]
    p10s = [sample_data[name][1] for _, name, _ in channels]
    return p99s, p10s


# ── HDF5 save ─────────────────────────────────────────────────────────────────

def save_dataset(
    out_path: Path,
    coords: np.ndarray,
    p99s: list[float],
    p10s: list[float],
    targets: np.ndarray,
    channels: list[tuple[int, str, int | None]],
    patch_size: int,
    patch_size_level0: int,
    ps_if: int,
    sample: str,
    x_if_offset: int = 0,
    y_if_offset: int = 0,
    token_grid: int = 16,
) -> None:
    N, C, G, _ = targets.shape
    marker_names    = [name    for _, name, _  in channels]
    channel_indices = [ch_idx  for ch_idx, _, _ in channels]
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("coords",          data=coords,  compression="gzip")
        f.create_dataset("p99s",            data=np.array(p99s, dtype=np.float32))
        f.create_dataset("p10s",            data=np.array(p10s, dtype=np.float32))
        f.create_dataset("targets",         data=targets, compression="gzip",
                         chunks=(min(256, N), C, G, G))
        f.create_dataset("channel_indices", data=np.array(channel_indices, dtype=np.int32))
        f.attrs["sample"]            = sample
        f.attrs["marker_names"]      = marker_names
        f.attrs["patch_size"]        = patch_size
        f.attrs["patch_size_level0"] = patch_size_level0
        f.attrs["ps_if"]             = ps_if
        f.attrs["x_if_offset"]       = x_if_offset
        f.attrs["y_if_offset"]       = y_if_offset
        f.attrs["token_grid"]        = token_grid

    mb = out_path.stat().st_size / 1e6
    print(f"\n  Saved → {out_path}  ({mb:.1f} MB)")
    print(f"    /coords   {coords.shape}")
    print(f"    /targets  {targets.shape}  mean={targets.mean():.4f}")
    print(f"    markers   {marker_names}")


# ── Per-sample pipeline ───────────────────────────────────────────────────────

def process_sample(info: dict, args: argparse.Namespace) -> None:
    sample   = info["sample"]
    he_path  = info["he_path"]
    if_path  = info["if_path"]
    zip_path = info["zip_path"]
    n_rois   = info.get("n_rois", 1)
    roi_n    = info.get("roi_n", 1)

    print(f"\n{'='*64}\n  {sample}\n{'='*64}")

    if sample.startswith('MEL01-1'):
        return

    print(f"  H&E : {he_path.name}")
    print(f"  IF  : {if_path.name}")
    print(f"  ZIP : {zip_path.name}")

    # ── CSV inspection ───────────────────────────────────────────────────────
    if args.inspect_csv:
        describe_csv(zip_path)

    channels = parse_csv_channels(zip_path)
    channels = filter_channels(channels, None if args.marker_set == "none" else args.marker_set)
    print(f"  channels ({len(channels)}): {[n for _, n, _ in channels]}")

    valis_dir  = Path(args.valis_dir)  / sample
    job_dir    = Path(args.job_dir)    / sample
    output_dir = Path(args.output_dir)
    out_path   = output_dir / f"{sample}_patch_dataset.h5"
    p99s_txt   = output_dir / "p99s_slide.txt"

    # ── 0a. Single-channel Valis crop when IF contains multiple ROIs ─────────
    if_stem    = if_path.name.split(".")[0]
    split_axis = IF_SPLIT_AXIS.get(if_stem, "h")
    x_frac     = IF_VALIS_X_CROP.get(sample, None)
    if n_rois > 1:
        crop_tag     = "_regcrop_xcrop" if x_frac is not None else "_regcrop"
        if_crop_path = if_path.parent / f"{if_stem}_roi{roi_n}{crop_tag}.tif"
        x_if_offset, y_if_offset = write_if_valis_crop(
            if_path, roi_n, n_rois, channels[0][0], if_crop_path,
            split_axis=split_axis, x_frac=x_frac,
        )
        active_if = if_crop_path
        print(f"  [if_crop] Valis crop: {active_if.name}  "
              f"x_offset={x_if_offset}  y_offset={y_if_offset}")
    else:
        active_if    = if_path
        x_if_offset  = 0
        y_if_offset  = 0

    # ── 0b. Pre-rotate H&E if orientation mismatches IF ──────────────────────
    rot_k = he_rotation_k(he_path, active_if)
    if rot_k:
        rot_label   = {1: "90ccw", 2: "180", 3: "90cw"}[rot_k]
        base        = he_path.name.split(".")[0]
        rot_he_path = he_path.parent / f"{base}_rot{rot_label}.tif"
        prerotate_he(he_path, rot_he_path, rot_k)
        active_he = rot_he_path
        print(f"  [prerotate] using rotated H&E: {active_he.name}")
    else:
        active_he = he_path

    # ── 1. Registration ──────────────────────────────────────────────────────
    mpp_he = get_mpp(he_path)   # original H&E always has OME metadata; rotated copy does not
    print(f"[MPP] H&E = {mpp_he:.4f} µm/px")
    existing_pickle = list(valis_dir.rglob("*.pickle"))
    if existing_pickle:
        print(f"[Valis] Pickle found ({existing_pickle[0].name}) — skipping registration.")
    elif not args.skip_valis:
        run_valis(active_he, active_if, valis_dir, dna_ch=channels[0][0], mpp=mpp_he)
    else:
        print("[Valis] Skipping (--skip_valis); loading existing pickle…")
    he_slide, if_slide = load_slides(valis_dir, active_he.name, active_if.name)
    from valis import registration as valis_reg
    valis_reg.kill_jvm()

    # ── 2. TRIDENT ───────────────────────────────────────────────────────────
    if args.skip_trident:
        h5_files = list(job_dir.rglob("*_patches.h5"))
        if not h5_files:
            raise FileNotFoundError(f"--skip_trident: no coords h5 under {job_dir}")
        coords_h5 = h5_files[0]
        print(f"[TRIDENT] Reusing {coords_h5}")
    else:
        coords_h5 = run_trident(
            active_he, job_dir, mpp_he,
            args.mag, args.patch_size, args.overlap, args.min_tissue,
            args.segmenter, args.seg_thresh, args.gpu,
        )

    coords, patch_size, target_mag = load_trident_coords(coords_h5)
    coords = coords[np.lexsort((coords[:, 0], coords[:, 1]))]

    mpp_if            = get_mpp(if_path)
    patch_size_level0 = round(patch_size * (10.0 / target_mag) / mpp_he)
    ps_if             = round(patch_size * (10.0 / target_mag) / mpp_if)
    print(f"  patch_size_level0={patch_size_level0} px (H&E)  ps_if={ps_if} px (IF)")

    # ── 3. Open IF zarr (always from original full IF) ───────────────────────
    if_arr, c_ax, h_ax, w_ax = open_zarr_level0(if_path)
    H_if = if_arr.shape[h_ax]
    W_if = if_arr.shape[w_ax]
    print(f"  IF shape: {if_arr.shape}  (H={H_if}, W={W_if})  "
          f"x_offset={x_if_offset}  y_offset={y_if_offset}")

    # ── 4. p99s ──────────────────────────────────────────────────────────────
    print(f"\n  Computing p99s…")
    lams, biases = build_af_arrays(channels, args.af_params, slide=sample)
    try:
        p99s, p10s = load_p99s_txt(sample, p99s_txt, channels)
        print("  Loaded cached p99s.")
    except Exception:
        p99s, p10s = compute_slide_p99s(
            if_arr, c_ax, h_ax, w_ax,
            channels, coords, patch_size_level0,
            he_slide, if_slide, H_if, W_if,
            lams=lams, biases=biases, max_patches=args.max_patches,
            x_offset=x_if_offset, y_offset=y_if_offset,
        )
        #save_p99s_txt(sample, p99s, p10s, channels, p99s_txt)

    # ── 5. Token-grid targets ────────────────────────────────────────────────
    print(f"\n  Computing {args.token_grid}×{args.token_grid} targets "
          f"({len(coords)} patches)…")
    targets = compute_token_grid_targets(
        if_arr, c_ax, h_ax, w_ax,
        channels, coords, patch_size_level0,
        p99s, p10s, he_slide, if_slide, H_if, W_if,
        lams=lams, biases=biases, token_grid=args.token_grid,
        x_offset=x_if_offset, y_offset=y_if_offset,
    )

    # ── 6. Filter patches with no IF tissue signal ───────────────────────────
    if args.min_if_signal > 0:
        dna_mean    = targets[:, 0].mean(axis=(-2, -1))
        tissue_mask = dna_mean > args.min_if_signal
        n_before    = len(coords)
        coords      = coords[tissue_mask]
        targets     = targets[tissue_mask]
        print(f"  [if_filter] kept {tissue_mask.sum()}/{n_before} patches "
              f"(DNA mean > {args.min_if_signal})")

    # ── 7. Alignment filter (H&E hematoxylin ↔ IF Hoechst SSIM) ──────────────
    if args.min_alignment_ssim > 0 or args.alignment_debug_dir:
        print(f"\n  Computing alignment scores ({len(coords)} patches)…")
        he_arr_z, _, h_ax_he, w_ax_he = open_zarr_level0(active_he)
        dna_raw_ch = channels[0][0]
        debug_dir = Path(args.alignment_debug_dir) / sample if args.alignment_debug_dir else None
        alignment_scores = compute_alignment_scores(
            he_arr_z, h_ax_he, w_ax_he,
            if_arr, c_ax, h_ax, w_ax,
            dna_raw_ch, coords, patch_size_level0,
            he_slide, if_slide, H_if, W_if,
            x_offset=x_if_offset, y_offset=y_if_offset,
            debug_dir=debug_dir,
            debug_min_ssim=args.min_alignment_ssim,
        )
        scores_path = output_dir / f"{sample}_alignment_scores.npy"
        np.save(scores_path, alignment_scores)
        print(f"  Alignment scores: mean={alignment_scores[alignment_scores>=0].mean():.3f}  "
              f"min={alignment_scores[alignment_scores>=0].min():.3f}  → {scores_path.name}")

        align_mask = alignment_scores >= args.min_alignment_ssim
        n_before   = len(coords)
        coords     = coords[align_mask]
        targets    = targets[align_mask]
        print(f"  [align_filter] kept {align_mask.sum()}/{n_before} patches "
              f"(SSIM >= {args.min_alignment_ssim})")



    # ── 8. Save ──────────────────────────────────────────────────────────────
    if len(coords) == 0:
        print(f"  [warn] no patches remain after filtering — skipping save for {sample}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(
        out_path, coords, p99s, p10s, targets,
        channels, patch_size, patch_size_level0, ps_if,
        sample, x_if_offset=x_if_offset, y_if_offset=y_if_offset,
        token_grid=args.token_grid,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Melanoma CyCIF patch dataset builder"
    )
    parser.add_argument(
        "--samples", default="all",
        help="'all' or comma-separated sample names e.g. MEL08-1-1-ROI1,MEL03-1-1-ROI2",
    )
    parser.add_argument("--list_samples",  action="store_true",
                        help="Print all discoverable sample names and exit")
    parser.add_argument("--inspect_csv",   action="store_true",
                        help="Print full CSV channel layout for each sample")
    parser.add_argument("--skip_valis",    action="store_true")
    parser.add_argument("--skip_trident",  action="store_true")
    parser.add_argument("--patch_size",    type=int,   default=224)
    parser.add_argument("--mag",           type=float, default=20.0)
    parser.add_argument("--overlap",       type=int,   default=0)
    parser.add_argument("--min_tissue",    type=float, default=0.1)
    parser.add_argument("--segmenter",     default="hest",
                        choices=["hest", "grandqc", "otsu"])
    parser.add_argument("--seg_thresh",    type=float, default=0.5)
    parser.add_argument("--gpu",           type=int,   default=1)
    parser.add_argument("--af_params",     default='af_params_melanoma.json',
                        help="Path to AF subtraction JSON with per-marker lambda/bias. "
                             "If omitted, no AF subtraction is applied. "
                             "See build_af_arrays() for the JSON schema.")
    parser.add_argument("--token_grid",    type=int,   default=16,
                        help="Token grid size (UNI2=16)")
    parser.add_argument("--max_patches",   type=int,   default=2000,
                        help="Max patches sampled for p99 estimation")
    parser.add_argument("--min_if_signal", type=float, default=0.04,
                        help="Drop patches where normalised DNA mean < this (0=off)")
    parser.add_argument("--min_alignment_ssim", type=float, default=0.6,
                        help="Drop patches where H&E hematoxylin ↔ IF Hoechst SSIM < this "
                             "(0=off); scores saved to {sample}_alignment_scores.npy")
    parser.add_argument("--alignment_debug_dir", default='melanoma_alignment_debug',
                        help="If set, save a debug montage (H&E | hematoxylin | DNA) "
                             "for the first 20 patches to this directory")
    parser.add_argument("--marker_set",    default="orion_crc",
                        choices=list(MARKER_SETS) + ["none"],
                        help="Restrict to a predefined marker set "
                             f"(available: {list(MARKER_SETS)}; 'none' = all markers)")
    parser.add_argument("--job_dir",       default=str(JOB_DIR))
    parser.add_argument("--output_dir",    default=str(OUTPUT_DIR))
    parser.add_argument("--valis_dir",     default=str(VALIS_DIR))
    parser.add_argument("--data_dir",      default=str(DATA_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    all_samples = discover_samples(data_dir)

    if args.list_samples:
        print(f"Discoverable samples ({len(all_samples)}):")
        for info in all_samples:
            print(f"  {info['sample']}")
            print(f"    H&E : {info['he_path'].name}")
            print(f"    IF  : {info['if_path'].name}")
            print(f"    ZIP : {info['zip_path'].name}")
        return

    if args.samples.lower() == "all":
        to_process = all_samples
    else:
        requested = {s.strip() for s in args.samples.split(",")}
        to_process = [s for s in all_samples if s["sample"] in requested]
        missing = requested - {s["sample"] for s in to_process}
        if missing:
            print(f"[warn] samples not found: {missing}")

    print(f"Processing {len(to_process)} sample(s):")
    for info in to_process:
        print(f"  {info['sample']}")

    for info in to_process:
        try:
            process_sample(info, args)
        except Exception as e:
            import traceback
            print(f"\n  ERROR {info['sample']}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()