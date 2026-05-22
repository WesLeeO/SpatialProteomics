"""
Build patch dataset from JEDI20033 (H&E) + JEDI20034 (MxIF).

JEDI20033.tif              — H&E WSI (60723×76524 RGB uint8, ~0.325 µm/px assumed)
JEDI20034_c{i}_{name}.tif  — 5 single-channel uint8 MxIF TIFFs at same resolution:
  c0_DNA, c1_CD20, c2_CD45, c3_CD3, c4_CD68

Pipeline
--------
1. Write inverted DNA TIFF for Valis registration (OD polarity matches H&E)
2. Valis: register H&E (reference) ↔ inverted DNA
3. TRIDENT: tissue segmentation + patch coords on H&E
4. Map H&E patch coords → MxIF space via Valis transform
5. Compute (N, C, G, G) token-grid targets
6. Save HDF5
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import argparse
import csv as _csv
import subprocess
import numpy as np
import cv2
import h5py
import tifffile
import zarr
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR       = Path("/mnt/ssd1/virtual_proteomics/data/JEDI_201207")
TRIDENT_SCRIPT = Path("TRIDENT/run_batch_of_slides.py")
JOB_DIR        = Path("jedi_trident_output")
OUTPUT_DIR     = Path("jedi_patch_dataset")
VALIS_DIR      = Path("jedi_valis")

HE_TIF = DATA_DIR / "JEDI20033.tif"
MPP_HE = 0.220
MPP_IF = 0.325   # µm/px — assumed (no metadata in either TIFF)

CHANNELS = [
    ("c0_DNA",  "DNA"),
    ("c1_CD20", "CD20"),
    ("c2_CD45", "CD45"),
    ("c3_CD3",  "CD3"),
    ("c4_CD68", "CD68"),
]

def mxif_path(suffix: str) -> Path:
    return DATA_DIR / f"JEDI20034_{suffix}.tif"


# ── H&E corner cleaning ────────────────────────────────────────────────────────

def write_he_white_corners(
    he_path: Path, out_path: Path,
    border_frac: float = 0.20, black_thresh: int = 50,
) -> None:
    """Replace black scanner-padding pixels near image borders with white."""
    if out_path.exists():
        print(f"  [he_clean] {out_path.name} already exists — skipping")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [he_clean] reading {he_path.name}…")
    he = tifffile.imread(str(he_path))          # (H, W, 3) uint8
    H, W = he.shape[:2]

    rows = np.arange(H)[:, None]               # (H, 1)
    cols = np.arange(W)[None, :]               # (1, W)
    border_mask = (
        (rows <  H * border_frac) | (rows >= H * (1 - border_frac)) |
        (cols <  W * border_frac) | (cols >= W * (1 - border_frac))
    )                                           # (H, W) bool
    black = np.all(he < black_thresh, axis=-1) # (H, W) bool
    filled = int((border_mask & black).sum())
    he[border_mask & black] = 255
    print(f"  [he_clean] filled {filled:,} pixels  →  writing {out_path.name}…")
    from valis.slide_io import create_ome_xml
    ome = create_ome_xml((W, H, 1, 1, 3), 'uint8', is_rgb=True,
                         pixel_physical_size_xyu=(MPP_HE, MPP_HE, 'µm'))
    tifffile.imwrite(str(out_path), he, photometric='rgb', compression='lzw',
                     bigtiff=True, description=ome.to_xml().encode('utf-8'))
    mb = out_path.stat().st_size / 1e6
    print(f"  [he_clean] done  ({mb:.0f} MB)")


def write_dna_with_mpp(src: Path, out_path: Path) -> None:
    """Write local copy of DNA TIFF with embedded OME MPP metadata."""
    if out_path.exists():
        print(f"  [dna_mpp] {out_path.name} already exists — skipping")
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [dna_mpp] copying {src.name} with MPP_IF={MPP_IF} µm/px…")
    dna = tifffile.imread(str(src))
    H, W = dna.shape[:2]
    from valis.slide_io import create_ome_xml
    ome = create_ome_xml((W, H, 1, 1, 1), 'uint8', is_rgb=False,
                         pixel_physical_size_xyu=(MPP_IF, MPP_IF, 'µm'))
    tifffile.imwrite(str(out_path), dna, compression='lzw',
                     description=ome.to_xml().encode('utf-8'))
    mb = out_path.stat().st_size / 1e6
    print(f"  [dna_mpp] done  ({mb:.0f} MB)")


# ── Valis registration ─────────────────────────────────────────────────────────

def run_valis(he_path: Path, dna_path: Path, valis_dir: Path) -> None:
    from valis import registration as valis_reg
    from valis.preprocessing import OD, ChannelGetter
    import valis.micro_rigid_registrar as _mrr
    from pqdm.threads import pqdm as _pqdm_threads
    from valis.micro_rigid_registrar import MicroRigidRegistrar
    _mrr.pqdm = _pqdm_threads
    from valis import valtils as _valtils
    _valtils.get_ncpus_available = lambda: 8

    valis_dir.mkdir(parents=True, exist_ok=True)
    processor_dict = {
        he_path.name:  OD,
        dna_path.name: [ChannelGetter, {"channel": 0, "adaptive_eq": True}],
    }
    print("[Valis] Registering…")
    registrar = valis_reg.Valis(
        str(valis_dir), str(valis_dir.parent),
        img_list=[str(he_path), str(dna_path)],
        reference_img_f=str(he_path),
        align_to_reference=True,
        micro_rigid_registrar_cls=MicroRigidRegistrar,
        micro_rigid_registrar_params={"scale": 0.0625, "tile_wh": 256, "roi": "mask"},
    )
    registrar.register(processor_dict=processor_dict)
    print(f"[Valis] Done → {valis_dir}")


def load_slides(valis_dir: Path, he_name: str, dna_name: str):
    from valis import registration as valis_reg
    pickles = list(valis_dir.rglob("*.pickle"))
    if not pickles:
        raise FileNotFoundError(f"No Valis pickle under {valis_dir}")
    reg = valis_reg.load_registrar(str(pickles[0]))
    he_slide = dna_slide = None
    for slide in reg.slide_dict.values():
        name = Path(slide.src_f).name
        if name == he_name:
            he_slide = slide
        elif name == dna_name:
            dna_slide = slide
    if he_slide  is None: raise KeyError(f"{he_name} not in registrar")
    if dna_slide is None: raise KeyError(f"{dna_name} not in registrar")
    return he_slide, dna_slide


# ── TRIDENT ────────────────────────────────────────────────────────────────────

def run_trident(he_path: Path, job_dir: Path, args: argparse.Namespace) -> Path:
    job_dir.mkdir(parents=True, exist_ok=True)
    csv_path = job_dir / "wsi_list.csv"
    with open(csv_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["wsi", "mpp"])
        w.writerow([he_path.name, MPP_HE])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", str(args.gpu))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": visible}
    base = [
        "python", str(TRIDENT_SCRIPT),
        "--wsi_dir",    str(he_path.parent),
        "--job_dir",    str(job_dir),
        "--gpu",        "0",
        "--segmenter",  args.segmenter,
        "--seg_conf_thresh", str(args.seg_thresh),
        "--mag",        str(args.mag),
        "--patch_size", str(args.patch_size),
        "--overlap",    str(args.overlap),
        "--min_tissue_proportion", str(args.min_tissue),
        "--wsi_ext",    ".tif",
        "--custom_list_of_wsis", str(csv_path),
    ]
    print("[TRIDENT] Segmenting…")
    subprocess.run(base + ["--task", "seg"],    check=True, env=env)
    print("[TRIDENT] Extracting coords…")
    subprocess.run(base + ["--task", "coords"], check=True, env=env)
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


# ── MxIF region reader ─────────────────────────────────────────────────────────

def open_mxif_maps() -> list:
    """Open all MxIF channel TIFFs as zarr arrays for lazy tile-level access.

    tifffile.memmap fails on DEFLATE-compressed TIFFs; zarr reads tiles on
    demand and supports the same [y0:y1, x0:x1] slicing interface. The zarr
    store retains a reference to the TiffFile, so no separate handle tracking
    is needed.
    """
    maps = []
    for suffix, name in CHANNELS:
        path = mxif_path(suffix)
        arr = zarr.open(tifffile.TiffFile(str(path)).aszarr(), mode='r')
        maps.append(arr)
        print(f"  [mxif] opened {path.name}  shape={arr.shape}")
    return maps


def read_mxif_region(maps: list, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    """Return (C, h, w) float32 crop from memory-mapped channel TIFFs."""
    return np.stack([ch[y0:y1, x0:x1].astype(np.float32) for ch in maps])


# ── Coord mapping ──────────────────────────────────────────────────────────────

def precompute_mxif_bboxes(he_slide, dna_slide, coords, ps_he, H_mxif, W_mxif):
    tl = coords.astype(float)
    all_corners = np.vstack([tl, tl + [ps_he, 0], tl + [0, ps_he], tl + [ps_he, ps_he]])
    mapped = he_slide.warp_xy_from_to(all_corners, dna_slide)   # (4N, 2)
    N = len(coords)
    mapped = mapped.reshape(4, N, 2).transpose(1, 0, 2)          # (N, 4, 2)

    bboxes = []
    for i in range(N):
        c = mapped[i]
        if np.any(np.isnan(c)):
            bboxes.append(None); continue
        x0 = max(int(np.floor(c[:, 0].min())), 0)
        y0 = max(int(np.floor(c[:, 1].min())), 0)
        x1 = min(int(np.ceil(c[:, 0].max())),  W_mxif)
        y1 = min(int(np.ceil(c[:, 1].max())),  H_mxif)
        bboxes.append(None if x1 <= x0 or y1 <= y0 else (x0, y0, x1, y1))
    return bboxes, mapped


# ── p99 / p10 ──────────────────────────────────────────────────────────────────

def compute_p99s(maps, coords, ps_he, he_slide, dna_slide,
                 H_mxif, W_mxif, max_patches=2000):
    bboxes, mapped = precompute_mxif_bboxes(
        he_slide, dna_slide, coords, ps_he, H_mxif, W_mxif)
    order = sorted((i for i, bb in enumerate(bboxes) if bb is not None),
                   key=lambda i: (bboxes[i][1], bboxes[i][0]))
    rng = np.random.default_rng(0)
    sel = set(rng.choice(order, min(max_patches, len(order)), replace=False).tolist())

    C = len(CHANNELS)
    dst_pts = np.float32([[0, 0], [224, 0], [0, 224], [224, 224]])
    hists = [np.zeros(256, dtype=np.float64) for _ in range(C)]

    for i in order:
        if i not in sel:
            continue
        x0, y0, x1, y1 = bboxes[i]
        region = read_mxif_region(maps, x0, y0, x1, y1)   # (C, h, w)
        vc = mapped[i]
        src_pts = np.float32([[vc[j, 0] - x0, vc[j, 1] - y0] for j in range(4)])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            region.transpose(1, 2, 0), M, (224, 224),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
        ).transpose(2, 0, 1)   # (C, 224, 224)
        for ci in range(C):
            fg = warped[ci].ravel().astype(np.uint8)
            fg = fg[fg > 0]
            if len(fg):
                h, _ = np.histogram(fg, bins=256, range=(0, 256))
                hists[ci] += h

    p99s, p20s = [], []
    for ci, (_, name) in enumerate(CHANNELS):
        total = hists[ci].sum()
        if total == 0:
            p99s.append(1.0); p20s.append(0.0)
            print(f"    {name:<10}  EMPTY")
            continue
        cdf = np.cumsum(hists[ci] / total)
        p99s.append(float(max(int(np.searchsorted(cdf, 0.99, side='right')), 1)))
        p20s.append(float(int(np.searchsorted(cdf, 0.20,  side='right'))))
        print(f"    {name:<10}  p99={p99s[-1]:.1f}  p20={p20s[-1]:.1f}")
    return p99s, p20s


# ── Token-grid targets ─────────────────────────────────────────────────────────

def compute_token_grid_targets(maps, coords, ps_he, p99s, p20s,
                                he_slide, dna_slide, H_mxif, W_mxif,
                                token_grid=16):
    N = len(coords)
    C = len(CHANNELS)
    targets  = np.zeros((N, C, token_grid, token_grid), dtype=np.float32)
    token_px = 224 // token_grid
    dst_pts  = np.float32([[0, 0], [224, 0], [0, 224], [224, 224]])

    p99s_arr = np.array(p99s, dtype=np.float32)[:, None, None]
    #p10s_arr = np.array(p10s, dtype=np.float32)[:, None, None]
    #ranges   = np.maximum(p99s_arr - p10s_arr, 1.0)

    bboxes, mapped = precompute_mxif_bboxes(
        he_slide, dna_slide, coords, ps_he, H_mxif, W_mxif)
    order = sorted((i for i, bb in enumerate(bboxes) if bb is not None),
                   key=lambda i: (bboxes[i][1], bboxes[i][0]))
    print(f"  valid bboxes: {len(order)}/{N}")

    for done, i in enumerate(order):
        if done % 500 == 0:
            print(f"  [{done}/{len(order)}] token targets…", flush=True)
        x0, y0, x1, y1 = bboxes[i]
        region = read_mxif_region(maps, x0, y0, x1, y1)
        if region.shape[1] < 4 or region.shape[2] < 4:
            continue

        normed = np.clip(
            np.log1p(region / p99s_arr), 0.0, 1.0
        )

        vc = mapped[i]
        src_pts = np.float32([[vc[j, 0] - x0, vc[j, 1] - y0] for j in range(4)])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(
            normed.transpose(1, 2, 0), M, (224, 224),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )   # (224, 224, C)

        targets[i] = (
            warped
            .reshape(token_grid, token_px, token_grid, token_px, C)
            .mean(axis=(1, 3))
            .transpose(2, 0, 1)
        )

    return targets


# ── HDF5 save ──────────────────────────────────────────────────────────────────

def save_dataset(out_path, coords, p99s, p20s, targets,
                 patch_size, patch_size_level0, token_grid):
    N, C, G, _ = targets.shape
    with h5py.File(str(out_path), "w") as f:
        f.create_dataset("coords",  data=coords,  compression="gzip")
        f.create_dataset("p99s",    data=np.array(p99s,  dtype=np.float32))
        f.create_dataset("p20s",    data=np.array(p20s,  dtype=np.float32))
        f.create_dataset("targets", data=targets, compression="gzip",
                         chunks=(min(256, N), C, G, G))
        f.attrs["sample"]            = "JEDI20034"
        f.attrs["marker_names"]      = [name for _, name in CHANNELS]
        f.attrs["patch_size"]        = patch_size
        f.attrs["patch_size_level0"] = patch_size_level0
        f.attrs["mpp_he"]               = MPP_HE
        f.attrs["mpp_if"]               = MPP_IF
        f.attrs["token_grid"]        = token_grid
    mb = out_path.stat().st_size / 1e6
    print(f"\n  Saved → {out_path}  ({mb:.1f} MB)")
    print(f"    /coords   {coords.shape}")
    print(f"    /targets  {targets.shape}  mean={targets.mean():.4f}")
    print(f"    markers   {[n for _, n in CHANNELS]}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="JEDI patch dataset builder")
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
    parser.add_argument("--token_grid",    type=int,   default=16)
    parser.add_argument("--max_patches",   type=int,   default=2000)
    parser.add_argument("--min_if_signal", type=float, default=0.01,
                        help="Drop patches where normalised DNA mean < this (0=off)")
    parser.add_argument("--job_dir",       default=str(JOB_DIR))
    parser.add_argument("--output_dir",    default=str(OUTPUT_DIR))
    parser.add_argument("--valis_dir",     default=str(VALIS_DIR))
    parser.add_argument("--overwrite",     action="store_true")
    args = parser.parse_args()

    out_path = Path(args.output_dir) / "JEDI20034_patch_dataset.h5"
    if out_path.exists() and not args.overwrite:
        print(f"Output already exists: {out_path}  (use --overwrite to rerun)")
        return

    valis_dir = Path(args.valis_dir)
    job_dir   = Path(args.job_dir)

    with tifffile.TiffFile(str(mxif_path("c0_DNA"))) as t:
        H_mxif, W_mxif = t.pages[0].shape[:2]

    patch_size_level0 = round(args.patch_size * 0.5 / MPP_HE)   # 20× equivalent

    print("=" * 60)
    print(f"  JEDI patch dataset builder")
    print(f"  H&E  : {HE_TIF.name}  (60723×76524)  → clean: JEDI20033_clean.tif")
    print(f"  MxIF : JEDI20034  ({H_mxif}×{W_mxif})")
    print(f"  channels : {[n for _, n in CHANNELS]}")
    print(f"  mpp  : {MPP_HE} µm/px")
    print(f"  patch_size_level0 : {patch_size_level0} px")
    print("=" * 60)

    # ── 1. Prepare Valis inputs (embedded MPP metadata) ───────────────────────
    valis_dir.mkdir(parents=True, exist_ok=True)
    he_clean  = valis_dir / "JEDI20033_clean.tif"
    dna_local = valis_dir / "JEDI20034_c0_DNA.tif"
    write_he_white_corners(HE_TIF, he_clean)
    write_dna_with_mpp(mxif_path("c0_DNA"), dna_local)

    # ── 2. Valis registration ──────────────────────────────────────────────────
    existing_pickle = list(valis_dir.rglob("*.pickle"))
    if existing_pickle:
        print(f"[Valis] Pickle found — skipping registration.")
    elif not args.skip_valis:
        run_valis(he_clean, dna_local, valis_dir)
    else:
        print("[Valis] --skip_valis set.")

    he_slide, dna_slide = load_slides(valis_dir, he_clean.name, dna_local.name)
    from valis import registration as valis_reg
    valis_reg.kill_jvm()

    # ── 3. TRIDENT ─────────────────────────────────────────────────────────────
    if args.skip_trident:
        h5_files = list(job_dir.rglob("*_patches.h5"))
        if not h5_files:
            raise FileNotFoundError(f"--skip_trident: no coords h5 under {job_dir}")
        coords_h5 = h5_files[0]
        print(f"[TRIDENT] Reusing {coords_h5}")
    else:
        coords_h5 = run_trident(HE_TIF, job_dir, args)

    coords, patch_size, target_mag = load_trident_coords(coords_h5)
    coords = coords[np.lexsort((coords[:, 0], coords[:, 1]))]

    # ── 4. Open MxIF memory maps ───────────────────────────────────────────────
    print("\nOpening MxIF memory maps…")
    maps = open_mxif_maps()

    # ── 5. p99s ────────────────────────────────────────────────────────────────
    print("\nComputing p99s…")
    p99s, p20s = compute_p99s(
        maps, coords, patch_size_level0,
        he_slide, dna_slide, H_mxif, W_mxif, args.max_patches,
    )

    # ── 6. Token-grid targets ──────────────────────────────────────────────────
    print(f"\nComputing {args.token_grid}×{args.token_grid} targets "
          f"({len(coords)} patches)…")
    targets = compute_token_grid_targets(
        maps, coords, patch_size_level0, p99s, p20s,
        he_slide, dna_slide, H_mxif, W_mxif, args.token_grid,
    )

    # ── 7. Filter patches with no IF tissue ────────────────────────────────────
    if args.min_if_signal > 0:
        dna_mean = targets[:, 0].mean(axis=(-2, -1))
        mask     = dna_mean > args.min_if_signal
        print(f"  [if_filter] kept {mask.sum()}/{len(coords)} patches "
              f"(DNA mean > {args.min_if_signal})")
        coords  = coords[mask]
        targets = targets[mask]

    # ── 8. Save ────────────────────────────────────────────────────────────────
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    save_dataset(
        out_path, coords, p99s, p20s, targets,
        patch_size, patch_size_level0, args.token_grid,
    )


if __name__ == "__main__":
    main()
