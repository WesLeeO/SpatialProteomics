import argparse
import h5py
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

ROOT     = Path("/mnt/ssd1/virtual_proteomics/data/immunoatlas_NOLN210920")
H5_PATH  = Path("datasets/immunoatlas_NOLN210920_trident_output/40.0x_224px_0px_overlap/patches/HandE_RGB_patches.h5")

TMA_COLS   = 10
TMA_ROWS   = 7
TMA_W      = 9600
TMA_H      = 5040
CELL_W     = TMA_W // TMA_COLS   # 960  (TMA px per core)
CELL_H     = TMA_H // TMA_ROWS   # 720
WEBP_SCALE = 2.0                  #   per-core webp files are stored at exactly 2× the resolution of the full TMA image.


def core_bbox_tma(core_name):
    """Returns (x0, y0, x1, y1) of the core cell in full-TMA pixel space."""
    idx = int(core_name.replace("core", "")) - 1
    col, row = idx % TMA_COLS, idx // TMA_COLS
    return col * CELL_W, row * CELL_H, (col + 1) * CELL_W, (row + 1) * CELL_H


def patches_for_core(coords, patch_size, core_name):
    """
    Filters coords to those whose centre falls inside the core's TMA cell,
    and returns them in webp-local pixel space.
    """
    x0, y0, x1, y1 = core_bbox_tma(core_name)
    half = patch_size / 2.0
    webp_patches = []
    for px, py in coords:
        if x0 <= px + half < x1 and y0 <= py + half < y1:
            lx = int((int(px) - x0) * WEBP_SCALE)
            ly = int((int(py) - y0) * WEBP_SCALE)
            pw = int(patch_size * WEBP_SCALE)
            webp_patches.append((lx, ly, lx + pw, ly + pw))
    return webp_patches


def visualise_core(core_name, coords, patch_size, out_dir):
    composite = next(ROOT.glob(f"{core_name}_*/composite.webp"), None)
    if composite is None:
        print(f"  [{core_name}] composite.webp not found — skipping")
        return

    img = Image.open(composite).convert("RGB")
    draw = ImageDraw.Draw(img)

    boxes = patches_for_core(coords, patch_size, core_name)
    for (bx0, by0, bx1, by1) in boxes:
        draw.rectangle([bx0, by0, bx1, by1], outline="red", width=2)

    out_path = Path(out_dir) / f"{core_name}_patches.jpg"
    img.save(out_path, quality=85)
    print(f"  [{core_name}] {len(boxes)} patches → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5",       type=str, default=str(H5_PATH))
    parser.add_argument("--cores",    type=str, nargs="+",
                        default=["core001", "core004"],
                        help="Core names to visualise")
    parser.add_argument("--out_dir",  type=str, default="sanity_webp")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True)

    with h5py.File(args.h5, "r") as f:
        key = "coords" if "coords" in f else list(f.keys())[0]
        coords     = f[key][:]
        patch_size = int(f[key].attrs.get("patch_size", 224))
    print(f"Loaded {len(coords)} coords (patch_size={patch_size})")

    for core_name in args.cores:
        visualise_core(core_name, coords, patch_size, args.out_dir)


if __name__ == "__main__":
    main()