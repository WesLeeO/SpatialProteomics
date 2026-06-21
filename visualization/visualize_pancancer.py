"""
Visualize pancancer (CODEX TMA) patches: H&E side-by-side with protein channels.

Reads a per-core HDF5 built by build_patch_dataset_pancancer.py, loads the
exported H&E RGB TIFF and the bestFocus protein hyperstack, then renders
random patches with optional 16×16 token grids for noise inspection.

Normalisation uses global p99s stored in the HDF5 (pooled across all cores
in the TMA at build time): clip(x / global_p99, 0, 1).

Usage:
    python visualize_pancancer.py --tma CRC_TMA_A --core reg001_X01_Y01
    python visualize_pancancer.py --tma CRC_TMA_A --core reg001_X01_Y01 \\
        --n_patches 8 --markers Hoechst CD20 CD3 --show_tokens
    python visualize_pancancer.py --list_cores --tma CRC_TMA_A
"""

import argparse
import numpy as np
import cv2
import h5py
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from build_patch_dataset_pancancer import (
    DATA_ROOT, JOB_DIR, TMAS,
    parse_channel_names,
    best_focus_z,
    load_hyperstack,
    export_he_rgb,
    discover_cores,
)

OUTPUT_DIR    = Path("visualization_out/pancancer")
PATCH_DATASET = Path("datasets/pancancer_patch_dataset")
MODEL_SIZE    = 224


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_he_rgb(tma_key: str, core_id: str, z_idx: int) -> np.ndarray:
    """Load (or export) the uint8 RGB H&E TIFF for this core."""
    he_rgb_tif = Path(JOB_DIR) / tma_key / core_id / f"{core_id}.tif"
    if not he_rgb_tif.exists():
        print(f"  Exporting H&E RGB → {he_rgb_tif} …")
        he_raw = DATA_ROOT / f"{TMAS[tma_key]}_HandE" / f"{core_id}.tif"
        if not he_raw.exists():
            raise FileNotFoundError(f"H&E source not found: {he_raw}")
        he_rgb_tif.parent.mkdir(parents=True, exist_ok=True)
        export_he_rgb(he_raw, z_idx, he_rgb_tif)
    return tifffile.imread(str(he_rgb_tif))   # (H, W, 3) uint8


def crop_he(he: np.ndarray, x: int, y: int, psz: int) -> np.ndarray:
    H, W = he.shape[:2]
    patch = he[y:min(y + psz, H), x:min(x + psz, W)]
    if patch.ndim == 3 and patch.shape[2] > 3:
        patch = patch[:, :, :3]
    return cv2.resize(patch, (MODEL_SIZE, MODEL_SIZE),
                      interpolation=cv2.INTER_LINEAR)


def crop_protein(hs: np.ndarray,
                 t: int, c: int,
                 x: int, y: int, psz: int,
                 p99: float) -> np.ndarray:
    H, W = hs.shape[2], hs.shape[3]
    region = hs[t, c, y:min(y + psz, H), x:min(x + psz, W)].astype(np.float32)
    resized = cv2.resize(region, (MODEL_SIZE, MODEL_SIZE),
                         interpolation=cv2.INTER_LINEAR)
    return np.clip(resized / max(p99, 1.0), 0.0, 1.0)


# ── Main ──────────────────────────────────────────────────────────────────────

def visualize(args: argparse.Namespace) -> None:
    tma_key = args.tma
    core_id = args.core

    # ── Load HDF5 ────────────────────────────────────────────────────────────
    h5_path = PATCH_DATASET / tma_key / f"{core_id}_patch_dataset.h5"
    if not h5_path.exists():
        raise FileNotFoundError(
            f"HDF5 not found: {h5_path}\n"
            f"Run build_patch_dataset_pancancer.py first."
        )

    with h5py.File(h5_path) as f:
        coords      = f["coords"][:]                  # (N, 2) int64
        p99s        = f["p99s"][:]                    # (C,) float32
        targets     = f["targets"][:] if args.show_tokens else None
        marker_names = list(f.attrs["marker_names"])
        patch_size_level0 = int(f.attrs["patch_size_level0"])
        token_grid        = int(f.attrs.get("token_grid", 16))

    print(f"  {len(coords)} patches  patch_size_level0={patch_size_level0}")
    print(f"  Markers ({len(marker_names)}): {marker_names}")
    print(f"  Global p99s (from HDF5): { {n: f'{p99s[i]:.1f}' for i, n in enumerate(marker_names)} }")

    # ── Resolve markers ───────────────────────────────────────────────────────
    sel_names = args.markers if args.markers else marker_names
    sel = []
    for name in sel_names:
        if name in marker_names:
            sel.append((marker_names.index(name), name))
        else:
            print(f"  [warn] '{name}' not in dataset — skipping")
    if not sel:
        raise ValueError("No valid markers selected")

    # ── Load images ───────────────────────────────────────────────────────────
    channels_txt = DATA_ROOT / f"{TMAS[tma_key]}_hyperstacks" / "channelNames.txt"
    channels     = parse_channel_names(channels_txt)   # [(t, c, name), ...]
    name_to_tc   = {name: (t, c) for t, c, name in channels}

    hs_path, z_idx = best_focus_z(tma_key, core_id)
    print(f"  Loading hyperstack {hs_path.name}  (Z={z_idx}) …")
    hs = load_hyperstack(hs_path, z_idx)   # (T_rounds, 4, H, W) uint16
    print(f"  Hyperstack shape: {hs.shape}")

    print(f"  Loading H&E …")
    he = load_he_rgb(tma_key, core_id, z_idx)   # (H, W, 3) uint8
    print(f"  H&E shape: {he.shape}")

    # ── Random patch selection ────────────────────────────────────────────────
    rng  = np.random.default_rng(args.seed)
    n    = min(args.n_patches, len(coords))
    pick = rng.choice(len(coords), n, replace=False)
    pick.sort()

    n_marker_cols = len(sel) * (2 if args.show_tokens else 1)
    n_cols = 1 + n_marker_cols

    fig, axes = plt.subplots(n, n_cols,
                             figsize=(n_cols * 2.5, n * 2.5),
                             squeeze=False)

    # column headers
    axes[0, 0].set_title("H&E", fontsize=9, fontweight="bold")
    col = 1
    for _, name in sel:
        axes[0, col].set_title(name, fontsize=9, fontweight="bold")
        col += 1
        if args.show_tokens:
            axes[0, col].set_title(f"{name}\n(tokens)", fontsize=9, fontweight="bold")
            col += 1

    for row_i, patch_idx in enumerate(pick):
        x, y = int(coords[patch_idx, 0]), int(coords[patch_idx, 1])

        # H&E
        he_patch = crop_he(he, x, y, patch_size_level0)
        axes[row_i, 0].imshow(he_patch)
        axes[row_i, 0].set_ylabel(f"#{patch_idx}", fontsize=7)

        col = 1
        for mi, name in sel:
            tc = name_to_tc.get(name)
            if tc is None:
                normed = np.zeros((MODEL_SIZE, MODEL_SIZE), dtype=np.float32)
                print(f"  [warn] '{name}' not in channelNames.txt")
            else:
                t_idx, c_idx = tc
                normed = crop_protein(hs, t_idx, c_idx, x, y,
                                      patch_size_level0, float(p99s[mi]))

            axes[row_i, col].imshow(normed, cmap="gray", vmin=0, vmax=1)
            col += 1

            if args.show_tokens:
                tok = targets[patch_idx, mi]   # (G, G)
                axes[row_i, col].imshow(
                    np.clip(tok, 0, 1), cmap="gray", vmin=0, vmax=1,
                    interpolation="nearest",
                )
                axes[row_i, col].set_xlabel(f"{tok.mean():.3f}", fontsize=6, labelpad=1)
                col += 1

    for ax in axes.ravel():
        ax.axis("off")

    marker_tag = "-".join(name for _, name in sel[:8])
    suffix     = "_tokens" if args.show_tokens else ""
    plt.suptitle(f"{tma_key}  /  {core_id}", fontsize=11, y=1.002)
    plt.tight_layout()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{tma_key}_{core_id}_{marker_tag}{suffix}_n{n}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize pancancer CODEX TMA patches"
    )
    parser.add_argument("--tma",         required=False,
                        choices=list(TMAS), default="CRC_TMA_A",
                        help=f"TMA key: {list(TMAS)}")
    parser.add_argument("--core",        default=None,
                        help="Core ID, e.g. reg001_X01_Y01 (random if omitted)")
    parser.add_argument("--list_cores",  action="store_true",
                        help="Print available cores for --tma and exit")
    parser.add_argument("--n_patches",   type=int,   default=8)
    parser.add_argument("--markers",     nargs="*",  default=None,
                        help="Protein markers to display (default: all)")
    parser.add_argument("--show_tokens", action="store_true")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    if args.list_cores:
        cores = discover_cores(args.tma)
        print(f"{len(cores)} cores in {args.tma}:")
        for c in cores:
            print(f"  {c}")
        return

    if args.core is None:
        cores = discover_cores(args.tma)
        rng   = np.random.default_rng(args.seed)
        args.core = rng.choice(cores)
        print(f"  Randomly selected core: {args.core}")

    visualize(args)


if __name__ == "__main__":
    main()