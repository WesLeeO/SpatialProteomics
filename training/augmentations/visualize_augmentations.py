"""
Sanity-check the training augmentations in dataset_orion_reg.py — visually.

Pulls real ORION H&E patches through OrionSpatialDataset (augment=False, clean
originals), then applies each augmentation exactly as __getitem__ does and renders
three figures. Augmentations act on the H&E image only, so this only shows H&E:

  1. <out>_isolated.png   — each augmentation in ISOLATION (forced on), one row per
                            sample patch. "Does each transform look like a plausible
                            H&E, not a destroyed image?"
  2. <out>_geometric.png  — the geometric variants (orig / hflip / vflip / rot90/180/270).
  3. <out>_pipeline.png   — the FULL pipeline sampled at the real training
                            probabilities (p_geom/p_hed/p_bc/p_blur/p_noise), several
                            draws of the same patch. "Reasonable variety, or too aggressive?"

Augmentation primitives are imported from dataset_orion_reg so this stays faithful to
training; the geometric op is replicated here (it lives inline in __getitem__).

Usage
-----
  python visualize_augmentations.py                      # defaults: first slide
  python visualize_augmentations.py --slide CRC02 --n 5 --seed 0
"""
import argparse
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_orion_reg import OrionSpatialDataset, _hed_jitter, _photometric_jitter

H5_DIR   = "datasets/orion_crc_patch_dataset_benchmark"
TIFF_DIR = "/mnt/ssd1/virtual_proteomics/data/ORION_CRC"


# ── geometric ops on the H&E (mirror dataset_orion_reg.__getitem__) ──────────────
def geom_variants(patch_u8: np.ndarray):
    """All geometric variants applied to the H&E, as training does to the centre patch."""
    out = [("orig", patch_u8)]
    out.append(("hflip", patch_u8[:, ::-1, :].copy()))
    out.append(("vflip", patch_u8[::-1, :, :].copy()))
    for k in (1, 2, 3):
        out.append((f"rot{90*k}", np.rot90(patch_u8, k, axes=(0, 1)).copy()))
    return out


def random_geom(patch_u8: np.ndarray):
    """One random geometric op (flip or rotation), as __getitem__ picks per call."""
    choice = np.random.randint(3)
    if choice == 0:
        return "hflip", patch_u8[:, ::-1, :].copy()
    if choice == 1:
        return "vflip", patch_u8[::-1, :, :].copy()
    k = np.random.randint(1, 4)
    return f"rot{90*k}", np.rot90(patch_u8, k, axes=(0, 1)).copy()


# ── raw patch loader (uint8, pre-normalisation) — mirrors __getitem__ loading ────
def load_raw_patch(ds: OrionSpatialDataset, idx: int) -> np.ndarray:
    slide_idx, x, y, psz = ds.patch_map[idx]
    arr   = ds._open_arr(slide_idx)
    patch = np.array(arr[y:y + psz, x:x + psz, :])
    patch = cv2.resize(patch, (ds.patch_size, ds.patch_size), interpolation=cv2.INTER_LINEAR)
    return patch.astype(np.uint8)


def main(args):
    np.random.seed(args.seed)
    ds = OrionSpatialDataset(H5_DIR, TIFF_DIR, augment=False,
                             slide_names=[args.slide] if args.slide else None,
                             num_slides=0 if args.slide else 1)
    if len(ds) == 0:
        raise SystemExit("No patches loaded — check --slide / dataset dirs.")
    print(f"{len(ds)} patches loaded")

    rng  = np.random.default_rng(args.seed)
    idxs = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False).tolist()
    out  = Path(args.out)

    # ── Figure 1: each augmentation isolated ─────────────────────────────────────
    cols = ["original", "HED jitter", "brightness/contrast", "Gaussian blur",
            "Gaussian noise", "geometric"]
    fig, axes = plt.subplots(len(idxs), len(cols),
                             figsize=(2.4 * len(cols), 2.4 * len(idxs)), squeeze=False)
    for r, idx in enumerate(idxs):
        p0 = load_raw_patch(ds, idx)
        gname, gimg = random_geom(p0)
        variants = [
            ("original", p0),
            ("HED jitter", _hed_jitter(p0)),
            ("brightness/contrast", _photometric_jitter(p0, p_bc=1.0, p_blur=0.0, p_noise=0.0)),
            ("Gaussian blur",       _photometric_jitter(p0, p_bc=0.0, p_blur=1.0, p_noise=0.0)),
            ("Gaussian noise",      _photometric_jitter(p0, p_bc=0.0, p_blur=0.0, p_noise=1.0)),
            (f"geometric: {gname}", gimg),
        ]
        for c, (name, img) in enumerate(variants):
            axes[r, c].imshow(img)
            axes[r, c].axis("off")
            axes[r, c].set_title(cols[c] if r == 0 else name, fontsize=9)
    fig.suptitle("Augmentations in isolation (each forced on)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    f1 = out.with_name(out.stem + "_isolated.png")
    plt.savefig(f1, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Saved → {f1}")

    # ── Figure 2: geometric variants ─────────────────────────────────────────────
    idx = idxs[0]
    p0  = load_raw_patch(ds, idx)
    variants = geom_variants(p0)
    fig, axes = plt.subplots(1, len(variants),
                             figsize=(2.4 * len(variants), 2.8), squeeze=False)
    for c, (name, pv) in enumerate(variants):
        axes[0, c].imshow(pv); axes[0, c].set_title(name, fontsize=10); axes[0, c].axis("off")
    fig.suptitle(f"Geometric variants (patch {idx})", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    f2 = out.with_name(out.stem + "_geometric.png")
    plt.savefig(f2, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Saved → {f2}")

    # ── Figure 3: full pipeline at training probabilities ────────────────────────
    idx = idxs[0]
    p0  = load_raw_patch(ds, idx)
    ncol = 8
    fig, axes = plt.subplots(1, ncol + 1, figsize=(2.2 * (ncol + 1), 2.6), squeeze=False)
    axes[0, 0].imshow(p0); axes[0, 0].set_title("original", fontsize=10); axes[0, 0].axis("off")
    for c in range(1, ncol + 1):
        # Mirror dataset_orion_reg.__getitem__ ordering, but log which ops fire so each
        # column self-documents. The photometric block replicates _photometric_jitter's
        # internal draws (same np.random call order → identical result) to expose them.
        p, ops = p0.copy(), []
        if np.random.random() < args.p_geom:
            gname, p = random_geom(p); ops.append(gname)
        if np.random.random() < args.p_hed:
            p = _hed_jitter(p); ops.append("HED")
        out_f = p.astype(np.float32)
        if np.random.random() < args.p_bc:
            alpha = 1.0 + np.random.uniform(-0.2, 0.2)
            beta  = np.random.uniform(-0.2, 0.2) * 255.0
            m = out_f.mean(); out_f = alpha * (out_f - m) + m + beta
            ops.append(f"bc α{alpha:.2f} β{beta/255:+.2f}")
        out_f = np.clip(out_f, 0, 255).astype(np.uint8)
        if np.random.random() < args.p_blur:
            sig = np.random.uniform(0.1, 1.5)
            out_f = cv2.GaussianBlur(out_f, (7, 7), sigmaX=sig); ops.append(f"blur σ{sig:.1f}")
        if np.random.random() < args.p_noise:
            std = np.random.uniform(0.02, 0.05) * 255.0
            out_f = np.clip(out_f.astype(np.float32) + np.random.normal(0, std, out_f.shape),
                            0, 255).astype(np.uint8); ops.append("noise")
        p = out_f
        title = f"aug {c}\n" + ("\n".join(ops) if ops else "none")
        axes[0, c].imshow(p); axes[0, c].axis("off"); axes[0, c].set_title(title, fontsize=7)
    fig.suptitle(f"Full pipeline @ training p (geom={args.p_geom} hed={args.p_hed} "
                 f"bc={args.p_bc} blur={args.p_blur} noise={args.p_noise})", fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    f3 = out.with_name(out.stem + "_pipeline.png")
    plt.savefig(f3, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"Saved → {f3}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slide", default="", help="slide base name (e.g. CRC01); blank = first slide")
    ap.add_argument("--n", type=int, default=4, help="number of sample patches in the isolated figure")
    ap.add_argument("--out", default="augmentation_check.png")
    ap.add_argument("--seed", type=int, default=0)
    # training probabilities (defaults match dataset_orion_reg.OrionSpatialDataset)
    ap.add_argument("--p_geom", type=float, default=0.7)
    ap.add_argument("--p_hed",  type=float, default=0.4)
    ap.add_argument("--p_bc",   type=float, default=0.5)
    ap.add_argument("--p_blur", type=float, default=0.1)
    ap.add_argument("--p_noise", type=float, default=0.1)
    main(ap.parse_args())
