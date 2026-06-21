#!/usr/bin/env python3
"""
Pixel-level intensity histograms per biomarker across orion_crc_patch_dataset_reg.
Each patch (N, C, 16, 16) contributes 256 pixel samples per channel.
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATASET_DIR = Path("/home/wesley/spatial_proteomics/hemit_patch_dataset")
H5_FILES    = sorted(DATASET_DIR.glob("*.h5"))
OUTPUT_DIR  = Path("/home/wesley/spatial_proteomics/biomarker_stats/hemit")
OUTPUT_DIR.mkdir(exist_ok=True)

N_BINS = 100


def main():
    marker_names: list[str] = []
    # Accumulate flat pixel arrays per marker across all slides
    token_vals: dict[int, list[np.ndarray]] = {}

    print("Loading slides...")
    for h5_path in H5_FILES:
        with h5py.File(h5_path, "r") as f:
            markers = list(f.attrs["marker_names"])
            targets = f["targets"][:]   # (N, C, 16, 16)
        slide = h5_path.stem.replace("_patch_dataset", "")
        n_patches, n_ch, H, W = targets.shape
        print(f"  {slide}: {n_patches:,} patches  →  {n_patches * H * W:,} tokens/marker")
        marker_names = markers
        # targets reshaped to (C, N*H*W)
        flat = targets.reshape(n_patches, n_ch, H * W)  # (N, C, 256)
        for m_idx in range(n_ch):
            token_vals.setdefault(m_idx, []).append(flat[:, m_idx, :].ravel())

    n_markers = len(marker_names)

    # Concatenate across slides once
    all_tokens = {m: np.concatenate(arrs) for m, arrs in token_vals.items()}

    # ── Print & save statistics ───────────────────────────────────────────────
    hdr = (f"{'Marker':<14} {'n_tokens':>12} {'mean':>8} {'std':>8} "
           f"{'median':>8} {'p5':>8} {'p25':>8} {'p75':>8} {'p95':>8} "
           f"{'min':>8} {'max':>8}")
    sep = "-" * 108
    print(f"\n{hdr}")
    print(sep)
    rows = []
    for m_idx, marker in enumerate(marker_names):
        v = all_tokens[m_idx]
        p5, p25, p75, p95 = np.percentile(v, [5, 25, 75, 95])
        row = (f"{marker:<14} {len(v):>12,} {v.mean():>8.4f} {v.std():>8.4f} "
               f"{np.median(v):>8.4f} {p5:>8.4f} {p25:>8.4f} {p75:>8.4f} {p95:>8.4f} "
               f"{v.min():>8.4f} {v.max():>8.4f}")
        print(row)
        rows.append(row)
    txt_path = OUTPUT_DIR / "stats.txt"
    txt_path.write_text("\n".join([hdr, sep] + rows) + "\n")
    print(f"Saved → {txt_path}")

    # ── Histograms ────────────────────────────────────────────────────────────
    ncols = 4
    nrows = int(np.ceil(n_markers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.0))
    axes = axes.flatten()

    for m_idx, marker in enumerate(marker_names):
        ax = axes[m_idx]
        v = all_tokens[m_idx]
        ax.hist(v, bins=N_BINS, range=(0, 1), color="steelblue", edgecolor="none", log=True)
        ax.axvline(v.mean(),   color="red",  lw=1.5, label=f"mean {v.mean():.3f}")
        ax.axvline(np.median(v), color="green", lw=1.5, linestyle="--", label=f"med {np.median(v):.3f}")
        ax.axvline(np.percentile(v, 99), color="blue", lw=1.5, linestyle="--", label=f"p99 {np.percentile(v, 99):.3f}")
        ax.set_title(marker, fontsize=10, fontweight="bold")
        ax.set_xlabel("Intensity (p99-norm)", fontsize=8)
        ax.set_ylabel("Count (log)", fontsize=8)
        ax.legend(fontsize=7, frameon=False)

    for ax in axes[n_markers:]:
        ax.set_visible(False)

    fig.suptitle("Token-level intensity distribution per biomarker (all slides)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "token_intensity_histograms_all_slides.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()