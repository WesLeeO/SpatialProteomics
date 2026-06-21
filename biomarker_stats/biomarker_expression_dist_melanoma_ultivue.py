#!/usr/bin/env python3
"""Token-level intensity histograms per biomarker across melanoma2_patch_dataset."""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATASET_DIR = Path("datasets/melanoma_ultivue_patch_dataset")
H5_FILES    = sorted(DATASET_DIR.glob("*/*_patches.h5"))
OUTPUT_DIR  = Path("biomarker_stats/melanoma_ultivue")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BINS = 100


def main():
    marker_names: list[str] = []
    token_vals: dict[int, list[np.ndarray]] = {}

    print("Loading slides...")
    for h5_path in H5_FILES:
        with h5py.File(h5_path, "r") as f:
            raw = f.attrs["marker_names"]
            markers = (raw.decode() if isinstance(raw, bytes) else raw).split(";")
            targets = f["targets"][:]          # (N, C, G, G)
        slide = h5_path.parent.name
        n_patches, n_ch, G, _ = targets.shape
        print(f"  {slide}: {n_patches:,} patches  →  {n_patches * G * G:,} tokens/marker")
        marker_names = markers
        flat = targets.reshape(n_patches, n_ch, G * G)
        for m_idx in range(n_ch):
            token_vals.setdefault(m_idx, []).append(flat[:, m_idx, :].ravel())

    all_tokens = {m: np.concatenate(arrs) for m, arrs in token_vals.items()}
    n_markers  = len(marker_names)

    hdr = (f"{'Marker':<14} {'n_tokens':>12} {'mean':>8} {'std':>8} "
           f"{'median':>8} {'p5':>8} {'p25':>8} {'p75':>8} {'p95':>8}")
    sep = "-" * 96
    print(f"\n{hdr}\n{sep}")
    rows = []
    for m_idx, marker in enumerate(marker_names):
        v  = all_tokens[m_idx]
        p5, p25, p75, p95 = np.percentile(v, [5, 25, 75, 95])
        row = (f"{marker:<14} {len(v):>12,} {v.mean():>8.4f} {v.std():>8.4f} "
               f"{np.median(v):>8.4f} {p5:>8.4f} {p25:>8.4f} {p75:>8.4f} {p95:>8.4f}")
        print(row)
        rows.append(row)
    txt_path = OUTPUT_DIR / "stats.txt"
    txt_path.write_text("\n".join([hdr, sep] + rows) + "\n")
    print(f"\nSaved → {txt_path}")

    ncols = 4
    nrows = int(np.ceil(n_markers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.0))
    axes = axes.flatten()

    for m_idx, marker in enumerate(marker_names):
        ax = axes[m_idx]
        v  = all_tokens[m_idx]
        ax.hist(v, bins=N_BINS, range=(0, 1), color="steelblue", edgecolor="none", log=True)
        ax.axvline(v.mean(),              color="red",   lw=1.5, label=f"mean {v.mean():.3f}")
        ax.axvline(np.median(v),          color="green", lw=1.5, linestyle="--", label=f"med {np.median(v):.3f}")
        ax.axvline(np.percentile(v, 95),  color="blue",  lw=1.5, linestyle="--", label=f"p95 {np.percentile(v,95):.3f}")
        ax.set_title(marker, fontsize=10, fontweight="bold")
        ax.set_xlabel("Intensity (log1p/p99)", fontsize=8)
        ax.set_ylabel("Count (log)", fontsize=8)
        ax.legend(fontsize=7, frameon=False)

    for ax in axes[n_markers:]:
        ax.set_visible(False)

    fig.suptitle("Token-level intensity distribution — melanoma2 (all slides)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "token_intensity_histograms.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()