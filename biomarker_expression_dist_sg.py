#!/usr/bin/env python3
"""
Pixel-level intensity histograms per biomarker across singular_genomics patch datasets.
Each patch (N, C, 16, 16) contributes 256 token samples per channel.

Differences from the ORION version:
  - valid_markers mask: missing JP2s are zero-filled and excluded from stats/plots
  - One slide per disease — per-disease stats are printed alongside the aggregate
  - Presence heatmap: shows which markers are present per disease
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

DATASET_DIR = Path("singular_genomics")
H5_FILES    = sorted(DATASET_DIR.glob("*_patch_dataset.h5"))
OUTPUT_DIR  = Path("biomarker_stats_out/sg")
OUTPUT_DIR.mkdir(exist_ok=True)

N_BINS = 100


def disease_name(h5_path: Path) -> str:
    return h5_path.stem.replace("_patch_dataset", "")


def main():
    if not H5_FILES:
        print(f"No h5 files found in {DATASET_DIR}")
        return

    marker_names: list[str] = []

    # Per-marker accumulation across all slides (only valid tokens)
    token_vals: dict[int, list[np.ndarray]] = {}

    # Per-disease stats for the summary table
    per_disease: list[dict] = []

    # Presence matrix: disease × marker
    disease_names: list[str] = []
    presence_rows: list[np.ndarray] = []

    print("Loading slides...")
    for h5_path in H5_FILES:
        disease = disease_name(h5_path)
        with h5py.File(h5_path, "r") as f:
            markers     = list(f.attrs["marker_names"])
            targets     = f["targets"][:]         # (N, C, 16, 16)
            valid_mask  = f["valid_markers"][:]   # (C,) bool

        n_patches, n_ch, H, W = targets.shape
        marker_names = markers
        n_valid = int(valid_mask.sum())
        print(f"  {disease}: {n_patches:,} patches  "
              f"{n_valid}/{n_ch} markers present")

        disease_names.append(disease)
        presence_rows.append(valid_mask.copy())

        # flat (N*H*W,) per channel — only for valid channels
        flat = targets.reshape(n_patches, n_ch, H * W)   # (N, C, 256)
        disease_stats = {"disease": disease, "markers": {}}
        for m_idx in range(n_ch):
            if not valid_mask[m_idx]:
                continue
            vals = flat[:, m_idx, :].ravel()
            token_vals.setdefault(m_idx, []).append(vals)
            disease_stats["markers"][m_idx] = vals
        per_disease.append(disease_stats)

    n_markers = len(marker_names)
    presence_mat = np.stack(presence_rows, axis=0)  # (n_diseases, C)

    # ── Aggregate across all slides ───────────────────────────────────────────
    all_tokens: dict[int, np.ndarray] = {}
    for m_idx, arrs in token_vals.items():
        all_tokens[m_idx] = np.concatenate(arrs)

    # ── Print aggregate statistics ────────────────────────────────────────────
    print(f"\n{'Marker':<14} {'n_diseases':>10} {'n_tokens':>12} {'mean':>8} "
          f"{'std':>8} {'median':>8} {'p5':>8} {'p95':>8} {'max':>8}")
    print("-" * 102)
    for m_idx, marker in enumerate(marker_names):
        if m_idx not in all_tokens:
            print(f"{marker:<14} {'MISSING':>10}")
            continue
        v = all_tokens[m_idx]
        n_dis = int(presence_mat[:, m_idx].sum())
        p5, p95 = np.percentile(v, [5, 95])
        print(f"{marker:<14} {n_dis:>10} {len(v):>12,} {v.mean():>8.4f} "
              f"{v.std():>8.4f} {np.median(v):>8.4f} {p5:>8.4f} {p95:>8.4f} "
              f"{v.max():>8.4f}")

    # ── Print per-disease means ───────────────────────────────────────────────
    print(f"\nPer-disease mean intensity (- = missing):")
    header = f"{'':20}" + "".join(f"{m[:6]:>8}" for m in marker_names)
    print(header)
    for ds in per_disease:
        row = f"{ds['disease']:<20}"
        for m_idx in range(n_markers):
            if m_idx in ds["markers"]:
                row += f"{ds['markers'][m_idx].mean():>8.4f}"
            else:
                row += f"{'  -':>8}"
        print(row)

    # ── Presence heatmap ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, n_markers * 0.55), len(disease_names) * 0.55 + 1.5))
    im = ax.imshow(presence_mat.astype(float), aspect="auto",
                   cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n_markers))
    ax.set_xticklabels(marker_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(disease_names)))
    ax.set_yticklabels(disease_names, fontsize=8)
    for i in range(len(disease_names)):
        for j in range(n_markers):
            ax.text(j, i, "✓" if presence_mat[i, j] else "✗",
                    ha="center", va="center", fontsize=7,
                    color="black")
    ax.set_title("Marker presence per disease (green=present, red=missing)", fontsize=10)
    plt.tight_layout()
    heatmap_path = OUTPUT_DIR / "marker_presence.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"\nSaved → {heatmap_path}")

    # ── Token intensity histograms ────────────────────────────────────────────
    ncols = 4
    nrows = int(np.ceil(n_markers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.0))
    axes = axes.flatten()

    for m_idx, marker in enumerate(marker_names):
        ax = axes[m_idx]
        if m_idx not in all_tokens:
            ax.set_facecolor("#f0f0f0")
            ax.text(0.5, 0.5, f"{marker}\n(missing in all diseases)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8, color="gray")
            ax.set_title(marker, fontsize=10, color="gray")
            ax.axis("off")
            continue

        v = all_tokens[m_idx]
        n_dis = int(presence_mat[:, m_idx].sum())
        ax.hist(v, bins=N_BINS, range=(0, 1),
                color="steelblue", edgecolor="none", log=True)
        ax.axvline(v.mean(),            color="red",   lw=1.5,
                   label=f"mean {v.mean():.3f}")
        ax.axvline(np.median(v),        color="green", lw=1.5, linestyle="--",
                   label=f"med {np.median(v):.3f}")
        ax.axvline(np.percentile(v, 99), color="blue", lw=1.5, linestyle=":",
                   label=f"p99 {np.percentile(v, 99):.3f}")
        ax.set_title(f"{marker}  ({n_dis}/{len(disease_names)} diseases)",
                     fontsize=9, fontweight="bold")
        ax.set_xlabel("Intensity (log1p/p99 norm)", fontsize=7)
        ax.set_ylabel("Count (log)", fontsize=7)
        ax.legend(fontsize=6, frameon=False)

    for ax in axes[n_markers:]:
        ax.set_visible(False)

    fig.suptitle("Token-level intensity distribution per biomarker — Singular Genomics",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    hist_path = OUTPUT_DIR / "token_intensity_histograms.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Saved → {hist_path}")

    # ── Per-disease histograms (one figure per marker with per-disease overlays) ──
    valid_markers = [m for m_idx, m in enumerate(marker_names) if m_idx in all_tokens]
    n_valid_markers = len(valid_markers)
    if n_valid_markers:
        ncols2 = 4
        nrows2 = int(np.ceil(n_valid_markers / ncols2))
        fig2, axes2 = plt.subplots(nrows2, ncols2,
                                   figsize=(ncols2 * 3.8, nrows2 * 3.0))
        axes2 = axes2.flatten()
        cmap  = plt.get_cmap("tab10", len(disease_names))

        plot_idx = 0
        for m_idx, marker in enumerate(marker_names):
            if m_idx not in all_tokens:
                continue
            ax = axes2[plot_idx]
            for di, ds in enumerate(per_disease):
                if m_idx not in ds["markers"]:
                    continue
                v = ds["markers"][m_idx]
                ax.hist(v, bins=N_BINS, range=(0, 1), log=True,
                        alpha=0.5, color=cmap(di), label=ds["disease"],
                        edgecolor="none")
            ax.set_title(marker, fontsize=9, fontweight="bold")
            ax.set_xlabel("Intensity", fontsize=7)
            ax.set_ylabel("Count (log)", fontsize=7)
            ax.legend(fontsize=5, frameon=False, ncol=2)
            plot_idx += 1

        for ax in axes2[plot_idx:]:
            ax.set_visible(False)

        fig2.suptitle("Per-disease intensity overlay per biomarker — Singular Genomics",
                      fontsize=12, fontweight="bold")
        plt.tight_layout()
        per_disease_path = OUTPUT_DIR / "token_intensity_per_disease.png"
        plt.savefig(per_disease_path, dpi=150)
        plt.close()
        print(f"Saved → {per_disease_path}")


if __name__ == "__main__":
    main()