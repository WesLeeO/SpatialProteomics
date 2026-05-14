#!/usr/bin/env python3
"""
Token-level intensity distributions for pancancer CODEX TMA patch datasets.

Covers four TMAs: CRC_TMA_A, CRC_TMA_B, Multi-tumor, Tonsil.
Each TMA has a different marker panel (~57-59 channels), so figures are
produced per-TMA.  A cross-TMA presence heatmap and per-marker overlay
(for markers shared across TMAs) are produced as additional outputs.

Outputs (saved to biomarker_stats/pancancer/)
--------------------------------------------
  {TMA}_token_intensity_histograms.png   — per-marker histograms for that TMA
  {TMA}_token_stats.csv                  — mean/std/percentile table per marker
  cross_tma_marker_presence.png          — heatmap of which markers appear in which TMA
  cross_tma_shared_markers_overlay.png   — per-marker overlaid distributions for
                                           markers present in ≥2 TMAs
"""

import csv
import math

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_DIR = Path("pancancer_patch_dataset")
OUTPUT_DIR  = Path("biomarker_stats/pancancer")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TMAS   = ["CRC_TMA_A", "CRC_TMA_B", "Multi-tumor", "Tonsil"]
N_BINS = 100
TMA_COLORS = {"CRC_TMA_A": "#E24A33", "CRC_TMA_B": "#348ABD",
              "Multi-tumor": "#988ED5", "Tonsil": "#8EBA42"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_tma(tma: str) -> tuple[list[str], dict[int, np.ndarray], int]:
    """
    Load all cores for a TMA.

    Returns
    -------
    marker_names : list[str]
    token_vals   : {marker_idx: flat float32 array of all token values [0,1]}
    total_patches: int
    """
    h5_files = sorted((DATASET_DIR / tma).glob("*_patch_dataset.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No H5 files found in {DATASET_DIR / tma}")

    marker_names: list[str] = []
    token_vals:   dict[int, list[np.ndarray]] = {}
    total_patches = 0

    for h5_path in h5_files:
        with h5py.File(h5_path) as f:
            if "marker_names" not in f.attrs or "targets" not in f:
                print(f"  Skipping incomplete file: {h5_path.name}")
                continue
            markers = list(f.attrs["marker_names"])
            targets = f["targets"][:]   # (N, C, G, G)
        marker_names = markers
        N, C, G, _ = targets.shape
        total_patches += N
        flat = targets.reshape(N, C, G * G)
        for m_idx in range(C):
            token_vals.setdefault(m_idx, []).append(flat[:, m_idx, :].ravel())

    aggregated = {m: np.concatenate(arrs) for m, arrs in token_vals.items()}
    return marker_names, aggregated, total_patches


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def compute_stats(token_vals: dict[int, np.ndarray],
                  marker_names: list[str]) -> list[dict]:
    rows = []
    for m_idx, marker in enumerate(marker_names):
        v = token_vals[m_idx]
        p5, p25, p75, p95, p99 = np.percentile(v, [5, 25, 75, 95, 99])
        rows.append(dict(
            marker=marker,
            n_tokens=len(v),
            mean=float(v.mean()),
            std=float(v.std()),
            median=float(np.median(v)),
            p5=float(p5), p25=float(p25),
            p75=float(p75), p95=float(p95), p99=float(p99),
            min=float(v.min()), max=float(v.max()),
        ))
    return rows


def print_stats_table(tma: str, stats: list[dict], total_patches: int) -> None:
    n_markers = len(stats)
    print(f"\n{'='*80}")
    print(f"  {tma}  —  {total_patches:,} patches  {n_markers} markers")
    print(f"{'='*80}")
    print(f"  {'Marker':<24} {'n_tokens':>12} {'mean':>8} {'std':>8} "
          f"{'median':>8} {'p5':>8} {'p95':>8} {'p99':>8} {'max':>8}")
    print("  " + "-" * 96)
    for r in stats:
        print(f"  {r['marker']:<24} {r['n_tokens']:>12,} {r['mean']:>8.4f} "
              f"{r['std']:>8.4f} {r['median']:>8.4f} {r['p5']:>8.4f} "
              f"{r['p95']:>8.4f} {r['p99']:>8.4f} {r['max']:>8.4f}")


def save_stats_csv(stats: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)


# ---------------------------------------------------------------------------
# Per-TMA histogram figure
# ---------------------------------------------------------------------------

def plot_tma_histograms(tma: str,
                         marker_names: list[str],
                         token_vals: dict[int, np.ndarray],
                         total_patches: int,
                         out_path: Path) -> None:
    n_markers = len(marker_names)
    ncols = 8
    nrows = math.ceil(n_markers / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.2, nrows * 2.6), dpi=120)
    axes = axes.flatten()

    for m_idx, marker in enumerate(marker_names):
        ax = axes[m_idx]
        v  = token_vals[m_idx]
        ax.hist(v, bins=N_BINS, range=(0, 1),
                color=TMA_COLORS.get(tma, "steelblue"),
                edgecolor="none", log=True)
        ax.axvline(v.mean(),             color="red",   lw=1.2,
                   label=f"μ {v.mean():.3f}")
        ax.axvline(np.median(v),         color="green", lw=1.2, ls="--",
                   label=f"med {np.median(v):.3f}")
        ax.axvline(np.percentile(v, 99), color="navy",  lw=1.2, ls=":",
                   label=f"p99 {np.percentile(v,99):.3f}")
        ax.set_title(marker, fontsize=8, fontweight="bold")
        ax.set_xlabel("Intensity (log1p/p99)", fontsize=6)
        ax.set_ylabel("Count (log)", fontsize=6)
        ax.legend(fontsize=5.5, frameon=False)
        ax.tick_params(labelsize=6)

    for ax in axes[n_markers:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Token-level intensity distributions — {tma}\n"
        f"({total_patches:,} patches  ·  {n_markers} markers)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Cross-TMA marker presence heatmap
# ---------------------------------------------------------------------------

def plot_presence_heatmap(tma_markers: dict[str, list[str]],
                           out_path: Path) -> None:
    """
    Heatmap: rows = TMAs, columns = union of all markers across TMAs,
    sorted so shared markers come first.
    """
    all_markers_ordered = []
    seen = set()
    # First pass: markers present in >1 TMA (sorted by name)
    from collections import Counter
    counts = Counter(m for markers in tma_markers.values() for m in markers)
    for m in sorted(counts, key=lambda x: (-counts[x], x)):
        if m not in seen:
            all_markers_ordered.append(m)
            seen.add(m)

    tma_list = list(tma_markers.keys())
    n_tma    = len(tma_list)
    n_marker = len(all_markers_ordered)

    presence = np.zeros((n_tma, n_marker), dtype=np.float32)
    for ti, tma in enumerate(tma_list):
        panel = set(tma_markers[tma])
        for mi, marker in enumerate(all_markers_ordered):
            presence[ti, mi] = float(marker in panel)

    fig_w = max(14, n_marker * 0.32)
    fig_h = max(3,  n_tma   * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=120)

    ax.imshow(presence, aspect="auto", cmap="RdYlGn",
              vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n_marker))
    ax.set_xticklabels(all_markers_ordered,
                        rotation=60, ha="right", fontsize=6.5)
    ax.set_yticks(range(n_tma))
    ax.set_yticklabels(tma_list, fontsize=9)

    for ti in range(n_tma):
        for mi in range(n_marker):
            sym = "✓" if presence[ti, mi] else "·"
            ax.text(mi, ti, sym, ha="center", va="center",
                    fontsize=6, color="black" if presence[ti, mi] else "#aaaaaa")

    # Annotate how many TMAs share each marker
    for mi, marker in enumerate(all_markers_ordered):
        n = int(presence[:, mi].sum())
        ax.text(mi, -0.65, str(n), ha="center", va="center",
                fontsize=6, color="black" if n > 1 else "#aaaaaa")

    ax.set_title(
        "Marker presence across pancancer TMAs  "
        "(numbers = how many TMAs share each marker)",
        fontsize=10, fontweight="bold", pad=18,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Cross-TMA overlay histograms (shared markers only)
# ---------------------------------------------------------------------------

def plot_cross_tma_overlays(
    tma_data: dict[str, tuple[list[str], dict[int, np.ndarray]]],
    min_tmas: int = 2,
    out_path: Path = OUTPUT_DIR / "cross_tma_shared_markers_overlay.png",
) -> None:
    """
    For each marker present in ≥ min_tmas TMAs, overlay per-TMA histograms
    so the shift in expression across tissue contexts is visible.
    """
    # Collect per-marker, per-TMA token arrays
    marker_tma: dict[str, dict[str, np.ndarray]] = {}
    for tma, (marker_names, token_vals) in tma_data.items():
        for m_idx, marker in enumerate(marker_names):
            marker_tma.setdefault(marker, {})[tma] = token_vals[m_idx]

    shared = sorted(m for m, d in marker_tma.items() if len(d) >= min_tmas)
    if not shared:
        print("  No shared markers found for cross-TMA overlay.")
        return

    n_shared = len(shared)
    ncols    = 8
    nrows    = math.ceil(n_shared / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.2, nrows * 2.8), dpi=120)
    axes = axes.flatten()

    for k, marker in enumerate(shared):
        ax = axes[k]
        tma_dict = marker_tma[marker]
        for tma, v in sorted(tma_dict.items()):
            ax.hist(v, bins=N_BINS, range=(0, 1), log=True,
                    alpha=0.55, color=TMA_COLORS.get(tma, "gray"),
                    label=f"{tma} μ={v.mean():.3f}", edgecolor="none")
        ax.set_title(f"{marker}  ({len(tma_dict)} TMAs)",
                     fontsize=8, fontweight="bold")
        ax.set_xlabel("Intensity", fontsize=6)
        ax.set_ylabel("Count (log)", fontsize=6)
        ax.legend(fontsize=5, frameon=False)
        ax.tick_params(labelsize=6)

    for ax in axes[n_shared:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Cross-TMA token intensity overlay — markers in ≥{min_tmas} TMAs  "
        f"({n_shared} markers)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tma_data:    dict[str, tuple[list[str], dict[int, np.ndarray]]] = {}
    tma_markers: dict[str, list[str]] = {}

    for tma in TMAS:
        tma_dir = DATASET_DIR / tma
        if not tma_dir.exists() or not list(tma_dir.glob("*_patch_dataset.h5")):
            print(f"Skipping {tma}: no H5 files found in {tma_dir}")
            continue

        print(f"\nLoading {tma}…", flush=True)
        marker_names, token_vals, total_patches = load_tma(tma)
        tma_data[tma]    = (marker_names, token_vals)
        tma_markers[tma] = marker_names

        stats = compute_stats(token_vals, marker_names)
        print_stats_table(tma, stats, total_patches)

        save_stats_csv(stats, OUTPUT_DIR / f"{tma}_token_stats.csv")

        plot_tma_histograms(
            tma, marker_names, token_vals, total_patches,
            out_path=OUTPUT_DIR / f"{tma}_token_intensity_histograms.png",
        )

    if len(tma_data) < 2:
        print("\nFewer than 2 TMAs loaded — skipping cross-TMA plots.")
        return

    print("\nBuilding cross-TMA outputs…")
    plot_presence_heatmap(tma_markers,
                           OUTPUT_DIR / "cross_tma_marker_presence.png")
    plot_cross_tma_overlays(tma_data,
                             min_tmas=2,
                             out_path=OUTPUT_DIR / "cross_tma_shared_markers_overlay.png")

    print(f"\nDone. All outputs in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()