import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

H5_PATH    = Path("nsclc_charite_patch_dataset.h5")
OUTPUT_DIR = Path("biomarker_stats/nsclc_charite")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BINS = 100


def main():
    print(f"Loading {H5_PATH.name}...")
    with h5py.File(H5_PATH, "r") as f:
        marker_names = list(f.attrs["marker_names"])
        targets      = f["targets"][:]   # (N, C, 16, 16)

    n_patches, n_ch, H, W = targets.shape
    print(f"  {n_patches:,} patches  ×  {n_ch} markers  ×  {H}×{W} tokens  "
          f"→  {n_patches * H * W:,} tokens/marker")

    flat = targets.reshape(n_patches, n_ch, H * W)   # (N, C, 256)

    hdr = (f"{'Idx':<4} {'Marker':<16} {'n_tokens':>12} {'mean':>8} {'std':>8} "
           f"{'median':>8} {'p5':>8} {'p25':>8} {'p75':>8} {'p95':>8} "
           f"{'min':>8} {'max':>8}")
    sep = "-" * 112
    print(f"\n{hdr}")
    print(sep)
    all_tokens = {}
    rows = []
    for m_idx, marker in enumerate(marker_names):
        v = flat[:, m_idx, :].ravel()
        all_tokens[m_idx] = v
        p5, p25, p75, p95 = np.percentile(v, [5, 25, 75, 95])
        row = (f"{m_idx:<4} {marker:<16} {len(v):>12,} {v.mean():>8.4f} {v.std():>8.4f} "
               f"{np.median(v):>8.4f} {p5:>8.4f} {p25:>8.4f} {p75:>8.4f} {p95:>8.4f} "
               f"{v.min():>8.4f} {v.max():>8.4f}")
        print(row)
        rows.append(row)
    txt_path = OUTPUT_DIR / "stats.txt"
    txt_path.write_text("\n".join([hdr, sep] + rows) + "\n")
    print(f"Saved → {txt_path}")

    ncols = 4
    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.8, nrows * 3.0))
    axes = axes.flatten()

    for m_idx, marker in enumerate(marker_names):
        ax = axes[m_idx]
        v  = all_tokens[m_idx]
        ax.hist(v, bins=N_BINS, range=(0, 1), color="steelblue", edgecolor="none", log=True)
        ax.axvline(v.mean(),             color="red",   lw=1.5, label=f"mean {v.mean():.3f}")
        ax.axvline(np.median(v),         color="green", lw=1.5, linestyle="--", label=f"med {np.median(v):.3f}")
        ax.axvline(np.percentile(v, 99), color="blue",  lw=1.5, linestyle=":",  label=f"p99 {np.percentile(v,99):.3f}")
        ax.set_title(f"{m_idx}. {marker}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Intensity (p99-norm)", fontsize=7)
        ax.set_ylabel("Count (log)", fontsize=7)
        ax.legend(fontsize=6, frameon=False)

    for ax in axes[n_ch:]:
        ax.set_visible(False)

    fig.suptitle(f"Token-level intensity distribution — nsclc_charite\n"
                 f"({n_patches:,} patches, {n_ch} markers)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "token_intensity_histograms.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()