import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

H5_PATH    = Path('datasets/hnscc_patch_dataset/hnscc_patch_dataset.h5')
OUTPUT_DIR = Path('biomarker_stats/hnscc')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_BINS = 100


def main():
    print(f'Loading {H5_PATH.name}…')
    with h5py.File(H5_PATH) as f:
        marker_names = list(f.attrs['marker_names'])
        targets      = f['targets'][:]   # (N, C, 16, 16)

    n_patches, n_ch, G, _ = targets.shape
    print(f'  {n_patches:,} sub-patches  ×  {n_ch} markers  ×  {G}×{G} tokens  '
          f'→  {n_patches * G * G:,} tokens/marker')

    flat = targets.reshape(n_patches, n_ch, G * G)   # (N, C, 256)

    hdr = (f"{'Idx':<4} {'Marker':<16} {'n_tokens':>12} {'mean':>8} {'std':>8} "
           f"{'median':>8} {'p5':>8} {'p25':>8} {'p75':>8} {'p95':>8} "
           f"{'frac>0':>8}")
    sep = '-' * 110
    print(f'\n{hdr}\n{sep}')

    rows       = []
    all_tokens = {}

    for ci, marker in enumerate(marker_names):
        v       = flat[:, ci, :].ravel()
        p5, p25, p75, p95 = np.percentile(v, [5, 25, 75, 95])
        frac_pos = (v > 0).mean()
        all_tokens[ci] = v
        row = (f'{ci:<4} {marker:<16} {len(v):>12,} {v.mean():>8.4f} {v.std():>8.4f} '
               f'{np.median(v):>8.4f} {p5:>8.4f} {p25:>8.4f} {p75:>8.4f} {p95:>8.4f} '
               f'{frac_pos:>8.4f}')
        print(row)
        rows.append(row)

    txt_path = OUTPUT_DIR / 'stats.txt'
    txt_path.write_text('\n'.join([hdr, sep] + rows) + '\n')
    print(f'\nSaved → {txt_path}')

    # ── Per-marker histogram ──────────────────────────────────────────────────
    ncols = min(n_ch, 3)
    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.0, nrows * 3.2))
    axes = np.array(axes).flatten()

    for ci, marker in enumerate(marker_names):
        ax = axes[ci]
        v  = all_tokens[ci]
        # full range histogram (log scale)
        ax.hist(v, bins=N_BINS, range=(0, 1), color='steelblue',
                edgecolor='none', log=True)
        ax.axvline(v.mean(),             color='red',    lw=1.5,
                   label=f'mean {v.mean():.3f}')
        ax.axvline(np.median(v),         color='green',  lw=1.5,
                   linestyle='--', label=f'med  {np.median(v):.3f}')
        ax.axvline(np.percentile(v, 95), color='orange', lw=1.5,
                   linestyle=':', label=f'p95  {np.percentile(v,95):.3f}')
        frac = (v > 0).mean()
        ax.set_title(f'{marker}   (frac>0: {frac:.3f})', fontsize=9,
                     fontweight='bold')
        ax.set_xlabel('Token intensity (normalised)', fontsize=7)
        ax.set_ylabel('Count (log)', fontsize=7)
        ax.legend(fontsize=6, frameon=False)

    for ax in axes[n_ch:]:
        ax.set_visible(False)

    fig.suptitle(
        f'Token-level intensity distribution — HNSCC\n'
        f'({n_patches:,} sub-patches, {n_ch} markers, {G}×{G} token grid)',
        fontsize=11, fontweight='bold')
    plt.tight_layout()
    hist_path = OUTPUT_DIR / 'token_intensity_histograms.png'
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f'Saved → {hist_path}')

    # ── Per-case mean expression bar chart ───────────────────────────────────
    with h5py.File(H5_PATH) as f:
        patch_ids = np.array([p.decode() for p in f['patch_ids'][:]])

    cases = sorted(set(p.split('_')[0] for p in patch_ids))
    case_means = np.zeros((len(cases), n_ch), dtype=np.float32)

    for ki, case in enumerate(cases):
        mask = np.array([p.startswith(case + '_') for p in patch_ids])
        case_means[ki] = targets[mask].mean(axis=(0, 2, 3))

    fig2, ax2 = plt.subplots(figsize=(max(8, len(cases) * 1.2), 4))
    x     = np.arange(len(cases))
    width = 0.8 / n_ch
    colors = plt.cm.tab10(np.linspace(0, 1, n_ch))

    for ci, marker in enumerate(marker_names):
        ax2.bar(x + ci * width, case_means[:, ci], width,
                label=marker, color=colors[ci], alpha=0.85)

    ax2.set_xticks(x + width * (n_ch - 1) / 2)
    ax2.set_xticklabels(cases, fontsize=8)
    ax2.set_ylabel('Mean token expression', fontsize=9)
    ax2.set_title('Per-case mean token expression — HNSCC', fontsize=10,
                  fontweight='bold')
    ax2.legend(fontsize=7, frameon=False, ncol=n_ch)
    plt.tight_layout()
    bar_path = OUTPUT_DIR / 'per_case_mean_expression.png'
    plt.savefig(bar_path, dpi=150)
    plt.close()
    print(f'Saved → {bar_path}')


if __name__ == '__main__':
    main()