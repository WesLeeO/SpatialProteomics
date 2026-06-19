"""
Overlay ORION-CRC nuclei / cell-type calls on top of benchmark patches.

For a chosen slide we:
  1. read patch coordinates from the benchmark HDF5
       orion_crc_patch_dataset_benchmark/<sample>_patch_dataset.h5
       (coords are top-left (x, y) in shared H&E/IF level-0 pixel space;
        each patch spans patch_size_level0 px/side — usually >224)
  2. read the matching H&E (or IF) pixels from the original ORION slide via the
     zarr loader in build_patch_dataset_orion_crc_reg.py, then resize the H&E
     DOWN onto the masks' native 20× grid (the mask is never resampled) with a
     16×16 token grid drawn on top
  3. composite per-cell info from
       ORIONCRC_dataset_tile_20x/
         csv_nuclei_pos/<slide>.csv   (label, x, y, <marker>_pos …)
         nuclei/<slide>_<X>_<Y>_0_512_512.tiff  (instance masks, label == csv label)
     The nucleus *footprint* is filled (not just the centroid) and optionally
     expanded by a radius to approximate the whole-cell body, coloured by
     marker positivity.

Example
-------
  python visualize_orion_cell_overlay.py --sample CRC02 \
      --patches random:6 --markers CD8a,FOXP3,CD20,Pan-CK
  python visualize_orion_cell_overlay.py --sample CRC02 --patches 0,500,1000 \
      --cell_radius_um 2 --modality if --if_marker CD8a --markers CD8a
"""

import argparse
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, Normalize
from matplotlib.cm import ScalarMappable
import cv2
from scipy import ndimage as ndi
from scipy.stats import pearsonr
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import watershed

# shared paths / geometry / cell↔token↔mask helpers (also puts project root on
# sys.path so the build_patch_dataset_orion_crc_reg import below resolves)
from utils import (
    BENCH_DIR, TILE_DIR, SLIDE_DF, NUCLEI_DIR, TILE_L0, TILE_MASK, MPP_20X, GRID,
    sample_rows, csv_marker_col, index_nuclei_tiles, load_patch_labels,
)
from build_patch_dataset_orion_crc_reg import (
    crc_paths, open_zarr_level0, ORION_CRC_CHANNELS, ORION_AF_RAW_CH, MPP_HE,
)

OUT_DIR   = Path("/home/wesley/spatial_proteomics/cell_cls/orion_cell_cls/visualize_orion_cells_out")
# Cached token-level model predictions + MIPHEI IF targets, written by
# visualize_orion_predictions.py (row order matches the benchmark h5 `coords`).
#   <PRED_DIR>/<sample>_preds.npy    (N, C, 16, 16)  model output
#   <PRED_DIR>/<sample>_targets.npy  (N, C, 16, 16)  MIPHEI ground-truth IF
#   <PRED_DIR>/<sample>_names.npy    (C,)            channel → marker name
PRED_DIR  = Path("outputs_orion_token_UNI2_baseline_bg0.2")
# MIPHEI-ViT token predictions (N, C, 16, 16), row-aligned to the benchmark coords,
# built by benchmarking; used to add a MIPHEI column and to rank disagreement patches.
MIPHEI_CACHE = Path("benchmarking/preds_cache/miphei-vit")

MARKER_COLORS = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324",
    "#fffac8", "#800000", "#aaffc3", "#808000",
]


# ── Pixel I/O ──────────────────────────────────────────────────────────────────

def read_he_patch(arr, c_ax, h_ax, w_ax, px, py, ps0, H, W) -> np.ndarray:
    idx = [slice(None)] * arr.ndim
    idx[h_ax] = slice(int(py), min(int(py) + ps0, H))
    idx[w_ax] = slice(int(px), min(int(px) + ps0, W))
    patch = np.asarray(arr[tuple(idx)])
    if c_ax == 0:
        patch = np.transpose(patch, (1, 2, 0))
    return patch[..., :3]


def read_if_marker_patch(arr, c_ax, h_ax, w_ax, raw_ch, px, py, ps0, H, W,
                         af_ch=ORION_AF_RAW_CH) -> np.ndarray:
    idx = [slice(None)] * arr.ndim
    idx[h_ax] = slice(int(py), min(int(py) + ps0, H))
    idx[w_ax] = slice(int(px), min(int(px) + ps0, W))
    if c_ax == 0:
        sig = arr[tuple([raw_ch] + idx[1:])].astype(np.float32)
        af  = arr[tuple([af_ch]  + idx[1:])].astype(np.float32)
    else:
        block = np.asarray(arr[tuple(idx)]).astype(np.float32)
        sig, af = block[..., raw_ch], block[..., af_ch]
    x  = np.maximum(sig - af, 0.0)
    hi = np.percentile(x, 99.5) or 1.0
    return np.clip(x / hi, 0, 1)


def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    m = float(img.max()) or 1.0
    return (img / m * 255).astype(np.uint8) if m > 1.0 else (img * 255).astype(np.uint8)


def expand_cells(labels: np.ndarray, radius_px: int) -> np.ndarray:
    """Approximate whole-cell regions by dilating nuclei (MIPHEI recipe).

    Dilate the binary nuclei footprint by a disk, then watershed-constrain it
    back to per-instance labels so neighbouring expanded cells don't merge.
    (MIPHEI-ViT/notebooks/inference.ipynb: "nuclei expansion to approximate cells")
    """
    if radius_px <= 0:
        return labels
    binary  = labels > 0
    dilated = binary_dilation(binary, footprint=disk(radius_px))
    dist    = ndi.distance_transform_edt(~binary)
    return watershed(-dist, markers=labels, mask=dilated, watershed_line=False)


def draw_token_grid(ax, h: int, w: int, n: int = GRID) -> None:
    for k in range(1, n):
        ax.axhline(k * h / n, color="black", lw=0.4, alpha=0.35)
        ax.axvline(k * w / n, color="black", lw=0.4, alpha=0.35)


# ── Main ───────────────────────────────────────────────────────────────────────

def pick_patches(spec: str, coords: np.ndarray, ps0: int,
                 cx: np.ndarray, cy: np.ndarray,
                 min_cells: int, seed: int) -> list[int]:
    """Resolve --patches to indices, optionally keeping only patches whose FOV
    holds >= min_cells centroids (near-empty patches are skipped — only useful
    for training the model not to hallucinate FPs, not for this comparison)."""
    n_total = len(coords)

    def n_in_fov(i: int) -> int:
        px, py = int(coords[i, 0]), int(coords[i, 1])
        return int(np.sum((cx >= px) & (cx < px + ps0) &
                          (cy >= py) & (cy < py + ps0)))

    if spec.startswith("random:"):
        k   = int(spec.split(":")[1])
        rng = np.random.default_rng(seed)
        out = []
        for i in rng.permutation(n_total):
            if min_cells <= 0 or n_in_fov(int(i)) >= min_cells:
                out.append(int(i))
                if len(out) == k:
                    break
        if len(out) < k:
            print(f"  only {len(out)}/{k} patches have >= {min_cells} cells")
        return sorted(out)

    idxs = [int(s) for s in spec.split(",") if s.strip()]
    if min_cells > 0:
        kept    = [i for i in idxs if n_in_fov(i) >= min_cells]
        dropped = [i for i in idxs if i not in kept]
        if dropped:
            print(f"  dropped patches {dropped} (< {min_cells} cells)")
        idxs = kept
    return idxs


def pick_diff_patches(k: int, ch: int, preds, miphei, tgt, coords, ps0,
                      cx, cy, min_cells: int, sig_thr: float):
    """Rank patches by where my model and MIPHEI disagree the most on this marker:
    score = mean |mine - MIPHEI| over the patch's tokens. Return the top-k patches
    (most disagreement) + a label per patch giving the magnitude and direction.
    Only patches with >= min_cells centroids in the FOV and GT-IF signal
    (max token > sig_thr) qualify, so we look where the marker is actually present."""
    n = len(coords)
    pm = preds[:, ch].reshape(n, -1)
    mm = miphei[:, ch].reshape(n, -1)
    disagree = np.abs(pm - mm).mean(1)            # ranking metric (magnitude)
    signed   = (pm - mm).mean(1)                  # >0: mine higher, <0: MIPHEI higher
    gt_max   = tgt[:, ch].reshape(n, -1).max(1)

    px, py = coords[:, 0].astype(int), coords[:, 1].astype(int)
    ncell = np.array([int(np.sum((cx >= px[i]) & (cx < px[i] + ps0) &
                                 (cy >= py[i]) & (cy < py[i] + ps0))) for i in range(n)])
    ok = (ncell >= min_cells) & (gt_max > sig_thr)
    cand = np.nonzero(ok)[0]
    if len(cand) == 0:
        return [], {}
    order = cand[np.argsort(disagree[cand])[::-1]]   # most disagreement first
    sel = [int(i) for i in order[:k]]
    print(f"  diff[{ch}]: {len(cand)} candidate patches; "
          f"max |mine-MIPHEI| {disagree[sel[0]]:.3f}..{disagree[sel[-1]]:.3f}")
    labels = {i: (f"|mine-MIPHEI|={disagree[i]:.3f}\n"
                  f"{'mine higher' if signed[i] > 0 else 'MIPHEI higher'}")
              for i in sel}
    return sel, labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="CRC02", help="orion_slide_id, e.g. CRC02")
    ap.add_argument("--patches", default="random:10",
                    help="'random:N', comma indices e.g. 0,500,1000, or 'diff:K' to pick "
                         "the K patches where my model most beats MIPHEI (vs GT) and the K "
                         "where it most loses, for --diff_marker.")
    ap.add_argument("--diff_marker", default=None,
                    help="marker used to rank 'diff:K' selection (default: first of --markers)")
    ap.add_argument("--diff_sig_thr", type=float, default=0.1,
                    help="min GT max-token expression for a patch to qualify in 'diff:K'")
    ap.add_argument("--miphei_cache", default=str(MIPHEI_CACHE),
                    help="folder with MIPHEI-ViT <sample>_preds.npy (adds a MIPHEI column)")
    ap.add_argument("--no_miphei", action="store_true", help="skip the MIPHEI column")
    ap.add_argument("--markers", default="CD8a,FOXP3,CD20,Pan-CK",
                    help="comma marker names to draw")
    ap.add_argument("--modality", choices=["he", "if"], default="he")
    ap.add_argument("--if_marker", default="Pan-CK",
                    help="marker channel for --modality if background")
    ap.add_argument("--cell_radius_um", type=float, default=2.0,
                    help="µm to dilate nuclei → whole-cell proxy (0 = nucleus only); "
                         "MIPHEI paper uses 2 µm")
    ap.add_argument("--alpha", type=float, default=0.55, help="cell fill opacity")
    ap.add_argument("--min_cells", type=int, default=0,
                    help="skip patches with fewer than this many cells in the FOV "
                         "(0 = keep all; e.g. 10 drops near-empty background patches)")
    ap.add_argument("--no_grid", action="store_true", help="hide 16×16 token grid")
    ap.add_argument("--pred_dir", default=str(PRED_DIR),
                    help="folder with cached <sample>_preds/_targets/_names.npy")
    ap.add_argument("--no_pred", action="store_true",
                    help="skip the model-prediction vs MIPHEI-target token columns")
    ap.add_argument("--heat_vmax", type=float, default=1.0,
                    help="upper bound of the viridis token-heatmap scale (0=blue → vmax=yellow); "
                         "lower it (e.g. 0.4) to make sparse immune markers pop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=str(OUT_DIR / PRED_DIR))
    args = ap.parse_args()

    sample  = args.sample
    markers = [m.strip() for m in args.markers.split(",") if m.strip()]

    # benchmark patch coords
    with h5py.File(BENCH_DIR / f"{sample}_patch_dataset.h5", "r") as f:
        coords = f["coords"][:]
        ps0    = int(f.attrs["patch_size_level0"])
    print(f"[{sample}] {len(coords)} patches  patch_size_level0={ps0}")

    # token-level model predictions + MIPHEI targets (row-aligned to coords)
    show_pred = not args.no_pred
    preds_arr = tgt_arr = None
    ch_of = {}
    if show_pred:
        pred_dir = Path(args.pred_dir)
        preds_arr = np.load(pred_dir / f"{sample}_preds.npy")     # (N, C, G, G)
        tgt_arr   = np.load(pred_dir / f"{sample}_targets.npy")   # (N, C, G, G)
        pred_names = [str(x) for x in np.load(pred_dir / f"{sample}_names.npy")]
        assert len(preds_arr) == len(coords), (
            f"{sample}: pred cache {len(preds_arr)} != coords {len(coords)} — "
            f"stale {pred_dir}, regenerate with visualize_orion_predictions.py")
        ch_of = {m: pred_names.index(m) for m in markers}
        print(f"  preds/targets ← {pred_dir} (channels {ch_of})")

    # optional MIPHEI-ViT prediction column (row-aligned to coords)
    miphei_arr = None
    if show_pred and not args.no_miphei:
        mph_path = Path(args.miphei_cache) / f"{sample}_preds.npy"
        if mph_path.exists():
            miphei_arr = np.load(mph_path)
            assert len(miphei_arr) == len(coords), (
                f"{sample}: MIPHEI cache {len(miphei_arr)} != coords {len(coords)}")
            print(f"  MIPHEI preds  ← {mph_path}")
        else:
            print(f"  no MIPHEI cache at {mph_path} — MIPHEI column skipped")

    # cells + masks
    row   = sample_rows(sample)
    cells = pd.read_csv(TILE_DIR / row.nuclei_csv_path)
    print(f"  {len(cells)} cells from {Path(row.nuclei_csv_path).name}")
    marker_cols = {m: csv_marker_col(cells, m) for m in markers}
    # label → bool array per marker (label values index directly)
    max_lab = int(cells.label.max())
    pos_lut = {}
    for m, col in marker_cols.items():
        lut = np.zeros(max_lab + 1, dtype=bool)
        lut[cells.label.values] = cells[col].values.astype(bool)
        pos_lut[m] = lut

    base = Path(row.nuclei_slide_path).stem
    tile_index = index_nuclei_tiles(base)
    print(f"  {len(tile_index)} mask tiles indexed")

    # cell centroids (level-0) for dot overlay
    cx_all, cy_all = cells.x.values, cells.y.values
    cell_scale = TILE_MASK / TILE_L0                # level-0 → 20× mask-px

    # slide pixels
    he_path, if_path = crc_paths(sample)
    slide_path = he_path if args.modality == "he" else if_path
    arr, c_ax, h_ax, w_ax = open_zarr_level0(slide_path)
    H, W = arr.shape[h_ax], arr.shape[w_ax]
    if args.modality == "if":
        raw_ch = dict((n, rc) for rc, n, _ in ORION_CRC_CHANNELS)[args.if_marker]

    diff_labels = {}
    if args.patches.startswith("diff:"):
        if miphei_arr is None:
            raise SystemExit("diff:K needs the MIPHEI cache (and predictions). "
                             "Drop --no_miphei / check --miphei_cache.")
        dm = args.diff_marker or markers[0]
        k  = int(args.patches.split(":")[1])
        idxs, diff_labels = pick_diff_patches(
            k, ch_of[dm], preds_arr, miphei_arr, tgt_arr, coords, ps0,
            cx_all, cy_all, max(args.min_cells, 1), args.diff_sig_thr)
        print(f"  max mine-vs-MIPHEI disagreement on '{dm}': {len(idxs)} patches")
    else:
        idxs = pick_patches(args.patches, coords, ps0, cx_all, cy_all,
                            args.min_cells, args.seed)
    if not idxs:
        print(f"  no patches matched (min_cells={args.min_cells}); nothing to draw")
        return
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # per marker columns: GT cells [+ mine pred (+ MIPHEI pred) token heatmaps]
    have_mph = miphei_arr is not None
    cols_per_marker = 1 + ((2 if have_mph else 1) if show_pred else 0)
    ncol = 1 + cols_per_marker * len(markers)
    rad_px = round(args.cell_radius_um / MPP_20X)   # mask-px for the dilation (20×)
    # accumulate per-marker pred↔target tokens across patches for a summary r
    r_acc = {m: ([], []) for m in markers}

    fig, axes = plt.subplots(len(idxs), ncol,
                             figsize=(3.0 * ncol, 3.2 * len(idxs)),
                             squeeze=False)

    for r, pi in enumerate(idxs):
        px, py = int(coords[pi, 0]), int(coords[pi, 1])

        # native 20× label crop — NEVER resampled
        lab = load_patch_labels(tile_index, px, py, ps0)
        mh, mw = lab.shape
        cell = expand_cells(lab, rad_px)               # 2 µm dilation on native mask

        # a cell belongs to this patch iff its centroid is inside the FOV
        m0, n0 = round(px * cell_scale), round(py * cell_scale)
        sel = ((cx_all >= px) & (cx_all < px + ps0) &
               (cy_all >= py) & (cy_all < py + ps0))
        fov_labels = cells.label.values[sel]
        fov_lut = np.zeros(max_lab + 1, dtype=bool)
        fov_lut[fov_labels] = True
        n_cells = int(sel.sum())
        lab = np.where(fov_lut[np.clip(lab,  0, max_lab)], lab,  0)
        cell = np.where(fov_lut[np.clip(cell, 0, max_lab)], cell, 0)

        # centroids → display (20× mask-px) coords
        dotx = cx_all[sel] * cell_scale - m0
        doty = cy_all[sel] * cell_scale - n0

        # background image read at level-0, resized DOWN onto the mask's 20× grid
        if args.modality == "he":
            img = read_he_patch(arr, c_ax, h_ax, w_ax, px, py, ps0, H, W)
        else:
            img = read_if_marker_patch(arr, c_ax, h_ax, w_ax, raw_ch, px, py, ps0, H, W)
        disp_img = cv2.resize(to_uint8(np.asarray(img)), (mw, mh),
                              interpolation=cv2.INTER_AREA)
        cmap = None if disp_img.ndim == 3 else "gray"

        # col 0: raw image + all nucleus outlines
        ax = axes[r][0]
        ax.imshow(disp_img, cmap=cmap)
        outline = (lab > 0) & (ndi.minimum_filter(lab, size=3) != lab)
        oy, ox = np.where(outline)
        ax.scatter(ox, oy, s=0.5, c="cyan", marker=".", alpha=0.6)
        ax.scatter(dotx, doty, s=1.5, c="black", marker=".", linewidths=0)
        ttl0 = f"patch {pi}  ({px},{py})\n{n_cells} cells"
        if pi in diff_labels:
            ttl0 += f"\n{diff_labels[pi]}"
        ax.set_title(ttl0, fontsize=8)
        if not args.no_grid:
            draw_token_grid(ax, mh, mw)
        ax.set_xlim(0, mw); ax.set_ylim(mh, 0)
        ax.set_xticks([]); ax.set_yticks([])

        # per marker: [GT cells | model pred tokens | MIPHEI target tokens]
        for k, m in enumerate(markers):
            base = 1 + k * cols_per_marker

            # GT cells filled
            ax = axes[r][base]
            ax.imshow(disp_img, cmap=cmap)
            pos_mask = pos_lut[m][np.clip(cell, 0, max_lab)] & (cell > 0)
            overlay = np.zeros((mh, mw, 4), dtype=np.float32)
            overlay[pos_mask] = (*to_rgb(MARKER_COLORS[k % len(MARKER_COLORS)]), args.alpha)
            ax.imshow(overlay)
            ax.scatter(dotx, doty, s=1.5, c="black", marker=".", linewidths=0)
            n_pos = len(np.unique(cell[pos_mask]))
            ax.set_title(f"{m}+ cells  ({n_pos})", fontsize=8)
            if not args.no_grid:
                draw_token_grid(ax, mh, mw)
            ax.set_xlim(0, mw); ax.set_ylim(mh, 0)
            ax.set_xticks([]); ax.set_yticks([])

            if not show_pred:
                continue

            # token heatmaps — upscale 16×16 → display grid, shared colour scale
            ch = ch_of[m]
            pt = preds_arr[pi, ch]            # (G, G) my model prediction
            tt = tgt_arr[pi, ch]             # (G, G) GT IF expression
            r_acc[m][0].append(pt.ravel()); r_acc[m][1].append(tt.ravel())
            rr = pearsonr(pt.ravel(), tt.ravel())[0] if (pt.std() > 1e-8 and tt.std() > 1e-8) else np.nan
            panels = [
                (pt, f"{m} mine  r={rr:.2f}" if np.isfinite(rr) else f"{m} mine"),
            ]
            if have_mph:
                mt = miphei_arr[pi, ch]      # (G, G) MIPHEI prediction
                rm = pearsonr(mt.ravel(), tt.ravel())[0] if (mt.std() > 1e-8 and tt.std() > 1e-8) else np.nan
                panels.append((mt, f"{m} MIPHEI  r={rm:.2f}" if np.isfinite(rm) else f"{m} MIPHEI"))
            for off, (hm, ttl) in enumerate(panels):
                ax2 = axes[r][base + 1 + off]
                hm_disp = cv2.resize(hm.astype(np.float32), (mw, mh),
                                     interpolation=cv2.INTER_NEAREST)
                # fixed 0→heat_vmax scale so pred/target panels are comparable everywhere
                ax2.imshow(hm_disp, cmap="viridis", vmin=0, vmax=args.heat_vmax)
                ax2.set_title(ttl, fontsize=8)
                if not args.no_grid:
                    draw_token_grid(ax2, mh, mw)
                ax2.set_xlim(0, mw); ax2.set_ylim(mh, 0)
                ax2.set_xticks([]); ax2.set_yticks([])

    suffix = "  | pred vs MIPHEI target tokens" if show_pred else ""
    fig.suptitle(f"{sample} — {args.modality.upper()} 20× patches (native-mask overlay) "
                 f"+ {args.cell_radius_um} µm dilation{suffix}", fontsize=11)
    fig.tight_layout(rect=[0, 0.05 if show_pred else 0, 1, 0.99])
    if show_pred:
        # one shared colorbar for every viridis token heatmap (pred + target)
        sm  = ScalarMappable(norm=Normalize(0, args.heat_vmax), cmap="viridis")
        cax = fig.add_axes([0.35, 0.02, 0.30, 0.012])
        cb  = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_label(f"token expression   (0 = blue → {args.heat_vmax:g} = yellow)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    tag = f"_diff-{args.diff_marker or markers[0]}" if args.patches.startswith("diff:") else ""
    out_path = out_dir / f"{sample}_{args.modality}_celloverlay{tag}.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\nSaved → {out_path}")

    # do the predictions align with the MIPHEI targets? pooled token r over the
    # shown patches, per marker.
    if show_pred:
        print(f"\n  pred ↔ MIPHEI-target token correlation over {len(idxs)} patches:")
        print(f"  {'Marker':<12} {'pearson r':>10}")
        print("  " + "-" * 24)
        for m in markers:
            p = np.concatenate(r_acc[m][0]); t = np.concatenate(r_acc[m][1])
            rr = pearsonr(p, t)[0] if (p.std() > 1e-8 and t.std() > 1e-8) else np.nan
            print(f"  {m:<12} {rr:>10.4f}" if np.isfinite(rr) else f"  {m:<12} {'nan':>10}")


if __name__ == "__main__":
    main()
