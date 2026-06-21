"""
Qualitative model comparison on the ORION token grid — *why* is one model better than
another for a given marker?  Built around the two failure modes we identified:

  1. OVERESTIMATION  — does the model keep true-background tokens near 0, or paint
                       spurious signal on empty tissue?
  2. FOREGROUND LIGHT-UP — does it actually raise its prediction on the rare tokens
                       where the marker IS present, or leave them at background level?

Pearson r (the benchmark metric) is driven by the positive tail, so these two behaviours
explain most of the per-marker / per-slide differences. This script makes them visible.

Two outputs per (slide, marker):
  • <out>/qual_<slide>_<marker>_gallery.png
      Token-heatmap gallery. Rows = example patches (the N_FG with the strongest GT
      signal, then N_BG near-empty patches); columns = [H&E] GT + each model. Shared
      colour scale per row → you can SEE whether a model lights up the positives (top
      rows) and whether it stays dark on background (bottom rows).
  • <out>/qual_<slide>_summary.png
      Per-marker bars across models: mean pred on background tokens (overestimation),
      mean pred on the true top-1% tokens vs the GT level (light-up), and Pearson r.

Model predictions are read from cached (N, C, G, G) npy (same caches the benchmark uses).

Usage
-----
  python compare_models_qualitative.py                       # default slides/markers
  python compare_models_qualitative.py --slides CRC11 CRC02 --markers FOXP3 CD8a --he
"""
import argparse
from pathlib import Path

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

HERE   = Path(__file__).resolve().parent
REPO   = HERE.parent.parent
GT_DIR = REPO / "datasets/orion_crc_patch_dataset_benchmark"
TIFF_DIR = Path("/mnt/ssd1/virtual_proteomics/data/ORION_CRC")

# model label → directory holding "<slide>_preds.npy" as (N, C, G, G). Edit freely.
MODELS = {
    "ours":       REPO / "training_outputs/outputs_orion_token_UNI2_cls_neighbours_bg0.1",
    "MIPHEI-ViT": HERE.parent / "preds_cache" / "miphei-vit",
    "MIPHEI-CNX": HERE.parent / "preds_cache" / "miphei-convnext",
    "Pix2Pix":    HERE.parent / "preds_cache" / "pix2pix",
    "HEMIT":      HERE.parent / "preds_cache" / "hemit",
}


def load_gt(slide):
    with h5py.File(GT_DIR / f"{slide}_patch_dataset.h5", "r") as f:
        gt     = f["targets"][:].astype(np.float32)               # (N, C, G, G)
        coords = f["coords"][:]
        psz    = int(f.attrs["patch_size_level0"])
        names  = [x.decode() if isinstance(x, bytes) else str(x) for x in f.attrs["marker_names"]]
    return gt, coords, psz, names


def load_preds(slide, n_expected, c_expected):
    """Return {label: (N, C, G, G)} for every model whose cache exists for this slide."""
    out = {}
    for label, d in MODELS.items():
        p = d / f"{slide}_preds.npy"
        if not p.exists():
            print(f"  [skip] {label}: no cache at {p}")
            continue
        a = np.load(p).astype(np.float32)
        if a.shape[0] != n_expected or a.shape[1] != c_expected:
            print(f"  [skip] {label}: shape {a.shape} != ({n_expected}, {c_expected}, …)")
            continue
        out[label] = a
    return out


def read_he(coords_xy, psz, slide, size=128):
    """Read H&E crops for the selected patches from the registered OME-TIFF (optional)."""
    import tifffile, zarr, cv2
    sd = TIFF_DIR / slide
    tif_paths = list(sd.glob("*-registered.ome.tif"))
    if not tif_paths:
        print(f"  [he] no registered.ome.tif under {sd} — skipping H&E column")
        return None
    z = zarr.open(tifffile.TiffFile(tif_paths[0]).aszarr(), mode="r")
    arr = z["0"] if isinstance(z, zarr.hierarchy.Group) else z
    crops = []
    for x, y in coords_xy:
        c = np.array(arr[int(y):int(y) + psz, int(x):int(x) + psz, :])
        crops.append(cv2.resize(c, (size, size)))
    return crops


def pick_patches(gt_m, n_fg, n_bg, seed=0):
    """Patch indices: strongest-foreground (by per-patch max), then near-empty ones."""
    pmax = gt_m.reshape(gt_m.shape[0], -1).max(1)
    fg = np.argsort(pmax)[::-1][:n_fg]
    rng = np.random.default_rng(seed)
    bg_pool = np.where(pmax <= np.percentile(pmax, 50))[0]
    bg = rng.choice(bg_pool, size=min(n_bg, len(bg_pool)), replace=False)
    return list(fg) + list(bg), len(fg)


def gallery(slide, marker, ci, gt, preds, coords, psz, n_fg, n_bg, he, out_dir):
    gt_m = gt[:, ci]                                              # (N, G, G)
    idxs, n_fg_real = pick_patches(gt_m, n_fg, n_bg)
    he_crops = read_he(coords[idxs], psz, slide) if he else None

    labels = list(preds.keys())
    cols   = (["H&E"] if he_crops else []) + ["GT"] + labels
    fig, axes = plt.subplots(len(idxs), len(cols),
                             figsize=(1.9 * len(cols), 1.9 * len(idxs)), squeeze=False)
    for r, pi in enumerate(idxs):
        # shared scale per row: cover GT and all model preds so under/over-shoot both show
        vmax = max(gt_m[pi].max(), max(preds[l][pi, ci].max() for l in labels), 1e-6)
        c = 0
        if he_crops:
            axes[r, c].imshow(he_crops[r]); axes[r, c].axis("off")
            if r == 0: axes[r, c].set_title("H&E", fontsize=10)
            c += 1
        for name, img in [("GT", gt_m[pi])] + [(l, preds[l][pi, ci]) for l in labels]:
            ax = axes[r, c]
            ax.imshow(img, cmap="magma", vmin=0, vmax=vmax); ax.axis("off")
            if r == 0: ax.set_title(name, fontsize=10)
            c += 1
        tag = "FG" if r < n_fg_real else "bg"
        axes[r, 0].set_ylabel(f"{tag} #{pi}", fontsize=8, rotation=0, ha="right", labelpad=18)
    fig.suptitle(f"{slide} · {marker} — token maps (shared scale per row; top={n_fg_real} "
                 f"strongest-FG patches, bottom={len(idxs)-n_fg_real} background)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    f = out_dir / f"qual_{slide}_{marker}_gallery.png"
    plt.savefig(f, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  saved → {f}")


def overlap_at_k(pred_flat, gt_pos_mask, k):
    """Matched top-k spatial overlap: take the model's k highest tokens (k = #GT
    positives) and intersect with the true-positive set. With matched sizes,
    precision == recall == TP/k. Returns (TP, FP, FN, precision)."""
    if k == 0:
        return 0, 0, 0, float("nan")
    topk = np.argpartition(pred_flat, -k)[-k:]      # model's k highest tokens
    tp   = int(gt_pos_mask[topk].sum())             # of those, how many truly positive
    fp   = k - tp                                   # model said top-k but GT background
    fn   = k - tp                                   # GT positive but missed by model's top-k
    return tp, fp, fn, tp / k


def summary(slide, markers, idxmap, gt, preds, out_dir):
    labels = list(preds.keys())
    fig, axes = plt.subplots(len(markers), 4, figsize=(17, 3.0 * len(markers)), squeeze=False)
    x = np.arange(len(labels))
    for r, m in enumerate(markers):
        ci = idxmap[m]
        g  = gt[:, ci].ravel()
        thr = g.mean(); bg = g <= thr
        gt_pos = g > thr                            # true-positive (foreground) tokens
        k = int(gt_pos.sum())                       # matched-k = number of true positives
        hi = g >= np.percentile(g, 99)
        bg_pred  = [preds[l][:, ci].ravel()[bg].mean() for l in labels]
        hi_pred  = [preds[l][:, ci].ravel()[hi].mean() for l in labels]
        rs       = [pearsonr(preds[l][:, ci].ravel(), g)[0] for l in labels]
        prec     = []
        print(f"  {slide} {m}: k(true+)={k}  (TP/FP/FN per model, matched top-k)")
        for l in labels:
            tp, fp, fn, p = overlap_at_k(preds[l][:, ci].ravel(), gt_pos, k)
            prec.append(p)
            print(f"     {l:<12} TP={tp:6d} FP={fp:6d} FN={fn:6d}  precision@k=recall@k={p:.3f}  r={pearsonr(preds[l][:,ci].ravel(),g)[0]:+.3f}")
        # 1) background (overestimation): lower = better, GT≈0 reference
        axes[r, 0].bar(x, bg_pred, color="tomato"); axes[r, 0].axhline(g[bg].mean(), color="k", ls="--", lw=0.8)
        axes[r, 0].set_title(f"{m}: mean pred on BACKGROUND\n(lower=better, GT={g[bg].mean():.4f})", fontsize=9)
        # 2) foreground light-up: closer to GT line = better
        axes[r, 1].bar(x, hi_pred, color="seagreen"); axes[r, 1].axhline(g[hi].mean(), color="k", ls="--", lw=0.8)
        axes[r, 1].set_title(f"{m}: mean pred on TRUE TOP-1%\n(closer to GT={g[hi].mean():.4f} dashed=better)", fontsize=9)
        # 3) spatial overlap@k — did it light up the RIGHT tokens (precision@k=recall@k)
        axes[r, 2].bar(x, prec, color="goldenrod"); axes[r, 2].set_ylim(0, 1)
        axes[r, 2].set_title(f"{m}: top-k spatial overlap\n(prec@k=rec@k, k={k} true+; higher=better)", fontsize=9)
        # 4) pearson r
        axes[r, 3].bar(x, rs, color="steelblue"); axes[r, 3].axhline(0, color="k", lw=0.6)
        axes[r, 3].set_title(f"{m}: Pearson r", fontsize=9)
        for cc in range(4):
            axes[r, cc].set_xticks(x); axes[r, cc].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.suptitle(f"{slide} — background calibration · foreground light-up · spatial overlap@k · r", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    f = out_dir / f"qual_{slide}_summary.png"
    plt.savefig(f, dpi=130, bbox_inches="tight"); plt.close(fig)
    print(f"  saved → {f}")


def main(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for slide in args.slides:
        print(f"\n=== {slide} ===")
        gt, coords, psz, names = load_gt(slide)
        idxmap = {n: i for i, n in enumerate(names)}
        markers = [m for m in args.markers if m in idxmap]
        if not markers:
            print(f"  none of {args.markers} in {names}"); continue
        preds = load_preds(slide, gt.shape[0], gt.shape[1])
        if not preds:
            print("  no model caches found"); continue
        for m in markers:
            gallery(slide, m, idxmap[m], gt, preds, coords, psz, args.n_fg, args.n_bg, args.he, out_dir)
        summary(slide, markers, idxmap, gt, preds, out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--slides",  nargs="+", default=["CRC02", "CRC11", "CRC30"])
    ap.add_argument("--markers", nargs="+", default=["FOXP3", "CD8a", "PD-L1"])
    ap.add_argument("--n_fg", type=int, default=6, help="strongest-foreground patches in gallery")
    ap.add_argument("--n_bg", type=int, default=2, help="background patches in gallery")
    ap.add_argument("--he", action="store_true", help="add H&E column (reads registered OME-TIFF)")
    ap.add_argument("--out_dir", default="results/qualitative")
    main(ap.parse_args())
