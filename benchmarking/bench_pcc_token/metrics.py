#!/usr/bin/env python
"""
metrics.py — sparsity-tiered token-level benchmark for the ORION virtual-stain models.

Global Pearson r is unreliable for sparse markers (FOXP3 ~2% positive tokens): the GT is a
spike-and-slab, so the correlation is dominated by background↔foreground SEPARATION, not by
whether the model got the positive cells right — and it swings ~±0.1 run-to-run. So instead
of one number we score each marker on TWO honest axes plus a structural check, and pick the
PRIMARY metric per marker by its prevalence tier:

  axis            question                                   metric
  ------------    ---------------------------------------    -------------------------
  localization    did you fire on the right tokens?          AUC-PR (average precision)
  intensity       given a positive token, is the level OK?   PCC+  (Pearson on GT-pos)
  structure       spatial pattern within the patch           SSIM  (dense markers; fg-masked)
  (calibration)   cross-token level                          global PCC  (dense primary)
  (background)    false positives — DIAGNOSTIC, not scored   bg_mass / bg_std on GT-neg

Positive definition (the crux): GT has no cell-gating, only continuous token intensity, so a
"positive" token is a threshold on GT intensity. The threshold is GT-ONLY (identical across
all models) and computed PER-SLIDE (default) via a 2-component GMM — the low mode is
background/AF, the high mode is real signal, the crossover is the cut. Per-slide decouples the
localization metrics from per-slide staining intensity. `> token-mean` (the training mask) is
NOT used here: the mean is dragged by the bright tail and sweeps in low-level noise.
Fallback chain if the GMM is degenerate: Otsu → token-mean.

AP caveat: a random classifier scores AP = prevalence, so we also report AP / prevalence
("AP_norm") for fair cross-marker comparison.

Robustness: we report each metric as MEAN ± STD across the scored slides (the cross-slide
spread IS the noise we care about), and optionally a slide-level bootstrap CI.

Usage
-----
  # baseline + all cached competitors, default 4 MIPHEI holdout slides:
  python metrics.py
  # add your new run once its preds are dumped (visualize_orion_predictions.py):
  python metrics.py --models ours-bg0.2,ours-2loss,miphei-vit,miphei-convnext,hemit,pix2pix
  python metrics.py --slides CRC30 --thresh otsu --bootstrap 1000

Pred sources: "ours-*" → REPO/outputs_orion_token_UNI2_<...>/{slide}_preds.npy ;
cached competitors → preds_cache/<key>/{slide}_preds.npy. GT targets come from --gt_dir
(default the bg0.2 baseline dir, which co-locates {slide}_targets.npy + {slide}_names.npy).
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "preds_cache"

# Default GT + "ours" baseline (co-locates {slide}_targets.npy + {slide}_names.npy).
GT_DIR = REPO / "training_outputs/outputs_orion_token_UNI2_baseline_bg0.2"
MIPHEI_HOLDOUT = ["CRC02", "CRC11", "CRC19", "CRC30"]   # val+test (excluded from training)

IMMUNE = {"CD4", "FOXP3", "CD8a", "CD45RO", "CD20", "PD-L1", "CD3e", "CD45", "CD163", "CD68"}

# model key → predictions directory (relative to REPO unless under preds_cache)
MODEL_DIRS = {
    "ours-bg0.2":      REPO / "training_outputs/outputs_orion_token_UNI2_baseline_bg0.2",
    "ours-2loss":      REPO / "training_outputs/outputs_orion_token_UNI2_baseline_unfreeze4_2loss_lbg8",
    "miphei-vit":      CACHE / "miphei-vit",
    "miphei-convnext": CACHE / "miphei-convnext",
    "hemit":           CACHE / "hemit",
    "pix2pix":         CACHE / "pix2pix",
}


def flat(a):
    """(N, C, G, G) → (N*G*G, C)."""
    N, C, G, _ = a.shape
    return a.transpose(0, 2, 3, 1).reshape(-1, C)


def load_gt(gt_dir, slide):
    tf = gt_dir / f"{slide}_targets.npy"
    nf = gt_dir / f"{slide}_names.npy"
    if not tf.exists():
        raise FileNotFoundError(f"GT targets missing: {tf}")
    names = [str(n) for n in np.load(nf, allow_pickle=True)] if nf.exists() else None
    return np.load(tf), names


def load_preds(model, slide):
    pf = MODEL_DIRS[model] / f"{slide}_preds.npy"
    if not pf.exists():
        return None
    return np.load(pf)


# ── positive-token threshold (GT-only, per-slide) ───────────────────────────────────────────
def gmm_threshold(x, rng):
    """2-component 1-D GMM crossover between the low (bg) and high (signal) modes.
    Returns (thr, ok). thr lies strictly between the two component means; ok=False if the
    fit is degenerate (components collapse) so the caller can fall back."""
    from sklearn.mixture import GaussianMixture
    xs = x if x.size <= 200_000 else rng.choice(x, 200_000, replace=False)
    xs = xs.reshape(-1, 1).astype(np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm = GaussianMixture(n_components=2, covariance_type="full",
                             n_init=2, random_state=0).fit(xs)
    mu = gm.means_.ravel()
    lo, hi = mu.min(), mu.max()
    if hi - lo < 1e-4:
        return None, False
    grid = np.linspace(lo, hi, 512).reshape(-1, 1)
    hi_comp = int(np.argmax(mu))
    post = gm.predict_proba(grid)[:, hi_comp]
    cross = np.where(post >= 0.5)[0]
    if cross.size == 0:
        return None, False
    return float(grid[cross[0], 0]), True


def positive_mask(gt_col, method, rng):
    """Boolean positive mask for one marker on one slide + the threshold used + its name."""
    if method == "mean":
        thr = float(gt_col.mean()); return gt_col > thr, thr, "mean"
    if method == "quantile":
        thr = float(np.quantile(gt_col, 0.95)); return gt_col > thr, thr, "q95"
    if method == "otsu":
        from skimage.filters import threshold_otsu
        thr = float(threshold_otsu(gt_col)); return gt_col > thr, thr, "otsu"
    # default: gmm with fallback chain
    thr, ok = gmm_threshold(gt_col, rng)
    if ok:
        return gt_col > thr, thr, "gmm"
    try:
        from skimage.filters import threshold_otsu
        thr = float(threshold_otsu(gt_col)); return gt_col > thr, thr, "otsu*"
    except Exception:
        thr = float(gt_col.mean()); return gt_col > thr, thr, "mean*"


# ── metric primitives ───────────────────────────────────────────────────────────────────────
def pcc(a, b):
    if a.size < 3 or a.std() < 1e-8 or b.std() < 1e-8:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def average_precision(y_true, score):
    if y_true.sum() == 0 or y_true.all():
        return np.nan
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y_true, score))


def patch_ssim(pred, gt, c, rng, n_patches, fg_only, thr):
    """Mean per-patch SSIM on the (G,G) token map for marker c, over a patch subsample.
    fg_only restricts the averaged SSIM map to patches containing ≥1 GT-positive token
    (avoids the all-background patches that make sparse-marker SSIM trivially ~1)."""
    from skimage.metrics import structural_similarity as ssim
    N = pred.shape[0]
    idx = np.arange(N)
    if fg_only:
        has_pos = (gt[:, c] > thr).reshape(N, -1).any(axis=1)
        idx = idx[has_pos]
        if idx.size == 0:
            return np.nan
    if idx.size > n_patches:
        idx = rng.choice(idx, n_patches, replace=False)
    vals = []
    for i in idx:
        p, g = pred[i, c], gt[i, c]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vals.append(ssim(g, p, data_range=1.0, win_size=7))
    return float(np.mean(vals)) if vals else np.nan


# ── per (model, slide, marker) ───────────────────────────────────────────────────────────────
def score_slide(model, slide, gt_dir, method, ssim_patches, rng):
    gt, names = load_gt(gt_dir, slide)
    pred = load_preds(model, slide)
    if pred is None:
        return None
    if pred.shape != gt.shape:
        raise ValueError(f"{model}/{slide}: pred {pred.shape} != gt {gt.shape}")
    gf, pf = flat(gt), flat(pred)                                  # (T, C)
    rows = []
    for c, m in enumerate(names):
        g, p = gf[:, c], pf[:, c]
        pos, thr, tname = positive_mask(g, method, rng)
        prev = float(pos.mean())
        neg = ~pos
        ap = average_precision(pos, p)
        rows.append(dict(
            model=model, slide=slide, marker=m, prevalence=prev, thr=thr, thr_method=tname,
            PCC=pcc(p, g),
            PCC_pos=pcc(p[pos], g[pos]) if pos.sum() >= 3 else np.nan,
            AUC_PR=ap,
            AP_norm=(ap / prev) if (ap == ap and prev > 0) else np.nan,
            SSIM=patch_ssim(pred, gt, c, rng, ssim_patches, fg_only=(m in IMMUNE), thr=thr),
            bg_mass=float(p[neg].mean()) if neg.any() else np.nan,
            bg_std=float(p[neg].std()) if neg.any() else np.nan,
        ))
    return rows


def tier(prev):
    return "dense" if prev >= 0.10 else ("sparse" if prev < 0.05 else "mid")


PRIMARY = {"dense": "PCC", "mid": "AUC_PR", "sparse": "AUC_PR"}
METRIC_COLS = ["PCC", "PCC_pos", "AUC_PR", "AP_norm", "SSIM", "bg_mass", "bg_std"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="ours-bg0.2,miphei-vit,miphei-convnext,hemit,pix2pix")
    ap.add_argument("--slides", default=",".join(MIPHEI_HOLDOUT))
    ap.add_argument("--gt_dir", default=str(GT_DIR))
    ap.add_argument("--thresh", default="gmm", choices=["gmm", "otsu", "mean", "quantile"])
    ap.add_argument("--ssim_patches", type=int, default=2000)
    ap.add_argument("--bootstrap", type=int, default=0, help="slide-level bootstrap iters (0=off)")
    ap.add_argument("--out", default=str(HERE / "results_metrics"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    models = [s.strip() for s in args.models.split(",") if s.strip()]
    slides = [s.strip() for s in args.slides.split(",") if s.strip()]
    gt_dir = Path(args.gt_dir)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_rows = []
    for model in models:
        if model not in MODEL_DIRS:
            print(f"  ! unknown model '{model}' — skip"); continue
        got = 0
        for slide in slides:
            r = score_slide(model, slide, gt_dir, args.thresh, args.ssim_patches, rng)
            if r is None:
                print(f"  - {model}/{slide}: no preds (skip)"); continue
            all_rows.extend(r); got += 1
        print(f"  ✓ {model}: scored {got}/{len(slides)} slides")
    if not all_rows:
        print("no predictions found for any model/slide — nothing scored."); return

    df = pd.DataFrame(all_rows)
    df.to_csv(out / "per_slide_metrics.csv", index=False)

    # aggregate across slides: mean ± std (cross-slide spread = the noise that matters)
    agg = (df.groupby(["model", "marker"])[METRIC_COLS + ["prevalence"]]
             .agg(["mean", "std"]))
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg = agg.reset_index()
    agg["tier"] = agg["prevalence_mean"].apply(tier)
    agg["primary_metric"] = agg["tier"].map(PRIMARY)
    agg["primary_value"] = agg.apply(lambda r: r[f"{r.primary_metric}_mean"], axis=1)
    agg.to_csv(out / "per_marker_metrics.csv", index=False)

    # console summary: primary metric per marker, models side by side
    marker_order = (agg.drop_duplicates("marker").sort_values("prevalence_mean")["marker"].tolist())
    piv = agg.pivot_table(index="marker", columns="model", values="primary_value")
    piv = piv.reindex(marker_order)
    tiers = agg.drop_duplicates("marker").set_index("marker")["tier"]
    prim = agg.drop_duplicates("marker").set_index("marker")["primary_metric"]
    prev = agg.drop_duplicates("marker").set_index("marker")["prevalence_mean"]
    print("\n=== PRIMARY metric per marker (dense→PCC, mid/sparse→AUC-PR), models side by side ===")
    hdr = f"{'marker':12}{'prev':>6}{'tier':>7}{'primary':>8}  " + "".join(f"{m:>16}" for m in piv.columns)
    print(hdr)
    for mk in piv.index:
        line = f"{mk:12}{prev[mk]:6.3f}{tiers[mk]:>7}{prim[mk]:>8}  " + "".join(
            f"{(piv.loc[mk, m] if not np.isnan(piv.loc[mk, m]) else float('nan')):16.3f}"
            for m in piv.columns)
        print(line)

    # tier-mean of the primary metric per model
    print("\n=== tier means of primary metric (per model) ===")
    tm = agg.copy()
    tmean = tm.pivot_table(index="tier", columns="model", values="primary_value", aggfunc="mean")
    print(tmean.round(3).to_string())

    if args.bootstrap > 0:
        boot_slide_ci(df, slides, models, args.bootstrap, rng, out)
    print(f"\nwrote → {out}/per_slide_metrics.csv, per_marker_metrics.csv")


def boot_slide_ci(df, slides, models, iters, rng, out):
    """Slide-level bootstrap CI on the primary metric, pooled per (model, marker).
    Resamples the scored slides with replacement (the honest unit, since cross-slide
    variance is the concern) and recomputes the slide-mean. Coarse with few slides."""
    rows = []
    for (model, marker), g in df.groupby(["model", "marker"]):
        prev = g["prevalence"].mean()
        col = PRIMARY[tier(prev)]
        vals = g.set_index("slide")[col]
        samp = vals.dropna()
        if samp.empty:
            continue
        bs = [samp.sample(len(samp), replace=True, random_state=rng.integers(1 << 31)).mean()
              for _ in range(iters)]
        rows.append(dict(model=model, marker=marker, metric=col, mean=float(samp.mean()),
                         ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5))))
    pd.DataFrame(rows).to_csv(out / "bootstrap_ci.csv", index=False)
    print(f"wrote → {out}/bootstrap_ci.csv ({iters} iters, slide-level)")


if __name__ == "__main__":
    main()