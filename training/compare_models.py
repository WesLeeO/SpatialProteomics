#!/usr/bin/env python
"""
compare_models.py — rigorous fat-val-split comparison of the Orion token models.

WHY: comparing models on the 2-slide MIPHEI val set is pseudoreplication — the unit
of independence is the slide, so 2 slides ≈ n=2 and no comparison is significant.
The fix here is NOT k-fold (too expensive given UNI2 forward cost) but a single
*fat* held-out split: hold out ~8 slides once, train each arm once, then compare on
PER-SLIDE Pearson r across those 8 slides. That gives ~8 paired observations per
arm — enough to see whether a difference is consistent or just noise.

The arms (matched factorial — see model.py SpatialModel vs NeighbourCLSModel):
  frozen_nonbr : frozen UNI2 + global-context transformer, neighbours MASKED   (B)
  frozen_nbr   : frozen UNI2 + global-context transformer, neighbours visible   (A)
  finetune4    : finetune last-4 UNI2 blocks (full) + per-token head            (C)
  lora         : LoRA adapters on UNI2 + per-token head                         (D)
  frozen_head  : frozen UNI2 + per-token head (no context transformer)          (C0, anchor)

Decisive, confound-free comparisons these support:
  frozen_nbr  vs frozen_nonbr  → neighbour effect      (head + backbone held)
  finetune4   vs lora          → full vs low-rank adapt (capacity / overfit check)
  finetune4   vs frozen_head   → backbone-adaptation effect (per-token head held)
  frozen_nonbr vs frozen_head  → context-transformer-head effect (frozen held)

Usage (run inside thesis_env):
  python compare_models.py train                 # train every arm (long, sequential)
  python compare_models.py train --only lora finetune4
  python compare_models.py eval                  # aggregate checkpoints → comparison
  python compare_models.py eval --baseline frozen_nbr
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Importing training_orion_reg is safe (parse_known_args ignores our argv) and gives us
# the constant dataset paths / dims for eval. Per-arm model config comes from each arm's
# saved config.yaml, not from T's defaults.
import training_orion_reg as T
from training_orion_reg import GroupedOnlinePearson
from dataset_orion_reg import OrionSpatialDataset
from model import SpatialModel, NeighbourCLSModel
from config import load_dict, deep_merge, save_config, to_namespace

BASE_CONFIG = "config.yaml"

# ── Comparison config (EDIT THESE) ──────────────────────────────────────────────
COMPARE_DIR = Path("training_outputs/outputs_cmp")          # one subdir per arm
NUM_EPOCHS  = 10
SEED        = 42
EVAL_BATCH  = 1024

# Fat held-out validation set: 8 slides (vs the 2-slide MIPHEI val). Keep the two
# MIPHEI val slides for continuity + 6 more. Avoid the test slides and the CRC33
# serial-section pair (near-duplicate → would leak across train/val). EDIT freely.
VAL_SLIDES  = ["CRC19", "CRC30", "CRC05", "CRC12", "CRC21", "CRC26", "CRC36", "CRC38"]
TEST_SLIDES = ["CRC11", "CRC02"]           # MIPHEI test set

# Slides the comparison is computed on. Both val AND test are held out of training
# (excluded from train_slides), so both are legitimately unseen → pool them for more
# paired observations (8 + 2 = 10 slides). NOTE: best_model.pt is selected on the val
# slides, so val r is slightly optimistic vs test; that bias is shared across arms
# (same selection rule, same slides) so it ~cancels in the per-slide Δ comparison.
EVAL_SLIDES = VAL_SLIDES + TEST_SLIDES

# OVERSAMPLE hard-marker patches during training — ON for EVERY arm so sampling is a
# matched constant (an earlier comparison was confounded by uneven oversampling).
OVERSAMPLE  = True

# Each arm = a config-override dict deep-merged onto config.yaml. Keys mirror the yaml
# structure (finetune.mode, neighbour.use, …), so an arm is just "what differs from base".
ARMS = {
    "frozen_nbr":   {"neighbour": {"use": True,  "mask_neighbours": False}},
    "frozen_nonbr": {"neighbour": {"use": True,  "mask_neighbours": True}},
    "finetune4":    {"neighbour": {"use": False},
                     "finetune": {"mode": "unfreeze", "unfreeze_last_n": 4}},
    # low-rank twin of finetune4: SAME blocks (last 4) + SAME Linears (attn+mlp), only
    # the rank differs → isolates full-rank capacity vs adaptation. lora≈finetune4 on
    # test ⇒ the win is adaptation, not the ~110M extra params. LoRA last-4 is activation-
    # heavier than unfreeze-4 → OOMs at batch 1024, so halve the batch.
    "lora":         {"neighbour": {"use": False},
                     "finetune": {"mode": "lora",
                                  "lora": {"last_n": 4, "rank": 16, "alpha": 32, "dropout": 0.05,
                                           "suffixes": ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]}},
                     "train": {"batch_size": 512}},
    # "frozen_head": {"neighbour": {"use": False}, "finetune": {"mode": "none"}},  # optional anchor
}
DEFAULT_BASELINE = "frozen_nbr"


def arm_dir(name: str) -> Path:
    return COMPARE_DIR / name


def arm_config(name: str):
    """Resolved config namespace for an arm: its saved config.yaml if trained, else the
    base config deep-merged with the arm's overrides (so eval matches what was trained)."""
    saved = arm_dir(name) / "config.yaml"
    d = load_dict(saved) if saved.exists() else deep_merge(load_dict(BASE_CONFIG), ARMS[name])
    return to_namespace(d)


# ── Training orchestration ──────────────────────────────────────────────────────
def train(only=None):
    COMPARE_DIR.mkdir(exist_ok=True)
    base = load_dict(BASE_CONFIG)
    arms = {k: v for k, v in ARMS.items() if (only is None or k in only)}
    print(f"Training {len(arms)} arm(s) on {len(VAL_SLIDES)} held-out val slides "
          f"(train pool = all − {len(VAL_SLIDES)} val − {len(TEST_SLIDES)} test).")
    for name, overrides in arms.items():
        out = arm_dir(name)
        if (out / "best_model.pt").exists():
            print(f"  [skip] {name}: {out/'best_model.pt'} already exists.")
            continue
        # arm overrides + comparison-level matched settings (same across every arm)
        cfg = deep_merge(base, overrides)
        cfg = deep_merge(cfg, {
            "data":  {"split_mode": "miphei", "val_slides": VAL_SLIDES,
                      "test_slides": TEST_SLIDES, "output_dir": str(out)},
            "train": {"oversample": OVERSAMPLE, "num_epochs": NUM_EPOCHS, "seed": SEED},
        })
        out.mkdir(parents=True, exist_ok=True)
        arm_yaml = out / "config.yaml"
        save_config(cfg, arm_yaml)
        print(f"\n=== arm '{name}' → {out} ===\n  overrides: {overrides}")
        subprocess.run([sys.executable, "training_orion_reg.py", "--config", str(arm_yaml)],
                       check=True)


# ── Model reconstruction for eval ───────────────────────────────────────────────
def build_model(cfg):
    """Rebuild an arm's architecture (from its config namespace) so best_model.pt loads
    cleanly. unfreeze only flips requires_grad (no extra params) → build with last_n=0;
    LoRA ADDS adapter params → must inject them to match the checkpoint."""
    if cfg.neighbour.use:
        return NeighbourCLSModel(T.MODEL_NAME, num_outputs=T.NUM_OUTPUTS,
                                 token_grid=T.TOKEN_GRID, unfreeze_last_n=0)
    lora = cfg.finetune.mode == "lora"
    return SpatialModel(
        T.MODEL_NAME, num_outputs=T.NUM_OUTPUTS, token_grid=T.TOKEN_GRID,
        unfreeze_last_n=0,
        lora_last_n=cfg.finetune.lora.last_n if lora else 0,
        lora_rank=cfg.finetune.lora.rank,
        lora_alpha=cfg.finetune.lora.alpha,
        lora_dropout=cfg.finetune.lora.dropout,
        lora_suffixes=tuple(cfg.finetune.lora.suffixes),
    )


def count_trainable(model, cfg) -> int:
    """#trainable params under the arm's FINAL training config. build_model returns
    the encoder frozen (and LoRA adapters re-frozen); here we flip requires_grad to
    mirror training_orion_reg's phase-2 setup, then count — so the number matches what
    was actually optimised."""
    use_nbr = cfg.neighbour.use
    mode    = cfg.finetune.mode
    if not use_nbr and mode == "unfreeze":
        n = cfg.finetune.unfreeze_last_n
        for blk in model.encoder.blocks[-n:]:
            for p in blk.parameters():
                p.requires_grad = True
        for p in model.encoder.norm.parameters():
            p.requires_grad = True
    elif not use_nbr and mode == "lora":
        for nm, p in model.named_parameters():
            if "lora_" in nm:
                p.requires_grad = True
        for p in model.encoder.norm.parameters():
            p.requires_grad = True
    # neighbour / "none": as built (encoder frozen, head/transformer trainable)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def eval_arm(name: str):
    """Return (per_slide {label: (C,) r}, marker_names, n_trainable) on held-out slides."""
    ckpt = arm_dir(name) / "best_model.pt"
    if not ckpt.exists():
        print(f"  [skip] {name}: no checkpoint at {ckpt}")
        return None, None, None
    cfg = arm_config(name)
    use_nbr  = cfg.neighbour.use
    mask_nbr = cfg.neighbour.mask_neighbours

    cache_dir = None
    if use_nbr:
        from cls_cache import build_cls_cache
        build_cls_cache(str(T.H5_DIR), str(T.TIFF_DIR), T.CLS_CACHE_DIR,
                        model_name=T.MODEL_NAME, device=str(T.device))
        cache_dir = T.CLS_CACHE_DIR

    # token_means only feeds the (here-unused) foreground mask → zeros is fine.
    ds = OrionSpatialDataset(
        str(T.H5_DIR), str(T.TIFF_DIR), augment=False, slide_names=EVAL_SLIDES,
        token_means=torch.zeros(T.NUM_OUTPUTS), use_neighbours=use_nbr,
        cls_cache_dir=cache_dir, return_slide_idx=True)
    loader = DataLoader(ds, batch_size=EVAL_BATCH, shuffle=False,
                        num_workers=T.NUM_WORKERS, pin_memory=True)

    model = build_model(cfg).to(T.device)
    model.load_state_dict(torch.load(ckpt, map_location=T.device), strict=False)
    n_trainable = count_trainable(model, cfg)
    model.eval()

    acc = GroupedOnlinePearson(T.NUM_OUTPUTS)
    for batch in loader:
        *rest, slide_ids = batch
        slide_ids = slide_ids.numpy()
        if use_nbr:
            patches, targets, _masks, nbr, present = rest
            present = present.to(T.device)
            if mask_nbr:
                present = torch.zeros_like(present)
            nbr = nbr.to(T.device)
            kw = (dict(neighbour_cls=nbr, neighbour_present=present) if nbr.dim() == 3
                  else dict(neighbours=nbr, neighbour_present=present))
            with torch.amp.autocast("cuda"):
                preds, _ = model(patches.to(T.device), **kw)
        else:
            patches, targets, _masks = rest
            with torch.amp.autocast("cuda"):
                preds, _ = model(patches.to(T.device))
        acc.update(preds.float().cpu().numpy(), targets.numpy(), slide_ids)

    # slide_idx → "CRCxx" label. Use the dataset's stored CRC base_name (the resolved
    # tiff filename is the long scan id, e.g. 19510_C19_..., which won't match VAL_SLIDES).
    label = lambda i: ds.slide_ids[i]
    per_slide = {label(i): r for i, r in acc.per_slide().items()}
    print(f"  {name}: {n_trainable:,} trainable params")
    return per_slide, list(ds.marker_names), n_trainable


def _bootstrap_ci(deltas: np.ndarray, n=5000, seed=0):
    """Percentile bootstrap CI for the mean of slide-level deltas."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(deltas), size=(n, len(deltas)))
    means = deltas[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _train_r_at_best(name: str):
    """Mean train Pearson r at the best-val epoch, read from the arm's saved curves.
    The overfitting signal: train_r ≫ unseen_r ⇒ memorising. NOTE train r is measured
    under training conditions (augment + dropout + oversampling), so it's a CONSERVATIVE
    estimate of fit — a large gap despite that is strong evidence of overfitting."""
    d = arm_dir(name)
    tp_f, vp_f = d / "train_pearsons.npy", d / "val_pearsons.npy"
    if not (tp_f.exists() and vp_f.exists()):
        return np.nan
    tp, vp = np.load(tp_f), np.load(vp_f)          # (epochs, markers)
    best_e = int(vp.mean(axis=1).argmax())          # same epoch best_model.pt was saved
    return float(tp[best_e].mean())


def evaluate(baseline=DEFAULT_BASELINE):
    rows = []          # long: arm, slide, marker, r
    marker_names = None
    n_params = {}      # arm → #trainable params
    for name in ARMS:
        per_slide, markers, n_trainable = eval_arm(name)
        if per_slide is None:
            continue
        marker_names = markers
        n_params[name] = n_trainable
        for slide, r in per_slide.items():
            for m, val in zip(markers, r):
                rows.append(dict(arm=name, slide=slide, marker=m, r=float(val)))
    if not rows:
        print("No trained arms found. Run `python compare_models.py train` first.")
        return
    df = pd.DataFrame(rows)
    COMPARE_DIR.mkdir(exist_ok=True)
    df.to_csv(COMPARE_DIR / "comparison_per_slide.csv", index=False)
    # Persist trainable-param counts so `summary` can rebuild the table without
    # reconstructing (and reloading the weights of) every model.
    pd.Series(n_params, name="n_trainable").to_csv(COMPARE_DIR / "comparison_nparams.csv")
    _summarize(df, baseline)


def _crc_label_map() -> dict:
    """Map long tiff scan-id stems (e.g. 19510_C19_…, 18459_LSP10364_…) → CRC base_name,
    by resolving each EVAL slide's registered tiff. Filesystem only — no torch, no patch
    loading — so `summary` can normalise an old per-slide CSV whose slide column holds the
    long scan ids instead of the CRC ids."""
    m = {}
    for crc in EVAL_SLIDES:
        hits = list((T.TIFF_DIR / crc).glob("*-registered.ome.tif"))
        if len(hits) == 1:
            m[hits[0].name.split("-registered")[0]] = crc
    return m


def _summarize(df: pd.DataFrame, baseline=DEFAULT_BASELINE):
    """Print the per-slide / summary / paired tables from a per-slide (arm,slide,marker,r)
    frame. Pure pandas — no GPU. `split` is recomputed here, never trusted from the CSV."""
    test_set = set(TEST_SLIDES)
    df = df.copy()
    df["split"] = df["slide"].map(lambda s: "test" if s in test_set else "val")
    marker_names = df["marker"].unique()
    n_params = {}
    npz = COMPARE_DIR / "comparison_nparams.csv"
    if npz.exists():
        n_params = pd.read_csv(npz, index_col=0)["n_trainable"].to_dict()

    # per-slide overall r = mean over markers (one number per arm × slide)
    slide_mean = df.groupby(["arm", "slide"])["r"].mean().unstack("arm")   # rows=slide, cols=arm
    arms_present = [a for a in ARMS if a in slide_mean.columns]
    # val first (selection set → the comparable one), test after (held fully out)
    val_idx  = [s for s in VAL_SLIDES  if s in slide_mean.index]
    test_idx = [s for s in TEST_SLIDES if s in slide_mean.index]
    slide_mean = slide_mean.loc[val_idx + test_idx]

    print("\n" + "=" * 78)
    print(f"PER-SLIDE mean Pearson r (mean over {len(marker_names)} markers)")
    print("=" * 78)
    disp = slide_mean[arms_present].copy()
    disp.insert(0, "split", ["test" if s in test_set else "val" for s in disp.index])
    print(disp.to_string(float_format=lambda x: f"{x:.4f}"))

    print("-" * 78)
    print("SUMMARY  (params · fit · unseen performance · overfitting)")
    train_r   = pd.Series({a: _train_r_at_best(a) for a in arms_present})
    test_mean = (slide_mean.loc[test_idx, arms_present].mean()
                 if test_idx else pd.Series(np.nan, index=arms_present))
    summary = pd.DataFrame({
        "trainable_M": pd.Series({a: n_params[a] / 1e6 for a in arms_present if a in n_params}),
        "train_r":   train_r,                                  # fit (best-val epoch)
        "val_mean":  slide_mean.loc[val_idx, arms_present].mean(),
        "test_mean": test_mean,                                # truly unseen
        "gen_gap":   train_r - test_mean,                      # overfitting: train − unseen
    })
    summary.to_csv(COMPARE_DIR / "comparison_summary.csv")
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"  trainable_M = M trained params · train_r = fit @best-val epoch "
          f"(augmented, conservative) · gen_gap = train_r − test_mean (↑ = more overfit)")
    print(f"  val n={len(val_idx)} (selection set) · test n={len(test_idx)} (unseen)")

    if baseline not in arms_present:
        baseline = arms_present[0]
    print("\n" + "=" * 78)
    print(f"PAIRED vs baseline '{baseline}'  (Δ = arm − baseline, per slide)")
    print(f"  primary = {len(val_idx)} VAL slides (bootstrap CI);  "
          f"test Δ = {len(test_idx)} slides shown for confirmation, no CI (too few)")
    print("=" * 78)
    base_val = slide_mean.loc[val_idx, baseline].values
    print(f"{'arm':<14}{'val Δr':>9}{'95% CI (val bootstrap)':>26}{'val wins':>10}{'test Δr':>10}")
    for a in arms_present:
        if a == baseline:
            continue
        d = slide_mean.loc[val_idx, a].values - base_val
        lo, hi = _bootstrap_ci(d)
        wins = int((d > 0).sum())
        flag = "" if lo <= 0 <= hi else "  *"   # CI excludes 0
        test_d = ((slide_mean.loc[test_idx, a] - slide_mean.loc[test_idx, baseline]).mean()
                  if test_idx else float("nan"))
        print(f"{a:<14}{d.mean():>+9.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>26}"
              f"{f'{wins}/{len(d)}':>10}{test_d:>+10.4f}{flag}")
    print("\n  * = 95% bootstrap CI of the mean per-slide VAL Δ excludes 0 (consistent "
          "direction).")
    print("  Trust a result only if the val CI excludes 0 AND the test Δ agrees in sign.")
    print(f"\nWrote per-slide/per-marker table → {COMPARE_DIR/'comparison_per_slide.csv'}")
    print("Per-marker breakdown: pivot that CSV (e.g. df.groupby(['arm','marker']).r.mean()).")


def summarize_from_csv(baseline=DEFAULT_BASELINE):
    """Rebuild the comparison tables from an existing comparison_per_slide.csv — no GPU,
    no patch loading. Normalises long scan-id slide labels back to CRC ids so older CSVs
    (written before slide_ids existed) still summarise correctly."""
    csv = COMPARE_DIR / "comparison_per_slide.csv"
    if not csv.exists():
        print(f"No {csv}. Run `python compare_models.py eval` first.")
        return
    df = pd.read_csv(csv)
    lab = _crc_label_map()
    df["slide"] = df["slide"].map(lambda s: lab.get(s, s))   # identity if already a CRC id
    _summarize(df, baseline)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pt = sub.add_parser("train", help="train each arm once on the fat split")
    pt.add_argument("--only", nargs="+", help="subset of arm names to train")
    pe = sub.add_parser("eval", help="aggregate checkpoints into a comparison (runs models)")
    pe.add_argument("--baseline", default=DEFAULT_BASELINE, help="arm to diff against")
    ps = sub.add_parser("summary", help="rebuild tables from comparison_per_slide.csv (no GPU)")
    ps.add_argument("--baseline", default=DEFAULT_BASELINE, help="arm to diff against")
    args = ap.parse_args()
    if args.cmd == "train":
        train(only=args.only)
    elif args.cmd == "summary":
        summarize_from_csv(baseline=args.baseline)
    else:
        evaluate(baseline=args.baseline)


if __name__ == "__main__":
    main()