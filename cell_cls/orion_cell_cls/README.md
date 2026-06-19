# orion_cell_cls — cell-level positivity from token predictions

Turn the model's 16×16 **token** predictions into **per-cell** marker calls and
score them (TP/FP/…) against the GT positivity labels, to see whether the
generator is good enough for cell-level readouts.

All scripts share **`utils.py`** (paths, grid/mask geometry, the cell↔token↔mask
helpers, the cached-pred loader, and the metrics). Run them from the repo root,
e.g. `python orion_cell_cls/eval_deterministic.py ...`.

## Pipeline

1. **`build_cell_token_features.py`** — per slide, walk every benchmark patch,
   read its `(16,16)` token prediction grid + the native-20× nuclei mask crop,
   and for each cell (attributed to the single patch holding its centroid) record
   per marker:
   - `mean_<marker>` — area-weighted mean predicted intensity over the tokens the
     cell footprint overlaps, weighting each token by the footprint area inside it
   plus `tile` (source patch idx, the bootstrap unit), `area_px`, `n_tokens`,
   centroid `x,y`, and `gt_<marker>_pos`.
   → one parquet of `cells × markers`.

2a. **`eval_deterministic.py`** — quick floor: a single threshold per marker.

2b. **`train_cell_classifier.py`** — MIPHEI-style strong eval (see below).

### Deterministic rule (floor)
```
cell is <marker>+   iff   mean_<marker> > threshold
```
Single threshold per marker (fixed `--threshold T`, or per-marker F1-max sweep).
Sensitive to per-channel intensity calibration — that's why MIPHEI prefers a probe.

### LR linear probe (MIPHEI-style, the reliable eval)
`train_cell_classifier.py` reproduces MIPHEI's protocol:
- features = `mean_<marker>` (predicted expression), labels = `gt_<marker>_pos`;
- `OneVsRestClassifier(LogisticRegression(class_weight="balanced"))` on a
  `StandardScaler` — multi-label, robust to channel-calibration differences;
- **train on VAL slides, evaluate on TEST slides** (OrionCRC: VAL=CRC19,CRC30;
  TEST=CRC11,CRC02);
- **tile bootstrap** (1000×, seed 42, resample `tile` groups with replacement) →
  AUPRC + F1 with 95% CIs.

## Usage
```bash
# 1) build per-slide features (val + test) — ~a few min mask I/O each
for s in CRC19 CRC30 CRC11 CRC02; do
  python orion_cell_cls/build_cell_token_features.py --sample $s \
      --pred_dir outputs_orion_token_UNI2_baseline_bg0.2 \
      --out orion_cell_cls/cell_token_features_$s.parquet
done

# 2a) quick floor (single slide ok)
python orion_cell_cls/eval_deterministic.py \
    --features orion_cell_cls/cell_token_features_CRC11.parquet

# 2b) strong eval: probe trained on VAL, scored on TEST + bootstrap CIs
python orion_cell_cls/train_cell_classifier.py \
    --train orion_cell_cls/cell_token_features_CRC19.parquet \
            orion_cell_cls/cell_token_features_CRC30.parquet \
    --test  orion_cell_cls/cell_token_features_CRC11.parquet \
            orion_cell_cls/cell_token_features_CRC02.parquet
```

> Predictions must be cached for every split slide (`<slide>_preds.npy` in
> `--pred_dir`). `bg0.2` has all four; other models may need
> `visualize_orion_predictions.py` run on the missing slides first.

## Relation to MIPHEI
MIPHEI's GT `_pos` labels (from Zenodo) come from **GMM-gating the real IF**
per-cell intensities (`preprocessings/single_cell_analysis/`). For *scoring
predictions* the MIPHEI benchmark trains a supervised classifier — an
`OneVsRestClassifier(XGBClassifier)` (also a logreg variant) on the **predicted**
per-cell expression (`regionprops` mean over each nucleus), reporting ROC-AUC /
AUPRC / Balanced-Acc / F1 (`benchmark/evaluators/utils.py:train_xgboost`).

This folder is the **deterministic, threshold-only** counterpart:
- same inputs (predicted per-cell expression vs GT `_pos`) and threshold-free
  scores (ROC-AUC, AUPRC) for an apples-to-apples comparison,
- but **no trained classifier** — a single threshold per marker. It answers
  "is the raw token signal already separable at the cell level?" If this is
  competitive, the XGBoost stage isn't buying much; if not, it quantifies the
  headroom a learned classifier could recover.
- aggregation is the token-resolution analog of MIPHEI's `regionprops` mean,
  since our model emits a coarse token grid rather than pixel-level IF.

## Caveats
- **Single slide / no split.** Only CRC30 has cached preds for the `unfreeze4`
  model, so this is a within-slide sanity check, not a generalization test.
  Build features for more slides → concatenate → hold a slide out for an honest
  evaluation (matching MIPHEI's slide-level split).
- **Headline metrics: AUPRC + F1.** Most markers are rare (immune ~2–7%
  prevalence), so ROC-AUC is over-optimistic — compare AUPRC against the no-skill
  baseline (= prevalence, printed in the summary). F1 is the operating-point score.
  ROC-AUC is kept in the table for reference only.
- **Best-F1 threshold peeks at the labels** — treat swept F1 as an upper bound.
  AUPRC is threshold-free. For a deployable rule, fix the threshold on a train
  slide and apply it to a test slide.