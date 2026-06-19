# pathocell_cls — cell-TYPE classification on PathoCell (external CRC CODEX)

Benchmark the token model on **PathoCell** (CRC FFPE CODEX, Schürch et al.) under
MIPHEI's exact protocol. Built to mirror `orion_cell_cls` (same file names, same
`utils.py` / `build_cell_token_features.py` / `eval_cell_auprc.py` /
`reproduce_miphei.py` / `plot_metrics.py` layout and conventions). The one
semantic difference: PathoCell is scored as **cell-type classification** (15 coarse
types) rather than per-marker positivity — the model predicts the 16 ORION markers
zero-shot and a logreg maps those to cell type, on MIPHEI's fixed **20% train / 80%
test** split.

We reuse MIPHEI's processed data so our cells are byte-aligned to theirs — only the
*predicted expression* differs.

## Data (on disk)

`/mnt/ssd/virtual_proteomics/data/pathocell/pathocell_hdf/<core>.hdf` — 112 cores
(`regNNN_A` / `regNNN_B`). Each holds:
- `img` (3,H,W uint16) brightfield H&E (best-Z) → **model input**
- `gt_inst` (1,H,W) nuclei instance mask; **label == MIPHEI `cell_id`**
- `gt_ct` / `gt_ct_coarse` per-pixel cell-type GT (we don't re-derive it)
- `ifl` (58,H,W) CODEX IF (unused here)

GT cell-type one-hot + the `split` column come straight from MIPHEI's released
`checkpoints/MIPHEI-vit/pathocell_cell_dataframe_logreg.parquet`.
**Alignment verified:** `gt_inst` labels of a core == the `cell_id`s in that parquet,
exactly (reg001_A: 1164 == 1164; full build merged all 220,977 cells, 0 dropped).

## Pipeline (mirrors orion_cell_cls)

1. **`build_cell_token_features.py`** — per core: `img` uint16→uint8 (per-channel
   p99 stretch, same as `build_patch_dataset_pancancer.export_he_rgb`), tile into
   `PATCH_SIZE_LEVEL0 = round(224·0.5/0.377) = 297` px crops (same 112 µm FOV the
   model trained on @ 0.5 µm/px; the HDF is native 0.377 µm/px), run the model →
   16×16 token grid per crop, and for each nucleus record `mean_<marker>` =
   **area-weighted mean** predicted intensity over the tokens its footprint overlaps
   (weight = footprint area per token). Whole-core non-overlapping grid, so a nucleus
   on a tile boundary correctly gets contributions from both tiles. Joins MIPHEI's GT
   one-hot + `split` on `(slide_name, cell_id)` → one parquet
   `cell_token_features_<tag>.parquet`. The bootstrap unit (`tile`) is the **core**.

2. **`eval_cell_auprc.py`** — MIPHEI's `src/metrics.train_logreg`: features = every
   `<marker>_pred` column, labels = the 15 types; `StandardScaler` +
   `OneVsRestClassifier(LogisticRegression(class_weight="balanced", random_state=42))`
   fit on `split==train`, scored on `split==test`. Per-type ROC-AUC / **AUPRC** /
   **F1** + the **MEAN15** headline, with a **test-core bootstrap** (seed 42) for
   95% CIs. Writes `eval_cell_auprc_<tag>.csv` (+ `cell_predictions_<tag>.csv`).
   `--source ours | parquet`.

3. **`reproduce_miphei.py`** — the trust check (below).

4. **`plot_metrics.py`** — bar charts from the `eval_cell_auprc_<tag>.csv` files:
   `model_comparison_<metric>.png` (mean-15 per model) and `--compare A B`
   (per-type, two models).

## Usage

Run from the repo root. The model needs the local UNI2 cache + a free GPU:

```bash
export HF_HOME=/home/wesley/spatial_proteomics/foundation_models HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1            # pick a GPU with headroom

# 1) build our per-cell features over all 112 cores
python cell_cls/pathocell_cls/build_cell_token_features.py \
    --pred_dir outputs_orion_token_UNI2_baseline_bg0.2 --tag bg0.2 --batch_size 16

# 2) score our model (logreg, 20/80 split, core bootstrap)
python cell_cls/pathocell_cls/eval_cell_auprc.py --tag bg0.2

# 3) score every MIPHEI baseline through the SAME harness (for the plots)
for m in MIPHEI-vit MIPHEI-convnext DiffusionFT Rosie-ORION Pix2Pix HEMIT; do
  python cell_cls/pathocell_cls/eval_cell_auprc.py --source parquet \
      --pred_parquet checkpoints/$m/pathocell_cell_dataframe_logreg.parquet
done

# 4) plots
python cell_cls/pathocell_cls/plot_metrics.py --metric auprc                 # per-model
python cell_cls/pathocell_cls/plot_metrics.py --metric auprc --compare bg0.2 MIPHEI-vit
```

## Is the edge real? — `reproduce_miphei.py`

This runs OUR exact logreg on each MIPHEI checkpoint's *own* released predictions
and checks we recover their published `pathocell_logreg.csv` numbers. If we do, the
harness is unbiased, so a higher score for our model is a real edge, not a scoring
artifact.

```bash
python cell_cls/pathocell_cls/reproduce_miphei.py
```

Result — **all 6 MIPHEI models reproduced** (max |Δ| ≤ 0.005, tol 0.01):

| model | their AUPRC | ours (repro) | max\|ΔAUPRC\| |
|---|---|---|---|
| MIPHEI-vit | 0.2495 | 0.2498 | 0.004 |
| MIPHEI-convnext | 0.1966 | 0.1971 | 0.005 |
| DiffusionFT | 0.1692 | 0.1693 | 0.001 |
| Rosie-ORION | 0.1173 | 0.1173 | 0.002 |
| Pix2Pix | 0.0971 | 0.0971 | 0.001 |
| HEMIT | 0.0940 | 0.0939 | 0.001 |

**Headline (mean-15 AUPRC, core-bootstrap 95% CI):**

| model | AUPRC |
|---|---|
| **ours (bg0.2)** | **0.251 [0.238, 0.263]** |
| MIPHEI-vit | 0.250 [0.237, 0.263] |
| MIPHEI-convnext | 0.198 |
| DiffusionFT | 0.170 |
| Rosie-ORION | 0.118 |
| Pix2Pix | 0.097 |
| HEMIT | 0.094 |

Ours edges the best MIPHEI baseline; per-type, the gains are on **Tumor cells**
(0.742 vs 0.697) and **B cells** (0.503 vs 0.456).

### Same-feature parity (E-Cadherin)

PathoCell's 58-ch CODEX panel has **no E-cadherin** (only `beta-catenin` /
`Cytokeratin`), so MIPHEI scores **15** markers while our model predicts **16**.
That extra epithelial feature could flatter our logreg, so re-run with it dropped:

```bash
python cell_cls/pathocell_cls/eval_cell_auprc.py --tag bg0.2_noEcad \
    --features cell_cls/pathocell_cls/cell_token_features_bg0.2.parquet \
    --drop_markers E-Cadherin
```

On the **identical 15-feature set**, ours = **0.250 [0.238, 0.262]** (vs 0.251 with
E-Cad; MIPHEI-vit 0.250) — a **tie on the mean**, E-Cadherin worth only ~0.001.
The per-type wins survive (Tumor 0.740, B cells 0.502): the tumor signal comes from
**Pan-CK**, not E-Cadherin.

## Files
- `utils.py` — paths, geometry (297 px crop), HDF/H&E loaders, model loader,
  `NUCLEI_CLASSES`, `NTOK`, `FEAT_DIR`.
- `build_cell_token_features.py` — per-cell area-weighted token aggregation → parquet.
- `eval_cell_auprc.py` — MIPHEI-protocol logreg eval (`--source ours|parquet`).
- `reproduce_miphei.py` — recover MIPHEI's own numbers (harness trust check).
- `plot_metrics.py` — model-comparison / per-type bar charts.

## Caveats
- **Zero-shot domain shift.** ORION-trained model (16 markers) applied to PathoCell
  H&E; PathoCell's native 58-ch CODEX panel is *not* used — we predict the ORION
  markers and classify cell type from them, exactly as MIPHEI does, for a fair
  cross-model comparison. Our feature set is 16 vs MIPHEI's 15 — PathoCell has no
  E-cadherin channel so MIPHEI drops it; `--drop_markers E-Cadherin` gives strict
  parity and the result holds (see "Same-feature parity" above).
- **Headline = mean AUPRC + F1.** Rare types (NK / Nerves / Dendritic <1%) make
  per-type AUPRC-vs-prevalence the honest read; ROC-AUC is optimistic.
- **Split is fixed in the parquet** (stratified, `test_size=0.8`, `random_state=42`),
  identical across all models — differences are the model's, not the split's.