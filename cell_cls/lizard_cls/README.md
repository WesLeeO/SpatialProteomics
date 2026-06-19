# lizard_cls — cell-TYPE classification on Lizard (external colon H&E)

Benchmark the token model on **Lizard** (colon H&E, 6 nuclei classes) under MIPHEI's
exact protocol. Mirrors `cell_cls/pathocell_cls` / `pannuke_cls`. Cell-type
classification: the ORION 16-marker model runs zero-shot on the H&E and a logreg maps
predicted marker expression to the nucleus class, on MIPHEI's fixed **20%/80%** split.

6 classes: Neutrophil · Epithelial · Lymphocyte · Plasma · Eosinophil · Connective_tissue.

## Data

Kaggle `aadimator/lizard-dataset`, downloaded via the Kaggle API (needs
`~/.kaggle/kaggle.json`, perms 600):
```
/mnt/ssd/virtual_proteomics/data/lizard/
  lizard_images{1,2}/Lizard_Images{1,2}/<stem>.png    variable-size RGB H&E @ 20x (0.5 µm/px)
  lizard_labels/Lizard_Labels/Labels/<stem>.mat       inst_map (==cell_id), id, class, ...
```
We read the `.mat` directly with scipy — **no MIPHEI/pyvips preprocessing needed**.
`inst_map` labels == `.mat id` == MIPHEI `cell_id` (verified; full build merged all
431,913 cells, 0 dropped). GT one-hot + `split` from
`checkpoints/MIPHEI-vit/lizard_cell_dataframe_logreg.parquet`, joined by `(slide_name, cell_id)`.

## Geometry — the easy one

Lizard is **already 20× (0.5 µm/px) = our training scale**, so `PATCH_SIZE_LEVEL0 =
round(224·0.5/0.5) = 224`: a native 224 crop *is* the model input, no rescale, no
padding gymnastics. Images are variable-size, so each is tiled into a non-overlapping
224 grid (edges padded white); a nucleus straddling a tile boundary gets area-weighted
contributions from both. `token = floor(offset·G/224)`.

## Usage
```bash
export HF_HOME=/home/wesley/spatial_proteomics/foundation_models HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
python cell_cls/lizard_cls/build_cell_token_features.py --tag bg0.2
python cell_cls/lizard_cls/eval_cell_auprc.py --tag bg0.2
for m in MIPHEI-vit MIPHEI-convnext DiffusionFT Rosie-ORION Pix2Pix HEMIT; do
  python cell_cls/lizard_cls/eval_cell_auprc.py --source parquet \
      --pred_parquet checkpoints/$m/lizard_cell_dataframe_logreg.parquet
done
python cell_cls/lizard_cls/plot_metrics.py --metric auprc
python cell_cls/lizard_cls/reproduce_miphei.py     # harness trust check
```

## Result (mean-6 AUPRC, image-bootstrap 95% CI) — our one LOSS

| model | AUPRC |
|---|---|
| MIPHEI-vit | **0.517** [0.504, 0.529] |
| **ours (bg0.2)** | 0.480 [0.469, 0.491] |
| MIPHEI-convnext | 0.467 |
| DiffusionFT | 0.451 |
| Rosie-ORION | 0.416 |
| Pix2Pix | 0.236 |
| HEMIT | 0.228 |

We're **2nd — below MIPHEI-vit, above every other baseline.** Per-class we *tie* on
Epithelial (0.976, Pan-CK nails it) but lose on the **small, densely-packed types**:
Connective_tissue (0.719 vs 0.800), Lymphocyte (0.697 vs 0.760), Neutrophil (0.123 vs
0.180). That's the token-resolution limit made concrete: adjacent small cells fall in
the same ~7 µm token and blur together, where MIPHEI's pixel resolution separates them.

**Caveat:** Lizard is at 0.5 µm/px = MIPHEI's *home* training scale, so this is their
best-case dataset (on PanNuke they ran off-scale at 40× and we won). `reproduce_miphei.py`
recovers all 6 MIPHEI `lizard_logreg.csv` numbers (harness unbiased — the loss is real).

### Matched-resolution diagnostic (`run_miphei_matched.py`)

Running MIPHEI-vit ourselves and aggregating two ways shows **the entire loss is pixel
resolution, not model quality**:

| | MEAN6 AUPRC |
|---|---|
| MIPHEI pixel (native, validates ≈ their 0.517) | 0.519 |
| **ours** (token, 7 µm) | **0.480** |
| MIPHEI token (~cell-scale + our aggregation) | 0.471 |

Forced to ~cell-scale tokens MIPHEI drops 0.519→0.471 (−0.048 ≈ the gap they beat us by),
and at matched resolution the models **tie** (0.480 vs 0.471, overlapping CIs). So the
Lizard loss is the token-resolution ceiling on dense small cells — pixel resolution
separates adjacent lymphocytes our 7 µm tokens blur — *not* an inferior model.

## Files
`utils.py` · `build_cell_token_features.py` · `eval_cell_auprc.py` ·
`reproduce_miphei.py` · `plot_metrics.py`. Headline = mean-6 AUPRC, image-bootstrap CI.
