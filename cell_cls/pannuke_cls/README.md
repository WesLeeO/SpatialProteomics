# pannuke_cls — cell-TYPE classification on PanNuke (external H&E nuclei)

Benchmark the token model on **PanNuke** (19-tissue H&E, 5 nuclei classes) under
MIPHEI's exact protocol. Mirrors `cell_cls/pathocell_cls` (same file layout +
conventions). Cell-type classification, **not** marker positivity: the ORION
16-marker model is run zero-shot on the H&E and a logreg maps predicted marker
expression to the nucleus class, on MIPHEI's fixed **20% train / 80% test** split.

## Data

Download (warwick, public) + MIPHEI's `convert_fold_npy_to_pngs`:
```
/mnt/ssd/virtual_proteomics/data/pannuke/
  orig_data/Fold {1,2,3}/...                       # downloaded npy folds
  process_data/<Fold>/images/img_<tissue>_<f>_<k>.png      256×256 RGB @ 40x (0.25 µm/px)
                     /inst_masks/inst_<tissue>_<f>_<k>.png  uint16, label == MIPHEI cell_id
```
Build: `bash MIPHEI-ViT/datasets/download/pannuke_download.sh /mnt/ssd/virtual_proteomics/data`
then run `convert_fold_npy_to_pngs(sorted(folds), ...)` (sort the folds so the fold
index matches MIPHEI's — that's what keeps cell_ids aligned). **Verified:** the
instance-mask labels equal the cell_ids in MIPHEI's parquet (full build merged all
189,865 cells, 0 dropped).

GT cell-type one-hot + `split` come from
`checkpoints/MIPHEI-vit/pannuke_cell_dataframe_logreg.parquet`.

5 classes: Neoplastic cells · Inflammatory · Connective/Soft tissue cells · Dead Cells · Epithelial.

## Geometry (and why `--ps0`)

PanNuke is 256 px @ 40× = **64 µm**, *smaller* than the model's 112 µm FOV. Two ways
to feed our fixed-224 UNI2, set by `--ps0`:
- `--ps0 256` (**default / canonical**): resize the whole 256 image to 224 (≈0.29 µm/px,
  ~native 40×). This **matches what MIPHEI does** — their PannukeBaseEvaluator does NOT
  rescale (CenterCrop to nearest power-of-2 = 256→256 no-op), feeding native 256@40× into
  their flexible-input model. So this is the apples-to-apples comparison.
- `--ps0 448`: pad the 256 image (white) up to 448 then resize 224, keeping the trained
  **0.5 µm/px** scale. Token id = `floor(offset·G/ps0)`.

Robustness: ours beats MIPHEI-vit at **both** (native 0.654, scale-matched-pad 0.641 vs
their 0.600) — the win is not a preprocessing artifact, and the token model is scale-robust.

## Usage
```bash
export HF_HOME=/home/wesley/spatial_proteomics/foundation_models HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
python cell_cls/pannuke_cls/build_cell_token_features.py --tag bg0.2
python cell_cls/pannuke_cls/eval_cell_auprc.py --tag bg0.2
for m in MIPHEI-vit MIPHEI-convnext DiffusionFT Rosie-ORION Pix2Pix HEMIT; do
  python cell_cls/pannuke_cls/eval_cell_auprc.py --source parquet \
      --pred_parquet checkpoints/$m/pannuke_cell_dataframe_logreg.parquet
done
python cell_cls/pannuke_cls/plot_metrics.py --metric auprc
python cell_cls/pannuke_cls/reproduce_miphei.py     # harness trust check
```

## Result (mean-5 AUPRC, image-bootstrap 95% CI; ours = canonical native-scale)

| model | AUPRC |
|---|---|
| **ours (bg0.2)** | **0.654 [0.640, 0.670]** |
| MIPHEI-vit | 0.600 |
| DiffusionFT | 0.498 |
| MIPHEI-convnext | 0.463 |
| Rosie-ORION | 0.399 |
| HEMIT | 0.345 |
| Pix2Pix | 0.305 |

**Ours beats every MIPHEI baseline on every class** (non-overlapping CIs). Per-class vs
MIPHEI-vit: Neoplastic 0.748/0.677, Inflammatory 0.769/0.744, Connective 0.262/0.249,
Dead 0.654/0.567, Epithelial 0.835/0.762. `reproduce_miphei.py` recovers all 6 MIPHEI
`pannuke_logreg.csv` numbers (max |Δ| ≤ 0.003) → the edge is real. Plots:
`model_comparison_{auprc,f1}.png`, `compare_bg0.2_vs_MIPHEI-vit_{auprc,f1}.png`.

## Files
`utils.py` · `build_cell_token_features.py` · `eval_cell_auprc.py` ·
`reproduce_miphei.py` · `plot_metrics.py`
