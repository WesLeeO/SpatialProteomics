#!/usr/bin/env python3
"""
Build HNSCC mIF/mIHC token-grid patch dataset.

Geometry
--------
  Source     : 512×512 px @ 0.5 µm/px (Vectra Polaris 20×)
  Sub-patches: 4 × 256×256 (2×2 non-overlapping), closest match to ORION 20×
  Model input: 256→224 resize at training time (png_crops loader)
  Eff. MPP   : 0.5 × 256/224 ≈ 0.571 µm/px

Normalisation  (per-source-image, mirrors visualize_hnscc_all_patches.py)
--------------------------------------------------------------------------
  1. p99_fg = 99th-percentile of non-zero pixels in the full 512×512 channel
  2. normed = clip(arr / p99_fg, 0, 1)
  3. normed[normed < BOUNDS[pid][marker]] = 0   (visual quality floor)
  4. Per sub-patch: crop 256×256 → resize to 224 → 16×16 block-mean

HDF5 layout  (png_crops-compatible with training_multisource.py)
----------------------------------------------------------------
  /coords    (N, 2)       int16   — (x, y) top-left within the 512×512 source
  /targets   (N, C, G, G) float32 — token-grid mean expression
  /patch_ids (N,)         bytes   — source patch ID, e.g. "Case1_M1_0_0"
  attrs: marker_names, patch_size_level0=256, token_grid,
         native_mpp, effective_mpp, normalisation
"""

import cv2
import h5py
import numpy as np
from pathlib import Path
from PIL import Image

MIF_DIR    = Path('/mnt/ssd1/virtual_proteomics/data/HNSCC/mIF_Data')
MIHC_DIR   = Path('/mnt/ssd1/virtual_proteomics/data/HNSCC/mIHC_Data')
OUTPUT_DIR = Path('datasets/hnscc_patch_dataset')

MARKERS       = ['CD3', 'CD8', 'FoxP3', 'PanCK', 'DAPI']
NATIVE_MPP    = 0.5
SOURCE_SIZE   = 512
SUB_SIZE      = 256   # 2×2 sub-patches
PATCH_SIZE    = 224   # model input (resize at training time)
TOKEN_GRID    = 16
TOKEN_PX      = PATCH_SIZE // TOKEN_GRID   # = 14

EFFECTIVE_MPP = NATIVE_MPP * (SUB_SIZE / PATCH_SIZE)   # ≈ 0.571 µm/px

# (x, y) top-left of each 256×256 sub-patch within the 512×512 source.
# png_crops loader does img[y:y+psz, x:x+psz], so (x=0,y=0) → top-left etc.
SUB_COORDS = [(0, 0), (SUB_SIZE, 0), (0, SUB_SIZE), (SUB_SIZE, SUB_SIZE)]

# Fallback thresholds for patches absent from BOUNDS
DEFAULT_BOUNDS = {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}

# Per-patch visual quality thresholds in clip(x/p99, 0, 1) space.
# Copied verbatim from visualize_hnscc_all_patches.py.
BOUNDS: dict[str, dict[str, float]] = {
    'Case1_M1_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_M1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_M1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.45, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_M1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.45, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_M2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_M2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_M2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.43, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_M2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_M3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_M3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_M3_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_M3_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_S2_0_0': {'CD3': 0.65, 'CD8': 0.70, 'FoxP3': 0.80, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case1_S3_0_0': {'CD3': 0.35, 'CD8': 0.55, 'FoxP3': 0.50, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case1_S3_0_1': {'CD3': 0.45, 'CD8': 0.65, 'FoxP3': 0.60, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case1_S3_1_0': {'CD3': 0.35, 'CD8': 0.60, 'FoxP3': 0.60, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case1_S3_1_1': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.55, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case1_T1_0_0': {'CD3': 0.50, 'CD8': 0.40, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.30},
    'Case1_T1_0_1': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.55, 'PanCK': 0.35, 'DAPI': 0.20},
    'Case1_T1_1_0': {'CD3': 0.35, 'CD8': 0.65, 'FoxP3': 0.60, 'PanCK': 0.35, 'DAPI': 0.20},
    'Case1_T1_1_1': {'CD3': 0.35, 'CD8': 0.55, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_T2_0_0': {'CD3': 0.99, 'CD8': 0.99, 'FoxP3': 0.99, 'PanCK': 0.99, 'DAPI': 0.40},
    'Case1_T2_0_1': {'CD3': 0.99, 'CD8': 0.99, 'FoxP3': 0.99, 'PanCK': 0.99, 'DAPI': 0.40},
    'Case1_T2_1_0': {'CD3': 0.80, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.80, 'DAPI': 0.30},
    'Case1_T2_1_1': {'CD3': 0.80, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.80, 'DAPI': 0.30},
    'Case1_T3_0_0': {'CD3': 0.35, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.60, 'DAPI': 0.20},
    'Case1_T3_0_1': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.65, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case1_T3_1_0': {'CD3': 0.35, 'CD8': 0.75, 'FoxP3': 0.75, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case1_T3_1_1': {'CD3': 0.35, 'CD8': 0.55, 'FoxP3': 0.40, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case2_M1_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case2_M1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.45, 'PanCK': 0.35, 'DAPI': 0.20},
    'Case2_M1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case2_M1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case2_M2_0_0': {'CD3': 0.35, 'CD8': 0.45, 'FoxP3': 0.85, 'PanCK': 0.55, 'DAPI': 0.20},
    'Case2_M2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.45, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case2_M2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.45, 'PanCK': 0.35, 'DAPI': 0.20},
    'Case2_M2_1_1': {'CD3': 0.35, 'CD8': 0.65, 'FoxP3': 0.65, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case2_M3_0_0': {'CD3': 0.35, 'CD8': 0.65, 'FoxP3': 0.75, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case2_M3_0_1': {'CD3': 0.35, 'CD8': 0.85, 'FoxP3': 0.85, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case2_M3_1_0': {'CD3': 0.35, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.35, 'DAPI': 0.20},
    'Case2_M3_1_1': {'CD3': 0.35, 'CD8': 0.85, 'FoxP3': 0.65, 'PanCK': 0.35, 'DAPI': 0.20},
    'Case2_S1_0_0': {'CD3': 0.35, 'CD8': 0.50, 'FoxP3': 0.60, 'PanCK': 0.75, 'DAPI': 0.20},
    'Case2_S1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.70, 'DAPI': 0.20},
    'Case2_S1_1_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.75, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case2_S1_1_1': {'CD3': 0.50, 'CD8': 0.70, 'FoxP3': 0.75, 'PanCK': 0.60, 'DAPI': 0.20},
    'Case2_S2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.60, 'DAPI': 0.20},
    'Case2_S2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case2_S2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case2_S2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.65, 'DAPI': 0.20},
    'Case2_S3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.60, 'DAPI': 0.20},
    'Case2_S3_0_1': {'CD3': 0.35, 'CD8': 0.65, 'FoxP3': 0.85, 'PanCK': 0.90, 'DAPI': 0.20},
    'Case2_S3_1_0': {'CD3': 0.35, 'CD8': 0.50, 'FoxP3': 0.70, 'PanCK': 0.99, 'DAPI': 0.20},
    'Case2_S3_1_1': {'CD3': 0.35, 'CD8': 0.65, 'FoxP3': 0.65, 'PanCK': 0.65, 'DAPI': 0.20},
    'Case2_T1_0_0': {'CD3': 0.35, 'CD8': 0.85, 'FoxP3': 0.85, 'PanCK': 0.45, 'DAPI': 0.20},
    'Case2_T1_0_1': {'CD3': 0.50, 'CD8': 0.85, 'FoxP3': 0.85, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case2_T1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case2_T1_1_1': {'CD3': 0.50, 'CD8': 0.85, 'FoxP3': 0.90, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case2_T2_0_0': {'CD3': 0.70, 'CD8': 0.90, 'FoxP3': 0.85, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case2_T2_0_1': {'CD3': 0.35, 'CD8': 0.60, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case2_T2_1_0': {'CD3': 0.35, 'CD8': 0.85, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case2_T2_1_1': {'CD3': 0.35, 'CD8': 0.50, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case2_T3_0_0': {'CD3': 0.90, 'CD8': 0.95, 'FoxP3': 0.90, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case2_T3_0_1': {'CD3': 0.90, 'CD8': 0.90, 'FoxP3': 0.90, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case2_T3_1_0': {'CD3': 0.90, 'CD8': 0.90, 'FoxP3': 0.90, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case2_T3_1_1': {'CD3': 0.90, 'CD8': 0.90, 'FoxP3': 0.90, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case3_M1_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.45, 'DAPI': 0.20},
    'Case3_M1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.90, 'PanCK': 0.45, 'DAPI': 0.20},
    'Case3_M1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.97, 'PanCK': 0.45, 'DAPI': 0.20},
    'Case3_M1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.45, 'DAPI': 0.20},
    'Case3_M2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.50, 'PanCK': 0.45, 'DAPI': 0.20},
    'Case3_M2_0_1': {'CD3': 0.35, 'CD8': 0.75, 'FoxP3': 0.80, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case3_M2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case3_M2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case3_M3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case3_M3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.50, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case3_M3_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case3_M3_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case3_S1_0_0': {'CD3': 0.85, 'CD8': 0.85, 'FoxP3': 0.85, 'PanCK': 0.70, 'DAPI': 0.20},
    'Case3_S1_0_1': {'CD3': 0.50, 'CD8': 0.50, 'FoxP3': 0.50, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case3_S1_1_0': {'CD3': 0.50, 'CD8': 0.85, 'FoxP3': 0.85, 'PanCK': 0.70, 'DAPI': 0.20},
    'Case3_S1_1_1': {'CD3': 0.65, 'CD8': 0.65, 'FoxP3': 0.65, 'PanCK': 0.60, 'DAPI': 0.20},
    'Case3_T1_0_0': {'CD3': 0.90, 'CD8': 0.90, 'FoxP3': 0.90, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case3_T1_0_1': {'CD3': 0.80, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case3_T1_1_0': {'CD3': 0.80, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case3_T1_1_1': {'CD3': 0.80, 'CD8': 0.85, 'FoxP3': 0.80, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case3_T2_0_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.70, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case3_T2_0_1': {'CD3': 0.35, 'CD8': 0.80, 'FoxP3': 0.85, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case3_T2_1_0': {'CD3': 0.35, 'CD8': 0.85, 'FoxP3': 0.70, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case3_T2_1_1': {'CD3': 0.35, 'CD8': 0.65, 'FoxP3': 0.60, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case3_T3_0_0': {'CD3': 0.35, 'CD8': 0.80, 'FoxP3': 0.90, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case3_T3_0_1': {'CD3': 0.70, 'CD8': 0.70, 'FoxP3': 0.80, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case3_T3_1_1': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.80, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case4_M1_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_M1_0_1': {'CD3': 0.35, 'CD8': 0.75, 'FoxP3': 0.75, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case4_M1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_M1_1_1': {'CD3': 0.80, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case4_M2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_M2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_M2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_M2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_M3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_M3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_M3_1_0': {'CD3': 0.80, 'CD8': 0.85, 'FoxP3': 0.95, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case4_M3_1_1': {'CD3': 0.75, 'CD8': 0.75, 'FoxP3': 0.85, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case4_S1_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.50, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_S1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_S1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_S1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_S2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.30},
    'Case4_S2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.85, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_S2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_S2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_S3_1_1': {'CD3': 0.35, 'CD8': 0.50, 'FoxP3': 0.40, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case4_T1_0_0': {'CD3': 0.35, 'CD8': 0.80, 'FoxP3': 0.60, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case4_T1_0_1': {'CD3': 0.35, 'CD8': 0.85, 'FoxP3': 0.60, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case4_T1_1_0': {'CD3': 0.70, 'CD8': 0.85, 'FoxP3': 0.85, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case4_T1_1_1': {'CD3': 0.35, 'CD8': 0.60, 'FoxP3': 0.50, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case4_T2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_T2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_T2_1_0': {'CD3': 0.35, 'CD8': 0.50, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_T2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_T3_0_0': {'CD3': 0.35, 'CD8': 0.60, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_T3_0_1': {'CD3': 0.80, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_T3_1_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case4_T3_1_1': {'CD3': 0.35, 'CD8': 0.65, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case5_M1_0_0': {'CD3': 0.50, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case5_M1_0_1': {'CD3': 0.50, 'CD8': 0.35, 'FoxP3': 0.90, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case5_M1_1_0': {'CD3': 0.50, 'CD8': 0.35, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case5_M1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.95, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case5_M2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.90, 'PanCK': 0.9999, 'DAPI': 0.20},
    'Case5_M2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case5_M2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case5_M2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case5_M3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case5_M3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case5_M3_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case5_M3_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case5_S1_0_0': {'CD3': 0.60, 'CD8': 0.70, 'FoxP3': 0.70, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case5_S1_0_1': {'CD3': 0.50, 'CD8': 0.70, 'FoxP3': 0.60, 'PanCK': 0.60, 'DAPI': 0.20},
    'Case5_S1_1_0': {'CD3': 0.70, 'CD8': 0.70, 'FoxP3': 0.70, 'PanCK': 0.70, 'DAPI': 0.20},
    'Case5_S1_1_1': {'CD3': 0.70, 'CD8': 0.80, 'FoxP3': 0.70, 'PanCK': 0.70, 'DAPI': 0.20},
    'Case5_S2_0_0': {'CD3': 0.60, 'CD8': 0.70, 'FoxP3': 0.70, 'PanCK': 0.60, 'DAPI': 0.20},
    'Case5_S2_0_1': {'CD3': 0.50, 'CD8': 0.60, 'FoxP3': 0.70, 'PanCK': 0.60, 'DAPI': 0.20},
    'Case5_S2_1_0': {'CD3': 0.50, 'CD8': 0.80, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case5_S2_1_1': {'CD3': 0.65, 'CD8': 0.70, 'FoxP3': 0.75, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case5_S3_0_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.50, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case5_S3_0_1': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.50, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case5_S3_1_0': {'CD3': 0.70, 'CD8': 0.70, 'FoxP3': 0.70, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case5_S3_1_1': {'CD3': 0.35, 'CD8': 0.65, 'FoxP3': 0.40, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case5_T1_0_0': {'CD3': 0.40, 'CD8': 0.70, 'FoxP3': 0.50, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case5_T1_0_1': {'CD3': 0.70, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case5_T1_1_0': {'CD3': 0.70, 'CD8': 0.70, 'FoxP3': 0.75, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case5_T1_1_1': {'CD3': 0.60, 'CD8': 0.75, 'FoxP3': 0.80, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case5_T2_0_0': {'CD3': 0.70, 'CD8': 0.70, 'FoxP3': 0.80, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case5_T2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case5_T2_1_0': {'CD3': 0.80, 'CD8': 0.70, 'FoxP3': 0.85, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case5_T2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case5_T3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.45, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case5_T3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.65, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case5_T3_1_0': {'CD3': 0.35, 'CD8': 0.50, 'FoxP3': 0.45, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case5_T3_1_1': {'CD3': 0.35, 'CD8': 0.50, 'FoxP3': 0.85, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_M1_0_0': {'CD3': 0.35, 'CD8': 0.40, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_M1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_M1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_M1_1_1': {'CD3': 0.35, 'CD8': 0.80, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_M2_0_0': {'CD3': 0.35, 'CD8': 0.60, 'FoxP3': 0.45, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_M2_0_1': {'CD3': 0.35, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_M2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_M2_1_1': {'CD3': 0.35, 'CD8': 0.40, 'FoxP3': 0.45, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_M3_0_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_M3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_M3_1_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_M3_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_S1_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_S1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.50, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_S1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_S1_1_1': {'CD3': 0.35, 'CD8': 0.60, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_S2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case6_S2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_S2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_S2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case6_S3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case6_S3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case6_S3_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case6_S3_1_1': {'CD3': 0.35, 'CD8': 0.40, 'FoxP3': 0.60, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case6_T1_0_0': {'CD3': 0.35, 'CD8': 0.75, 'FoxP3': 0.75, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_T1_0_1': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_T1_1_0': {'CD3': 0.75, 'CD8': 0.75, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_T1_1_1': {'CD3': 0.50, 'CD8': 0.75, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_T2_0_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_T2_0_1': {'CD3': 0.70, 'CD8': 0.70, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_T2_1_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_T2_1_1': {'CD3': 0.70, 'CD8': 0.70, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_T3_0_0': {'CD3': 0.75, 'CD8': 0.60, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_T3_0_1': {'CD3': 0.70, 'CD8': 0.60, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_T3_1_0': {'CD3': 0.35, 'CD8': 0.60, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case6_T3_1_1': {'CD3': 0.60, 'CD8': 0.60, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_M1_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_M1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_M1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_M1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_M2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_M2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_M2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_M2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_M3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_M3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_M3_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_M3_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_S1_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_S1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_S1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_S1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_S2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_S2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_S2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_S2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_S3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_S3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_S3_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_S3_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_T1_0_0': {'CD3': 0.35, 'CD8': 0.50, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_T1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_T1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_T1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_T2_0_0': {'CD3': 0.35, 'CD8': 0.50, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_T2_0_1': {'CD3': 0.50, 'CD8': 0.60, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_T2_1_1': {'CD3': 0.35, 'CD8': 0.40, 'FoxP3': 0.55, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case7_T3_0_0': {'CD3': 0.90, 'CD8': 0.90, 'FoxP3': 0.95, 'PanCK': 0.40, 'DAPI': 0.30},
    'Case7_T3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.55, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case7_T3_1_0': {'CD3': 0.50, 'CD8': 0.65, 'FoxP3': 0.80, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case7_T3_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_M1_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_M1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_M1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_M1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_M2_0_0': {'CD3': 0.35, 'CD8': 0.60, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_M2_0_1': {'CD3': 0.35, 'CD8': 0.60, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_M2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_M2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_M3_0_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_M3_0_1': {'CD3': 0.45, 'CD8': 0.70, 'FoxP3': 0.90, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_M3_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_M3_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_S1_0_0': {'CD3': 0.60, 'CD8': 0.60, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_S1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_S1_1_0': {'CD3': 0.35, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.80, 'DAPI': 0.20},
    'Case8_S1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.50, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_S2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_S2_0_1': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_S2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.50, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_S2_1_1': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.80, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case8_S3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.50, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_S3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.50, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_S3_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.50, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_S3_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.55, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case8_T1_0_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_T1_0_1': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_T1_1_0': {'CD3': 0.60, 'CD8': 0.75, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_T1_1_1': {'CD3': 0.35, 'CD8': 0.65, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case8_T2_0_0': {'CD3': 0.90, 'CD8': 0.90, 'FoxP3': 0.90, 'PanCK': 0.50, 'DAPI': 0.35},
    'Case8_T2_0_1': {'CD3': 0.50, 'CD8': 0.65, 'FoxP3': 0.75, 'PanCK': 0.40, 'DAPI': 0.20},
    'Case8_T2_1_0': {'CD3': 0.85, 'CD8': 0.85, 'FoxP3': 0.85, 'PanCK': 0.45, 'DAPI': 0.40},
    'Case8_T2_1_1': {'CD3': 0.85, 'CD8': 0.75, 'FoxP3': 0.90, 'PanCK': 0.50, 'DAPI': 0.30},
    'Case8_T3_0_0': {'CD3': 0.90, 'CD8': 0.90, 'FoxP3': 0.90, 'PanCK': 0.60, 'DAPI': 0.30},
    'Case8_T3_0_1': {'CD3': 0.50, 'CD8': 0.70, 'FoxP3': 0.50, 'PanCK': 0.60, 'DAPI': 0.30},
    'Case8_T3_1_0': {'CD3': 0.80, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case8_T3_1_1': {'CD3': 0.80, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.55, 'DAPI': 0.20},
}


def get_source_patch_ids() -> list[str]:
    return sorted(set(
        p.stem.rsplit('_', 1)[0]
        for p in MIF_DIR.glob('*_CD3.png')
    ))


def compute_token_targets(pid: str) -> np.ndarray:
    """
    Returns (C, 4, TOKEN_GRID, TOKEN_GRID) float32 — targets for all 4 sub-patches.

    Pipeline per channel:
      1. Load full 512×512 mIF PNG
      2. p99_fg = 99th-pct of non-zero pixels in the 512×512
      3. normed = clip(arr / p99_fg, 0, 1)
      4. normed[normed < BOUNDS[pid][marker]] = 0   (noise floor)
      5. Per sub-patch (x, y):
           crop  = normed[y:y+256, x:x+256]          (256×256)
           sized = cv2.resize(crop, (224, 224))       (matches training loader)
           token = sized.reshape(G, 14, G, 14).mean(axis=(1,3))
    """
    bounds = BOUNDS.get(pid, DEFAULT_BOUNDS)
    C = len(MARKERS)
    targets = np.zeros((C, len(SUB_COORDS), TOKEN_GRID, TOKEN_GRID), dtype=np.float32)

    for ci, marker in enumerate(MARKERS):
        arr = np.array(Image.open(MIF_DIR / f'{pid}_{marker}.png')).astype(np.float32)

        nz  = arr[arr > 0]
        p99 = float(np.percentile(nz, 99)) if len(nz) else 1.0
        p99 = max(p99, 1.0)

        normed = np.clip(arr / p99, 0.0, 1.0)
        lo = bounds.get(marker, DEFAULT_BOUNDS[marker])
        normed[normed < lo] = 0.0

        for si, (x, y) in enumerate(SUB_COORDS):
            crop   = normed[y:y + SUB_SIZE, x:x + SUB_SIZE]
            sized  = cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE),
                                interpolation=cv2.INTER_LINEAR)
            targets[ci, si] = (
                sized
                .reshape(TOKEN_GRID, TOKEN_PX, TOKEN_GRID, TOKEN_PX)
                .mean(axis=(1, 3))
            )

    return targets  # (C, 4, G, G)


def main() -> None:
    source_ids = get_source_patch_ids()
    n_missing  = sum(1 for pid in source_ids if pid not in BOUNDS)
    print(f"Source patches : {len(source_ids)}")
    print(f"Sub-patches    : {len(source_ids) * len(SUB_COORDS)}")
    print(f"Using DEFAULT_BOUNDS for {n_missing} patches not in BOUNDS")
    print(f"Effective MPP  : {EFFECTIVE_MPP:.3f} µm/px  "
          f"(native {NATIVE_MPP} × {SUB_SIZE}/{PATCH_SIZE})")

    all_coords   : list[tuple[int, int]] = []
    all_targets  : list[np.ndarray]      = []   # each (C, G, G)
    all_patch_ids: list[str]             = []

    for i, pid in enumerate(source_ids):
        if i % 50 == 0:
            print(f"  [{i}/{len(source_ids)}] {pid}…", flush=True)

        tgt = compute_token_targets(pid)   # (C, 4, G, G)

        for si, (x, y) in enumerate(SUB_COORDS):
            all_coords.append((x, y))
            all_targets.append(tgt[:, si, :, :])
            all_patch_ids.append(pid)

    N         = len(all_coords)
    C         = len(MARKERS)
    targets_arr = np.stack(all_targets, axis=0)   # (N, C, G, G)
    coords_arr  = np.array(all_coords, dtype=np.int16)

    max_len = max(len(s) for s in all_patch_ids)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / 'hnscc_patch_dataset.h5'

    with h5py.File(out_path, 'w') as f:
        f.create_dataset('coords',    data=coords_arr, compression='gzip')
        f.create_dataset('targets',   data=targets_arr, compression='gzip',
                         chunks=(min(256, N), C, TOKEN_GRID, TOKEN_GRID))
        f.create_dataset('patch_ids', data=np.array(all_patch_ids, dtype=f'S{max_len}'),
                         compression='gzip')
        f.attrs['marker_names']      = MARKERS
        f.attrs['patch_size_level0'] = SUB_SIZE
        f.attrs['token_grid']        = TOKEN_GRID
        f.attrs['native_mpp']        = NATIVE_MPP
        f.attrs['effective_mpp']     = float(EFFECTIVE_MPP)
        f.attrs['normalisation']     = 'per_source_p99: clip(x/p99_fg,0,1) + BOUNDS floor'

    mb = out_path.stat().st_size / 1e6
    print(f"\nSaved → {out_path}  ({mb:.1f} MB)")
    print(f"  /coords    {coords_arr.shape}")
    print(f"  /targets   {targets_arr.shape}  mean={targets_arr.mean():.4f}")
    print(f"  /patch_ids {len(all_patch_ids)}")


if __name__ == '__main__':
    main()