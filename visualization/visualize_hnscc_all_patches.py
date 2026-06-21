#!/usr/bin/env python3
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

MIF_DIR  = Path('/mnt/ssd1/virtual_proteomics/data/HNSCC/mIF_Data')
MIHC_DIR = Path('/mnt/ssd1/virtual_proteomics/data/HNSCC/mIHC_Data')
OUT      = Path('visualization_out/hnscc/all_patches')

IF_MARKERS = ['CD3', 'CD8', 'FoxP3', 'PanCK', 'DAPI']
CMAPS = {
    'CD3':   LinearSegmentedColormap.from_list('cd3',   ['black', 'limegreen']),
    'CD8':   LinearSegmentedColormap.from_list('cd8',   ['black', 'cyan']),
    'FoxP3': LinearSegmentedColormap.from_list('foxp3', ['black', 'lime']),
    'PanCK': LinearSegmentedColormap.from_list('pck',   ['black', 'magenta']),
    'DAPI':  LinearSegmentedColormap.from_list('dapi',  ['black', 'cornflowerblue']),
}

# Per-patch thresholds in clip(x/p99, 0, 1) space — tune visually.
# Keys are patch IDs: Case{N}_{region}_{x}_{y}
BOUNDS = {
    'Case1_M1_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_M1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_M1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.45, 'PanCK': 0.30, 'DAPI': 0.20}, # foxp3 change
    'Case1_M1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.45, 'PanCK': 0.30, 'DAPI': 0.20}, # foxp3 change
    'Case1_M2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_M2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_M2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.43, 'PanCK': 0.30, 'DAPI': 0.20}, #foxp3 change
    'Case1_M2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_M3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_M3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_M3_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_M3_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case1_S2_0_0': {'CD3': 0.65, 'CD8': 0.7, 'FoxP3': 0.8, 'PanCK': 0.5, 'DAPI': 0.20}, # 13 change
    'Case1_S3_0_0': {'CD3': 0.35, 'CD8': 0.55, 'FoxP3': 0.5, 'PanCK': 0.5, 'DAPI': 0.20}, # 14 change
    'Case1_S3_0_1': {'CD3': 0.45, 'CD8': 0.65, 'FoxP3': 0.6, 'PanCK': 0.40, 'DAPI': 0.20}, # 15 change
    'Case1_S3_1_0': {'CD3': 0.35, 'CD8': 0.6, 'FoxP3': 0.6, 'PanCK': 0.40, 'DAPI': 0.20}, # 16 change
    'Case1_S3_1_1': {'CD3': 0.35, 'CD8': 0.7, 'FoxP3': 0.55, 'PanCK': 0.40, 'DAPI': 0.20}, # 17 change
    'Case1_T1_0_0': {'CD3': 0.50, 'CD8': 0.40, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.30},  # 18 change
    'Case1_T1_0_1': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.55, 'PanCK': 0.35, 'DAPI': 0.20},  # 19 change
    'Case1_T1_1_0': {'CD3': 0.35, 'CD8': 0.65, 'FoxP3': 0.60, 'PanCK': 0.35, 'DAPI': 0.20},  # 20 change
    'Case1_T1_1_1': {'CD3': 0.35, 'CD8': 0.55, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 21 change
    'Case1_T2_0_0': {'CD3': 0.99, 'CD8': 0.99, 'FoxP3': 0.99, 'PanCK': 0.99, 'DAPI': 0.40},  # 22 change -> potential remove
    'Case1_T2_0_1': {'CD3': 0.99, 'CD8': 0.99, 'FoxP3': 0.99, 'PanCK': 0.99, 'DAPI': 0.40}, # 23 change -> potential remove
    'Case1_T2_1_0': {'CD3': 0.80, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.80, 'DAPI': 0.30},  # 24 change -> potential remove
    'Case1_T2_1_1': {'CD3': 0.80, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.80, 'DAPI': 0.30}, # 25 change -> potential remove
    'Case1_T3_0_0': {'CD3': 0.35, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.60, 'DAPI': 0.20}, # 26 change
    'Case1_T3_0_1': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.65, 'PanCK': 0.50, 'DAPI': 0.20}, # 27 change
    'Case1_T3_1_0': {'CD3': 0.35, 'CD8': 0.75, 'FoxP3': 0.75, 'PanCK': 0.50, 'DAPI': 0.20}, # 28 change
    'Case1_T3_1_1': {'CD3': 0.35, 'CD8': 0.55, 'FoxP3': 0.40, 'PanCK': 0.50, 'DAPI': 0.20}, # 29 change
    'Case2_M1_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, #  1
    'Case2_M1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.45, 'PanCK': 0.35, 'DAPI': 0.20}, # 2 change
    'Case2_M1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case2_M1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},
    'Case2_M2_0_0': {'CD3': 0.35, 'CD8': 0.45, 'FoxP3': 0.85, 'PanCK': 0.55, 'DAPI': 0.20}, # 5 change
    'Case2_M2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.45, 'PanCK': 0.30, 'DAPI': 0.20}, # 6 change
    'Case2_M2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.45, 'PanCK': 0.35, 'DAPI': 0.20}, # 7 change
    'Case2_M2_1_1': {'CD3': 0.35, 'CD8': 0.65, 'FoxP3': 0.65, 'PanCK': 0.40, 'DAPI': 0.20}, # 8 change
    'Case2_M3_0_0': {'CD3': 0.35, 'CD8': 0.65, 'FoxP3': 0.75, 'PanCK': 0.30, 'DAPI': 0.20}, # 9 change
    'Case2_M3_0_1': {'CD3': 0.35, 'CD8': 0.85, 'FoxP3': 0.85, 'PanCK': 0.30, 'DAPI': 0.20}, # 10 change
    'Case2_M3_1_0': {'CD3': 0.35, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.35, 'DAPI': 0.20},  # 11 change
    'Case2_M3_1_1': {'CD3': 0.35, 'CD8': 0.85, 'FoxP3': 0.65, 'PanCK': 0.35, 'DAPI': 0.20},  # 12 change
    'Case2_S1_0_0': {'CD3': 0.35, 'CD8': 0.5, 'FoxP3': 0.60, 'PanCK': 0.75, 'DAPI': 0.20}, # 13 change
    'Case2_S1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.70, 'DAPI': 0.20}, # 14 change
    'Case2_S1_1_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.75, 'PanCK': 0.50, 'DAPI': 0.20}, # 15 change
    'Case2_S1_1_1': {'CD3': 0.50, 'CD8': 0.70, 'FoxP3': 0.75, 'PanCK': 0.60, 'DAPI': 0.20}, # 16 change
    'Case2_S2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.60, 'DAPI': 0.20},  # 17 change
    'Case2_S2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.50, 'DAPI': 0.20},  # 18 change
    'Case2_S2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.50, 'DAPI': 0.20},  # 19 change
    'Case2_S2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.65, 'DAPI': 0.20}, # 20 change
    'Case2_S3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.60, 'DAPI': 0.20},  # 21 change
    'Case2_S3_0_1': {'CD3': 0.35, 'CD8': 0.65, 'FoxP3': 0.85, 'PanCK': 0.90, 'DAPI': 0.20},  # 22 change panck problematic
    'Case2_S3_1_0': {'CD3': 0.35, 'CD8': 0.50, 'FoxP3': 0.70, 'PanCK': 0.99, 'DAPI': 0.20},  # 23 change panck problematic
    'Case2_S3_1_1': {'CD3': 0.35, 'CD8': 0.65, 'FoxP3': 0.65, 'PanCK': 0.65, 'DAPI': 0.20}, # 24 change
    'Case2_T1_0_0': {'CD3': 0.35, 'CD8': 0.85, 'FoxP3': 0.85, 'PanCK': 0.45, 'DAPI': 0.20}, # 25 change
    'Case2_T1_0_1': {'CD3': 0.50, 'CD8': 0.85, 'FoxP3': 0.85, 'PanCK': 0.40, 'DAPI': 0.20}, # 26 change
    'Case2_T1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.40, 'DAPI': 0.20},# 27 change
    'Case2_T1_1_1': {'CD3': 0.50, 'CD8': 0.85, 'FoxP3': 0.90, 'PanCK': 0.40, 'DAPI': 0.20}, # 28 change
    'Case2_T2_0_0': {'CD3': 0.70, 'CD8': 0.90, 'FoxP3': 0.85, 'PanCK': 0.50, 'DAPI': 0.20}, # 29 change
    'Case2_T2_0_1': {'CD3': 0.35, 'CD8': 0.60, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20},  # 30 change
    'Case2_T2_1_0': {'CD3': 0.35, 'CD8': 0.85, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20},   # 31 change
    'Case2_T2_1_1': {'CD3': 0.35, 'CD8': 0.50, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20},    # 32 change
    'Case2_T3_0_0': {'CD3': 0.90, 'CD8': 0.95, 'FoxP3': 0.90, 'PanCK': 0.30, 'DAPI': 0.20},  # 33 change
    'Case2_T3_0_1': {'CD3': 0.90, 'CD8': 0.90, 'FoxP3': 0.90, 'PanCK': 0.30, 'DAPI': 0.20}, # 34 change
    'Case2_T3_1_0': {'CD3': 0.90, 'CD8': 0.90, 'FoxP3': 0.90, 'PanCK': 0.30, 'DAPI': 0.20}, # 35 change
    'Case2_T3_1_1': {'CD3': 0.90, 'CD8': 0.90, 'FoxP3': 0.90, 'PanCK': 0.30, 'DAPI': 0.20}, # 36 change
    'Case3_M1_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.45, 'DAPI': 0.20}, # 1
    'Case3_M1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.90, 'PanCK': 0.45, 'DAPI': 0.20},
    'Case3_M1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.97, 'PanCK': 0.45, 'DAPI': 0.20},
    'Case3_M1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.45, 'DAPI': 0.20},
    'Case3_M2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.50, 'PanCK': 0.45, 'DAPI': 0.20}, # 5
    'Case3_M2_0_1': {'CD3': 0.35, 'CD8': 0.75, 'FoxP3': 0.80, 'PanCK': 0.40, 'DAPI': 0.20}, # 6
    'Case3_M2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20}, # 7
    'Case3_M2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 8
    'Case3_M3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.40, 'DAPI': 0.20}, # 9
    'Case3_M3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.50, 'PanCK': 0.30, 'DAPI': 0.20}, # 10
    'Case3_M3_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 11
    'Case3_M3_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 12 
    'Case3_S1_0_0': {'CD3': 0.85, 'CD8': 0.85, 'FoxP3': 0.85, 'PanCK': 0.70, 'DAPI': 0.20}, # 13
    'Case3_S1_0_1': {'CD3': 0.50, 'CD8': 0.50, 'FoxP3': 0.50, 'PanCK': 0.30, 'DAPI': 0.20}, # 14
    'Case3_S1_1_0': {'CD3': 0.50, 'CD8': 0.85, 'FoxP3': 0.85, 'PanCK': 0.70, 'DAPI': 0.20}, # 15
    'Case3_S1_1_1': {'CD3': 0.65, 'CD8': 0.65, 'FoxP3': 0.65, 'PanCK': 0.60, 'DAPI': 0.20}, # 16
    'Case3_T1_0_0': {'CD3': 0.90, 'CD8': 0.90, 'FoxP3': 0.90, 'PanCK': 0.50, 'DAPI': 0.20}, # 17
    'Case3_T1_0_1': {'CD3': 0.80, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.50, 'DAPI': 0.20}, # 18
    'Case3_T1_1_0': {'CD3': 0.80, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.50, 'DAPI': 0.20}, # 19
    'Case3_T1_1_1': {'CD3': 0.80, 'CD8': 0.85, 'FoxP3': 0.80, 'PanCK': 0.40, 'DAPI': 0.20}, # 20
    'Case3_T2_0_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.70, 'PanCK': 0.40, 'DAPI': 0.20}, # 21
    'Case3_T2_0_1': {'CD3': 0.35, 'CD8': 0.80, 'FoxP3': 0.85, 'PanCK': 0.40, 'DAPI': 0.20}, # 22
    'Case3_T2_1_0': {'CD3': 0.35, 'CD8': 0.85, 'FoxP3': 0.70, 'PanCK': 0.40, 'DAPI': 0.20}, #23
    'Case3_T2_1_1': {'CD3': 0.35, 'CD8': 0.65, 'FoxP3': 0.60, 'PanCK': 0.40, 'DAPI': 0.20}, # 24
    'Case3_T3_0_0': {'CD3': 0.35, 'CD8': 0.80, 'FoxP3': 0.90, 'PanCK': 0.40, 'DAPI': 0.20}, #25
    'Case3_T3_0_1': {'CD3': 0.70, 'CD8': 0.70, 'FoxP3': 0.80, 'PanCK': 0.40, 'DAPI': 0.20}, # 26
    'Case3_T3_1_1': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.80, 'PanCK': 0.40, 'DAPI': 0.20}, # 27
    'Case4_M1_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 1
    'Case4_M1_0_1': {'CD3': 0.35, 'CD8': 0.75, 'FoxP3': 0.75, 'PanCK': 0.40, 'DAPI': 0.20}, # 2
    'Case4_M1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 3
    'Case4_M1_1_1': {'CD3': 0.80, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.40, 'DAPI': 0.20}, # 4
    'Case4_M2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 5
    'Case4_M2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 6
    'Case4_M2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 7
    'Case4_M2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 8
    'Case4_M3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 9
    'Case4_M3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 10
    'Case4_M3_1_0': {'CD3': 0.8, 'CD8': 0.85, 'FoxP3': 0.95, 'PanCK': 0.40, 'DAPI': 0.20}, # 11
    'Case4_M3_1_1': {'CD3': 0.75, 'CD8': 0.75, 'FoxP3': 0.85, 'PanCK': 0.40, 'DAPI': 0.20}, # 12
    'Case4_S1_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.50, 'PanCK': 0.30, 'DAPI': 0.20}, # 13
    'Case4_S1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 14
    'Case4_S1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 15
    'Case4_S1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 16
    'Case4_S2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.30}, # 17
    'Case4_S2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.85, 'PanCK': 0.30, 'DAPI': 0.20}, # 18
    'Case4_S2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20}, # 19
    'Case4_S2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 20
    'Case4_S3_1_1': {'CD3': 0.35, 'CD8': 0.50, 'FoxP3': 0.40, 'PanCK': 0.40, 'DAPI': 0.20}, # 21
    'Case4_T1_0_0': {'CD3': 0.35, 'CD8': 0.80, 'FoxP3': 0.60, 'PanCK': 0.40, 'DAPI': 0.20}, # 22
    'Case4_T1_0_1': {'CD3': 0.35, 'CD8': 0.85, 'FoxP3': 0.60, 'PanCK': 0.40, 'DAPI': 0.20}, # 23
    'Case4_T1_1_0': {'CD3': 0.70, 'CD8': 0.85, 'FoxP3': 0.85, 'PanCK': 0.40, 'DAPI': 0.20}, # 24
    'Case4_T1_1_1': {'CD3': 0.35, 'CD8': 0.60, 'FoxP3': 0.50, 'PanCK': 0.40, 'DAPI': 0.20}, # 25
    'Case4_T2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 26
    'Case4_T2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 27
    'Case4_T2_1_0': {'CD3': 0.35, 'CD8': 0.50, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 28
    'Case4_T2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 29
    'Case4_T3_0_0': {'CD3': 0.35, 'CD8': 0.60, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 30
    'Case4_T3_0_1': {'CD3': 0.80, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20}, # 31
    'Case4_T3_1_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 32
    'Case4_T3_1_1': {'CD3': 0.35, 'CD8': 0.65, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20}, # 33
    'Case5_M1_0_0': {'CD3': 0.50, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20}, # 1
    'Case5_M1_0_1': {'CD3': 0.50, 'CD8': 0.35, 'FoxP3': 0.90, 'PanCK': 0.30, 'DAPI': 0.20}, # 2
    'Case5_M1_1_0': {'CD3': 0.50, 'CD8': 0.35, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20}, # 3
    'Case5_M1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.95, 'PanCK': 0.30, 'DAPI': 0.20}, # 4
    'Case5_M2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.90, 'PanCK': 0.9999, 'DAPI': 0.20}, # 5 # pan ck wierd
    'Case5_M2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 6
    'Case5_M2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20}, # 7
    'Case5_M2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20}, # 8
    'Case5_M3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 9
    'Case5_M3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 10
    'Case5_M3_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 11
    'Case5_M3_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 12
    'Case5_S1_0_0': {'CD3': 0.60, 'CD8': 0.70, 'FoxP3': 0.70, 'PanCK': 0.50, 'DAPI': 0.20}, # 13
    'Case5_S1_0_1': {'CD3': 0.50, 'CD8': 0.70, 'FoxP3': 0.60, 'PanCK': 0.60, 'DAPI': 0.20}, # 14
    'Case5_S1_1_0': {'CD3': 0.70, 'CD8': 0.70, 'FoxP3': 0.70, 'PanCK': 0.70, 'DAPI': 0.20}, # 15
    'Case5_S1_1_1': {'CD3': 0.70, 'CD8': 0.80, 'FoxP3': 0.70, 'PanCK': 0.70, 'DAPI': 0.20}, # 16
    'Case5_S2_0_0': {'CD3': 0.60, 'CD8': 0.70, 'FoxP3': 0.70, 'PanCK': 0.60, 'DAPI': 0.20}, # 17
    'Case5_S2_0_1': {'CD3': 0.50, 'CD8': 0.60, 'FoxP3': 0.70, 'PanCK': 0.60, 'DAPI': 0.20}, # 18
    'Case5_S2_1_0': {'CD3': 0.50, 'CD8': 0.80, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20}, # 19
    'Case5_S2_1_1': {'CD3': 0.65, 'CD8': 0.70, 'FoxP3': 0.75, 'PanCK': 0.50, 'DAPI': 0.20}, #20
    'Case5_S3_0_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.50, 'PanCK': 0.50, 'DAPI': 0.20}, # 21
    'Case5_S3_0_1': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.50, 'PanCK': 0.50, 'DAPI': 0.20}, # 22
    'Case5_S3_1_0': {'CD3': 0.70, 'CD8': 0.70, 'FoxP3': 0.70, 'PanCK': 0.50, 'DAPI': 0.20}, # 23
    'Case5_S3_1_1': {'CD3': 0.35, 'CD8': 0.65, 'FoxP3': 0.40, 'PanCK': 0.50, 'DAPI': 0.20}, # 24
    'Case5_T1_0_0': {'CD3': 0.40, 'CD8': 0.70, 'FoxP3': 0.50, 'PanCK': 0.50, 'DAPI': 0.20}, # 25
    'Case5_T1_0_1': {'CD3': 0.70, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.50, 'DAPI': 0.20}, # 26
    'Case5_T1_1_0': {'CD3': 0.70, 'CD8': 0.70, 'FoxP3': 0.75, 'PanCK': 0.40, 'DAPI': 0.20}, # 27
    'Case5_T1_1_1': {'CD3': 0.60, 'CD8': 0.75, 'FoxP3': 0.80, 'PanCK': 0.40, 'DAPI': 0.20}, # 28
    'Case5_T2_0_0': {'CD3': 0.70, 'CD8': 0.70, 'FoxP3': 0.80, 'PanCK': 0.40, 'DAPI': 0.20}, # 29
    'Case5_T2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.40, 'DAPI': 0.20}, # 30
    'Case5_T2_1_0': {'CD3': 0.80, 'CD8': 0.70, 'FoxP3': 0.85, 'PanCK': 0.40, 'DAPI': 0.20}, # 31
    'Case5_T2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20}, # 32
    'Case5_T3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.45, 'PanCK': 0.30, 'DAPI': 0.20}, # 33
    'Case5_T3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.65, 'PanCK': 0.30, 'DAPI': 0.20}, # 34
    'Case5_T3_1_0': {'CD3': 0.35, 'CD8': 0.50, 'FoxP3': 0.45, 'PanCK': 0.30, 'DAPI': 0.20}, # 35
    'Case5_T3_1_1': {'CD3': 0.35, 'CD8': 0.50, 'FoxP3': 0.85, 'PanCK': 0.30, 'DAPI': 0.20}, # 36
    'Case6_M1_0_0': {'CD3': 0.35, 'CD8': 0.40, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 1
    'Case6_M1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 2 
    'Case6_M1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 3
    'Case6_M1_1_1': {'CD3': 0.35, 'CD8': 0.80, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 4
    'Case6_M2_0_0': {'CD3': 0.35, 'CD8': 0.60, 'FoxP3': 0.45, 'PanCK': 0.30, 'DAPI': 0.20}, # 5
    'Case6_M2_0_1': {'CD3': 0.35, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20}, # 6
    'Case6_M2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20}, # 7
    'Case6_M2_1_1': {'CD3': 0.35, 'CD8': 0.40, 'FoxP3': 0.45, 'PanCK': 0.30, 'DAPI': 0.20}, # 8
    'Case6_M3_0_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 9
    'Case6_M3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, #10
    'Case6_M3_1_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20}, # 11
    'Case6_M3_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 12
    'Case6_S1_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20}, # 13
    'Case6_S1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.50, 'PanCK': 0.30, 'DAPI': 0.20}, # 14
    'Case6_S1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20}, # 15
    'Case6_S1_1_1': {'CD3': 0.35, 'CD8': 0.60, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20}, # 16
    'Case6_S2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.40, 'DAPI': 0.20}, # 17
    'Case6_S2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 18
    'Case6_S2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 19
    'Case6_S2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.40, 'DAPI': 0.20}, # 20
    'Case6_S3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.40, 'DAPI': 0.20}, # 21
    'Case6_S3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.50, 'DAPI': 0.20}, # 22
    'Case6_S3_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.40, 'DAPI': 0.20}, # 23
    'Case6_S3_1_1': {'CD3': 0.35, 'CD8': 0.40, 'FoxP3': 0.60, 'PanCK': 0.50, 'DAPI': 0.20}, # 24
    'Case6_T1_0_0': {'CD3': 0.35, 'CD8': 0.75, 'FoxP3': 0.75, 'PanCK': 0.30, 'DAPI': 0.20}, # 25
    'Case6_T1_0_1': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20}, # 26
    'Case6_T1_1_0': {'CD3': 0.75, 'CD8': 0.75, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20}, # 27
    'Case6_T1_1_1': {'CD3': 0.50, 'CD8': 0.75, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20}, # 28 
    'Case6_T2_0_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20}, # 29
    'Case6_T2_0_1': {'CD3': 0.70, 'CD8': 0.70, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20}, # 30
    'Case6_T2_1_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20}, # 31
    'Case6_T2_1_1': {'CD3': 0.70, 'CD8': 0.70, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20}, # 32
    'Case6_T3_0_0': {'CD3': 0.75, 'CD8': 0.6, 'FoxP3': 0.6, 'PanCK': 0.30, 'DAPI': 0.20}, # 33 remove
    'Case6_T3_0_1': {'CD3': 0.7, 'CD8': 0.6, 'FoxP3': 0.6, 'PanCK': 0.30, 'DAPI': 0.20}, # 34 remove
    'Case6_T3_1_0': {'CD3': 0.35, 'CD8': 0.60, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20}, # 35 
    'Case6_T3_1_1': {'CD3': 0.60, 'CD8': 0.60, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20}, # 36 remove
    'Case7_M1_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 1
    'Case7_M1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 2
    'Case7_M1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 3
    'Case7_M1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 4
    'Case7_M2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 5
    'Case7_M2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 6
    'Case7_M2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 7
    'Case7_M2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 8
    'Case7_M3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 9
    'Case7_M3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 10
    'Case7_M3_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 11
    'Case7_M3_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 12
    'Case7_S1_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20}, # 13
    'Case7_S1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20}, # 14
    'Case7_S1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 15
    'Case7_S1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 16
    'Case7_S2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 17
    'Case7_S2_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 18
    'Case7_S2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 19
    'Case7_S2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 20
    'Case7_S3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 21
    'Case7_S3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 22
    'Case7_S3_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20}, # 23
    'Case7_S3_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 24
    'Case7_T1_0_0': {'CD3': 0.35, 'CD8': 0.50, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20}, # 25
    'Case7_T1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20},#26
    'Case7_T1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 27
    'Case7_T1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 28
    'Case7_T2_0_0': {'CD3': 0.35, 'CD8': 0.50, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20}, # 29
    'Case7_T2_0_1': {'CD3': 0.50, 'CD8': 0.60, 'FoxP3': 0.60, 'PanCK': 0.30, 'DAPI': 0.20}, # 30
    'Case7_T2_1_1': {'CD3': 0.35, 'CD8': 0.40, 'FoxP3': 0.55, 'PanCK': 0.30, 'DAPI': 0.20}, # 31
    'Case7_T3_0_0': {'CD3': 0.90, 'CD8': 0.90, 'FoxP3': 0.95, 'PanCK': 0.40, 'DAPI': 0.30}, # 32
    'Case7_T3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.55, 'PanCK': 0.40, 'DAPI': 0.20}, # 33
    'Case7_T3_1_0': {'CD3': 0.50, 'CD8': 0.65, 'FoxP3': 0.80, 'PanCK': 0.40, 'DAPI': 0.20}, # 34
    'Case7_T3_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 35
    'Case8_M1_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 1
    'Case8_M1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 2
    'Case8_M1_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 3
    'Case8_M1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 4
    'Case8_M2_0_0': {'CD3': 0.35, 'CD8': 0.60, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 5
    'Case8_M2_0_1': {'CD3': 0.35, 'CD8': 0.60, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20}, # 6
    'Case8_M2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 7
    'Case8_M2_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 8
    'Case8_M3_0_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20}, # 9
    'Case8_M3_0_1': {'CD3': 0.45, 'CD8': 0.70, 'FoxP3': 0.90, 'PanCK': 0.30, 'DAPI': 0.20}, # 10
    'Case8_M3_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 11
    'Case8_M3_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 12
    'Case8_S1_0_0': {'CD3': 0.60, 'CD8': 0.60, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20}, # 13 # foxp3 weird
    'Case8_S1_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 14
    'Case8_S1_1_0': {'CD3': 0.35, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.80, 'DAPI': 0.20}, # 15 remove
    'Case8_S1_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.50, 'PanCK': 0.30, 'DAPI': 0.20}, # 16
    'Case8_S2_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.40, 'PanCK': 0.30, 'DAPI': 0.20}, # 17
    'Case8_S2_0_1': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20}, # 18
    'Case8_S2_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.50, 'PanCK': 0.30, 'DAPI': 0.20}, # 19
    'Case8_S2_1_1': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.80, 'PanCK': 0.50, 'DAPI': 0.20}, # 20 ??
    'Case8_S3_0_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.50, 'PanCK': 0.30, 'DAPI': 0.20}, # 21
    'Case8_S3_0_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.50, 'PanCK': 0.30, 'DAPI': 0.20}, # 22
    'Case8_S3_1_0': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.50, 'PanCK': 0.30, 'DAPI': 0.20}, # 23 drop
    'Case8_S3_1_1': {'CD3': 0.35, 'CD8': 0.35, 'FoxP3': 0.55, 'PanCK': 0.40, 'DAPI': 0.20}, # 24
    'Case8_T1_0_0': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20}, # 25
    'Case8_T1_0_1': {'CD3': 0.35, 'CD8': 0.70, 'FoxP3': 0.70, 'PanCK': 0.30, 'DAPI': 0.20}, # 26
    'Case8_T1_1_0': {'CD3': 0.60, 'CD8': 0.75, 'FoxP3': 0.80, 'PanCK': 0.30, 'DAPI': 0.20}, # 27
    'Case8_T1_1_1': {'CD3': 0.35, 'CD8': 0.65, 'FoxP3': 0.70,  'PanCK': 0.30, 'DAPI': 0.20}, # 28
    'Case8_T2_0_0': {'CD3': 0.90, 'CD8': 0.90, 'FoxP3': 0.90, 'PanCK': 0.50, 'DAPI': 0.35}, # 29
    'Case8_T2_0_1': {'CD3': 0.50, 'CD8': 0.65, 'FoxP3': 0.75, 'PanCK': 0.40, 'DAPI': 0.20}, # 30
    'Case8_T2_1_0': {'CD3': 0.85, 'CD8': 0.85, 'FoxP3': 0.85, 'PanCK': 0.45, 'DAPI': 0.40}, # 31
    'Case8_T2_1_1': {'CD3': 0.85, 'CD8': 0.75, 'FoxP3': 0.90, 'PanCK': 0.50, 'DAPI': 0.30}, # 32
    'Case8_T3_0_0': {'CD3': 0.90, 'CD8': 0.90, 'FoxP3': 0.90, 'PanCK': 0.60, 'DAPI': 0.30}, # 33
    'Case8_T3_0_1': {'CD3': 0.50, 'CD8': 0.70, 'FoxP3': 0.50, 'PanCK': 0.60, 'DAPI': 0.30}, # 34
    'Case8_T3_1_0': {'CD3': 0.80, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.50, 'DAPI': 0.20},
    'Case8_T3_1_1': {'CD3': 0.80, 'CD8': 0.80, 'FoxP3': 0.80, 'PanCK': 0.55, 'DAPI': 0.20},
}

THUMB = 128  # px per panel in grid


def get_patch_ids(case: str):
    ids = set()
    for f in MIHC_DIR.glob(f'{case}_*_Hematoxylin.png'):
        parts = f.stem.split('_')
        ids.add('_'.join(parts[:4]))
    return sorted(ids)


def visualize_case(case: str):
    ids = get_patch_ids(case)
    ncols = 1 + len(IF_MARKERS)
    nrows = len(ids)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.8, nrows * 2.0))
    if nrows == 1:
        axes = axes[None, :]

    for c, label in enumerate(['H&E'] + IF_MARKERS):
        axes[0, c].set_title(label, fontsize=9, fontweight='bold')

    for row, pid in enumerate(ids):
        bounds = BOUNDS.get(pid, {})
        he = np.array(Image.open(MIHC_DIR / f'{pid}_Hematoxylin.png'))
        axes[row, 0].imshow(he)
        axes[row, 0].set_ylabel(pid.split('_', 1)[1], fontsize=7, rotation=0,
                                labelpad=55, va='center')
        axes[row, 0].axis('off')

        for col, m in enumerate(IF_MARKERS, start=1):
            arr = np.array(Image.open(MIF_DIR / f'{pid}_{m}.png')).astype(np.float32)
            nz  = arr[arr > 0]
            p99 = float(np.percentile(nz, 99)) if len(nz) else 1.0
            normed = np.clip(arr / max(p99, 1.0), 0.0, 1.0)
            lo = bounds.get(m)
            if lo is not None:
                normed[normed < lo] = 0.0
            sig_pct = 100 * (normed > 0).mean()
            axes[row, col].imshow(normed, cmap=CMAPS[m], vmin=0, vmax=1)
            axes[row, col].set_xlabel(f'{sig_pct:.0f}%', fontsize=6, labelpad=1)
            axes[row, col].axis('off')

    plt.suptitle(case, fontsize=10, y=1.001)
    plt.tight_layout(h_pad=0.3, w_pad=0.2)
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f'{case}_all_patches.png'
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'Saved → {out}')

# 9
for case in [f'Case{i}' for i in range(8, 9)]:
    print(f'Processing {case}…')
    visualize_case(case)