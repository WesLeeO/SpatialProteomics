"""
Multi-source token-level regression training.

Canonical panel (19 markers)
-----------------------------
  0–15  ORION markers (identical order)
  16–18 NSCLC-Charité additions: CD56, Granzyme_B, PD-1

Panel misalignment is handled via a per-sample channel_mask (N_CANONICAL,) bool:
every patch carries the mask of the dataset it came from. Only measured channels
contribute to the loss — the model head always predicts all 19 outputs.

Split strategy
--------------
  ORION       : slide-level; val=CRC01/CRC02, test=CRC03/CRC04, rest=train
  HEMIT       : pre-split (train.h5 / val.h5 / test.h5) — respected as-is
  SG          : file-level (each tissue = one unit); 1 val + 1 test file
  JEDI        : single file → 80/10/10 patch-level split
  NSCLC       : spot-level random split
  Immunoatlas : core-level random split
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"

import sys, math
import numpy as np
import cv2
import h5py
import tifffile, zarr
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from scipy.stats import spearmanr
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import Dataset, DataLoader

from model import SpatialModel

# ── Canonical panel ────────────────────────────────────────────────────────────
CANONICAL_MARKERS = [
    # ORION (indices 0-15, same order)
    "Hoechst", "CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a", "CD45RO",
    "CD20", "PD-L1", "CD3e", "CD163", "E-Cadherin", "Ki-67", "Pan-CK", "SMA",
    # NSCLC-Charité additions (indices 16-18)
    "CD56", "Granzyme_B", "PD-1",
]
N_CANONICAL = len(CANONICAL_MARKERS)   # 19
_CANON_IDX  = {m: i for i, m in enumerate(CANONICAL_MARKERS)}

# Maps any dataset-specific name to its canonical form.
# Unmapped names that also aren't canonical are silently dropped.
NAME_TO_CANONICAL = {
    # NSCLC Charité
    "CD3":        "CD3e",
    "CD8":        "CD8a",
    "FoxP3":      "FOXP3",
    "CK":         "Pan-CK",
    # Immunoatlas
    "GranzymeB":  "Granzyme_B",
    "Cytokeratin": "Pan-CK",
    # HEMIT / JEDI
    "Dapi":       "Hoechst",
    "DNA":        "Hoechst",
    # Singular Genomics
    "aSMA":       "SMA",
    "KI67":       "Ki-67",
    "PanCK":      "Pan-CK",
    "PD1":        "PD-1",
    "PDL1":       "PD-L1",
}

# ── Dataset configs ────────────────────────────────────────────────────────────
DATASET_CONFIGS = [
    dict(
        name   = "orion",
        type   = "orion",
        h5_dir = "orion_crc_patch_dataset_reg",
        he_dir = "/mnt/ssd1/virtual_proteomics/data/ORION_CRC",
    ),
    dict(
        name        = "hemit",
        type        = "hemit",           # pre-split, H&E path stored per-patch in /sources
        h5_dir      = "hemit_patch_dataset",
    ),
    dict(
        name    = "sg",
        type    = "sg",                  # per-tissue-file, valid_markers per file
        sg_dir  = "singular_genomics",
        he_root = "/mnt/ssd1/virtual_proteomics/data/singular_genomics",
    ),
    dict(
        name    = "jedi",
        type    = "jedi",
        h5_path = "jedi_patch_dataset/JEDI20034_patch_dataset.h5",
        he_tif  = "/mnt/ssd1/virtual_proteomics/data/JEDI_201207/JEDI20033.tif",
    ),
    dict(
        name      = "nsclc_charite",
        type      = "jpeg_crops",
        h5_path   = "nsclc_charite_patch_dataset.h5",
        he_dir    = "/mnt/ssd1/virtual_proteomics/data/nsclc_charite/extracted/spot_center_crops",
        he_suffix = "_he.jpg",
        id_key    = "spot_ids",
    ),
    dict(
        name      = "immunoatlas",
        type      = "png_crops",
        h5_path   = "immunoatlas_NOLN210920_patch_dataset.h5",
        he_dir    = "immunoatlas_70_png",
        he_suffix = ".png",
        id_key    = "core_ids",
    ),
]

# ── ORION fixed val / test slides ──────────────────────────────────────────────
ORION_VAL_SLIDES  = ["CRC01", "CRC02"]
ORION_TEST_SLIDES = ["CRC03", "CRC04"]
SG_CANCER_ONLY    = True   # exclude normal/control tissues from SG

# ── Training config ────────────────────────────────────────────────────────────
MODEL_NAME       = "UNI2"
OUTPUT_DIR       = Path(f"outputs_multisource_{MODEL_NAME}")
TOKEN_GRID       = 16
VAL_FRAC         = 0.15      # for datasets without a fixed split
BATCH_SIZE       = 512
NUM_EPOCHS       = 30
LR               = 1e-4
NUM_WORKERS      = 4
SEED             = 42
PHASE1_EPOCHS    = 2
UNFREEZE_LAST_N  = 4
WARMUP_STEPS     = 500

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Channel mapping ────────────────────────────────────────────────────────────

def build_channel_map(dataset_markers: list[str],
                      valid_mask: np.ndarray | None = None
                      ) -> tuple[list[int], list[int], np.ndarray]:
    """
    Map a dataset's marker list to canonical indices.

    valid_mask : optional (len(dataset_markers),) bool — per-marker quality flag
                 (used by SG which stores a /valid_markers array per H5 file).
                 When provided, markers flagged False are excluded even if they
                 have a canonical mapping.

    Returns
    -------
    src_idx   : positions in dataset_markers with a canonical match
    dst_idx   : corresponding canonical indices
    chan_mask : (N_CANONICAL,) bool — True for matched + valid channels
    """
    src_idx, dst_idx = [], []
    for di, name in enumerate(dataset_markers):
        if valid_mask is not None and not valid_mask[di]:
            continue
        cname = NAME_TO_CANONICAL.get(name, name)
        if cname in _CANON_IDX:
            src_idx.append(di)
            dst_idx.append(_CANON_IDX[cname])
    chan_mask = np.zeros(N_CANONICAL, dtype=bool)
    chan_mask[dst_idx] = True
    return src_idx, dst_idx, chan_mask


# ── Augmentation (geometric only) ─────────────────────────────────────────────

def augment_patch(patch: np.ndarray, target: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """patch: (H,W,3) uint8 | target: (C,G,G) float32 → augmented pair."""
    k = np.random.randint(4)
    if k > 0:
        patch  = np.rot90(patch,  k, axes=(0, 1)).copy()
        target = np.rot90(target, k, axes=(1, 2)).copy()
    if np.random.random() < 0.5:
        patch  = patch[:, ::-1, :].copy()
        target = target[:, :, ::-1].copy()
    if np.random.random() < 0.5:
        patch  = patch[::-1, :, :].copy()
        target = target[:, ::-1, :].copy()
    return patch, target


# ── Dataset ────────────────────────────────────────────────────────────────────

class MultiSourcePatchDataset(Dataset):
    """
    Each item:
      patch        (3, 224, 224) float32  ImageNet-normalised H&E
      targets      (N_CANONICAL, G, G) float32  zeros for unmeasured channels
      channel_mask (N_CANONICAL,) bool   True for channels this source measured

    channel_mask is dataset-level (same object for all patches from one source),
    not patch-level.  It reflects which canonical markers the source dataset
    measured, optionally intersected with per-file valid_markers flags (SG).
    """

    def __init__(self, configs: list[dict],
                 split: str = "train",          # "train" | "val" | "test"
                 split_ids: dict | None = None, # {dataset_name: [ids]}  overrides default
                 augment: bool = False):
        self.augment = augment
        self.split   = split
        # items: (loader_type, *loader_args, canon_targets, chan_mask)
        self._items:   list[tuple] = []
        self._item_ds: list[str]   = []   # dataset name per item, for sampling weights
        self._orion_slides: list[str] = []
        self._jedi_tif: str | None = None

        for cfg in configs:
            ids = split_ids.get(cfg["name"]) if split_ids else None
            self._load(cfg, split, ids)

        print(f"[{split}] MultiSourcePatchDataset: {len(self._items):,} patches total")

    # ── per-source loaders ─────────────────────────────────────────────────────

    def _load(self, cfg, split, ids):
        t = cfg["type"]
        if   t == "orion":       self._load_orion(cfg, ids)
        elif t == "hemit":       self._load_hemit(cfg, split)
        elif t == "sg":          self._load_sg(cfg, ids)
        elif t == "jedi":        self._load_jedi(cfg, ids)
        elif t in ("jpeg_crops", "png_crops"): self._load_crops(cfg, ids)
        else: raise ValueError(f"Unknown type: {t}")

    def _load_orion(self, cfg, ids):
        h5_dir = Path(cfg["h5_dir"])
        he_dir = Path(cfg["he_dir"])
        src_idx = dst_idx = chan_mask = None
        n = 0
        for h5_path in sorted(h5_dir.glob("*_patch_dataset.h5")):
            slide = h5_path.stem.replace("_patch_dataset", "")
            if ids is not None and slide not in ids:
                continue
            tiff_matches = list((he_dir / slide).glob("*-registered.ome.tif"))
            if not tiff_matches:
                print(f"  [orion] no tiff for {slide} — skipping"); continue
            try:
                with h5py.File(h5_path) as f:
                    coords  = f["coords"][:]
                    targets = f["targets"][:]
                    markers = list(f.attrs["marker_names"])
                    psz     = int(f.attrs["patch_size_level0"])
            except Exception as e:
                print(f"  [orion] {h5_path.name}: {e} — skipping"); continue
            if src_idx is None:
                src_idx, dst_idx, chan_mask = build_channel_map(markers)
                print(f"  [orion] {len(src_idx)}/{len(markers)} markers → canonical")
            si = len(self._orion_slides)
            self._orion_slides.append(str(tiff_matches[0]))
            for i in range(len(coords)):
                ct = np.zeros((N_CANONICAL, TOKEN_GRID, TOKEN_GRID), dtype=np.float32)
                ct[dst_idx] = targets[i][src_idx]
                self._items.append(("orion", si, int(coords[i,0]), int(coords[i,1]), psz, ct, chan_mask))
                self._item_ds.append("orion")
                n += 1
        print(f"  [orion] loaded {n:,} patches")

    def _load_hemit(self, cfg, split):
        """HEMIT is pre-split; load the matching H5 directly."""
        h5_path = Path(cfg["h5_dir"]) / f"{split}.h5"
        if not h5_path.exists():
            print(f"  [hemit] {h5_path} not found — skipping"); return
        with h5py.File(h5_path) as f:
            coords     = f["coords"][:]
            targets    = f["targets"][:]
            markers    = list(f.attrs["marker_names"])
            sources    = [s.decode() if isinstance(s, bytes) else s for s in f["sources"][:]]
            psz        = int(f.attrs["crop_size"])
            resize_to  = int(f.attrs.get("resize_to", 0))
        src_idx, dst_idx, chan_mask = build_channel_map(markers)
        print(f"  [hemit/{split}] {len(src_idx)}/{len(markers)} markers → canonical")
        for i in range(len(coords)):
            ct = np.zeros((N_CANONICAL, TOKEN_GRID, TOKEN_GRID), dtype=np.float32)
            ct[dst_idx] = targets[i][src_idx]
            self._items.append(
                ("hemit", sources[i], int(coords[i,0]), int(coords[i,1]),
                 psz, resize_to, ct, chan_mask)
            )
            self._item_ds.append("hemit")
        print(f"  [hemit/{split}] loaded {len(coords):,} patches")

    def _load_sg(self, cfg, ids):
        sg_dir  = Path(cfg["sg_dir"])
        he_root = Path(cfg["he_root"])
        n = 0
        for h5_path in sorted(sg_dir.glob("*_patch_dataset.h5")):
            disease = h5_path.stem.replace("_patch_dataset", "")
            if SG_CANCER_ONLY and "cancer" not in disease:
                continue
            if ids is not None and disease not in ids:
                continue
            try:
                with h5py.File(h5_path) as f:
                    coords       = f["coords"][:]
                    targets      = f["targets"][:]
                    markers      = list(f.attrs["marker_names"])
                    psz          = int(f.attrs["patch_size_level0"])
                    valid_mask   = f["valid_markers"][:] if "valid_markers" in f else None
            except Exception as e:
                print(f"  [sg] {h5_path.name}: {e} — skipping"); continue
            src_idx, dst_idx, chan_mask = build_channel_map(markers, valid_mask)
            # H&E: *_HE.ome.tiff in he_root/{disease}/
            he_matches = list((he_root / disease).rglob("*_HE.ome.tiff"))
            if not he_matches:
                print(f"  [sg] no H&E for {disease} — skipping"); continue
            si = len(self._orion_slides)   # reuse zarr cache for SG (same ome.tif format)
            self._orion_slides.append(str(he_matches[0]))
            print(f"  [sg/{disease}] {len(src_idx)}/{len(markers)} markers → canonical")
            for i in range(len(coords)):
                ct = np.zeros((N_CANONICAL, TOKEN_GRID, TOKEN_GRID), dtype=np.float32)
                ct[dst_idx] = targets[i][src_idx]
                self._items.append(("orion", si, int(coords[i,0]), int(coords[i,1]), psz, ct, chan_mask))
                self._item_ds.append("sg")
                n += 1
        print(f"  [sg] loaded {n:,} patches")

    def _load_jedi(self, cfg, ids):
        h5_path = Path(cfg["h5_path"])
        he_tif  = cfg["he_tif"]
        if not h5_path.exists():
            print(f"  [jedi] {h5_path} not found — skipping"); return
        with h5py.File(h5_path) as f:
            coords  = f["coords"][:]
            targets = f["targets"][:]
            markers = list(f.attrs["marker_names"])
            psz     = int(f.attrs["patch_size_level0"])
        src_idx, dst_idx, chan_mask = build_channel_map(markers)
        print(f"  [jedi] {len(src_idx)}/{len(markers)} markers → canonical")
        self._jedi_tif = he_tif
        si = len(self._orion_slides)
        self._orion_slides.append(he_tif)
        # ids contains integer patch indices for the requested split
        patch_indices = ids if ids is not None else range(len(coords))
        n = 0
        for i in patch_indices:
            ct = np.zeros((N_CANONICAL, TOKEN_GRID, TOKEN_GRID), dtype=np.float32)
            ct[dst_idx] = targets[i][src_idx]
            self._items.append(("orion", si, int(coords[i,0]), int(coords[i,1]), psz, ct, chan_mask))
            self._item_ds.append("jedi")
            n += 1
        print(f"  [jedi] loaded {n:,} patches")

    def _load_crops(self, cfg, ids):
        h5_path = Path(cfg["h5_path"])
        if not h5_path.exists():
            print(f"  [{cfg['name']}] {h5_path} not found — skipping"); return
        he_dir = Path(cfg["he_dir"])
        with h5py.File(h5_path) as f:
            coords     = f["coords"][:]
            targets    = f["targets"][:]
            markers    = list(f.attrs["marker_names"])
            psz        = int(f.attrs["patch_size_level0"])
            sample_ids = np.array([
                s.decode() if isinstance(s, bytes) else s for s in f[cfg["id_key"]][:]
            ])
        src_idx, dst_idx, chan_mask = build_channel_map(markers)
        print(f"  [{cfg['name']}] {len(src_idx)}/{len(markers)} markers → canonical")
        ids_set = set(ids) if ids is not None else None
        n = 0
        for i in range(len(coords)):
            if ids_set is not None and sample_ids[i] not in ids_set:
                continue
            he_path = str(he_dir / f"{sample_ids[i]}{cfg['he_suffix']}")
            ct = np.zeros((N_CANONICAL, TOKEN_GRID, TOKEN_GRID), dtype=np.float32)
            ct[dst_idx] = targets[i][src_idx]
            self._items.append(
                (cfg["type"], he_path, int(coords[i,0]), int(coords[i,1]), psz, ct, chan_mask)
            )
            self._item_ds.append(cfg["name"])
            n += 1
        print(f"  [{cfg['name']}] loaded {n:,} patches")

    # ── H&E loading ───────────────────────────────────────────────────────────

    def _open_zarr(self, slide_idx: int):
        if not hasattr(self, "_zarr_cache"):
            self._zarr_cache = {}
        if slide_idx not in self._zarr_cache:
            tif   = tifffile.TiffFile(self._orion_slides[slide_idx])
            store = tif.aszarr()
            z     = zarr.open(store, mode="r")
            self._zarr_cache[slide_idx] = z["0"] if isinstance(z, zarr.hierarchy.Group) else z
        return self._zarr_cache[slide_idx]

    def _load_he(self, item: tuple) -> np.ndarray:
        """Returns (224, 224, 3) uint8."""
        src = item[0]
        if src == "orion":
            _, si, x, y, psz = item[:5]
            arr   = self._open_zarr(si)
            patch = np.array(arr[y:y+psz, x:x+psz, :])
        elif src == "hemit":
            _, src_path, x, y, psz, resize_to = item[:6]
            img = np.array(tifffile.imread(src_path))
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            elif img.shape[0] in (1, 3):    # (C, H, W) → (H, W, C)
                img = img.transpose(1, 2, 0)
                if img.shape[2] == 1:
                    img = np.concatenate([img]*3, axis=2)
            if resize_to > 0 and img.shape[0] != resize_to:
                img = cv2.resize(img, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
            patch = img[y:y+psz, x:x+psz]
        else:   # jpeg_crops / png_crops
            _, he_path, x, y, psz = item[:5]
            img   = np.array(Image.open(he_path))
            patch = img[y:y+psz, x:x+psz]
            if patch.ndim == 2:
                patch = np.stack([patch]*3, axis=-1)
        return cv2.resize(patch, (224, 224), interpolation=cv2.INTER_LINEAR)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        item      = self._items[idx]
        patch     = self._load_he(item)
        target    = item[-2].copy() if self.augment else item[-2]
        chan_mask  = item[-1]

        if self.augment:
            patch, target = augment_patch(patch, target)

        patch  = (patch.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        patch  = torch.from_numpy(patch.transpose(2, 0, 1))
        target = torch.from_numpy(target)
        mask   = torch.from_numpy(chan_mask)
        return patch, target, mask

    def compute_marker_stds(self) -> torch.Tensor:
        """Per-canonical-marker std, computed only over patches that measured each channel."""
        total_sum   = np.zeros(N_CANONICAL, dtype=np.float64)
        total_sumsq = np.zeros(N_CANONICAL, dtype=np.float64)
        total_n     = np.zeros(N_CANONICAL, dtype=np.float64)
        for item in self._items:
            t, m = item[-2], item[-1]
            flat = t.reshape(N_CANONICAL, -1)
            n    = flat.shape[1]
            total_sum[m]   += flat[m].sum(1)
            total_sumsq[m] += (flat[m] ** 2).sum(1)
            total_n[m]     += n
        mean = np.where(total_n > 0, total_sum / np.maximum(total_n, 1), 0.0)
        var  = np.where(total_n > 0, total_sumsq / np.maximum(total_n, 1) - mean**2, 1.0)
        return torch.from_numpy(np.sqrt(np.maximum(var, 1e-8)).astype(np.float32))

    def compute_token_means(self) -> np.ndarray:
        """
        Per-canonical-marker mean token value, computed only over training patches
        that measured each channel (chan_mask=True). Returns (N_CANONICAL,) float32.
        Used as the informative-token threshold: tokens above this mean contribute
        to the loss; background tokens below it are masked out.
        """
        total_sum = np.zeros(N_CANONICAL, np.float64)
        total_n   = np.zeros(N_CANONICAL, np.float64)
        for item in self._items:
            t, m = item[-2], item[-1]   # (N_CANONICAL,G,G) float32, (N_CANONICAL,) bool
            flat = t.reshape(N_CANONICAL, -1)   # (N_CANONICAL, G*G)
            total_sum[m] += flat[m].sum(1)
            total_n[m]   += flat.shape[1]
        means = np.where(total_n > 0, total_sum / np.maximum(total_n, 1), 0.0)
        return means.astype(np.float32)

    def sampling_weights(self) -> torch.Tensor:
        """
        Inverse-frequency weights so every dataset contributes equally per epoch.
        Small datasets (NSCLC, immunoatlas) get upsampled; large ones (ORION) downsampled.
        """
        from collections import Counter
        counts  = Counter(self._item_ds)
        n_ds    = len(counts)
        weights = np.array(
            [1.0 / (n_ds * counts[ds]) for ds in self._item_ds], dtype=np.float32
        )
        return torch.from_numpy(weights)


# ── Split helpers ──────────────────────────────────────────────────────────────

def make_splits(configs: list[dict], val_frac: float, seed: int
                ) -> tuple[dict, dict, dict]:
    """
    Returns (train_ids, val_ids, test_ids) dicts keyed by dataset name.
    Each value is a list of slide/spot/core/patch IDs (or None = all).
    HEMIT is excluded — it has its own pre-split files.
    """
    rng = np.random.default_rng(seed)
    train_ids, val_ids, test_ids = {}, {}, {}

    for cfg in configs:
        name = cfg["name"]

        if name == "hemit":
            # Pre-split: no IDs needed; the split is encoded in which H5 to open
            train_ids[name] = val_ids[name] = test_ids[name] = None
            continue

        if name == "orion":
            h5_dir  = Path(cfg["h5_dir"])
            all_ids = sorted(p.stem.replace("_patch_dataset", "")
                             for p in h5_dir.glob("*_patch_dataset.h5"))
            test_set = set(ORION_TEST_SLIDES)
            val_set  = set(ORION_VAL_SLIDES)
            test_ids[name]  = [s for s in all_ids if s in test_set]
            val_ids[name]   = [s for s in all_ids if s in val_set]
            train_ids[name] = [s for s in all_ids if s not in test_set and s not in val_set]
            print(f"  [orion] train={len(train_ids[name])}  val={len(val_ids[name])}  "
                  f"test={len(test_ids[name])}")
            continue

        if name == "sg":
            all_ids = sorted(p.stem.replace("_patch_dataset", "")
                             for p in Path(cfg["sg_dir"]).glob("*_patch_dataset.h5"))
            if SG_CANCER_ONLY:
                all_ids = [s for s in all_ids if "cancer" in s]
            # all cancer slides go to training only
            train_ids[name] = all_ids
            val_ids[name]   = []
            test_ids[name]  = []
            print(f"  [sg] train={len(train_ids[name])}  val=0  test=0  (cancer-only, all train)")
            continue

        if name == "jedi":
            h5_path = Path(cfg["h5_path"])
            if not h5_path.exists():
                train_ids[name] = val_ids[name] = test_ids[name] = []
                continue
            with h5py.File(h5_path) as f:
                N = f["coords"].shape[0]
            # single slide — all patches for training
            train_ids[name] = list(range(N))
            val_ids[name]   = []
            test_ids[name]  = []
            print(f"  [jedi] train={N}  val=0  test=0  (single slide, all train)")
            continue

        # Generic: split by unique sample IDs from the H5
        h5_path = Path(cfg.get("h5_path", ""))
        if not h5_path.exists():
            train_ids[name] = val_ids[name] = test_ids[name] = []
            continue
        with h5py.File(h5_path) as f:
            id_key  = cfg.get("id_key", "spot_ids")
            all_ids = sorted(set(
                s.decode() if isinstance(s, bytes) else s for s in f[id_key][:]
            ))
        arr = np.array(all_ids); rng.shuffle(arr)
        if name == "nsclc_charite":
            n_test, n_val = 3, 4
        else:
            n_val  = max(1, round(len(arr) * val_frac))
            n_test = max(1, round(len(arr) * val_frac))
        test_ids[name]  = list(arr[:n_test])
        val_ids[name]   = list(arr[n_test:n_test+n_val])
        train_ids[name] = list(arr[n_test+n_val:])
        print(f"  [{name}] train={len(train_ids[name])}  val={len(val_ids[name])}  "
              f"test={len(test_ids[name])}")

    return train_ids, val_ids, test_ids


# ── Metrics ────────────────────────────────────────────────────────────────────

class OnlinePearson:
    def __init__(self, C):
        self.n = self.sx = self.sy = self.sxx = self.syy = self.sxy = None
        self.C = C

    def update(self, preds, targets, channel_mask):
        """preds/targets: (B,C,G,G)  channel_mask: (B,C) bool"""
        B, C, G, _ = preds.shape
        p = preds.transpose(0,2,3,1).reshape(-1, C).astype(np.float64)
        t = targets.transpose(0,2,3,1).reshape(-1, C).astype(np.float64)
        m = np.repeat(channel_mask, G*G, axis=0).astype(np.float64)
        if self.n is None:
            self.n   = np.zeros(C, np.float64)
            self.sx  = np.zeros(C, np.float64); self.sy  = np.zeros(C, np.float64)
            self.sxx = np.zeros(C, np.float64); self.syy = np.zeros(C, np.float64)
            self.sxy = np.zeros(C, np.float64)
        self.n   += m.sum(0)
        self.sx  += (p*m).sum(0); self.sy  += (t*m).sum(0)
        self.sxx += (p*p*m).sum(0); self.syy += (t*t*m).sum(0)
        self.sxy += (p*t*m).sum(0)

    def compute(self):
        if self.n is None:
            return np.zeros(self.C)
        num = self.n * self.sxy - self.sx * self.sy
        den = np.sqrt(np.maximum(
            (self.n*self.sxx - self.sx**2) * (self.n*self.syy - self.sy**2), 0.0))
        return np.where(den > 0, num/den, 0.0)


class OnlineSpearman:
    def __init__(self, C, buf=5_000_000):
        self.C = C; self.buf = buf
        self._p = np.empty((buf, C), np.float32)
        self._t = np.empty((buf, C), np.float32)
        self._m = np.zeros((buf, C), bool)
        self._n = 0

    def update(self, preds, targets, channel_mask):
        if self._n >= self.buf: return
        B, C, G, _ = preds.shape
        p = preds.transpose(0,2,3,1).reshape(-1,C).astype(np.float32)
        t = targets.transpose(0,2,3,1).reshape(-1,C).astype(np.float32)
        m = np.repeat(channel_mask, G*G, axis=0)
        fill = min(len(p), self.buf - self._n)
        self._p[self._n:self._n+fill] = p[:fill]
        self._t[self._n:self._n+fill] = t[:fill]
        self._m[self._n:self._n+fill] = m[:fill]
        self._n += fill

    def compute(self):
        n = min(self._n, self.buf)
        out = np.zeros(self.C)
        for c in range(self.C):
            valid = self._m[:n, c]
            if valid.sum() > 1:
                r, _ = spearmanr(self._p[:n,c][valid], self._t[:n,c][valid])
                out[c] = float(r) if np.isfinite(r) else 0.0
        return out


# ── Training loop ──────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer=None, scaler=None, scheduler=None,
              marker_stds=None, token_means=None):
    """
    token_means : (N_CANONICAL,) tensor on device — per-marker mean from training data.
                  Tokens with target > token_means are considered informative and
                  included in the loss. If None, all measured tokens contribute.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    total_mse = 0.0
    pm_acc    = None
    pearson   = OnlinePearson(N_CANONICAL)
    spearman  = OnlineSpearman(N_CANONICAL)

    with torch.set_grad_enabled(training):
        for i, (patches, targets_cpu, mask_cpu) in enumerate(loader):
            patches = patches.to(device)
            targets = targets_cpu.to(device)
            chan_mask = mask_cpu.to(device)   # (B, N_CANONICAL) bool

            with torch.amp.autocast("cuda"):
                preds, _ = model(patches)     # (B, N_CANONICAL, G, G)
                chan_exp = chan_mask.unsqueeze(-1).unsqueeze(-1).expand_as(preds)
                if token_means is not None:
                    # a spatial position is informative if at least one *measured*
                    # marker exceeds its training mean there (any-active semantics)
                    above_mean   = targets > token_means[None, :, None, None]  # (B,C,G,G)
                    any_active   = above_mean.any(dim=1, keepdim=True)  # (B,1,G,G)
                    mask_exp     = chan_exp & any_active.expand_as(preds)
                else:
                    mask_exp  = chan_exp
                sq_err   = (preds - targets) ** 2
                pm_mse   = (sq_err * mask_exp).sum((0,2,3)) / mask_exp.sum((0,2,3)).clamp(min=1)
                loss     = (pm_mse / marker_stds).mean()

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler: scheduler.step()

            total_mse += loss.item()
            pm_np = pm_mse.detach().float().cpu().numpy()
            if pm_acc is None:
                pm_acc = np.zeros(N_CANONICAL, np.float64)
            pm_acc += pm_np

            p_np = preds.detach().float().cpu().numpy()
            t_np = targets_cpu.numpy()
            m_np = mask_cpu.numpy()
            pearson.update(p_np, t_np, m_np)
            spearman.update(p_np, t_np, m_np)

            if i % 200 == 0:
                print(f"  [{i+1}/{len(loader)}] mse={loss.item():.5f}", flush=True)

    return total_mse/len(loader), pm_acc/len(loader), pearson.compute(), spearman.compute()


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_curves(train_l, val_l, train_p, val_p, train_s, val_s,
                train_pm, val_pm):
    epochs = range(1, len(train_l)+1)
    C = N_CANONICAL; nc = 6; nr = math.ceil(C/nc)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(epochs, train_l, label="Train"); ax[0].plot(epochs, val_l, label="Val")
    ax[0].set_title("MSE Loss"); ax[0].legend()
    ax[1].plot(epochs, [p.mean() for p in train_p], label="Train")
    ax[1].plot(epochs, [p.mean() for p in val_p],   label="Val")
    ax[1].set_title("Mean Pearson r"); ax[1].legend()
    plt.tight_layout(); plt.savefig(OUTPUT_DIR/"training_curves.png", dpi=150); plt.close()

    for metric, tm, vm, fname in [
        ("Pearson r",  train_p,  val_p,  "per_marker_pearson.png"),
        ("Spearman ρ", train_s,  val_s,  "per_marker_spearman.png"),
        ("MSE",        train_pm, val_pm, "per_marker_loss.png"),
    ]:
        mv = np.stack(vm); mt = np.stack(tm)
        fig, axes = plt.subplots(nr, nc, figsize=(nc*3, nr*2.5), squeeze=False)
        for j, name in enumerate(CANONICAL_MARKERS):
            ax = axes[j//nc][j%nc]
            ax.plot(epochs, mt[:,j], label="Train")
            ax.plot(epochs, mv[:,j], label="Val")
            ax.set_title(name, fontsize=8)
            if metric != "MSE":
                ax.set_ylim(-1,1); ax.axhline(0, color="gray", lw=0.5, ls="--")
        for j in range(C, nr*nc): axes[j//nc][j%nc].set_visible(False)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", fontsize=8)
        plt.suptitle(f"Per-marker {metric}", fontsize=10)
        plt.tight_layout(); plt.savefig(OUTPUT_DIR/fname, dpi=150); plt.close()


# ── Logger ─────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "a")
    def write(self, msg):
        self.terminal.write(msg); self.log.write(msg); self.log.flush()
    def flush(self): pass


# ── Main ───────────────────────────────────────────────────────────────────────

def train():
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    sys.stdout = Logger(OUTPUT_DIR / "training_log.txt")

    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    print(f"Canonical panel ({N_CANONICAL}): {CANONICAL_MARKERS}\n")

    print("Building splits…")
    train_ids, val_ids, test_ids = make_splits(DATASET_CONFIGS, VAL_FRAC, SEED)

    print("\nBuilding train dataset…")
    train_ds = MultiSourcePatchDataset(DATASET_CONFIGS, split="train",
                                       split_ids=train_ids, augment=True)
    print("\nBuilding val dataset…")
    val_ds   = MultiSourcePatchDataset(DATASET_CONFIGS, split="val",
                                       split_ids=val_ids, augment=False)

    print("\nComputing marker stds and token means…")
    marker_stds  = train_ds.compute_marker_stds().to(device)
    token_means_np = train_ds.compute_token_means()
    token_means  = torch.from_numpy(token_means_np).to(device)
    for name, s, mu in zip(CANONICAL_MARKERS, marker_stds.cpu(), token_means_np):
        print(f"  {name:<20s}  std={s:.4f}  mean={mu:.4f}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model  = SpatialModel(MODEL_NAME, num_outputs=N_CANONICAL,
                          token_grid=TOKEN_GRID, unfreeze_last_n=0).to(device)
    scaler = torch.amp.GradScaler("cuda")

    train_l, val_l   = [], []
    train_p, val_p   = [], []
    train_s, val_s   = [], []
    train_pm, val_pm = [], []
    best_val_r = -np.inf

    def _epoch(epoch, opt, sched=None):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        tr = run_epoch(model, train_loader, opt, scaler, sched, marker_stds, token_means)
        va = run_epoch(model, val_loader, marker_stds=marker_stds, token_means=token_means)
        train_l.append(tr[0]);  val_l.append(va[0])
        train_p.append(tr[2]);  val_p.append(va[2])
        train_s.append(tr[3]);  val_s.append(va[3])
        train_pm.append(tr[1]); val_pm.append(va[1])
        print(f"  train mse={tr[0]:.4f}  r={tr[2].mean():.4f}  ρ={tr[3].mean():.4f}")
        print(f"  val   mse={va[0]:.4f}  r={va[2].mean():.4f}  ρ={va[3].mean():.4f}")
        for name, p, pm in zip(CANONICAL_MARKERS, va[2], va[1]):
            print(f"    {name:<20s}  pearson={p:.4f}  val_mse={pm:.6f}")
        nonlocal best_val_r
        if va[2].mean() > best_val_r:
            best_val_r = va[2].mean()
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            print(f"  → best saved (r={best_val_r:.4f})")
        for arr, fname in [
            (train_l,"train_losses"), (val_l,"val_losses"),
            (train_p,"train_pearsons"), (val_p,"val_pearsons"),
            (train_s,"train_spearmans"), (val_s,"val_spearmans"),
            (train_pm,"train_pm_losses"), (val_pm,"val_pm_losses"),
        ]:
            np.save(OUTPUT_DIR/f"{fname}.npy",
                    np.array(arr) if not isinstance(arr[0], np.ndarray) else np.stack(arr))
        np.save(OUTPUT_DIR/"marker_names.npy", np.array(CANONICAL_MARKERS))
        plot_curves(train_l, val_l, train_p, val_p, train_s, val_s, train_pm, val_pm)

    # Phase 1: head only
    opt1 = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    print(f"\n── Phase 1: head only ({PHASE1_EPOCHS} epochs) ──")
    for ep in range(1, PHASE1_EPOCHS + 1):
        _epoch(ep, opt1)

    # Phase 2: unfreeze last N encoder blocks
    print(f"\n── Phase 2: unfreeze last {UNFREEZE_LAST_N} blocks ──")
    for blk in model.encoder.blocks[-UNFREEZE_LAST_N:]:
        for p in blk.parameters(): p.requires_grad = True
    for p in model.encoder.norm.parameters(): p.requires_grad = True

    head_p    = [p for n,p in model.named_parameters() if p.requires_grad and "encoder" not in n]
    enc_p     = [p for n,p in model.named_parameters() if p.requires_grad and "encoder" in n]
    opt2      = torch.optim.Adam([{"params": head_p, "lr": LR},
                                   {"params": enc_p,  "lr": LR*0.2}])
    p2_steps  = (NUM_EPOCHS - PHASE1_EPOCHS) * len(train_loader)
    sched2    = torch.optim.lr_scheduler.LambdaLR(opt2, [
        lambda s: 0.5*(1+math.cos(math.pi*min(s/p2_steps, 1.0))),
        lambda s: min(s/WARMUP_STEPS,1.0) * 0.5*(1+math.cos(
            math.pi*min(max(s-WARMUP_STEPS,0)/max(p2_steps-WARMUP_STEPS,1),1.0))),
    ])

    for ep in range(PHASE1_EPOCHS + 1, NUM_EPOCHS + 1):
        _epoch(ep, opt2, sched2)

    print(f"\nDone. Best val Pearson: {best_val_r:.4f}")


if __name__ == "__main__":
    train()