"""
Zero-shot dataset: every (patch, marker) pair is an independent training sample.

For N patches across all slides and M described markers, the dataset has N×M
items.  Markers are interleaved in the index layout so that a shuffled
DataLoader produces batches that naturally mix multiple markers.

__getitem__ returns:
    patch          : (3, 224, 224) float32 — ImageNet-normalised H&E
    conch_text_emb : (512,)        float32 — pre-encoded, L2-normed CONCH text emb
    target         : (1, G, G)     float32 — single-marker expression map
    marker_idx     : int           — index into self.marker_names
    mask           : (G, G)        bool    — True where token exceeds per-marker mean
"""

import os
import h5py
import numpy as np
import cv2
import torch
import tifffile
import zarr
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ── Augmentation (identical to OrionSpatialDataset) ───────────────────────────

def _hed_jitter(patch_uint8: np.ndarray) -> np.ndarray:
    from skimage.color import rgb2hed, hed2rgb
    hed = rgb2hed(patch_uint8.astype(np.float32) / 255.0)
    hed[:, :, 0] *= 1.0 + np.random.uniform(-0.05, 0.05)
    hed[:, :, 0] +=       np.random.uniform(-0.02, 0.02)
    hed[:, :, 1] *= 1.0 + np.random.uniform(-0.05, 0.05)
    hed[:, :, 1] +=       np.random.uniform(-0.05, 0.05)
    return (np.clip(hed2rgb(hed), 0.0, 1.0) * 255).astype(np.uint8)


_color_jitter = transforms.ColorJitter(
    brightness=0.25, contrast=0.25, saturation=0.4, hue=0.04,
)


def _find_tiff(slide_dir: Path, pattern: str) -> Path:
    matches = list(slide_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected 1 file matching {slide_dir}/{pattern}, found {matches}")
    return matches[0]


# ── Dataset ───────────────────────────────────────────────────────────────────

class ZeroShotOrionDataset(Dataset):
    """
    Parameters
    ----------
    h5_dir : str | Path
        Directory of *_patch_dataset.h5 files.
    tiff_dir : str | Path
        Root directory containing per-slide H&E TIFFs.
    marker_text_embs : dict[str, np.ndarray]
        Pre-computed CONCH text embeddings: {marker_name: (512,) float32}.
        Only markers present in BOTH the H5 files and this dict are used.
        Pre-compute with ZeroShotSpatialModel.encode_text_conch() in the
        training script so CONCH is never loaded inside worker processes.
    slide_names : list[str] | None
        If given, restrict to these slides (slide-level split support).
    num_slides : int
        If slide_names is None, use first num_slides H5 files (0 = all).
    augment : bool
        Enable geometric + colour augmentation.
    token_means : torch.Tensor | None
        (M,) per-marker mean expression across all tokens — used to build the
        active-token mask.  If None, computed from the loaded targets.
    """

    def __init__(
        self,
        h5_dir: str,
        tiff_dir: str,
        marker_text_embs: dict,           # {marker_name: np.ndarray (512,)}
        slide_names: list = None,
        num_slides: int = 0,
        augment: bool = False,
        p_geom: float = 0.5,
        p_color: float = 0.3,
        p_brightness: float = 0.2,
        token_means: torch.Tensor = None,
    ):
        self.augment      = augment
        self.p_geom       = p_geom
        self.p_color      = p_color
        self.p_brightness = p_brightness

        h5_dir   = Path(h5_dir)
        tiff_dir = Path(tiff_dir)
        h5_names = sorted(f for f in os.listdir(h5_dir) if f.endswith('.h5'))

        if slide_names is not None:
            slide_set = set(slide_names)
            h5_names  = [h for h in h5_names
                         if h.replace('_patch_dataset.h5', '') in slide_set]
        elif num_slides:
            h5_names = h5_names[:num_slides]

        # Will be determined from first H5
        self.patch_size   = None
        self.token_grid   = None
        self.marker_names = None   # only the described markers, in consistent order

        # Core storage
        # patch_map[i] = (slide_idx, x, y, psz)
        # targets[i]   = (M, G, G) for all described markers for patch i
        self._patch_map   = []    # (slide_idx, x, y, psz)
        self._targets     = []    # list of (M, G, G) float32 arrays
        self._slides      = []    # TIFF paths, opened lazily per worker

        described_markers_set = set(marker_text_embs.keys())

        for h5_name in h5_names:
            h5_path   = h5_dir / h5_name
            base_name = h5_name.replace('_patch_dataset.h5', '')
            try:
                tiff_path = _find_tiff(tiff_dir / base_name, "*-registered.ome.tif")
            except FileNotFoundError as e:
                print(f"  Skipping {base_name}: {e}")
                continue

            try:
                with h5py.File(h5_path, 'r') as f:
                    if 'targets' not in f:
                        print(f"  Warning: {h5_name} has no /targets, skipping.")
                        continue
                    coords            = f['coords'][:]
                    targets_all       = f['targets'][:]              # (N, C_full, G, G)
                    h5_marker_names   = list(f.attrs['marker_names'])
                    patch_size_level0 = int(f.attrs['patch_size_level0'])
                    if self.patch_size is None:
                        self.patch_size = int(f.attrs['patch_size'])
                    if self.token_grid is None:
                        self.token_grid = int(f.attrs.get('token_grid', 16))
            except Exception as e:
                print(f"  Could not load {h5_path}: {e}")
                continue

            # Determine described-marker ordering (consistent across all slides)
            if self.marker_names is None:
                self.marker_names = [m for m in h5_marker_names
                                     if m in described_markers_set]
                if not self.marker_names:
                    raise ValueError(
                        "No overlap between H5 marker_names and marker_text_embs keys."
                    )

            col_idx = [h5_marker_names.index(m) for m in self.marker_names]

            print(f"  Loading {base_name} ({len(coords)} patches, "
                  f"{len(self.marker_names)} described markers)…")
            slide_idx = len(self._slides)
            self._slides.append(str(tiff_path))

            for i in range(len(coords)):
                self._patch_map.append((slide_idx,
                                         int(coords[i, 0]), int(coords[i, 1]),
                                         patch_size_level0))
                # Only keep described markers
                self._targets.append(
                    targets_all[i][col_idx].astype(np.float32)   # (M, G, G)
                )

        if not self._patch_map:
            raise RuntimeError("No patches loaded — check h5_dir / tiff_dir paths.")

        M = len(self.marker_names)
        print(f"Dataset: {len(self._patch_map)} patches × {M} markers "
              f"= {len(self._patch_map) * M} samples across {len(self._slides)} slide(s).")

        # Pre-encoded CONCH text embeddings stored as tensor buffer
        # Shape: (M, 512) — one per described marker in self.marker_names order
        self.conch_text_embs = torch.from_numpy(
            np.stack([marker_text_embs[m] for m in self.marker_names])
        )   # (M, 512) float32

        # Per-marker token mean for active-token mask
        if token_means is not None:
            self.token_means = token_means           # (M,)
        else:
            total_sum = np.zeros(M, dtype=np.float64)
            total_n   = 0
            for t in self._targets:
                total_sum += t.reshape(M, -1).sum(axis=1)
                total_n   += t.shape[1] * t.shape[2]
            self.token_means = torch.from_numpy(
                (total_sum / total_n).astype(np.float32)
            )   # (M,)

    def __len__(self) -> int:
        # Every patch paired with every described marker
        return len(self._patch_map) * len(self.marker_names)

    def _open_arr(self, slide_idx: int):
        """Lazy per-worker zarr handle — opened after fork."""
        if not hasattr(self, '_arr_cache'):
            self._arr_cache = {}
        if slide_idx not in self._arr_cache:
            tif   = tifffile.TiffFile(self._slides[slide_idx])
            store = tif.aszarr()
            z     = zarr.open(store, mode="r")
            self._arr_cache[slide_idx] = (
                z["0"] if isinstance(z, zarr.hierarchy.Group) else z
            )
        return self._arr_cache[slide_idx]

    def __getitem__(self, idx: int):
        M          = len(self.marker_names)
        patch_idx  = idx // M
        marker_idx = idx %  M

        slide_idx, x, y, psz = self._patch_map[patch_idx]
        arr   = self._open_arr(slide_idx)
        patch = np.array(arr[y:y + psz, x:x + psz, :])          # (H, W, 3) uint8
        patch = cv2.resize(patch, (self.patch_size, self.patch_size),
                           interpolation=cv2.INTER_LINEAR)

        target = self._targets[patch_idx][marker_idx].copy()     # (G, G) float32

        if self.augment:
            if np.random.random() < self.p_geom:
                choice = np.random.randint(3)
                if choice == 0:
                    patch  = patch[:, ::-1, :].copy()
                    target = target[:, ::-1].copy()
                elif choice == 1:
                    patch  = patch[::-1, :, :].copy()
                    target = target[::-1, :].copy()
                else:
                    k      = np.random.randint(1, 4)
                    patch  = np.rot90(patch,  k, axes=(0, 1)).copy()
                    target = np.rot90(target, k, axes=(0, 1)).copy()
            if np.random.random() < self.p_color:
                patch = _hed_jitter(patch)
            if np.random.random() < self.p_brightness:
                patch = np.array(_color_jitter(Image.fromarray(patch)))

        patch  = (patch.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        patch  = torch.from_numpy(patch.transpose(2, 0, 1))          # (3, 224, 224)
        target = torch.from_numpy(target).unsqueeze(0)                # (1, G, G)
        mask   = (target[0] > self.token_means[marker_idx]).bool()    # (G, G)

        return (
            patch,
            self.conch_text_embs[marker_idx],   # (512,)
            target,                              # (1, G, G)
            marker_idx,                          # int
            mask,                                # (G, G)
        )
