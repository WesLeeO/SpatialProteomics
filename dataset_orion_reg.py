import os
import h5py
import torch
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import tifffile
import zarr

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _hed_jitter(patch_uint8: np.ndarray) -> np.ndarray:
    """
    Perturb H&E in HED colour space to simulate staining / scanner variation.
    Only H (nuclei) and E (cytoplasm) channels are perturbed — D (DAB) is left
    untouched since standard H&E slides carry no brown DAB signal.
    patch_uint8 : (H, W, 3) uint8 RGB  →  (H, W, 3) uint8 RGB
    """
    from skimage.color import rgb2hed, hed2rgb
    hed = rgb2hed(patch_uint8.astype(np.float32) / 255.0)
    # H channel (nuclei): moderate perturbation
    hed[:, :, 0] *= 1.0 + np.random.uniform(-0.05, 0.05)
    hed[:, :, 0] +=       np.random.uniform(-0.02, 0.02)
    # E channel (eosin/cytoplasm): larger perturbation — dominant gap between ORION and SG
    hed[:, :, 1] *= 1.0 + np.random.uniform(-0.05, 0.05)
    hed[:, :, 1] +=       np.random.uniform(-0.05, 0.05)
    return (np.clip(hed2rgb(hed), 0.0, 1.0) * 255).astype(np.uint8)


_color_jitter = transforms.ColorJitter(
    brightness=0.25,
    contrast=0.25,
    saturation=0.4,
    hue=0.04,
)


def _find_tiff(slide_dir: Path, pattern: str) -> Path:
    """Glob for a single TIFF in slide_dir; raise if not exactly one match."""
    matches = list(slide_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected 1 file matching {slide_dir}/{pattern}, found {matches}")
    return matches[0]


class OrionSpatialDataset(Dataset):
    """
    Loads pre-computed (N, C, G, G) token-grid targets from HDF5 files produced
    by build_patch_dataset_orion_crc_reg.py.

    __getitem__ returns:
      patch  : (3, 224, 224) float32  — normalised H&E
      target : (C, G, G)    float32  — mean IF expression per token cell
    """

    def __init__(self, h5_dir: str, tiff_dir: str, num_slides: int = 0,
                 augment: bool = False,
                 p_geom: float = 0.7,
                 p_color: float = 0.4,
                 p_brightness: float = 0.4,
                 slide_names: list = None,
                 token_means: torch.Tensor = None):
        self.patch_map    = []
        self.targets      = []
        self.slides       = []
        self.marker_names = None
        self.patch_size   = None
        self.token_grid   = None
        self.augment      = augment
        self.p_geom       = p_geom        # probability of picking one geometric transform
        self.p_color      = p_color       # probability of HED jitter
        self.p_brightness = p_brightness  # probability of brightness/contrast jitter

        h5_dir   = Path(h5_dir)
        tiff_dir = Path(tiff_dir)
        h5_names = sorted(f for f in os.listdir(h5_dir) if f.endswith('.h5'))

        if slide_names is not None:
            # Slide-level split: keep only the requested slides (matched by base name)
            slide_names_set = set(slide_names)
            h5_names = [h for h in h5_names
                        if h.replace('_patch_dataset.h5', '') in slide_names_set]
        elif num_slides:
            h5_names = h5_names[:num_slides]

        for h5_name in h5_names:
            h5_path   = h5_dir / h5_name
            base_name = h5_name.replace('_patch_dataset.h5', '')
            tiff_path   = _find_tiff(tiff_dir / base_name, "*-registered.ome.tif")

            try:
                with h5py.File(h5_path, 'r') as f:
                    if 'targets' not in f:
                        print(f"Warning: {h5_name} has no /targets, skipping.")
                        continue
                    coords            = f['coords'][:]
                    targets           = f['targets'][:]           # (N, C, G, G)
                    patch_size_level0 = int(f.attrs['patch_size_level0'])
                    if self.marker_names is None:
                        self.marker_names = list(f.attrs['marker_names'])
                    if self.patch_size is None:
                        self.patch_size = int(f.attrs['patch_size'])
                    if self.token_grid is None:
                        self.token_grid = int(f.attrs.get('token_grid', 16))
            except:
                print(f'{h5_path} not existing')
                continue
        
            print(f"Loading {base_name} ({len(coords)} patches)…")
            slide_idx = len(self.slides)
            self.slides.append(str(tiff_path))  # store path; open lazily per worker

            for i in range(len(coords)):
                self.patch_map.append((slide_idx, int(coords[i, 0]), int(coords[i, 1]),
                                       patch_size_level0))
                self.targets.append(targets[i].astype(np.float32))   # (C, G, G)

        print(f"Dataset: {len(self.patch_map)} patches across {len(self.slides)} slide(s).")

        # Global mean per marker across every token in the dataset — used to build
        # active-token masks so training loss is restricted to informative tokens.
        if token_means is not None:
            self.token_means = token_means
        elif self.targets:
            C = self.targets[0].shape[0]
            total_sum = np.zeros(C, dtype=np.float64)
            total_n   = 0
            for t in self.targets:
                total_sum += t.reshape(C, -1).sum(axis=1)
                total_n   += t.shape[1] * t.shape[2]
            self.token_means = torch.from_numpy((total_sum / total_n).astype(np.float32))  # (C,)
        else:
            self.token_means = None

    def compute_marker_stds(self) -> torch.Tensor:
        """Per-marker std across every token in the dataset. Shape: (C,)"""
        C           = self.targets[0].shape[0]
        total_sum   = np.zeros(C, dtype=np.float64)
        total_sumsq = np.zeros(C, dtype=np.float64)
        total_n     = 0
        for t in self.targets:
            flat         = t.reshape(C, -1)       # (C, G*G)
            total_sum   += flat.sum(axis=1)
            total_sumsq += (flat ** 2).sum(axis=1)
            total_n     += flat.shape[1]
        mean = total_sum / total_n
        var  = total_sumsq / total_n - mean ** 2
        return torch.from_numpy(np.sqrt(np.maximum(var, 0.0)).astype(np.float32))

    def compute_sampling_weights(
        self,
        hard_marker_names: list[str] | None = None,
        sparse_threshold: float = 0.01,
        top_pct: float = 99,
        cap: float = 10.0,
    ) -> torch.Tensor:
        """
        Per-patch sampling weight for WeightedRandomSampler.

        Binary scheme: a patch is "positive" for marker j if its max token
        expression exceeds the top_pct percentile of patch_max for that marker.
        Weight = clip(1 + n_positive_markers, 1, cap).

        Positive for 0 markers → weight 1 (baseline).
        Positive for 1 marker  → weight 2.
        Positive for all 5     → weight 6 (or cap if lower).

        hard_marker_names : explicit marker names, or None to auto-select markers
                            with token mean < sparse_threshold.
        top_pct           : percentile threshold per marker (default 99 → top 1%).
        cap               : maximum weight.
        """
        N = len(self.targets)
        if N == 0:
            return torch.ones(0)

        C = self.targets[0].shape[0]

        if hard_marker_names is not None:
            name_to_idx = {n: j for j, n in enumerate(self.marker_names)}
            hard_idx = [name_to_idx[n] for n in hard_marker_names if n in name_to_idx]
        elif self.token_means is not None:
            means = self.token_means.numpy()
            hard_idx = [j for j in range(C) if means[j] < sparse_threshold]
        else:
            hard_idx = list(range(C))

        if not hard_idx:
            print("compute_sampling_weights: no hard markers found, uniform weights.")
            return torch.ones(N, dtype=torch.float32)

        hard_names = [self.marker_names[j] for j in hard_idx] if self.marker_names else hard_idx
        print(f"compute_sampling_weights: hard markers ({len(hard_idx)}) = {hard_names}")

        # patch-level max per marker, computed in chunks to avoid memory spike
        chunk = 8192
        patch_max = np.empty((N, C), dtype=np.float32)
        for start in range(0, N, chunk):
            end   = min(start + chunk, N)
            batch = np.stack(self.targets[start:end])              # (chunk, C, G, G)
            patch_max[start:end] = batch.reshape(end - start, C, -1).max(axis=2)

        # count how many hard markers each patch is "positive" for
        n_pos = np.zeros(N, dtype=np.float32)
        for j in hard_idx:
            thresh = float(np.percentile(patch_max[:, j], top_pct))
            n_pos += (patch_max[:, j] > thresh).astype(np.float32)

        weights = np.clip(1.0 + n_pos, 1.0, cap).astype(np.float32)
        n_up = (weights > 1.0).sum()
        print(f"  weight stats  min={weights.min():.2f}  p50={np.median(weights):.2f}  "
              f"p95={np.percentile(weights, 95):.2f}  max={weights.max():.2f}")
        print(f"  upweighted patches: {n_up:,}/{N:,} ({100.0*n_up/N:.1f}%)")
        print(f"  mean weight of upweighted: {weights[weights > 1.0].mean():.3f}")
        return torch.from_numpy(weights)

    def __len__(self) -> int:
        return len(self.patch_map)

    def _open_arr(self, slide_idx: int):
        """Open zarr array lazily after fork so each worker gets its own file handle."""
        if not hasattr(self, '_arr_cache'):
            self._arr_cache = {}
        if slide_idx not in self._arr_cache:
            tif   = tifffile.TiffFile(self.slides[slide_idx])
            store = tif.aszarr()
            z     = zarr.open(store, mode="r")
            self._arr_cache[slide_idx] = z["0"] if isinstance(z, zarr.hierarchy.Group) else z
        return self._arr_cache[slide_idx]

    def __getitem__(self, idx: int):
        """
        Returns:
          patch  : (3, 224, 224) float32  — normalised H&E
          target : (C, G, G)    float32  — mean IF expression per token cell
          mask   : (G, G)       bool     — True where ≥1 marker exceeds its global mean
        """
        slide_idx, x, y, psz = self.patch_map[idx]
        arr   = self._open_arr(slide_idx)
        patch = np.array(arr[y:y + psz, x:x + psz, :])  # (H, W, 3)  — H&E is YXS
        patch = cv2.resize(patch, (self.patch_size, self.patch_size),
                           interpolation=cv2.INTER_LINEAR)  # still uint8

        if self.augment:
            target = self.targets[idx].copy()    # (C, G, G) — must copy before any in-place ops

            # 1. Geometric — one randomly chosen transform applied with probability p_geom.
            #    patch: (H, W, 3)  axes (0,1) are spatial
            #    target: (C, G, G) axes (1,2) are spatial — axis 0 is biomarker channels
            if np.random.random() < self.p_geom:
                choice = np.random.randint(3)    # 0 = h-flip, 1 = v-flip, 2 = rotation
                if choice == 0:
                    patch  = patch[:, ::-1, :].copy()      # flip W axis
                    target = target[:, :, ::-1].copy()     # flip token-col axis
                elif choice == 1:
                    patch  = patch[::-1, :, :].copy()      # flip H axis
                    target = target[:, ::-1, :].copy()     # flip token-row axis
                else:
                    k = np.random.randint(1, 4)            # 90 / 180 / 270°
                    patch  = np.rot90(patch,  k, axes=(0, 1)).copy()
                    target = np.rot90(target, k, axes=(1, 2)).copy()

            # 2. HED colour jitter — H&E only, targets unchanged
            if np.random.random() < self.p_color:
                patch = _hed_jitter(patch)

            # 3. Brightness / contrast / saturation — H&E only, targets unchanged
            if np.random.random() < self.p_brightness:
                patch = np.array(_color_jitter(Image.fromarray(patch)))
        else:
            target = self.targets[idx]

        patch  = (patch.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        patch  = torch.from_numpy(patch.transpose(2, 0, 1))         # (3, 224, 224)
        target = torch.from_numpy(target)                            # (C, G, G)
        mask   = (target > self.token_means[:, None, None]).any(dim=0)  # (G, G) bool
        return patch, target, mask
