import os
import h5py
import torch
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import Dataset
import tifffile
import zarr

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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

    def __init__(self, h5_dir: str, tiff_dir: str, num_slides: int = 5):
        self.patch_map    = []
        self.targets      = []
        self.slides       = []
        self.marker_names = None
        self.patch_size   = None
        self.token_grid   = None

        h5_dir   = Path(h5_dir)
        tiff_dir = Path(tiff_dir)
        h5_names = sorted(f for f in os.listdir(h5_dir) if f.endswith('.h5'))

        if num_slides:
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
        if self.targets:
            C = self.targets[0].shape[0]
            total_sum = np.zeros(C, dtype=np.float64)
            total_n   = 0
            for t in self.targets:
                total_sum += t.reshape(C, -1).sum(axis=1)
                total_n   += t.shape[1] * t.shape[2]
            self.token_means = torch.from_numpy((total_sum / total_n).astype(np.float32))  # (C,)
        else:
            self.token_means = None

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
                           interpolation=cv2.INTER_LINEAR)
        patch = (patch.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        patch  = torch.from_numpy(patch.transpose(2, 0, 1))     # (3, 224, 224)
        target = torch.from_numpy(self.targets[idx])            # (C, G, G)
        mask   = (target > self.token_means[:, None, None]).any(dim=0)  # (G, G) bool
        return patch, target, mask
