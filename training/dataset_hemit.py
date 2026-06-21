import h5py
import cv2
import tifffile
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class HEMITDataset(Dataset):
    """Flat-target dataset (v1 builder). targets shape: (N, C)."""

    def __init__(self, h5_path):
        h5_path = Path(h5_path)

        with h5py.File(h5_path, "r") as f:
            self.coords       = f["coords"][:]
            self.targets      = f["targets"][:]
            sources_raw       = [s.decode() for s in f["sources"][:]]
            self.crop_size    = int(f.attrs["crop_size"])
            self.patch_size   = int(f.attrs["patch_size"])
            self.marker_names = list(f.attrs["marker_names"])

        # Preload all unique source TIFs into RAM
        unique_paths = sorted(set(sources_raw))
        print(f"Loading {len(unique_paths)} source TIFs into RAM ({h5_path.stem})...")
        self.source_imgs = {}
        for p in unique_paths:
            img = tifffile.imread(p)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            self.source_imgs[p] = img[:, :, :3]   # ensure RGB, keep uint8

        self.sources = sources_raw
        print(f"  Ready: {len(self.coords)} patches, markers: {self.marker_names}")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = int(self.coords[idx, 0]), int(self.coords[idx, 1])
        img  = self.source_imgs[self.sources[idx]]

        crop = img[y:y + self.crop_size, x:x + self.crop_size, :].astype(np.float32)
        crop = cv2.resize(crop, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
        crop = (crop / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        patch  = torch.from_numpy(crop.transpose(2, 0, 1))   # (3, H, W)
        target = torch.from_numpy(self.targets[idx])
        return patch, target


class HEMITTokenDataset(Dataset):
    """Token-grid dataset (v2 builder). targets shape: (N, C, G, G)."""

    def __init__(self, h5_path):
        h5_path = Path(h5_path)

        with h5py.File(h5_path, "r") as f:
            self.coords       = f["coords"][:]
            self.targets      = f["targets"][:]          # (N, C, G, G)
            sources_raw       = [s.decode() for s in f["sources"][:]]
            self.crop_size    = int(f.attrs["crop_size"])
            self.patch_size   = int(f.attrs["patch_size"])
            self.token_grid   = int(f.attrs.get("token_grid", 16))
            self.marker_names = list(f.attrs["marker_names"])

        unique_paths = sorted(set(sources_raw))
        print(f"Loading {len(unique_paths)} source TIFs into RAM ({h5_path.stem})...")
        self.source_imgs = {}
        for p in unique_paths:
            img = tifffile.imread(p)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            self.source_imgs[p] = img[:, :, :3]

        self.sources = sources_raw
        print(f"  Ready: {len(self.coords)} patches  targets={self.targets.shape}  "
              f"markers={self.marker_names}")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        x, y = int(self.coords[idx, 0]), int(self.coords[idx, 1])
        img  = self.source_imgs[self.sources[idx]]

        crop = img[y:y + self.crop_size, x:x + self.crop_size, :].astype(np.float32)
        crop = cv2.resize(crop, (self.patch_size, self.patch_size), interpolation=cv2.INTER_LINEAR)
        crop = (crop / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        patch  = torch.from_numpy(crop.transpose(2, 0, 1))   # (3, patch_size, patch_size)
        target = torch.from_numpy(self.targets[idx])          # (C, G, G)
        return patch, target