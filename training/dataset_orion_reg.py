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

# Canonical neighbour order — must match NeighbourhoodSpatialModel._NEIGHBOUR_BLOCK.
# (dx, dy): dx shifts column (x), dy shifts row (y).
NUM_NEIGHBOURS    = 8
_NEIGHBOUR_DELTAS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _hed_jitter(patch_uint8: np.ndarray) -> np.ndarray:
    from skimage.color import rgb2hed, hed2rgb
    hed = rgb2hed(patch_uint8.astype(np.float32) / 255.0)
    # Multiplicative-only stain jitter. 
    hed[:, :, 0] *= 1.0 + np.random.uniform(-0.5, 0.5)   # hematoxylin scale
    hed[:, :, 1] *= 1.0 + np.random.uniform(-0.5, 0.5)   # eosin scale
    return (np.clip(hed2rgb(hed), 0.0, 1.0) * 255).astype(np.uint8)


def _photometric_jitter(patch_uint8: np.ndarray,
                        p_bc: float, p_blur: float, p_noise: float) -> np.ndarray:
    """Colour-only augmentations ported from MIPHEI-ViT (src/dataset.py color pipeline):
    RandomBrightnessContrast(±0.2), GaussianBlur(k=7, σ∈[0.1,1.5]) and
    GaussNoise(σ∈[0.02,0.05], reduced from MIPHEI's [0.05,0.1] — was too grainy for
    these H&E patches). Image-only — targets and spatial layout are untouched —
    so unlike geometric aug these are SAFE in neighbour mode. Each fires independently
    with its own probability (matching albumentations' Compose semantics)."""
    out = patch_uint8.astype(np.float32)
    if np.random.random() < p_bc:                      # brightness / contrast
        alpha = 1.0 + np.random.uniform(-0.2, 0.2)     # contrast (around the patch mean)
        beta  = np.random.uniform(-0.2, 0.2) * 255.0   # brightness shift
        mean  = out.mean()
        out   = alpha * (out - mean) + mean + beta
    out = np.clip(out, 0, 255).astype(np.uint8)
    if np.random.random() < p_blur:                    # Gaussian blur
        out = cv2.GaussianBlur(out, (7, 7), sigmaX=np.random.uniform(0.1, 1.5))
    if np.random.random() < p_noise:                   # Gaussian noise
        std   = np.random.uniform(0.02, 0.05) * 255.0
        noise = np.random.normal(0.0, std, out.shape).astype(np.float32)
        out   = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


# 3×3 grid of neighbour-slot ids laid out by (row=dy+1, col=dx+1); centre (0,0) = -1.
# Order matches _NEIGHBOUR_DELTAS. Geometric aug remaps neighbour slots through this
# grid so their directional positional embeddings (cls_pos) stay consistent with the
# flipped/rotated centre patch.
_SLOT_GRID = np.array([[0, 3, 5],
                       [1, -1, 6],
                       [2, 4, 7]], dtype=np.int64)


def _neighbour_perm(geom) -> np.ndarray:
    """Apply the SAME geometric op `geom` used on the centre patch to the slot grid and
    return a length-8 permutation: after the transform, neighbour slot s takes the value
    of slot perm[s] (so the layout flips/rotates together with the centre)."""
    moved = geom(_SLOT_GRID)
    perm  = np.empty(NUM_NEIGHBOURS, dtype=np.int64)
    for r in range(3):
        for c in range(3):
            s = _SLOT_GRID[r, c]
            if s >= 0:
                perm[s] = moved[r, c]
    return perm


def _find_tiff(slide_dir: Path, pattern: str) -> Path:
    matches = list(slide_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected 1 file matching {slide_dir}/{pattern}, found {matches}")
    return matches[0]


class OrionSpatialDataset(Dataset):
    """
    Loads pre-computed (N, C, G, G) token-grid targets from HDF5 files produced
    by build_patch_dataset_orion_crc_reg.py.
    """

    def __init__(self, h5_dir: str, tiff_dir: str, num_slides: int = 0,
                 augment: bool = False,
                 p_geom: float = 0.7,
                 p_hed: float = 0.6,          # stain jitter — #1 H&E domain-shift axis
                 p_bc: float = 0.5,           # brightness/contrast (MIPHEI default)
                 p_blur: float = 0.25,        # Gaussian blur — scanner focus varies
                 p_noise: float = 0.25,       # Gaussian noise — sensor noise varies
                 slide_names: list = None,
                 token_means: torch.Tensor = None,
                 use_neighbours: bool = False,
                 cls_cache_dir: str = None,
                 fg_mode: str = "token_mean",   # "token_mean" | "zero" — loss foreground def
                 return_slide_idx: bool = False):
        self.patch_map      = []
        self.targets        = []
        self.slides         = []   # resolved tiff paths (long scan ids)
        self.slide_ids      = []   # CRC base_name per slide, aligned to self.slides
        self.marker_names   = None
        self.patch_size     = None
        self.token_grid     = None
        self.augment        = augment
        self.p_geom         = p_geom
        self.p_hed          = p_hed
        self.p_bc           = p_bc
        self.p_blur         = p_blur
        self.p_noise        = p_noise
        self.use_neighbours = use_neighbours
        self.fg_mode        = fg_mode
        self.return_slide_idx = return_slide_idx
        self.cls_cache_dir  = Path(cls_cache_dir) if cls_cache_dir else None
        _cls_chunks         = []   # per-slide CLS arrays, aligned to patch_map order

        h5_dir   = Path(h5_dir)
        tiff_dir = Path(tiff_dir)
        h5_names = sorted(f for f in os.listdir(h5_dir) if f.endswith('.h5'))

        if slide_names is not None:
            slide_names_set = set(slide_names)
            h5_names = [h for h in h5_names
                        if h.replace('_patch_dataset.h5', '') in slide_names_set]
        elif num_slides:
            h5_names = h5_names[:num_slides]

        for h5_name in h5_names:
            h5_path   = h5_dir / h5_name
            base_name = h5_name.replace('_patch_dataset.h5', '')
            tiff_path = _find_tiff(tiff_dir / base_name, "*-registered.ome.tif")

            try:
                with h5py.File(h5_path, 'r') as f:
                    if 'targets' not in f:
                        print(f"Warning: {h5_name} has no /targets, skipping.")
                        continue
                    coords            = f['coords'][:]
                    targets           = f['targets'][:]
                    patch_size_level0 = int(f.attrs['patch_size_level0'])
                    if self.marker_names is None:
                        self.marker_names = list(f.attrs['marker_names'])
                    if self.patch_size is None:
                        self.patch_size = int(f.attrs['patch_size'])
                    if self.token_grid is None:
                        self.token_grid = int(f.attrs.get('token_grid', 16))
            except Exception:
                print(f'{h5_path} not found or unreadable — skipping')
                continue

            print(f"Loading {base_name} ({len(coords)} patches)…")
            slide_idx = len(self.slides)
            self.slides.append(str(tiff_path))
            self.slide_ids.append(base_name)

            for i in range(len(coords)):
                self.patch_map.append((slide_idx, int(coords[i, 0]), int(coords[i, 1]),
                                       patch_size_level0))
                self.targets.append(targets[i].astype(np.float32))

            if self.cls_cache_dir is not None:
                cls_path = self.cls_cache_dir / f"{base_name}_cls.npy"
                if not cls_path.exists():
                    raise FileNotFoundError(
                        f"CLS cache missing for {base_name}: {cls_path}. "
                        f"Run cls_cache.build_cls_cache(...) first.")
                cls = np.load(cls_path)
                assert len(cls) == len(coords), (
                    f"{base_name}: cls cache len {len(cls)} != coords {len(coords)}")
                _cls_chunks.append(cls)

        print(f"Dataset: {len(self.patch_map)} patches across {len(self.slides)} slide(s).")

        # (total_patches, embed_dim) float16, aligned to patch_map global index.
        self.cls_cache = np.concatenate(_cls_chunks, axis=0) if _cls_chunks else None
        if self.cls_cache is not None:
            print(f"  CLS cache loaded: {self.cls_cache.shape} ({self.cls_cache.dtype})")

        if token_means is not None:
            self.token_means = token_means
        elif self.targets:
            C = self.targets[0].shape[0]
            total_sum = np.zeros(C, dtype=np.float64)
            total_n   = 0
            for t in self.targets:
                total_sum += t.reshape(C, -1).sum(axis=1)
                total_n   += t.shape[1] * t.shape[2]
            self.token_means = torch.from_numpy((total_sum / total_n).astype(np.float32))
        else:
            self.token_means = None

        # Spatial index: (slide_idx, x, y) → position in patch_map, for neighbour lookup.
        self._coord_to_idx: dict[tuple[int, int, int], int] = {}
        if use_neighbours:
            for idx, (si, x, y, _) in enumerate(self.patch_map):
                self._coord_to_idx[(si, x, y)] = idx
            print(f"  Spatial index built: {len(self._coord_to_idx):,} patches indexed")

    # ── zarr helpers ─────────────────────────────────────────────────────────

    def _open_arr(self, slide_idx: int):
        if not hasattr(self, '_arr_cache'):
            self._arr_cache = {}
        if slide_idx not in self._arr_cache:
            tif   = tifffile.TiffFile(self.slides[slide_idx])
            store = tif.aszarr()
            z     = zarr.open(store, mode="r")
            self._arr_cache[slide_idx] = z["0"] if isinstance(z, zarr.hierarchy.Group) else z
        return self._arr_cache[slide_idx]

    def _load_norm_patch(self, slide_idx: int, x: int, y: int, psz: int) -> torch.Tensor:
        """Load one patch → (3, 224, 224) float32, ImageNet-normalised, no augmentation."""
        arr   = self._open_arr(slide_idx)
        patch = np.array(arr[y:y + psz, x:x + psz, :])
        patch = cv2.resize(patch, (self.patch_size, self.patch_size),
                           interpolation=cv2.INTER_LINEAR)
        patch = (patch.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(patch.transpose(2, 0, 1))

    # ── stats helpers ─────────────────────────────────────────────────────────

    def compute_marker_stds(self) -> torch.Tensor:
        C           = self.targets[0].shape[0]
        total_sum   = np.zeros(C, dtype=np.float64)
        total_sumsq = np.zeros(C, dtype=np.float64)
        total_n     = 0
        for t in self.targets:
            flat         = t.reshape(C, -1)
            total_sum   += flat.sum(axis=1)
            total_sumsq += (flat ** 2).sum(axis=1)
            total_n     += flat.shape[1]
        mean = total_sum / total_n
        var  = total_sumsq / total_n - mean ** 2
        return torch.from_numpy(np.sqrt(np.maximum(var, 0.0)).astype(np.float32))

    def compute_marker_prevalence(self) -> torch.Tensor:
        """Per-marker positive fraction p_c = frac of tokens where marker c exceeds its own
        token-mean — the SAME per-marker foreground definition as the loss mask. Used to set
        prevalence-balanced background weights w_bg(c)=balance·p_c/(1-p_c) in training, so each
        marker's rare positives aren't drowned by its abundant negatives. Computed once over
        the whole set (a single batch can have 0 positives for a sparse marker → unstable)."""
        C   = self.targets[0].shape[0]
        thr = self.token_means.numpy()[:, None, None]            # (C,1,1)
        pos = np.zeros(C, dtype=np.float64)
        tot = 0
        for t in self.targets:
            pos += (t > thr).reshape(C, -1).sum(axis=1)
            tot += t.reshape(C, -1).shape[1]
        return torch.from_numpy((pos / max(tot, 1)).astype(np.float32))

    def compute_sampling_weights(
        self,
        hard_marker_names: list[str] | None = None,
        sparse_threshold: float = 0.01,
        cap: float = 4.0,
    ) -> torch.Tensor:
        """Per-patch oversampling weights that target hard/sparse markers.

        Each patch holds token_grid² tokens, so we reduce (C, G, G) → one scalar by
        COUNTING positive tokens per hard marker — positive := token > that marker's
        global token-mean (the same foreground definition used for the loss mask). Each
        marker's count is normalised by the 95th-percentile count over its POSITIVE
        patches (a robust "strongly-positive patch" anchor; guarded ≥1 for ultra-sparse
        markers whose all-patch percentiles collapse to 0), clipped to 0–1 so it
        contributes comparably, then summed across hard markers and offset/capped.

        Count-based (not max-based) so the weight reflects HOW MUCH rare signal a patch
        carries — a patch with a few rare-marker tokens gets a modest bump; one rich in
        several hard markers gets the most — instead of firing on a single bright token.
        """
        N = len(self.targets)                       # total number of patches
        name_to_idx = {(n.decode() if isinstance(n, bytes) else str(n)): j
                           for j, n in enumerate(self.marker_names)}
        hard_idx = [name_to_idx[n] for n in hard_marker_names if n in name_to_idx]


        hm     = np.asarray(hard_idx)
        thr    = self.token_means.numpy()[hm][:, None, None]                  # (n_hard,1,1)
        counts = (np.stack([t[hm] for t in self.targets]) > thr).reshape(     # (N, n_hard) #  (N, 3, 16, 16)  > thr is 3D -> 4D with 1 in front
                     N, len(hm), -1).sum(2).astype(np.float32)                # positive-token count
        # normalise each marker by the 95th-pct count over its positive patches (guard ≥1)
        norm    = np.array([max(np.percentile(c[c > 0], 95) if (c > 0).any() else 0, 1)
                            for c in counts.T], dtype=np.float32)
        score   = np.clip(counts / norm, 0.0, 1.0).sum(1)                     # 0..1 per marker, summed
        weights = np.clip(1.0 + score, 1.0, cap).astype(np.float32)
        print(f"  Sampling weights: hard={[self.marker_names[j] for j in hard_idx]} "
              f"cap={cap} mean_w={weights.mean():.3f} frac>1={(weights > 1.0).mean():.3f}")
        return torch.from_numpy(weights)

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.patch_map)

    def __getitem__(self, idx: int):
        """
        Returns (patch, target, mask) or, when use_neighbours=True, a 5-tuple:
          patch             : (3, 224, 224)       float32 — normalised H&E
          target            : (C, G, G)           float32 — mean IF expression per token
          mask              : (G, G)              bool    — True where ≥1 marker > global mean
          neighbour_present : (8,)                bool    — True where a neighbour exists
          5th element is EITHER:
            neighbour_cls   : (8, D)              float32 — cached neighbour CLS (cls_cache set), OR
            neighbours      : (8, 3, 224, 224)    float32 — raw neighbour patches (no cache)

        NOTE: geometric augmentation is disabled in neighbour mode — flipping/rotating
        the center patch would break its spatial alignment with the neighbours.
        """
        slide_idx, x, y, psz = self.patch_map[idx]
        arr   = self._open_arr(slide_idx)
        patch = np.array(arr[y:y + psz, x:x + psz, :])
        patch = cv2.resize(patch, (self.patch_size, self.patch_size),
                           interpolation=cv2.INTER_LINEAR)

        nbr_perm = None
        if self.augment:
            target = self.targets[idx].copy()
            # Geometric aug — now also in neighbour mode: the same flip/rotation is applied
            # to the neighbour slot layout (nbr_perm) so directional pos-embeddings stay
            # consistent with the transformed centre patch.
            if np.random.random() < self.p_geom:
                choice = np.random.randint(3)
                if choice == 0:                        # horizontal flip
                    patch  = patch[:, ::-1, :].copy()
                    target = target[:, :, ::-1].copy()
                    geom   = lambda g: g[:, ::-1]
                elif choice == 1:                      # vertical flip
                    patch  = patch[::-1, :, :].copy()
                    target = target[:, ::-1, :].copy()
                    geom   = lambda g: g[::-1, :]
                else:                                  # rotation 90 / 180 / 270°
                    k = np.random.randint(1, 4)
                    patch  = np.rot90(patch,  k, axes=(0, 1)).copy()
                    target = np.rot90(target, k, axes=(1, 2)).copy()
                    geom   = lambda g, k=k: np.rot90(g, k, axes=(0, 1))
                if self.use_neighbours:
                    nbr_perm = _neighbour_perm(geom)
            # HED colour jitter — H&E only, targets unchanged
            if np.random.random() < self.p_hed:
                patch = _hed_jitter(patch)
            # MIPHEI colour augs (brightness/contrast, blur, noise) — image-only.
            patch = _photometric_jitter(patch, self.p_bc, self.p_blur, self.p_noise)
        else:
            target = self.targets[idx]

        patch  = (patch.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        patch  = torch.from_numpy(patch.transpose(2, 0, 1))
        target = torch.from_numpy(target)
        # OLD (per-token "any marker present" mask, (G,G)) — broadcast identically to all 16
        # channels in the loss, so a sparse marker's foreground was dominated by tokens where
        # OTHER markers were present but it was absent → no gradient on its rare positives.
        # mask   = (target > self.token_means[:, None, None]).any(dim=0)
        # NEW: PER-MARKER foreground mask (C,G,G) — channel c is foreground where marker c is
        # "present". fg_mode="zero": present = target>0 (background = true zeros only → the
        # ×λ bg term drives them to 0 = darker bg; valid because AF-subtraction leaves 92-99%
        # of tokens exactly 0). fg_mode="token_mean": present = target>its own token-mean.
        # The loss weights each marker's positives vs
        # negatives independently (prevalence-balanced bg weight in training_orion_reg).
        if self.fg_mode == "zero":
            mask = (target > 0)                                          # (C, G, G) per-marker
        else:
            mask = (target > self.token_means[:, None, None])            # (C, G, G) per-marker

        # When requested, append the slide index as the LAST tuple element so
        # downstream code can compute per-slide metrics. Default off keeps the
        # return signature unchanged for other consumers (cls_cache, viz).
        extra = (slide_idx,) if self.return_slide_idx else ()

        if not self.use_neighbours:
            return (patch, target, mask) + extra

        present = torch.zeros(NUM_NEIGHBOURS, dtype=torch.bool)

        if self.cls_cache is not None:
            # Cached path: return neighbour CLS vectors (no image loading / encoding).
            nbr_cls = torch.zeros(NUM_NEIGHBOURS, self.cls_cache.shape[1], dtype=torch.float32)
            for k, (dx, dy) in enumerate(_NEIGHBOUR_DELTAS):
                nidx = self._coord_to_idx.get((slide_idx, x + dx * psz, y + dy * psz))
                if nidx is not None:
                    nbr_cls[k] = torch.from_numpy(self.cls_cache[nidx].astype(np.float32))
                    present[k] = True
            if nbr_perm is not None:                    # geometric aug → remap slots
                pt      = torch.as_tensor(nbr_perm)
                nbr_cls = nbr_cls[pt]
                present = present[pt]
            return (patch, target, mask, nbr_cls, present) + extra

        # Fallback: return raw neighbour patches (encoded on the fly by the model).
        neighbours = torch.zeros(NUM_NEIGHBOURS, 3, self.patch_size, self.patch_size,
                                 dtype=torch.float32)
        for k, (dx, dy) in enumerate(_NEIGHBOUR_DELTAS):
            nidx = self._coord_to_idx.get((slide_idx, x + dx * psz, y + dy * psz))
            if nidx is not None:
                nsi, nx, ny, npsz = self.patch_map[nidx]
                neighbours[k] = self._load_norm_patch(nsi, nx, ny, npsz)
                present[k]    = True

        if nbr_perm is not None:                        # geometric aug → remap slots
            pt         = torch.as_tensor(nbr_perm)
            neighbours = neighbours[pt]
            present    = present[pt]

        return (patch, target, mask, neighbours, present) + extra