"""
Precompute frozen-UNI2 CLS tokens for every ORION patch.

Because the encoder is frozen, each patch's CLS is deterministic — compute it once
and the neighbour-model training never has to re-encode neighbours. One file per
slide, aligned to that slide's H5 coords order:

    <cache_dir>/<slide>_cls.npy   (N_patches, embed_dim)  float16

Run standalone:
    python cls_cache.py --h5_dir orion_crc_patch_dataset_benchmark \
                        --tiff_dir /mnt/ssd1/virtual_proteomics/data/ORION_CRC \
                        --cache_dir orion_crc_patch_dataset_benchmark/cls_cache
or call build_cls_cache(...) from the training script (auto-builds missing slides).
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from dataset_orion_reg import OrionSpatialDataset
from model import foundation_model


@torch.no_grad()
def build_cls_cache(h5_dir: str, tiff_dir: str, cache_dir: str,
                    model_name: str = "UNI2", device: str = "cuda",
                    batch_size: int = 1024, num_workers: int = 4) -> None:
    """Build per-slide CLS caches for every slide in h5_dir that isn't cached yet."""
    h5_dir    = Path(h5_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    slides = sorted(f.replace("_patch_dataset.h5", "")
                    for f in os.listdir(h5_dir) if f.endswith(".h5"))
    todo = [s for s in slides if not (cache_dir / f"{s}_cls.npy").exists()]
    if not todo:
        print(f"[cls_cache] all {len(slides)} slides already cached → {cache_dir}")
        return

    print(f"[cls_cache] building {len(todo)}/{len(slides)} slides → {cache_dir}")
    encoder = foundation_model(model_name).to(device).eval()
    embed_dim = encoder.embed_dim

    for s in todo:
        ds = OrionSpatialDataset(str(h5_dir), str(tiff_dir), slide_names=[s],
                                 augment=False, use_neighbours=False)
        if len(ds) == 0:
            print(f"[cls_cache] {s}: 0 patches — skipping")
            continue
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        cls_all = np.empty((len(ds), embed_dim), dtype=np.float16)
        n = 0
        for batch in loader:
            patch = batch[0].to(device)                       # (B, 3, 224, 224)
            with torch.amp.autocast("cuda"):
                feats = encoder.forward_features(patch)
            cls = feats[:, 0].float().cpu().numpy().astype(np.float16)
            cls_all[n:n + len(cls)] = cls
            n += len(cls)
            print(f"    [{s}] {n}/{len(ds)}", end="\r", flush=True)
        np.save(cache_dir / f"{s}_cls.npy", cls_all)
        print(f"\n[cls_cache] saved {s}: {cls_all.shape}")

    del encoder
    torch.cuda.empty_cache()
    print("[cls_cache] done.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5_dir",    default="datasets/orion_crc_patch_dataset_benchmark")
    ap.add_argument("--tiff_dir",  default="/mnt/ssd1/virtual_proteomics/data/ORION_CRC")
    ap.add_argument("--cache_dir", default="datasets/orion_crc_patch_dataset_benchmark/cls_cache")
    ap.add_argument("--model",     default="UNI2")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    build_cls_cache(args.h5_dir, args.tiff_dir, args.cache_dir,
                    model_name=args.model, device=device,
                    batch_size=args.batch_size, num_workers=args.num_workers)


if __name__ == "__main__":
    main()