"""
Run the ORION token model on the HEMIT v4 patches → predictions h5.

Zero-shot, exactly like PathoCell: the ORION-trained 16-marker model is applied to
HEMIT H&E; we keep its Pan-CK / CD3e (≈CD3) / Hoechst (≈DAPI) channels for the
cell eval. This produces the `/preds` h5 that build_cell_token_features.py --pred_h5
consumes — same native-448 geometry as build_patch_dataset_hemit_token.py v4, so the
patch coords line up with the dataset h5 and the nuclei masks.

For each patch (NATIVE top-left in the dataset h5): crop 448 native from the source
H&E, pad the edge to 448 with WHITE (255), resize to 224, ImageNet-normalise, run
the model → (16, 16, 16) token grid.

Example
-------
  HF_HOME=/home/wesley/spatial_proteomics/foundation_models HF_HUB_OFFLINE=1 \
  CUDA_VISIBLE_DEVICES=1 python cell_cls/hemit_cell_cls/run_inference_hemit.py \
      --split test --pred_dir outputs_orion_token_UNI2_baseline_bg0.2 --batch_size 16
"""
import os
os.environ.setdefault("HF_HOME", "/home/wesley/spatial_proteomics/foundation_models")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from model import SpatialModel                                    # noqa: E402
from build_patch_dataset_hemit_token import crop_pad_resize       # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(pred_dir: Path):
    import yaml
    marker_names = [str(m) for m in np.load(pred_dir / "marker_names.npy")]
    # LoRA checkpoints hold peft adapters (base_layer + lora_A/B); a plain SpatialModel
    # + strict=False would SILENTLY DROP them. Rebuild with the saved LoRA config.
    cfg_path = pred_dir / "config.yaml"
    lora_kw = {}
    if cfg_path.exists():
        ft = yaml.safe_load(cfg_path.read_text()).get("finetune", {})
        if ft.get("mode") == "lora":
            lc = ft["lora"]
            lora_kw = dict(lora_last_n=lc["last_n"], lora_rank=lc["rank"],
                           lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                           lora_suffixes=tuple(lc["suffixes"]))
    model = SpatialModel("UNI2", num_outputs=len(marker_names),
                         token_grid=16, freeze=True, fds_cfg=None, **lora_kw)
    missing, unexpected = model.load_state_dict(
        torch.load(pred_dir / "best_model.pt", map_location=DEVICE), strict=False)
    if [k for k in missing if "lora_" in k] or unexpected:
        raise RuntimeError(f"bad load: lora_missing={[k for k in missing if 'lora_' in k][:3]} "
                           f"unexpected={unexpected[:3]}")
    model.to(DEVICE).eval()
    print(f"  model ← {pred_dir/'best_model.pt'}  ({len(marker_names)} markers)")
    return model, marker_names


def normalise_patch(he_uint8: np.ndarray, x: int, y: int) -> np.ndarray:
    """Native-448 crop (edge padded white) → 224 → ImageNet norm → (3,224,224)."""
    crop = crop_pad_resize(he_uint8, x, y, pad_value=255).astype(np.float32)  # (224,224,3)
    crop = (crop / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return crop.transpose(2, 0, 1)


def run_batches(model, patches: np.ndarray, batch_size: int) -> np.ndarray:
    loader = DataLoader(TensorDataset(torch.from_numpy(patches)),
                        batch_size=batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for (batch,) in loader:
            with torch.autocast("cuda", enabled=torch.cuda.is_available()):
                pred, _ = model(batch.to(DEVICE))
            out.append(pred.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--dataset_h5", default=None,
                    help="default hemit_patch_dataset/<split>.h5")
    ap.add_argument("--pred_dir", default="training_outputs/outputs_orion_token_UNI2_baseline_bg0.2")
    ap.add_argument("--out", default=None, help="default <split>_preds.h5 next to dataset")
    ap.add_argument("--batch_size", type=int, default=1024)
    args = ap.parse_args()

    ds_h5 = Path(args.dataset_h5 or REPO_ROOT / f"datasets/hemit_patch_dataset/{args.split}.h5")
    out_h5 = Path(args.out or ds_h5.with_name(f"{args.split}_preds.h5"))

    with h5py.File(ds_h5, "r") as f:
        coords  = f["coords"][:].astype(np.int64)
        sources = f["sources"][:].astype(str)
        ps0     = int(f.attrs["patch_size_level0"])
        G       = int(f.attrs["token_grid"])
    assert ps0 == 448, f"expected v4 geometry (ps0=448), got {ps0}"

    model, marker_names = load_model(Path(args.pred_dir))
    C = len(marker_names)
    print(f"[{args.split}] {len(coords)} patches over {len(np.unique(sources))} tiles "
          f"| ps0={ps0} | out={out_h5.name}")

    # stream per source tile (load each H&E once), keep rows in dataset-h5 order
    preds = np.zeros((len(coords), C, G, G), dtype=np.float32)
    buf_patches, buf_rows = [], []

    def flush():
        if not buf_rows:
            return
        arr = np.stack(buf_patches).astype(np.float32)
        preds[np.asarray(buf_rows)] = run_batches(model, arr, args.batch_size)
        buf_patches.clear(); buf_rows.clear()

    order = np.argsort(sources, kind="stable")                   # group rows by tile
    cur_src, he = None, None
    for n, r in enumerate(order):
        if n % 1000 == 0:
            print(f"  {n}/{len(order)}")
        src = sources[r]
        if src != cur_src:
            flush()
            he = tifffile.imread(src)
            if he.ndim == 2:
                he = np.stack([he] * 3, -1)
            he = np.ascontiguousarray(he[..., :3].astype(np.uint8))
            cur_src = src
        x, y = int(coords[r, 0]), int(coords[r, 1])
        buf_patches.append(normalise_patch(he, x, y))
        buf_rows.append(r)
        if len(buf_rows) >= 512:
            flush()
    flush()

    max_len = max(len(s) for s in sources)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("coords",  data=coords.astype(np.int16), compression="gzip")
        f.create_dataset("sources", data=np.array(sources, dtype=f"S{max_len}"),
                         compression="gzip")
        f.create_dataset("preds",   data=preds, compression="gzip",
                         chunks=(min(256, len(preds)), C, G, G))
        f.attrs["marker_names"]      = marker_names
        f.attrs["patch_size_level0"] = ps0
        f.attrs["token_grid"]        = G
        f.attrs["pred_dir"]          = str(args.pred_dir)
    print(f"saved → {out_h5}  /preds {preds.shape}  mean={preds.mean():.4f}")


if __name__ == "__main__":
    main()