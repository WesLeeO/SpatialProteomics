"""Cache LoRA-model token predictions on the BENCHMARK patch grid.

The checkpoint in --model_dir is a peft-injected state (base_layer + lora_A/B).
A plain SpatialModel + load_state_dict(strict=False) would silently DROP every
LoRA weight, so we rebuild the encoder with the same LoRA config (read from the
model dir's config.yaml) before loading.

Predictions are row-aligned to each slide's orion_crc_patch_dataset_benchmark h5
coords (what build_cell_token_features.py / eval_cell_auprc.py expect), and saved
next to the checkpoint as <slide>_preds.npy (+ _targets.npy, _names.npy).
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from model import SpatialModel
from build_patch_dataset_orion_crc_reg import crc_paths, open_zarr_level0
# reuse the exact HE preprocessing + batched forward used by the benchmark eval
import visualize_orion_predictions as vp
from visualize_orion_predictions import extract_patches, run_inference

vp.BATCH_SIZE = 256   # GPU is shared; keep inference batch modest

BENCH = Path("datasets/orion_crc_patch_dataset_benchmark")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_lora_model(model_dir: Path) -> torch.nn.Module:
    cfg = yaml.safe_load((model_dir / "config.yaml").read_text())
    m, ft = cfg["model"], cfg["finetune"]
    lc = ft["lora"]
    assert ft["mode"] == "lora", f"config mode={ft['mode']} is not lora"
    model = SpatialModel(
        m["name"], num_outputs=m["num_outputs"], token_grid=m["token_grid"],
        freeze=True, fds_cfg=None,
        lora_last_n=lc["last_n"], lora_rank=lc["rank"],
        lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
        lora_suffixes=tuple(lc["suffixes"]),
    )
    state = torch.load(model_dir / "best_model.pt", map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # sanity: a correct load has NO lora_ in missing and NO unexpected keys
    lora_missing = [k for k in missing if "lora_" in k]
    print(f"  loaded: {len(state)} ckpt tensors | missing={len(missing)} "
          f"(lora_missing={len(lora_missing)}) | unexpected={len(unexpected)}")
    if lora_missing or unexpected:
        raise RuntimeError(f"bad load: lora_missing={lora_missing[:4]} "
                           f"unexpected={unexpected[:4]}")
    return model.to(device).eval()


def cache_slide(model, slide: str, out_dir: Path):
    with h5py.File(BENCH / f"{slide}_patch_dataset.h5", "r") as f:
        coords  = f["coords"][:]
        targets = f["targets"][:]
        psz     = int(f.attrs["patch_size_level0"])
        names   = [str(n) for n in f.attrs["marker_names"]]
    he_path, _ = crc_paths(slide)
    he_arr, _, _, _ = open_zarr_level0(he_path)
    print(f"[{slide}] {len(coords)} patches  psz={psz}")
    patches = extract_patches(he_arr, coords, psz)
    preds = run_inference(model, patches)            # (N, C, G, G)
    assert len(preds) == len(coords)
    np.save(out_dir / f"{slide}_preds.npy", preds)
    np.save(out_dir / f"{slide}_targets.npy", targets)
    np.save(out_dir / f"{slide}_names.npy", np.array(names))
    # quick sanity: per-marker pred<->target pearson (scale-invariant)
    P = preds.reshape(len(preds), preds.shape[1], -1)
    T = targets.reshape(len(targets), targets.shape[1], -1)
    rs = []
    for c in range(P.shape[1]):
        a, b = P[:, c].ravel(), T[:, c].ravel()
        rs.append(np.corrcoef(a, b)[0, 1])
    print(f"  saved preds -> {out_dir}  | mean pred-target r = {np.nanmean(rs):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="training_outputs/outputs_orion_token_UNI2_baseline_lora8x16mlp_2loss_lbg8_fg0")
    ap.add_argument("--slides", nargs="+", default=["CRC19", "CRC30", "CRC11", "CRC02"])
    args = ap.parse_args()
    out_dir = Path(args.model_dir)
    print(f"Loading LoRA model from {out_dir}/best_model.pt …")
    model = load_lora_model(out_dir)
    for s in args.slides:
        cache_slide(model, s, out_dir)


if __name__ == "__main__":
    main()