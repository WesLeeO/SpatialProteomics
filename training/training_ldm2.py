from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as skimage_ssim
from torch.utils.data import DataLoader, Subset, random_split

from dataset_orion_ldm import OrionLDMDataset
from ldm2 import LDM2


ORION_DATA_DIR = Path("/mnt/ssd1/virtual_proteomics/data/ORION_CRC")
H5_DIR = Path("orion_crc_patch_dataset")
P99_FILE = H5_DIR / "p99s_slide.txt"
LAMBDA_JSON = Path("MIPHEI-ViT/preprocessings/mif_cleaning/lambda_settings/orion.json")

PRETRAINED_MODEL = "sd2-community/stable-diffusion-2-1-base"

SLIDE_SPLIT = False
VAL_FRAC = 0.2
BATCH_SIZE = 128
NUM_EPOCHS = 20
LR = 5e-5
NUM_WORKERS = 8
SEED = 42
DISPLAY_SIZE = 224
GEN_BATCH_SIZE = 16
N_VIZ = 8
N_EVAL_PATCHES = 256


MARKERS = ["Hoechst", "SMA", "Pan-CK", "CD3e", "CD4"] 
NUM_SLIDES = 3
NOISE_TYPE = "gaussian"
NUM_INFERENCE_STEPS = 50

OUTPUT_DIR = Path(f"training_outputs/outputs_ldm2_0shot_{Path(PRETRAINED_MODEL).name}_LR{LR}_EPOCHS{NUM_EPOCHS}")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def plot_loss(train_losses, val_losses=None, path=OUTPUT_DIR / "plot_loss.png"):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train", color="red")
    if val_losses:
        plt.plot(epochs, val_losses, label="Val", color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Noise MSE (latent space)")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    pred_np = pred.squeeze(1).cpu().numpy()
    gt_np = gt.squeeze(1).cpu().numpy()

    ssims, psnrs, pccs = [], [], []
    for p, g in zip(pred_np, gt_np):
        ssims.append(skimage_ssim(p, g, data_range=1.0))

        mse = np.mean((p - g) ** 2)
        psnrs.append(float("inf") if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse)))

        pf, gf = p.ravel(), g.ravel()
        if gf.std() < 1e-6 or pf.std() < 1e-6:
            pccs.append(0.0)
        else:
            pccs.append(pearsonr(pf, gf).statistic)

    return {
        "ssim": float(np.mean(ssims)),
        "psnr": float(np.mean(psnrs)),
        "pcc": float(np.mean(pccs)),
    }


def plot_metrics(history: list[dict], path, marker_names: list[str] | None = None):
    epochs = range(1, len(history) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, label in zip(
        axes,
        ["ssim", "psnr", "pcc"],
        ["SSIM", "PSNR (dB)", "Pearson CC"],
    ):
        if marker_names:
            for marker in marker_names:
                mk_key = f"{marker}_{key}"
                vals = [h.get(mk_key) for h in history]
                if any(v is not None for v in vals):
                    ax.plot(epochs, vals, linewidth=1, alpha=0.6, label=marker)

        ax.plot(
            epochs,
            [h[key] for h in history],
            color="black",
            linewidth=2,
            marker="o",
            label="mean",
        )
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(True)
        if marker_names:
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_prediction_figure(
    he_patches,
    predictions: list,
    protein_maps,
    out_path,
    marker_name: str = "Hoechst",
):
    num_pred = len(predictions)
    fig, axes = plt.subplots(
        num_pred,
        3,
        figsize=(3 * DISPLAY_SIZE / 72, num_pred * DISPLAY_SIZE / 72),
        squeeze=False,
    )

    for i in range(num_pred):
        he_display = he_patches[i].permute(1, 2, 0).cpu().numpy()
        he_display = (he_display * IMAGENET_STD) + IMAGENET_MEAN
        he_display = np.clip(he_display, 0, 1)

        axes[i, 0].imshow(he_display)
        axes[i, 0].set_title("H&E Patch", fontsize=9)

        axes[i, 1].imshow(predictions[i].squeeze().cpu().numpy(), cmap="gray")
        axes[i, 1].set_title(f"Pred: {marker_name}", fontsize=9)

        axes[i, 2].imshow(protein_maps[i].squeeze().cpu().numpy(), cmap="gray")
        axes[i, 2].set_title("GT Protein", fontsize=9)

    for row in axes:
        for ax in row:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {out_path}")


def validate_epoch(ldm_model: LDM2, val_loader, device, marker_names: list[str]):
    ldm_model.eval()

    acc = {
        m: {"ssim": [], "psnr": [], "pcc": [], "he": [], "pred": [], "gt": [], "n": 0}
        for m in marker_names
    }

    with torch.no_grad():
        for he_patches, protein_maps, batch_markers, _ in val_loader:
            if all(acc[m]["n"] >= N_EVAL_PATCHES for m in marker_names):
                break

            he_patches = he_patches.to(device)
            protein_maps = protein_maps.to(device)

            for marker in marker_names:
                if acc[marker]["n"] >= N_EVAL_PATCHES:
                    continue

                idx = [i for i, m in enumerate(batch_markers) if m == marker]
                if not idx:
                    continue

                idx_t = torch.tensor(idx, device=device)
                he_m = he_patches[idx_t]
                gt_m = protein_maps[idx_t]

                pred_chunks = []
                for start in range(0, len(he_m), GEN_BATCH_SIZE):
                    he_chunk = he_m[start : start + GEN_BATCH_SIZE]
                    pred_chunks.append(
                        ldm_model.generate(
                            he_chunk,
                            marker_names=[marker] * len(he_chunk),
                            num_inference_steps=NUM_INFERENCE_STEPS,
                        )
                    )
                preds = torch.cat(pred_chunks, dim=0)

                m_dict = compute_metrics(preds, gt_m)
                acc[marker]["ssim"].append(m_dict["ssim"])
                acc[marker]["psnr"].append(m_dict["psnr"])
                acc[marker]["pcc"].append(m_dict["pcc"])
                acc[marker]["n"] += len(he_m)

                if len(acc[marker]["he"]) < N_VIZ:
                    n_take = min(N_VIZ - len(acc[marker]["he"]), len(he_m))
                    acc[marker]["he"].append(he_m[:n_take].cpu())
                    acc[marker]["pred"].append(preds[:n_take].cpu())
                    acc[marker]["gt"].append(gt_m[:n_take].cpu())

    per_marker = {}
    for marker in marker_names:
        a = acc[marker]
        if not a["ssim"]:
            continue
        per_marker[marker] = {
            "ssim": float(np.mean(a["ssim"])),
            "psnr": float(np.mean(a["psnr"])),
            "pcc": float(np.mean(a["pcc"])),
            "n": a["n"],
            "he": torch.cat(a["he"], dim=0) if a["he"] else None,
            "pred": torch.cat(a["pred"], dim=0) if a["pred"] else None,
            "gt": torch.cat(a["gt"], dim=0) if a["gt"] else None,
        }

    mean_metrics = {
        "ssim": float(np.mean([v["ssim"] for v in per_marker.values()])),
        "psnr": float(np.mean([v["psnr"] for v in per_marker.values()])),
        "pcc": float(np.mean([v["pcc"] for v in per_marker.values()])),
    }
    return mean_metrics, per_marker


def build_datasets():
    dataset = OrionLDMDataset(
        h5_dir=H5_DIR,
        tiff_dir=ORION_DATA_DIR,
        p99_file=P99_FILE,
        lambda_json=LAMBDA_JSON,
        markers=MARKERS,
        num_slides=NUM_SLIDES,
    )

    if SLIDE_SPLIT:
        n_slides = len(dataset.slide_names)
        rng = np.random.default_rng(SEED)
        order = rng.permutation(n_slides).tolist()
        n_val = max(1, int(n_slides * VAL_FRAC))
        val_set = set(order[:n_val])
        train_idx, val_idx = [], []
        for i, (slide_idx, *_) in enumerate(dataset.patch_map):
            (val_idx if slide_idx in val_set else train_idx).append(i)

        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        print(f"Slide-level split: val slides {sorted(val_set)} | train {len(train_ds)} val {len(val_ds)}")
    else:
        n_val = int(len(dataset) * VAL_FRAC)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(SEED),
        )
    return dataset, train_ds, val_ds


def make_loader(dataset, shuffle: bool):
    persistent_workers = NUM_WORKERS > 0
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )


def train_ldm2():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:1")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    dataset, train_ds, val_ds = build_datasets()
    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)

    print(f"\n{'─'*55}")
    print(f"  Dataset       : {len(dataset)} patches total")
    print(f"  Train patches : {len(train_ds)}  ({len(train_loader)} batches × bs={BATCH_SIZE})")
    print(f"  Val patches   : {len(val_ds)}  ({len(val_loader)} batches × bs={BATCH_SIZE})")
    print(f"  Val noise loss: all {len(val_ds)} val patches")
    print(f"  Val metrics   : at least {N_EVAL_PATCHES} patches per marker")
    print(f"{'─'*55}\n")

    marker_names = dataset.marker_names

    print("Marker names", marker_names)

    ldm = LDM2.from_pretrained(
        PRETRAINED_MODEL,
        marker_names=marker_names,
        noise_type=NOISE_TYPE,

    ).to(device)

    optimizer = torch.optim.AdamW(ldm.trainable_parameters(), lr=LR)

    num_parameters = sum(p.numel() for p in ldm.trainable_parameters())
    print(f"Num trainable parameters = {num_parameters}")

    log_path = OUTPUT_DIR / "training_log.csv"
    best_mean_ssim = -1.0
    train_losses = []
    val_losses = []
    val_metrics_history = []

    per_marker_cols = (
        [f"{m}_ssim" for m in marker_names]
        + [f"{m}_psnr" for m in marker_names]
        + [f"{m}_pcc" for m in marker_names]
    )
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "mean_ssim", "mean_psnr", "mean_pcc"] + per_marker_cols
        )

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Training epoch {epoch} ...")
        ldm.train()
        loss_epoch = 0.0

        for i, (he_patches, protein_maps, batch_marker_names, marker_weights) in enumerate(train_loader):
            he_patches = he_patches.to(device)
            protein_maps = protein_maps.to(device)
            marker_weights = marker_weights.to(device)

            out = ldm(
                he_patches,
                protein_maps,
                list(batch_marker_names),
                marker_weights=marker_weights,
            )
            loss = out.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item() * len(he_patches)
            print(f"End training batch {i + 1}/{len(train_loader)}")

        train_loss = loss_epoch / len(train_loader.dataset)
        train_losses.append(train_loss)

        ldm.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            print("Validation ...")
            for i, (he_patches, protein_maps, batch_marker_names, marker_weights) in enumerate(val_loader):
                he_patches = he_patches.to(device)
                protein_maps = protein_maps.to(device)
                marker_weights = marker_weights.to(device)

                out = ldm(
                    he_patches,
                    protein_maps,
                    list(batch_marker_names),
                    marker_weights=marker_weights,
                )
                val_loss_epoch += out.loss.item() * len(he_patches)
                print(f"End validation batch {i + 1}/{len(val_loader)}")

        val_loss = val_loss_epoch / len(val_loader.dataset)
        val_losses.append(val_loss)

        plot_loss(train_losses, val_losses)
        print(
            f"Epoch {epoch}/{NUM_EPOCHS} train={train_loss:.5f} "
            f"val={val_loss:.5f} - running image metrics..."
        )

        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(SEED)

        mean_metrics, per_marker = validate_epoch(ldm, val_loader, device, marker_names=marker_names)

        row_metrics = dict(mean_metrics)
        for m, v in per_marker.items():
            row_metrics[f"{m}_ssim"] = v["ssim"]
            row_metrics[f"{m}_psnr"] = v["psnr"]
            row_metrics[f"{m}_pcc"] = v["pcc"]
        val_metrics_history.append(row_metrics)
        plot_metrics(val_metrics_history, OUTPUT_DIR / "val_metrics.png", marker_names=marker_names)

        with open(log_path, "a", newline="") as f:
            def _fmt(marker, key):
                return f"{per_marker.get(marker, {}).get(key, float('nan')):.4f}"

            pm_vals = (
                [_fmt(m, "ssim") for m in marker_names]
                + [_fmt(m, "psnr") for m in marker_names]
                + [_fmt(m, "pcc") for m in marker_names]
            )
            csv.writer(f).writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{mean_metrics['ssim']:.4f}",
                    f"{mean_metrics['psnr']:.4f}",
                    f"{mean_metrics['pcc']:.4f}",
                ]
                + pm_vals
            )

        print(
            f"           mean metrics  SSIM={mean_metrics['ssim']:.4f}  "
            f"PSNR={mean_metrics['psnr']:.2f}dB  PCC={mean_metrics['pcc']:.4f}"
        )
        for m, v in per_marker.items():
            print(
                f"             {m:15s}  SSIM={v['ssim']:.4f}  "
                f"PSNR={v['psnr']:.2f}dB  PCC={v['pcc']:.4f}  "
                f"(n={v['n']})"
            )

        if mean_metrics["ssim"] > best_mean_ssim:
            best_mean_ssim = mean_metrics["ssim"]
            torch.save(
                {
                    "epoch": epoch,
                    "unet": ldm.unet.state_dict(),
                    "conditioner": ldm.conditioner.state_dict(),
                    "metrics": mean_metrics,
                    "per_marker": {
                        m: {k: v[k] for k in ("ssim", "psnr", "pcc")}
                        for m, v in per_marker.items()
                    },
                    "marker_names": marker_names,
                    "pretrained_model": PRETRAINED_MODEL,
                    "noise_type": NOISE_TYPE,
                },
                OUTPUT_DIR / "best_model.pt",
            )
            print(f"           -> best model saved (mean SSIM={best_mean_ssim:.4f})")

        for marker, v in per_marker.items():
            if v["he"] is None:
                continue
            n_viz = min(N_VIZ, len(v["he"]))
            save_prediction_figure(
                v["he"][:n_viz],
                [v["pred"][i].unsqueeze(0) for i in range(n_viz)],
                v["gt"][:n_viz],
                OUTPUT_DIR / f"val_epoch{epoch:03d}_{marker}.png",
                marker_name=marker,
            )

    torch.save(
        {
            "unet": ldm.unet.state_dict(),
            "conditioner": ldm.conditioner.state_dict(),
            "marker_names": marker_names,
            "pretrained_model": PRETRAINED_MODEL,
            "noise_type": NOISE_TYPE,
        },
        OUTPUT_DIR / "last_model.pt",
    )


if __name__ == "__main__":
    train_ldm2()
