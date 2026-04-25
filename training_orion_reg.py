import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader, random_split, Subset

from dataset_orion_reg import OrionSpatialDataset
from model import SpatialModel




# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = 'UNI2'
H5_DIR      = Path("orion_crc_patch_dataset_reg")
TIFF_DIR    = Path("/mnt/ssd1/virtual_proteomics/data/ORION_CRC")
OUTPUT_DIR  = Path(f"outputs_orion_token_{MODEL_NAME}_finetuning")
TOKEN_GRID  = 16          # must match build_patch_dataset_orion_crc_reg.py
NUM_OUTPUTS = 16          # number of IF markers
VAL_FRAC    = 0.2
BATCH_SIZE  = 512
NUM_EPOCHS  = 30
LR          = 1e-4
NUM_WORKERS = 4
SEED        = 42

# ── FDS config ────────────────────────────────────────────────────────────────
# Labels are in [0, 1].  50 uniform bins → bin width 0.02.
# start_smooth=1: begin calibrating features from the 2nd epoch onward
# (1st epoch is used to warm up the running statistics).
FDS_CFG = dict(
    bucket_num   = 50,
    bucket_start = 0,
    start_update = 0,
    start_smooth = 1,
    kernel       = 'gaussian',
    ks           = 5,
    sigma        = 2,
    momentum     = 0.9,
)

UNFREEZE_LAST_N  = 4    # UNI2 blocks to unfreeze in phase 2
PHASE1_EPOCHS    = 2    # epochs with head only (encoder fully frozen)
WARMUP_STEPS     = 500  # linear warmup steps for encoder LR at phase-2 start


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Logger:
    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path, "a")
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
        self.log.flush()
    def flush(self):
        pass


# ── Metrics ───────────────────────────────────────────────────────────────────

def pearson_per_marker(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    preds / targets: (N, C, G, G)  — flatten spatial dims before computing Pearson
    so each (patch, token) pair is treated as an independent observation.
    Returns (C,) array of Pearson r values.
    """
    N, C, G, _ = preds.shape
    p = preds.transpose(0, 2, 3, 1).reshape(-1, C)    # (N*G*G, C)
    t = targets.transpose(0, 2, 3, 1).reshape(-1, C)
    return np.array([pearsonr(p[:, j], t[:, j])[0] for j in range(C)])


def spearman_per_marker(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    N, C, G, _ = preds.shape
    p = preds.transpose(0, 2, 3, 1).reshape(-1, C)
    t = targets.transpose(0, 2, 3, 1).reshape(-1, C)
    return np.array([spearmanr(p[:, j], t[:, j])[0] for j in range(C)])


# ── FDS helpers ───────────────────────────────────────────────────────────────

def _fds_accumulate(h_cpu, targets_cpu, fds_list, fds_count, fds_sum, fds_sumsq):
    """
    Accumulate per-bucket feature moments from one batch.

    h_cpu:       (N, 128)    penultimate features on CPU
    targets_cpu: (B, C, G, G) IF labels in [0, 1] on CPU
    fds_count/sum/sumsq: accumulators of shape (C, bucket_num) / (C, bucket_num, 128)
    """
    N, D = h_cpu.shape
    B, C, G, _ = targets_cpu.shape
    labels_flat = targets_cpu.permute(0, 2, 3, 1).reshape(N, C).numpy()  # (N, C)

    for j, fds_j in enumerate(fds_list):
        buckets  = fds_j._bucket_idx(labels_flat[:, j])             # (N,) numpy int64
        b_tensor = torch.from_numpy(buckets.astype(np.int64))       # (N,)
        idx_exp  = b_tensor.unsqueeze(1).expand(-1, D)              # (N, D)

        fds_count[j].scatter_add_(0, b_tensor, torch.ones(N))
        fds_sum[j].scatter_add_(0, idx_exp, h_cpu)
        fds_sumsq[j].scatter_add_(0, idx_exp, h_cpu * h_cpu)


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer=None, scaler=None, scheduler=None, epoch=None):
    """
    epoch: 0-indexed epoch number.  Pass during training to enable FDS accumulation
           and feature smoothing.  Omit (or pass None) for validation.
    """
    training = optimizer is not None
    model.train() if training else model.eval()

    use_fds = training and getattr(model, 'fds', None) is not None and epoch is not None
    if use_fds:
        C          = model.num_outputs
        bucket_num = model.fds[0].bucket_num
        feat_dim   = model.fds[0].feature_dim
        fds_count  = torch.zeros(C, bucket_num)
        fds_sum    = torch.zeros(C, bucket_num, feat_dim)
        fds_sumsq  = torch.zeros(C, bucket_num, feat_dim)

    total_loss = 0.0
    all_preds, all_targets = [], []
    count = 0

    with torch.set_grad_enabled(training):
        for i, (patches, targets_cpu, masks_cpu) in enumerate(loader):
            patches = patches.to(device)
            targets = targets_cpu.to(device)
            masks   = masks_cpu.to(device)                           # (B, G, G) bool

            with torch.amp.autocast('cuda'):
                if use_fds:
                    preds, h = model(patches, labels=targets, epoch=epoch)
                    _fds_accumulate(h.float().cpu(), targets_cpu, model.fds,
                                    fds_count, fds_sum, fds_sumsq)
                else:
                    preds, _ = model(patches)

                mask_exp = masks.unsqueeze(1).expand_as(preds)      # (B, C, G, G)
                loss = criterion(preds[mask_exp], targets[mask_exp])

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

            active_elements = mask_exp.sum().item()
            total_loss += loss.item() * active_elements
            count += active_elements

            all_preds.append(preds.detach().float().cpu().numpy())
            all_targets.append(targets_cpu.numpy())
            print(f'End batch {i+1}/{len(loader)}')

    # After training epoch: update FDS running stats then snapshot + smooth
    if use_fds:
        for j, fds_j in enumerate(model.fds):
            fds_j.update_running_stats_from_moments(
                fds_count[j], fds_sum[j], fds_sumsq[j], epoch
            )
            fds_j.update_last_epoch_stats(epoch + 1)

    all_preds   = np.concatenate(all_preds,   axis=0)   # (N, C, G, G)
    all_targets = np.concatenate(all_targets, axis=0)
    mean_loss   = total_loss / count
    p_per_marker = pearson_per_marker(all_preds, all_targets)
    s_per_marker = spearman_per_marker(all_preds, all_targets)
    return mean_loss, p_per_marker, s_per_marker


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_curves(train_losses, val_losses, train_pearsons, val_pearsons,
                train_spearmans, val_spearmans, marker_names):
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    ax1.plot(epochs, train_losses, label="Train")
    ax1.plot(epochs, val_losses,   label="Val")
    ax1.set_title("MSE Loss"); ax1.legend()

    ax2.plot(epochs, [p.mean() for p in train_pearsons], label="Train")
    ax2.plot(epochs, [p.mean() for p in val_pearsons],   label="Val")
    ax2.set_title("Mean Pearson r"); ax2.legend()

    ax3.plot(epochs, [s.mean() for s in train_spearmans], label="Train")
    ax3.plot(epochs, [s.mean() for s in val_spearmans],   label="Val")
    ax3.set_title("Mean Spearman r"); ax3.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close()

    # Per-marker Pearson grid
    val_mat = np.stack(val_pearsons)   # (epochs, C)
    C = val_mat.shape[1]
    ncols = 6
    nrows = int(np.ceil(C / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5), squeeze=False)
    for j in range(C):
        ax = axes[j // ncols][j % ncols]
        ax.plot(epochs, [r[j] for r in train_pearsons], label="Train")
        ax.plot(epochs, val_mat[:, j], label="Val")
        ax.set_title(marker_names[j] if marker_names else f"M{j}", fontsize=8)
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    for j in range(C, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8)
    plt.suptitle("Per-marker Pearson r", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_marker_pearson.png", dpi=150)
    plt.close()

    # Per-marker Spearman grid
    val_mat = np.stack(val_spearmans)   # (epochs, C)
    C = val_mat.shape[1]
    ncols = 6
    nrows = int(np.ceil(C / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5), squeeze=False)
    for j in range(C):
        ax = axes[j // ncols][j % ncols]
        ax.plot(epochs, [r[j] for r in train_spearmans], label="Train")
        ax.plot(epochs, val_mat[:, j], label="Val")
        ax.set_title(marker_names[j] if marker_names else f"M{j}", fontsize=8)
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    for j in range(C, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8)
    plt.suptitle("Per-marker Spearman Rank", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_marker_spearman.png", dpi=150)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    sys.stdout = Logger(OUTPUT_DIR / "training_log.txt")

    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    dataset      = OrionSpatialDataset(str(H5_DIR), str(TIFF_DIR))
    marker_names = dataset.marker_names

    n_val   = int(len(dataset) * VAL_FRAC)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Device: {device}")

    # Encoder fully frozen for phase 1; we unfreeze last N blocks manually at phase 2.
    model     = SpatialModel(MODEL_NAME, num_outputs=NUM_OUTPUTS,
                              token_grid=TOKEN_GRID,
                              fds_cfg=None,
                              unfreeze_last_n=0).to(device)
    criterion = torch.nn.MSELoss()
    scaler    = torch.amp.GradScaler('cuda')

    train_losses,   val_losses    = [], []
    train_pearsons, val_pearsons  = [], []
    train_spearmans, val_spearmans = [], []
    best_val_pearson = -np.inf

    def _run_one_epoch(epoch, optimizer, scheduler=None):
        epoch_0idx = epoch - 1
        print(f'Epoch {epoch}...')
        print('Training...')
        train_loss, train_p, train_s = run_epoch(
            model, train_loader, criterion, optimizer,
            scaler=scaler, scheduler=scheduler, epoch=epoch_0idx,
        )
        print('Validating...')
        val_loss, val_p, val_s = run_epoch(model, val_loader, criterion)

        train_losses.append(train_loss);     val_losses.append(val_loss)
        train_pearsons.append(train_p);      val_pearsons.append(val_p)
        train_spearmans.append(train_s);     val_spearmans.append(val_s)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"train loss {train_loss:.4f}  r {train_p.mean():.4f} | "
              f"val loss {val_loss:.4f}  r {val_p.mean():.4f}")
        names = marker_names or [f"M{j}" for j in range(len(val_p))]
        for name, p, s in zip(names, val_p, val_s):
            print(f"    {name:<20s}  pearson {p:.4f}  spearman {s:.4f}")

        nonlocal best_val_pearson
        if val_p.mean() > best_val_pearson:
            best_val_pearson = val_p.mean()
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            print(f"  → best model saved (val r={best_val_pearson:.4f})")

        np.save(OUTPUT_DIR / "train_losses.npy",    np.array(train_losses))
        np.save(OUTPUT_DIR / "val_losses.npy",      np.array(val_losses))
        np.save(OUTPUT_DIR / "train_pearsons.npy",  np.stack(train_pearsons))
        np.save(OUTPUT_DIR / "val_pearsons.npy",    np.stack(val_pearsons))
        np.save(OUTPUT_DIR / "train_spearmans.npy", np.stack(train_spearmans))
        np.save(OUTPUT_DIR / "val_spearmans.npy",   np.stack(val_spearmans))
        np.save(OUTPUT_DIR / "marker_names.npy",    np.array(marker_names))
        plot_curves(train_losses, val_losses, train_pearsons, val_pearsons,
                    train_spearmans, val_spearmans, marker_names)

    # ── Phase 1: head only ────────────────────────────────────────────────────
    head_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_p1 = torch.optim.Adam(head_params, lr=LR)
    print(f"\n── Phase 1: head only ({PHASE1_EPOCHS} epochs) ──")
    for epoch in range(1, PHASE1_EPOCHS + 1):
        _run_one_epoch(epoch, optimizer_p1)

    # ── Phase 2: unfreeze last N blocks ───────────────────────────────────────
    print(f"\n── Phase 2: unfreezing last {UNFREEZE_LAST_N} blocks ──")
    for block in model.encoder.blocks[-UNFREEZE_LAST_N:]:
        for p in block.parameters():
            p.requires_grad = True
    for p in model.encoder.norm.parameters():
        p.requires_grad = True

    head_params    = [p for n, p in model.named_parameters()
                      if p.requires_grad and 'encoder' not in n]
    encoder_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and 'encoder' in n]
    optimizer_p2 = torch.optim.Adam([
        {'params': head_params,    'lr': LR},
        {'params': encoder_params, 'lr': LR * 0.2},   # 2e-5
    ])

    phase2_epochs = NUM_EPOCHS - PHASE1_EPOCHS
    phase2_steps  = phase2_epochs * len(train_loader)

    def _head_schedule(step):
        return 0.5 * (1 + math.cos(math.pi * min(step / phase2_steps, 1.0)))

    def _encoder_schedule(step):
        if step < WARMUP_STEPS:
            return step / max(WARMUP_STEPS, 1)
        progress = (step - WARMUP_STEPS) / max(phase2_steps - WARMUP_STEPS, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler_p2 = torch.optim.lr_scheduler.LambdaLR(
        optimizer_p2, [_head_schedule, _encoder_schedule]
    )

    for epoch in range(PHASE1_EPOCHS + 1, NUM_EPOCHS + 1):
        _run_one_epoch(epoch, optimizer_p2, scheduler=scheduler_p2)

    print(f"\nDone. Best val Pearson: {best_val_pearson:.4f}")


if __name__ == "__main__":
    train()