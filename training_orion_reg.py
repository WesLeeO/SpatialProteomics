import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader, random_split, Subset, WeightedRandomSampler

from dataset_orion_reg import OrionSpatialDataset
from model import SpatialModel
from constraints import BiologicalConstraintLoss
from marker_descriptions import MARKER_DESCRIPTIONS




# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME  = 'UNI2'
H5_DIR      = Path("orion_crc_patch_dataset_reg")
TIFF_DIR    = Path("/mnt/ssd1/virtual_proteomics/data/ORION_CRC")
OUTPUT_DIR  = Path(f"outputs_orion_token_{MODEL_NAME}_finetuning_full_lossv2")
TOKEN_GRID  = 16          # must match build_patch_dataset_orion_crc_reg.py
NUM_OUTPUTS = 16          # number of IF markers
VAL_FRAC    = 0.15
TEST_FRAC   = 0.15
BATCH_SIZE  = 512 # 1024
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

# ── Bio loss config ───────────────────────────────────────────────────────────
# lambda_max: maximum weight of bio loss relative to MSE (reached after warmup)
# warmup_steps: ramp bio loss from 0 → lambda_max over this many training steps
LAMBDA_BIO           = 0
BIO_WARMUP_STEPS     = 500
WEIGHTED_MSE_BASE    = 5   # alpha for median-std marker; others scale inversely with std
WEIGHTED_MSE_MAX     = 20  # cap to avoid instability on extremely sparse markers

# ── CONCH refinement config ───────────────────────────────────────────────────
# Set to True to add a per-marker patch-level cosine-similarity bias on top of
# UNI2 token predictions.  CONCH is always frozen; only sim_scale/sim_bias train.
USE_CONCH_REFINEMENT = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def encode_marker_descriptions(marker_names: list[str]) -> torch.Tensor:
    """
    Encode MARKER_DESCRIPTIONS for the given markers with CONCH's text encoder.
    Returns (C, 512) L2-normalised text embeddings on CPU.
    CONCH is loaded, used, then discarded — the model keeps only the buffer.
    """
    from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer, tokenize as conch_tokenize
    print("Encoding marker descriptions with CONCH text encoder...")
    conch, _ = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/CONCH")
    conch = conch.to(device).eval()
    tokenizer = get_tokenizer()
    texts  = [MARKER_DESCRIPTIONS[m] for m in marker_names]
    tokens = conch_tokenize(texts=texts, tokenizer=tokenizer).to(device)
    with torch.no_grad():
        embs = F.normalize(conch.encode_text(tokens), dim=-1).cpu()
    del conch
    torch.cuda.empty_cache()
    print(f"  text embeddings: {embs.shape}")
    return embs


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


# ── Biology-aware loss ────────────────────────────────────────────────────────

class BioLoss(BiologicalConstraintLoss):
    """
    Thin wrapper that adds compute_bio_only() so the bio terms can be added
    on top of the existing masked MSE without touching that loss path.
    Indices in constraints.py already match the sequential 0-15 model output.
    """

    def compute_bio_only(self, preds: torch.Tensor) -> dict:
        """
        preds : (B, C, G, G) — raw model output in [0, 1] (MSE-supervised).
        Returns dict with grad-connected 'bio_loss' and detached component scalars.
        """
        B, C, G, _ = preds.shape
        p = preds.permute(0, 2, 3, 1).reshape(B * G * G, C).clamp(0.0, 1.0)

        hard = self._hard_excl_loss(p)
        soft = self._soft_excl_loss(p)
        coex = self._coexpr_loss(p)
        ecad = self._ecad_panck_loss(p)

        lam = self._lambda()
        bio = (self.w_hard   * hard +
               self.w_soft   * soft +
               self.w_coexpr * coex +
               self.w_ecad   * ecad)

        if self.training:
            self._step += 1

        return {
            "bio_loss": lam * bio,
            "hard":     hard.detach(),
            "soft":     soft.detach(),
            "coexpr":   coex.detach(),
            "ecad":     ecad.detach(),
            "lambda":   lam,
        }


# ── Metrics ───────────────────────────────────────────────────────────────────


class OnlinePearson:
    """Streaming Pearson r — accumulates 5 sufficient statistics per marker."""
    def __init__(self, C: int):
        self.n    = np.zeros(C, dtype=np.float64)
        self.sx   = np.zeros(C, dtype=np.float64)
        self.sy   = np.zeros(C, dtype=np.float64)
        self.sxx  = np.zeros(C, dtype=np.float64)
        self.syy  = np.zeros(C, dtype=np.float64)
        self.sxy  = np.zeros(C, dtype=np.float64)

    def update(self, preds: np.ndarray, targets: np.ndarray):
        """preds / targets: (B, C, G, G) float32"""
        B, C, G, _ = preds.shape
        p = preds.transpose(0, 2, 3, 1).reshape(-1, C).astype(np.float64)   # (N, C)
        t = targets.transpose(0, 2, 3, 1).reshape(-1, C).astype(np.float64)
        self.n   += p.shape[0]
        self.sx  += p.sum(0);  self.sy  += t.sum(0)
        self.sxx += (p * p).sum(0);  self.syy += (t * t).sum(0)
        self.sxy += (p * t).sum(0)

    def compute(self) -> np.ndarray:
        num  = self.n * self.sxy - self.sx * self.sy
        den  = np.sqrt(np.maximum((self.n * self.sxx - self.sx**2) *
                                   (self.n * self.syy - self.sy**2), 0.0))
        return np.where(den > 0, num / den, 0.0)


class OnlineSpearman:
    """First-N sampled Spearman r — fills buf_size tokens then stops. Representative
    when the dataloader is shuffled (each epoch sees a different random prefix)."""
    def __init__(self, C: int, buf_size: int = 10_000_000):
        self.C        = C
        self.buf_size = buf_size
        self._p       = np.empty((buf_size, C), dtype=np.float32)
        self._t       = np.empty((buf_size, C), dtype=np.float32)
        self._n       = 0

    def update(self, preds: np.ndarray, targets: np.ndarray):
        """preds / targets: (B, C, G, G) float32"""
        if self._n >= self.buf_size:
            return
        B, C, G, _ = preds.shape
        p    = preds.transpose(0, 2, 3, 1).reshape(-1, C).astype(np.float32)
        t    = targets.transpose(0, 2, 3, 1).reshape(-1, C).astype(np.float32)
        fill = min(len(p), self.buf_size - self._n)
        self._p[self._n:self._n + fill] = p[:fill]
        self._t[self._n:self._n + fill] = t[:fill]
        self._n += fill

    def compute(self) -> np.ndarray:
        n = min(self._n, self.buf_size)
        out = np.zeros(self.C)
        for c in range(self.C):
            r, _ = spearmanr(self._p[:n, c], self._t[:n, c])
            out[c] = float(r) if np.isfinite(r) else 0.0
        return out


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

def run_epoch(model, loader, criterion, optimizer=None, scaler=None,
              scheduler=None, epoch=None, bio_loss_fn=None,
              marker_stds=None):
    """
    epoch: 0-indexed epoch number.  Pass during training to enable FDS accumulation
           and feature smoothing.  Omit (or pass None) for validation.
    bio_loss_fn: BioLoss instance; applied only during training.
    """
    training = optimizer is not None
    model.train() if training else model.eval()
    if bio_loss_fn is not None:
        bio_loss_fn.train() if training else bio_loss_fn.eval()

    use_fds = training and getattr(model, 'fds', None) is not None and epoch is not None
    if use_fds:
        C          = model.num_outputs
        bucket_num = model.fds[0].bucket_num
        feat_dim   = model.fds[0].feature_dim
        fds_count  = torch.zeros(C, bucket_num)
        fds_sum    = torch.zeros(C, bucket_num, feat_dim)
        fds_sumsq  = torch.zeros(C, bucket_num, feat_dim)

    total_mse            = 0.0
    total_bio            = 0.0
    per_marker_loss_acc  = None   # (C,) accumulated raw MSE per marker
    bio_batch_count      = 0
    pearson_acc          = None
    spearman_acc         = None

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

                # plain masked MSE
                # mse_loss = criterion(preds[mask_exp], targets[mask_exp])

         
                # MIPHEI: per-marker MSE normalised by train std → equal gradient scale across markers
                sq_err         = (preds - targets) ** 2                                                              # (B, C, G, G)
                per_marker_mse = (sq_err * mask_exp).sum(dim=(0, 2, 3)) / mask_exp.sum(dim=(0, 2, 3)).clamp(min=1)  # (C,) raw MSE per marker
                mse_loss       = (per_marker_mse / marker_stds).mean()                                              # mean over C markers

                bio_dict = None
                loss     = mse_loss
                if bio_loss_fn is not None:
                    bio_dict = bio_loss_fn.compute_bio_only(preds)
                    if training:
                        loss = mse_loss + bio_dict["bio_loss"]

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()

            # token-level aggregation (used with plain/weighted MSE)
            # active_elements = mask_exp.sum().item()
            # total_mse += mse_loss.item() * active_elements
            # count     += active_elements

            total_mse += mse_loss.item()
            pm_np      = per_marker_mse.detach().float().cpu().numpy()   # (C,) raw MSE per marker
            if per_marker_loss_acc is None:
                per_marker_loss_acc = np.zeros(len(pm_np), dtype=np.float64)
            per_marker_loss_acc += pm_np

            if bio_dict is not None:
                total_bio       += bio_dict["bio_loss"].item()
                bio_batch_count += 1

            p_np = preds.detach().float().cpu().numpy()
            t_np = targets_cpu.numpy()

            if pearson_acc is None:
                pearson_acc  = OnlinePearson(p_np.shape[1])
                spearman_acc = OnlineSpearman(p_np.shape[1])

            pearson_acc.update(p_np, t_np)
            spearman_acc.update(p_np, t_np)

            if i % 100 == 0 and bio_dict is not None:
                print(f'  batch {i+1}/{len(loader)}  '
                      f'mse={mse_loss.item():.5f}  '
                      f'bio_loss={bio_dict["bio_loss"].item():.5f}'
                )
      
            print(f'End batch {i+1}/{len(loader)}')

    # After training epoch: update FDS running stats then snapshot + smooth
    if use_fds:
        for j, fds_j in enumerate(model.fds):
            fds_j.update_running_stats_from_moments(
                fds_count[j], fds_sum[j], fds_sumsq[j], epoch
            )
            fds_j.update_last_epoch_stats(epoch + 1)

    mean_mse            = total_mse / len(loader)
    mean_bio            = total_bio / bio_batch_count if bio_batch_count > 0 else 0.0
    mean_per_marker_mse = per_marker_loss_acc / len(loader)   # (C,) mean raw MSE per marker
    p_per_marker        = pearson_acc.compute()
    s_per_marker        = spearman_acc.compute()
    return mean_mse, mean_bio, mean_per_marker_mse, p_per_marker, s_per_marker


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_curves(train_losses, val_losses, train_pearsons, val_pearsons,
                train_spearmans, val_spearmans, marker_names,
                train_bio_losses=None, val_bio_losses=None,
                train_per_marker_losses=None, val_per_marker_losses=None):
    epochs  = range(1, len(train_losses) + 1)
    use_bio = train_bio_losses is not None

    ncols = 3 if use_bio else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    axes[0].plot(epochs, train_losses, label="Train")
    axes[0].plot(epochs, val_losses,   label="Val")
    axes[0].set_title("MSE Loss"); axes[0].legend()

    if use_bio:
        axes[1].plot(epochs, train_bio_losses, label="Train")
        axes[1].plot(epochs, val_bio_losses,   label="Val")
        axes[1].set_title("Bio Loss"); axes[1].legend()

    ax_p = axes[2] if use_bio else axes[1]
    ax_p.plot(epochs, [p.mean() for p in train_pearsons], label="Train")
    ax_p.plot(epochs, [p.mean() for p in val_pearsons],   label="Val")
    ax_p.set_title("Mean Pearson r"); ax_p.legend()

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
    val_smat = np.stack(val_spearmans)   # (epochs, C)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5), squeeze=False)
    for j in range(C):
        ax = axes[j // ncols][j % ncols]
        ax.plot(epochs, [s[j] for s in train_spearmans], label="Train")
        ax.plot(epochs, val_smat[:, j], label="Val")
        ax.set_title(marker_names[j] if marker_names else f"M{j}", fontsize=8)
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    for j in range(C, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8)
    plt.suptitle("Per-marker Spearman ρ", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "per_marker_spearman.png", dpi=150)
    plt.close()

    # Per-marker MSE loss grid
    if train_per_marker_losses is not None and val_per_marker_losses is not None:
        train_lmat = np.stack(train_per_marker_losses)   # (epochs, C)
        val_lmat   = np.stack(val_per_marker_losses)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5), squeeze=False)
        for j in range(C):
            ax = axes[j // ncols][j % ncols]
            ax.plot(epochs, train_lmat[:, j], label="Train")
            ax.plot(epochs, val_lmat[:, j],   label="Val")
            ax.set_title(marker_names[j] if marker_names else f"M{j}", fontsize=8)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        for j in range(C, nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", fontsize=8)
        plt.suptitle("Per-marker MSE loss", fontsize=10)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "per_marker_loss.png", dpi=150)
        plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    sys.stdout = Logger(OUTPUT_DIR / "training_log.txt")

    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    # ── Slide-level train / val / test split ─────────────────────────────────────
    all_slides = sorted(f.replace('_patch_dataset.h5', '')
                        for f in os.listdir(H5_DIR) if f.endswith('.h5'))
    rng = np.random.default_rng(SEED)
    rng.shuffle(all_slides)

    n_test  = max(1, round(len(all_slides) * TEST_FRAC))
    n_val   = max(1, round(len(all_slides) * VAL_FRAC))
    test_slides  = all_slides[:n_test]
    val_slides   = all_slides[n_test:n_test + n_val]
    train_slides = all_slides[n_test + n_val:]

    print(f"Slides — train: {train_slides}")
    print(f"         val:   {val_slides}")
    print(f"         test:  {test_slides}")

    train_dataset = OrionSpatialDataset(str(H5_DIR), str(TIFF_DIR),
                                        augment=True,  slide_names=train_slides)
    val_dataset   = OrionSpatialDataset(str(H5_DIR), str(TIFF_DIR),
                                        augment=False, slide_names=val_slides,
                                        token_means=train_dataset.token_means)
    test_dataset  = OrionSpatialDataset(str(H5_DIR), str(TIFF_DIR),
                                        augment=False, slide_names=test_slides,
                                        token_means=train_dataset.token_means)
    marker_names  = train_dataset.marker_names

    train_stds = train_dataset.compute_marker_stds().to(device)   # (C,)
    print("Train marker stds (used for MIPHEI loss normalisation):")
    for name, s in zip(marker_names, train_stds.cpu()):
        print(f"  {name:<20s}  std={s:.4f}")

    """
    sample_weights = train_dataset.compute_sampling_weights(
        hard_marker_names=['FOXP3', 'CD8a', 'CD163', 'PD-L1', 'CD31', 'CD68']
    )

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )
    """

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    #print(f"Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}| Device: {device}")

    # Encoder fully frozen for phase 1; we unfreeze last N blocks manually at phase 2.
    conch_text_embs = encode_marker_descriptions(marker_names) if USE_CONCH_REFINEMENT else None
    model     = SpatialModel(MODEL_NAME, num_outputs=NUM_OUTPUTS,
                              token_grid=TOKEN_GRID,
                              fds_cfg=None,
                              unfreeze_last_n=0,
                              conch_text_embs=conch_text_embs).to(device)
    criterion = torch.nn.MSELoss()
    scaler    = torch.amp.GradScaler('cuda')

    bio_loss_fn = BioLoss(
        lambda_max=LAMBDA_BIO,
        warmup_steps=BIO_WARMUP_STEPS,
    ).to(device) if LAMBDA_BIO > 0 else None
    if bio_loss_fn is not None:
        print(f"BioLoss: lambda_max={LAMBDA_BIO}, warmup_steps={BIO_WARMUP_STEPS}")

    train_losses,   val_losses   = [], []
    train_pearsons, val_pearsons = [], []
    train_spearmans, val_spearmans = [], []
    train_per_marker_losses, val_per_marker_losses = [], []
    train_bio_losses, val_bio_losses = ([], []) if bio_loss_fn is not None else (None, None)
    best_val_pearson = -np.inf

    def _run_one_epoch(epoch, optimizer, scheduler=None):
        epoch_0idx = epoch - 1
        print(f'Epoch {epoch}...')
        print('Training...')
        train_mse, train_bio, train_pm_loss, train_p, train_s = run_epoch(
            model, train_loader, criterion, optimizer,
            scaler=scaler, scheduler=scheduler, epoch=epoch_0idx,
            bio_loss_fn=bio_loss_fn, marker_stds=train_stds,
        )
        print('Validating...')
        val_mse, val_bio, val_pm_loss, val_p, val_s = run_epoch(
            model, val_loader, criterion, bio_loss_fn=bio_loss_fn,
            marker_stds=train_stds,
        )

        train_losses.append(train_mse);              val_losses.append(val_mse)
        train_pearsons.append(train_p);              val_pearsons.append(val_p)
        train_spearmans.append(train_s);             val_spearmans.append(val_s)
        train_per_marker_losses.append(train_pm_loss)
        val_per_marker_losses.append(val_pm_loss)
        if bio_loss_fn is not None:
            train_bio_losses.append(train_bio)
            val_bio_losses.append(val_bio)

        bio_str = f"  bio {train_bio:.4f}/{val_bio:.4f}" if bio_loss_fn is not None else ""
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"train mse {train_mse:.4f}  r {train_p.mean():.4f}  ρ {train_s.mean():.4f} | "
              f"val mse {val_mse:.4f}  r {val_p.mean():.4f}  ρ {val_s.mean():.4f}{bio_str}")
        names = marker_names or [f"M{j}" for j in range(len(val_p))]
        for name, p, tl, vl in zip(names, val_p, train_pm_loss, val_pm_loss):
            print(f"    {name:<20s}  pearson {p:.4f}  train_mse {tl:.6f}  val_mse {vl:.6f}")

        if USE_CONCH_REFINEMENT and hasattr(model, 'sim_scale'):
            scales = model.sim_scale.detach().cpu().numpy()
            biases = model.sim_bias.detach().cpu().numpy()
            print("  CONCH sim_scale/bias:")
            for name, sc, bi in zip(names, scales, biases):
                print(f"    {name:<20s}  scale {sc:+.4f}  bias {bi:+.4f}")

        nonlocal best_val_pearson
        if val_p.mean() > best_val_pearson:
            best_val_pearson = val_p.mean()
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            print(f"  → best model saved (val r={best_val_pearson:.4f})")

        np.save(OUTPUT_DIR / "train_losses.npy",           np.array(train_losses))
        np.save(OUTPUT_DIR / "val_losses.npy",             np.array(val_losses))
        np.save(OUTPUT_DIR / "train_pearsons.npy",         np.stack(train_pearsons))
        np.save(OUTPUT_DIR / "val_pearsons.npy",           np.stack(val_pearsons))
        np.save(OUTPUT_DIR / "train_spearmans.npy",        np.stack(train_spearmans))
        np.save(OUTPUT_DIR / "val_spearmans.npy",          np.stack(val_spearmans))
        np.save(OUTPUT_DIR / "train_per_marker_losses.npy", np.stack(train_per_marker_losses))
        np.save(OUTPUT_DIR / "val_per_marker_losses.npy",   np.stack(val_per_marker_losses))
        np.save(OUTPUT_DIR / "marker_names.npy",           np.array(marker_names))
        if bio_loss_fn is not None:
            np.save(OUTPUT_DIR / "train_bio_losses.npy", np.array(train_bio_losses))
            np.save(OUTPUT_DIR / "val_bio_losses.npy",   np.array(val_bio_losses))
        plot_curves(train_losses, val_losses, train_pearsons, val_pearsons,
                    train_spearmans, val_spearmans, marker_names,
                    train_bio_losses, val_bio_losses,
                    train_per_marker_losses, val_per_marker_losses)

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

    _conch_names   = {'sim_scale', 'sim_bias'}
    conch_params   = [p for n, p in model.named_parameters()
                      if p.requires_grad and n in _conch_names]
    head_params    = [p for n, p in model.named_parameters()
                      if p.requires_grad and 'encoder' not in n and n not in _conch_names]
    encoder_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and 'encoder' in n]

    param_groups = [
        {'params': head_params,    'lr': LR},
        {'params': encoder_params, 'lr': LR * 0.2},
    ]
    if conch_params:
        param_groups.append({'params': conch_params, 'lr': LR * 5})
    optimizer_p2 = torch.optim.Adam(param_groups)

    phase2_epochs = NUM_EPOCHS - PHASE1_EPOCHS
    phase2_steps  = phase2_epochs * len(train_loader)

    def _head_schedule(step):
        return 0.5 * (1 + math.cos(math.pi * min(step / phase2_steps, 1.0)))

    def _encoder_schedule(step):
        if step < WARMUP_STEPS:
            return step / max(WARMUP_STEPS, 1)
        progress = (step - WARMUP_STEPS) / max(phase2_steps - WARMUP_STEPS, 1)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    schedules = [_head_schedule, _encoder_schedule]
    if conch_params:
        schedules.append(_head_schedule)  # same cosine decay as head
    scheduler_p2 = torch.optim.lr_scheduler.LambdaLR(optimizer_p2, schedules)

    for epoch in range(PHASE1_EPOCHS + 1, NUM_EPOCHS + 1):
        _run_one_epoch(epoch, optimizer_p2, scheduler=scheduler_p2)

    print(f"\nDone. Best val Pearson: {best_val_pearson:.4f}")

    # ── Test evaluation (best checkpoint) ─────────────────────────────────────
    """
    print("\n── Test set evaluation ──")
    best_state = torch.load(OUTPUT_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(best_state, strict=False)
    test_mse, _, test_p, test_s = run_epoch(model, test_loader, criterion)
    print(f"Test MSE {test_mse:.4f} | mean Pearson r {test_p.mean():.4f} | mean Spearman ρ {test_s.mean():.4f}")
    names = marker_names or [f"M{j}" for j in range(len(test_p))]
    for name, p in zip(names, test_p):
        print(f"    {name:<20s}  pearson {p:.4f}")
    np.save(OUTPUT_DIR / "test_pearsons.npy", test_p)
    np.save(OUTPUT_DIR / "test_slides.npy",   np.array(test_slides))
    """


if __name__ == "__main__":
    train()