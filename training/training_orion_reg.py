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
from model import SpatialModel, NeighbourhoodSpatialModel, NeighbourCLSModel
from constraints import BiologicalConstraintLoss
from marker_descriptions import MARKER_DESCRIPTIONS




# ── Config (loaded from config.yaml; override with --config / --set) ────────────
from config import load_config, ns_to_dict

CFG = load_config()

# Unpack into the module-level names the training body (and compare_models) already use.
MODEL_NAME  = CFG.model.name
TOKEN_GRID  = CFG.model.token_grid
NUM_OUTPUTS = CFG.model.num_outputs

H5_DIR      = Path(CFG.data.h5_dir)
TIFF_DIR    = Path(CFG.data.tiff_dir)
SPLIT_MODE  = CFG.data.split_mode
VAL_SLIDES  = list(CFG.data.val_slides)
TEST_SLIDES = list(CFG.data.test_slides)
VAL_FRAC    = CFG.data.val_frac
TEST_FRAC   = CFG.data.test_frac

# finetuning (non-neighbour model only; neighbour model keeps UNI2 frozen)
FINETUNE_MODE   = CFG.finetune.mode
UNFREEZE_LAST_N = CFG.finetune.unfreeze_last_n
GRAD_CHECKPOINT = CFG.finetune.grad_checkpoint
LORA_LAST_N   = CFG.finetune.lora.last_n
LORA_RANK     = CFG.finetune.lora.rank
LORA_ALPHA    = CFG.finetune.lora.alpha
LORA_DROPOUT  = CFG.finetune.lora.dropout
LORA_SUFFIXES = tuple(CFG.finetune.lora.suffixes)
LORA_LR_SCALE = CFG.finetune.lora.lr_scale
LORA_TRAIN_NORMS = CFG.finetune.lora.train_norms   # also train LayerNorms+LayerScales of adapted blocks
_LORA_MLP     = any("mlp" in s for s in LORA_SUFFIXES)

# neighbourhood model
USE_NEIGHBOURS       = CFG.neighbour.use
NEIGHBOUR_ARCH       = CFG.neighbour.arch
NEIGHBOUR_BATCH_SIZE = CFG.neighbour.batch_size
MASK_NEIGHBOURS      = CFG.neighbour.mask_neighbours
CLS_CACHE_DIR = str(H5_DIR / "cls_cache")
USE_CLS_CACHE = USE_NEIGHBOURS and NEIGHBOUR_ARCH == "cls"

# loss: ACTIVE two-term per-marker loss  L_fg + LAMBDA_BG * L_bg
LAMBDA_BG         = CFG.loss.lambda_bg
FG_MODE           = CFG.loss.fg_mode      # "zero" (target>0) | "token_mean"
LAMBDA_BIO        = CFG.loss.bio.lambda_bio
BIO_WARMUP_STEPS  = CFG.loss.bio.warmup_steps
WEIGHTED_MSE_BASE = CFG.loss.weighted_mse.base
WEIGHTED_MSE_MAX  = CFG.loss.weighted_mse.max

# training
BATCH_SIZE    = CFG.train.batch_size
NUM_EPOCHS    = CFG.train.num_epochs
LR            = CFG.train.lr
WEIGHT_DECAY  = CFG.train.weight_decay
NUM_WORKERS   = CFG.train.num_workers
SEED          = CFG.train.seed
OVERSAMPLE    = CFG.train.oversample
PHASE1_EPOCHS = CFG.train.phase1_epochs
WARMUP_STEPS  = CFG.train.warmup_steps
SELECT        = CFG.train.select          # best | last | swa_lastk
SELECT_WINDOW = CFG.train.select_window

FDS_CFG = ns_to_dict(CFG.fds)

# experimental (off by default)
USE_CONCH_REFINEMENT = CFG.experimental.use_conch_refinement
CLS_FILM             = CFG.experimental.cls_film

# ── derived output dir / run tags ───────────────────────────────────────────────
_NB_TAG = ((f'{NEIGHBOUR_ARCH}_' + ('no_neighbours' if MASK_NEIGHBOURS else 'neighbours'))
           if USE_NEIGHBOURS else 'baseline')
_FT_TAG = (f"_lora{LORA_LAST_N}x{LORA_RANK}{'mlp' if _LORA_MLP else ''}" if FINETUNE_MODE == "lora"
           else f"_unfreeze{UNFREEZE_LAST_N}" if FINETUNE_MODE == "unfreeze" and UNFREEZE_LAST_N > 0
           else "")
_BG_TAG = f"_2loss_lbg{LAMBDA_BG:g}" + ("_fg0" if FG_MODE == "zero" else "")  # two-term per-marker loss
_OS_TAG = "_os" if OVERSAMPLE else ""
OUTPUT_DIR = (Path(CFG.data.output_dir) if CFG.data.output_dir
              else Path(f"training_outputs/outputs_orion_token_{MODEL_NAME}_{_NB_TAG}{_FT_TAG}{_BG_TAG}{_OS_TAG}"))


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


class GroupedOnlinePearson:
    """
    Per-slide Pearson r, averaged across slides per marker.

    Keeps one OnlinePearson per slide id and, on compute(), returns the
    unweighted mean across slides of each marker's per-slide r. This avoids the
    cross-slide pooling that lets between-slide intensity offsets (which a
    patch-level model cannot predict) dominate the metric — see the per-slide vs
    pooled discussion: a constant per-slide affine shift leaves per-slide r
    unchanged but tanks pooled r.
    """
    def __init__(self, C: int):
        self.C      = C
        self.groups: dict[int, OnlinePearson] = {}

    def update(self, preds: np.ndarray, targets: np.ndarray, slide_ids: np.ndarray):
        """preds / targets: (B, C, G, G);  slide_ids: (B,) int."""
        for sid in np.unique(slide_ids):
            acc = self.groups.get(int(sid))
            if acc is None:
                acc = self.groups[int(sid)] = OnlinePearson(self.C)
            m = slide_ids == sid
            acc.update(preds[m], targets[m])

    def compute(self) -> np.ndarray:
        if not self.groups:
            return np.zeros(self.C)
        per_slide = np.stack([acc.compute() for acc in self.groups.values()])  # (S, C)
        return per_slide.mean(axis=0)                                          # (C,)

    def per_slide(self) -> dict:
        """{slide_id: (C,) per-marker Pearson r} — the un-averaged breakdown
        used for paired across-slide model comparison (compare_models.py)."""
        return {int(sid): acc.compute() for sid, acc in self.groups.items()}


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
              marker_stds=None, use_neighbours=False):
    """
    epoch:          0-indexed epoch number for FDS; None for validation.
    bio_loss_fn:    BioLoss instance; applied only during training.
    use_neighbours: when True the loader yields 5-tuples and neighbour patches are
                    passed to NeighbourhoodSpatialModel (it encodes them internally).
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
    total_fg             = 0.0
    total_bg             = 0.0
    total_bio            = 0.0
    per_marker_loss_acc  = None
    bio_batch_count      = 0
    pearson_acc          = None
    spearman_acc         = None

    with torch.set_grad_enabled(training):
        for i, batch in enumerate(loader):
            # Datasets are built with return_slide_idx=True → slide id is the
            # last element of every batch (used for per-slide Pearson).
            *batch, slide_ids = batch
            slide_ids = slide_ids.numpy()
            if use_neighbours:
                patches, targets_cpu, masks_cpu, nbr_cpu, present_cpu = batch
                present = present_cpu.to(device)                     # (B, 8) bool
                if MASK_NEIGHBOURS:
                    present = torch.zeros_like(present)              # run A: hide neighbours
                nbr = nbr_cpu.to(device)
                # (B, 8, D) → cached CLS ;  (B, 8, 3, 224, 224) → raw patches
                if nbr.dim() == 3:
                    nbr_kw = dict(neighbour_cls=nbr, neighbour_present=present)
                else:
                    nbr_kw = dict(neighbours=nbr, neighbour_present=present)
            else:
                patches, targets_cpu, masks_cpu = batch
                nbr_kw = {}

            patches = patches.to(device)
            targets = targets_cpu.to(device)
            masks   = masks_cpu.to(device)                           # (B, C, G, G) bool, per-marker

            with torch.amp.autocast('cuda'):
                if use_fds:
                    preds, h = model(patches, labels=targets, epoch=epoch)
                    _fds_accumulate(h.float().cpu(), targets_cpu, model.fds,
                                    fds_count, fds_sum, fds_sumsq)
                elif use_neighbours:
                    preds, _ = model(patches, **nbr_kw)
                else:
                    preds, _ = model(patches)

                sq_err = (preds - targets) ** 2                          # (B, C, G, G)
                # ── OLD: per-token any-marker mask, one scalar bg_weight for every marker ──
                # The (B,G,G) mask was broadcast identically to all 16 channels, so a sparse
                # marker's foreground was dominated by tokens where it was ABSENT (other markers
                # present) → "predict 0" won → poor + slide-variant Pearson on sparse markers.
                # m_fg   = masks.unsqueeze(1).float().expand_as(sq_err)
                # m_bg   = 1.0 - m_fg
                # weight = bg_weight + (1.0 - bg_weight) * m_fg          # fg=1, bg=bg_weight
                # ── NEW: PER-MARKER foreground mask (from dataset, (B,C,G,G)) + per-marker
                # prevalence-balanced background weight. masks[:,c] is foreground only where
                # marker c itself is present; bg_weights[c]=balance·p_c/(1-p_c) balances each
                # marker's positives vs negatives independently (set in train(), passed in). ──
                m_fg   = masks.float()                                   # (B, C, G, G) per-marker
                m_bg   = 1.0 - m_fg
                # ── OLD per-token weighted MSE (replaced by the two-term loss below) ──
                # bgw    = bg_weights.view(1, -1, 1, 1)                  # (1, C, 1, 1) per marker
                # weight = bgw + (1.0 - bgw) * m_fg                      # fg=1, bg=w_bg(c)
                # per_marker_mse  = (sq_err * weight).sum(dim=(0,2,3)) / weight.sum(dim=(0,2,3)).clamp(min=1)
                # per_marker_nmse = per_marker_mse / marker_stds; mse_loss = per_marker_nmse.mean()
                # ── ACTIVE: TWO-TERM per-marker loss  L = L_fg + LAMBDA_BG · L_bg ──
                # Per-marker MSE on each marker's OWN foreground / background tokens, each
                # count-normalised within its group then std-normalised. Summed with λ on bg.
                fg_pm = (sq_err * m_fg).sum(dim=(0, 2, 3)) / m_fg.sum(dim=(0, 2, 3)).clamp(min=1)  # (C,)
                bg_pm = (sq_err * m_bg).sum(dim=(0, 2, 3)) / m_bg.sum(dim=(0, 2, 3)).clamp(min=1)  # (C,)
                fg_nmse = fg_pm / marker_stds                            # (C,) std-normalised
                bg_nmse = bg_pm / marker_stds                            # (C,)
                L_fg = fg_nmse.mean()
                L_bg = bg_nmse.mean()
                per_marker_nmse = fg_nmse + LAMBDA_BG * bg_nmse          # (C,) per-marker objective (logged)
                mse_loss        = L_fg + LAMBDA_BG * L_bg               # == per_marker_nmse.mean(); what BP sees

                bio_dict = None
                loss     = mse_loss
                if bio_loss_fn is not None:
                    bio_dict = bio_loss_fn.compute_bio_only(preds)
                    if training:
                        loss = mse_loss + bio_dict["bio_loss"]

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # Skip the scheduler step when AMP skipped the optimizer step
                # (scale drops on inf/nan grads) — avoids the "scheduler before
                # optimizer" warning during early loss-scale calibration.
                stepped = scaler.get_scale() >= scale_before
                if scheduler is not None and stepped:
                    scheduler.step()

            # token-level aggregation (used with plain/weighted MSE)
            # active_elements = mask_exp.sum().item()
            # total_mse += mse_loss.item() * active_elements
            # count     += active_elements

            total_mse += mse_loss.item()
            total_fg  += L_fg.item()
            total_bg  += LAMBDA_BG * L_bg.item()   # log the WEIGHTED bg contribution (λ·L_bg),
                                                   # so the fg/bg curves are directly comparable
                                                   # as shares of the BP loss (fg + λ·bg)
            pm_np      = per_marker_nmse.detach().float().cpu().numpy()   # (C,) std-normalised per-marker loss (matches BP)
            if per_marker_loss_acc is None:
                per_marker_loss_acc = np.zeros(len(pm_np), dtype=np.float64)
            per_marker_loss_acc += pm_np

            if bio_dict is not None:
                total_bio       += bio_dict["bio_loss"].item()
                bio_batch_count += 1

            p_np = preds.detach().float().cpu().numpy()
            t_np = targets_cpu.numpy()

            if pearson_acc is None:
                pearson_acc  = GroupedOnlinePearson(p_np.shape[1])
                spearman_acc = OnlineSpearman(p_np.shape[1])

            pearson_acc.update(p_np, t_np, slide_ids)
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
    mean_fg             = total_fg  / len(loader)
    mean_bg             = total_bg  / len(loader)
    mean_bio            = total_bio / bio_batch_count if bio_batch_count > 0 else 0.0
    mean_per_marker_mse = per_marker_loss_acc / len(loader)   # (C,) mean std-normalised per-marker loss (matches BP)
    p_per_marker        = pearson_acc.compute()
    s_per_marker        = spearman_acc.compute()
    return mean_mse, mean_bio, mean_per_marker_mse, p_per_marker, s_per_marker, mean_fg, mean_bg


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_curves(train_losses, val_losses, train_pearsons, val_pearsons,
                train_spearmans, val_spearmans, marker_names,
                train_bio_losses=None, val_bio_losses=None,
                train_per_marker_losses=None, val_per_marker_losses=None,
                train_fg_losses=None, val_fg_losses=None,
                train_bg_losses=None, val_bg_losses=None):
    epochs   = range(1, len(train_losses) + 1)
    use_bio  = train_bio_losses is not None
    use_fgbg = train_fg_losses is not None

    ncols = 2 + int(use_bio) + int(use_fgbg)
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))
    i = 0

    axes[i].plot(epochs, train_losses, label="Train")
    axes[i].plot(epochs, val_losses,   label="Val")
    axes[i].set_title("std-normalised MSE loss (BP loss)"); axes[i].legend(); i += 1

    if use_fgbg:
        axes[i].plot(epochs, train_fg_losses, color="tab:blue",   label="fg train")
        axes[i].plot(epochs, val_fg_losses,   color="tab:blue",   linestyle="--", label="fg val")
        axes[i].plot(epochs, train_bg_losses, color="tab:orange", label="λ·bg train")
        axes[i].plot(epochs, val_bg_losses,   color="tab:orange", linestyle="--", label="λ·bg val")
        axes[i].set_title("fg vs λ·bg loss (shares of BP loss = fg + λ·bg)"); axes[i].legend(); i += 1

    if use_bio:
        axes[i].plot(epochs, train_bio_losses, label="Train")
        axes[i].plot(epochs, val_bio_losses,   label="Val")
        axes[i].set_title("Bio Loss"); axes[i].legend(); i += 1

    axes[i].plot(epochs, [p.mean() for p in train_pearsons], label="Train")
    axes[i].plot(epochs, [p.mean() for p in val_pearsons],   label="Val")
    axes[i].set_title("Mean Pearson r"); axes[i].legend()

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

    # Per-marker std-normalised MSE loss grid (the per-marker term that BP averages)
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
        plt.suptitle("Per-marker std-normalised MSE loss (BP loss)", fontsize=10)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "per_marker_loss.png", dpi=150)
        plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)
    from config import save_config
    save_config(CFG, OUTPUT_DIR / "config.yaml")   # resolved config for reproducibility / eval
    sys.stdout = Logger(OUTPUT_DIR / "training_log.txt")

    load_dotenv()
    # UNI2 is cached locally, so login is optional. Under HF_HUB_OFFLINE=1, login()
    # would hit the HF API to validate the token and crash — skip it when offline,
    # and never let an auth hiccup stop a run that needs no downloads.
    offline = os.getenv("HF_HUB_OFFLINE", "0") not in ("0", "", "false", "False")
    if os.getenv("HF_TOKEN") and not offline:
        try:
            login(token=os.getenv("HF_TOKEN"))
        except Exception as e:
            print(f"  (HF login skipped: {e})")

    # ── Slide-level train / val / test split ─────────────────────────────────────
    all_slides = sorted(f.replace('_patch_dataset.h5', '')
                        for f in os.listdir(H5_DIR) if f.endswith('.h5'))

    if SPLIT_MODE == "miphei":
        test_slides  = [s for s in TEST_SLIDES if s in all_slides]
        val_slides   = [s for s in VAL_SLIDES  if s in all_slides]
        train_slides = [s for s in all_slides
                        if s not in TEST_SLIDES and s not in VAL_SLIDES]
    else:
        rng = np.random.default_rng(SEED)
        rng.shuffle(all_slides)
        n_test       = max(1, round(len(all_slides) * TEST_FRAC))
        n_val        = max(1, round(len(all_slides) * VAL_FRAC))
        test_slides  = all_slides[:n_test]
        val_slides   = all_slides[n_test:n_test + n_val]
        train_slides = all_slides[n_test + n_val:]

    print(f"Slides — train: {train_slides}")
    print(f"         val:   {val_slides}")
    print(f"         test:  {test_slides}")

    # Build the frozen-UNI2 CLS cache once (covers all slides), reused across runs.
    cache_dir = None
    if USE_CLS_CACHE:
        from cls_cache import build_cls_cache
        print("\nBuilding / verifying CLS cache…")
        build_cls_cache(str(H5_DIR), str(TIFF_DIR), CLS_CACHE_DIR,
                        model_name=MODEL_NAME, device=str(device))
        cache_dir = CLS_CACHE_DIR

    train_dataset = OrionSpatialDataset(str(H5_DIR), str(TIFF_DIR),
                                        augment=True,  slide_names=train_slides,
                                        use_neighbours=USE_NEIGHBOURS,
                                        cls_cache_dir=cache_dir,
                                        fg_mode=FG_MODE,
                                        return_slide_idx=True)
    val_dataset   = OrionSpatialDataset(str(H5_DIR), str(TIFF_DIR),
                                        augment=False, slide_names=val_slides,
                                        token_means=train_dataset.token_means,
                                        use_neighbours=USE_NEIGHBOURS,
                                        cls_cache_dir=cache_dir,
                                        fg_mode=FG_MODE,
                                        return_slide_idx=True)
    test_dataset  = OrionSpatialDataset(str(H5_DIR), str(TIFF_DIR),
                                        augment=False, slide_names=test_slides,
                                        token_means=train_dataset.token_means,
                                        use_neighbours=USE_NEIGHBOURS,
                                        cls_cache_dir=cache_dir,
                                        fg_mode=FG_MODE,
                                        return_slide_idx=True)
    marker_names  = train_dataset.marker_names

    train_stds = train_dataset.compute_marker_stds().to(device)   # (C,)
    print("Train marker stds (used for MIPHEI loss normalisation):")
    for name, s in zip(marker_names, train_stds.cpu()):
        print(f"  {name:<20s}  std={s:.4f}")


    bs = NEIGHBOUR_BATCH_SIZE if USE_NEIGHBOURS else BATCH_SIZE
    if OVERSAMPLE:
        sample_weights = train_dataset.compute_sampling_weights(
            hard_marker_names=['FOXP3', 'PD-L1', 'CD8a'], cap=3.0)
        print('Sample weights:', sample_weights.max(), sample_weights.min())
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_dataset),
                                        replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=bs, sampler=sampler,
                                  num_workers=NUM_WORKERS, pin_memory=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=bs, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=bs, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    #print(f"Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}| Device: {device}")

    # Encoder fully frozen for phase 1; we unfreeze last N blocks manually at phase 2.
    conch_text_embs = encode_marker_descriptions(marker_names) if USE_CONCH_REFINEMENT else None
    if USE_NEIGHBOURS:
        _NbModel = NeighbourCLSModel if NEIGHBOUR_ARCH == "cls" else NeighbourhoodSpatialModel
        model = _NbModel(MODEL_NAME, num_outputs=NUM_OUTPUTS,
                         token_grid=TOKEN_GRID,
                         unfreeze_last_n=0).to(device)
    else:
        model = SpatialModel(MODEL_NAME, num_outputs=NUM_OUTPUTS,
                             token_grid=TOKEN_GRID,
                             fds_cfg=None,
                             unfreeze_last_n=0,
                             conch_text_embs=conch_text_embs,
                             cls_film=CLS_FILM,
                             lora_last_n=(LORA_LAST_N if FINETUNE_MODE == "lora" else 0),
                             lora_rank=LORA_RANK, lora_alpha=LORA_ALPHA,
                             lora_dropout=LORA_DROPOUT,
                             lora_suffixes=LORA_SUFFIXES,
                             grad_checkpoint=GRAD_CHECKPOINT).to(device)
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
    train_fg_losses, val_fg_losses = [], []
    train_bg_losses, val_bg_losses = [], []
    train_bio_losses, val_bio_losses = ([], []) if bio_loss_fn is not None else (None, None)
    best_val_pearson = -np.inf
    swa_sum: dict = {}      # name → summed trainable tensor (swa_lastk)
    swa_n = 0

    def _finalize_checkpoint():
        """Write best_model.pt for SELECT in {last, swa_lastk}; 'best' already wrote it."""
        if SELECT == "best":
            return
        sd = model.state_dict()                              # last-epoch weights (full)
        if SELECT == "swa_lastk":
            if swa_n == 0:
                print("  ⚠ swa_lastk: no epochs accumulated — saving last epoch instead")
            else:
                for n in swa_sum:
                    sd[n] = (swa_sum[n] / swa_n).to(sd[n].dtype).to(sd[n].device)
                print(f"  → SWA over last {swa_n} epochs saved → best_model.pt")
        if SELECT == "last" or (SELECT == "swa_lastk" and swa_n == 0):
            vm = [float(np.mean(p)) for p in val_pearsons]
            if len(vm) >= 3 and vm[-1] < max(vm[-3:]) - 0.01:
                print(f"  ⚠ last-epoch val r={vm[-1]:.4f} below recent max {max(vm[-3:]):.4f} "
                      f"— val still declining, 'last' may be suboptimal")
            print(f"  → last-epoch model saved → best_model.pt")
        torch.save(sd, OUTPUT_DIR / "best_model.pt")

    def _run_one_epoch(epoch, optimizer, scheduler=None):
        epoch_0idx = epoch - 1
        print(f'Epoch {epoch}...')
        print(f'Training... (per-marker fg loss, fg_mode={FG_MODE})')
        train_mse, train_bio, train_pm_loss, train_p, train_s, train_fg, train_bg = run_epoch(
            model, train_loader, criterion, optimizer,
            scaler=scaler, scheduler=scheduler, epoch=epoch_0idx,
            bio_loss_fn=bio_loss_fn, marker_stds=train_stds,
            use_neighbours=USE_NEIGHBOURS,
        )
        print('Validating...')
        val_mse, val_bio, val_pm_loss, val_p, val_s, val_fg, val_bg = run_epoch(
            model, val_loader, criterion, bio_loss_fn=bio_loss_fn,
            marker_stds=train_stds,
            use_neighbours=USE_NEIGHBOURS,
        )

        train_fg_losses.append(train_fg);            val_fg_losses.append(val_fg)
        train_bg_losses.append(train_bg);            val_bg_losses.append(val_bg)
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
        improved = val_p.mean() > best_val_pearson
        if improved:
            best_val_pearson = val_p.mean()
        # SELECT == "best": save the running best epoch (legacy). last/swa_lastk write
        # best_model.pt once at the end in _finalize_checkpoint(), so skip here.
        if SELECT == "best" and improved:
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            print(f"  → best model saved (val r={best_val_pearson:.4f})")
        # swa_lastk: accumulate the TRAINABLE params of the last `select_window` epochs
        # (frozen base is identical across epochs, so only trainable params need averaging).
        if SELECT == "swa_lastk" and epoch > NUM_EPOCHS - SELECT_WINDOW:
            sd = model.state_dict()
            for n, p in model.named_parameters():
                if p.requires_grad:
                    t = sd[n].detach().float().cpu()
                    swa_sum[n] = t.clone() if n not in swa_sum else swa_sum[n] + t
            nonlocal swa_n
            swa_n += 1
            print(f"  → SWA: accumulated epoch {epoch} ({swa_n} so far)")

        np.save(OUTPUT_DIR / "train_losses.npy",           np.array(train_losses))
        np.save(OUTPUT_DIR / "val_losses.npy",             np.array(val_losses))
        np.save(OUTPUT_DIR / "train_pearsons.npy",         np.stack(train_pearsons))
        np.save(OUTPUT_DIR / "val_pearsons.npy",           np.stack(val_pearsons))
        np.save(OUTPUT_DIR / "train_spearmans.npy",        np.stack(train_spearmans))
        np.save(OUTPUT_DIR / "val_spearmans.npy",          np.stack(val_spearmans))
        np.save(OUTPUT_DIR / "train_per_marker_losses.npy", np.stack(train_per_marker_losses))
        np.save(OUTPUT_DIR / "val_per_marker_losses.npy",   np.stack(val_per_marker_losses))
        np.save(OUTPUT_DIR / "train_fg_losses.npy",        np.array(train_fg_losses))
        np.save(OUTPUT_DIR / "val_fg_losses.npy",          np.array(val_fg_losses))
        np.save(OUTPUT_DIR / "train_bg_losses.npy",        np.array(train_bg_losses))
        np.save(OUTPUT_DIR / "val_bg_losses.npy",          np.array(val_bg_losses))
        np.save(OUTPUT_DIR / "marker_names.npy",           np.array(marker_names))
        if bio_loss_fn is not None:
            np.save(OUTPUT_DIR / "train_bio_losses.npy", np.array(train_bio_losses))
            np.save(OUTPUT_DIR / "val_bio_losses.npy",   np.array(val_bio_losses))
        plot_curves(train_losses, val_losses, train_pearsons, val_pearsons,
                    train_spearmans, val_spearmans, marker_names,
                    train_bio_losses, val_bio_losses,
                    train_per_marker_losses, val_per_marker_losses,
                    train_fg_losses, val_fg_losses, train_bg_losses, val_bg_losses)

    if USE_NEIGHBOURS:
        # ── Single phase: frozen UNI2 + trainable transformer on top ──────────
        # UNI2 is frozen; the trainable "encoder" here is the fresh transformer stack
        # (proj + positional embeds + attention layers) that mixes tokens on top of it.
        # Warm up ONLY that stack over WARMUP_STEPS from epoch 1 (random-init attention +
        # Adam's noisy early second-moment want warmup); the decoder head trains at a
        # constant LR — mirroring the no-neighbour encoder-warmup / head-constant split.
        # It's fresh (not pretrained) → full LR, no 0.2× discriminative scale.
        head_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and n.startswith("head")]
        tfm_params  = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith("head")]
        const       = lambda step: 1.0
        warmup_flat = lambda step: min(step / max(WARMUP_STEPS, 1), 1.0)   # 0→1 then flat
        optimizer   = torch.optim.AdamW(
            [{'params': head_params, 'lr': LR},
             {'params': tfm_params,  'lr': LR}], weight_decay=WEIGHT_DECAY)
        scheduler   = torch.optim.lr_scheduler.LambdaLR(optimizer, [const, warmup_flat])
        n_train     = sum(p.numel() for p in head_params + tfm_params)
        print(f"\n── Single-phase AdamW (UNI2 frozen): {n_train:,} trainable params · "
              f"transformer warmup {WARMUP_STEPS} → constant LR {LR:g} (head constant) ──")
        for epoch in range(1, NUM_EPOCHS + 1):
            _run_one_epoch(epoch, optimizer, scheduler=scheduler)

    else:
        # ── Phase 1: head only ────────────────────────────────────────────────
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer_p1 = torch.optim.AdamW(head_params, lr=LR, weight_decay=WEIGHT_DECAY)
        print(f"\n── Phase 1: head only ({PHASE1_EPOCHS} epochs) ──")
        for epoch in range(1, PHASE1_EPOCHS + 1):
            _run_one_epoch(epoch, optimizer_p1)

        # ── Phase 2: activate encoder adaptation (unfreeze blocks OR LoRA adapters) ──
        # NOTE: the unfreeze guard is required — blocks[-0:] would unfreeze the WHOLE encoder.
        if FINETUNE_MODE == "unfreeze" and UNFREEZE_LAST_N > 0:
            print(f"\n── Phase 2: unfreezing last {UNFREEZE_LAST_N} blocks ──")
            for block in model.encoder.blocks[-UNFREEZE_LAST_N:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in model.encoder.norm.parameters():
                p.requires_grad = True
        elif FINETUNE_MODE == "lora":
            n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
            print(f"\n── Phase 2: activating LoRA adapters (last {LORA_LAST_N} blocks, "
                  f"{n_lora:,} params) + final norm ──")
            for n, p in model.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True
            for p in model.encoder.norm.parameters():
                p.requires_grad = True
            # also train the LayerNorms (norm1/norm2) + LayerScales (ls1/ls2) of the
            # LoRA-adapted blocks — LoRA can't change how much a block contributes
            # (ls gates the residual), so unfreezing these removes that bottleneck.
            if LORA_TRAIN_NORMS:
                n_nrm = 0
                for block in model.encoder.blocks[-LORA_LAST_N:]:
                    for nm, p in block.named_parameters():
                        if nm.split(".")[0] in ("norm1", "norm2", "ls1", "ls2"):
                            p.requires_grad = True
                            n_nrm += p.numel()
                print(f"  + training norms/scales of last {LORA_LAST_N} blocks ({n_nrm:,} params)")
        else:
            print(f"\n── Phase 2: UNI2 fully frozen (training head only) ──")

        _conch_names   = {'sim_scale', 'sim_bias'}
        conch_params   = [p for n, p in model.named_parameters()
                          if p.requires_grad and n in _conch_names]
        head_params    = [p for n, p in model.named_parameters()
                          if p.requires_grad and 'encoder' not in n and n not in _conch_names]
        encoder_params = [p for n, p in model.named_parameters()
                          if p.requires_grad and 'encoder' in n]

        # Discriminative LR: LoRA adapters start at Δ=0 and are small → full LR; full
        # block-unfreeze uses 0.2× to protect pretrained weights. The ENCODER group is
        # warmed up linearly over WARMUP_STEPS from the start of phase 2 (its first time
        # training) then held constant; head/conch train at constant LR (no warmup) —
        # the same warmup-the-encoder / head-constant split as the neighbour branch.
        enc_lr_scale = LORA_LR_SCALE if FINETUNE_MODE == "lora" else 0.2
        # structured schedule: warmup→cosine decay over the phase-2 optimizer steps.
        # head decays from step 0 (no warmup); encoder warms up then decays together.
        total_p2 = max(len(train_loader) * (NUM_EPOCHS - PHASE1_EPOCHS), 1)
        def _cosine(step, warmup):
            if step < warmup:
                return step / max(warmup, 1)
            prog = (step - warmup) / max(total_p2 - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
        cosine        = lambda step: _cosine(step, 0)             # head: cosine, no warmup
        warmup_cosine = lambda step: _cosine(step, WARMUP_STEPS)  # encoder: warmup→cosine
        const         = lambda step: 1.0
        # weight decay ONLY on 2-D weight matrices (Linear, LoRA A/B). 1-D params —
        # LayerNorm weight/bias, LayerScale γ, all biases — get WD=0 (decaying γ→0 would
        # close the residual gates we unfroze; decaying norms/biases is an anti-pattern).
        param_groups, schedules = [], []
        def _add(params, lr, sched):
            decay    = [p for p in params if p.ndim >= 2]
            no_decay = [p for p in params if p.ndim < 2]
            if decay:
                param_groups.append({'params': decay, 'lr': lr, 'weight_decay': WEIGHT_DECAY})
                schedules.append(sched)
            if no_decay:
                param_groups.append({'params': no_decay, 'lr': lr, 'weight_decay': 0.0})
                schedules.append(sched)
        _add(head_params, LR, cosine)
        if encoder_params:                                    # empty in "none" mode
            _add(encoder_params, LR * enc_lr_scale, warmup_cosine)  # warmup ONLY the encoder
        if conch_params:
            _add(conch_params, LR * 5, const)
        optimizer_p2 = torch.optim.AdamW(param_groups, weight_decay=0.0)  # per-group WD set above
        scheduler_p2 = torch.optim.lr_scheduler.LambdaLR(optimizer_p2, schedules)

        for epoch in range(PHASE1_EPOCHS + 1, NUM_EPOCHS + 1):
            _run_one_epoch(epoch, optimizer_p2, scheduler=scheduler_p2)

    _finalize_checkpoint()
    print(f"\nDone. Best val Pearson: {best_val_pearson:.4f}  (select={SELECT})")

    # ── Test evaluation (best checkpoint) ─────────────────────────────────────
    """
    print("\n── Test set evaluation ──")
    best_state = torch.load(OUTPUT_DIR / "best_model.pt", map_location=device)
    model.load_state_dict(best_state, strict=False)
    test_mse, _, _, test_p, test_s, _, _ = run_epoch(
        model, test_loader, criterion, marker_stds=train_stds,
        use_neighbours=USE_NEIGHBOURS, bg_weight=BG_WEIGHT, bg_weights=bg_weights)
    print(f"Test MSE {test_mse:.4f} | mean Pearson r {test_p.mean():.4f} | mean Spearman ρ {test_s.mean():.4f}")
    names = marker_names or [f"M{j}" for j in range(len(test_p))]
    for name, p in zip(names, test_p):
        print(f"    {name:<20s}  pearson {p:.4f}")
    np.save(OUTPUT_DIR / "test_pearsons.npy", test_p)
    np.save(OUTPUT_DIR / "test_slides.npy",   np.array(test_slides))
    """


if __name__ == "__main__":
    train()