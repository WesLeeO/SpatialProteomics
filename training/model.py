import os
# Must be set before any huggingface_hub import resolves — prevents all network HEAD checks.
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm import create_model
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer


# ── Feature Distribution Smoothing ────────────────────────────────────────────

def _calibrate_mean_var(matrix, m1, v1, m2, v2, clip_min=0.1, clip_max=10.0):
    """Shift features from old distribution (m1, v1) to smoothed distribution (m2, v2)."""
    if v1.sum() < 1e-10:
        return matrix
    v1p = v1.clamp(min=0.)
    v2p = v2.clamp(min=0.)
    if (v1p == 0.).any():
        valid = v1p != 0.
        if valid.any():
            factor = (v2p[valid] / v1p[valid]).clamp(clip_min, clip_max)
            matrix[:, valid] = (matrix[:, valid] - m1[valid]) * factor.sqrt() + m2[valid]
        return matrix
    factor = (v2p / v1p).clamp(clip_min, clip_max)
    return (matrix - m1) * factor.sqrt() + m2


class FDS(nn.Module):
    """
    Feature Distribution Smoothing for one output marker.

    Maintains per-bucket running statistics (mean, var) over the 128-dim
    penultimate features, smooths them across adjacent label buckets with a
    1-D Gaussian convolution, then calibrates each token's features to the
    smoothed distribution during training.

    Labels are assumed to be in [0, 1].  50 uniform bins → bucket width 0.02.
    """
    def __init__(self, feature_dim, bucket_num=50, bucket_start=0,
                 start_update=0, start_smooth=1,
                 kernel='gaussian', ks=5, sigma=2, momentum=0.9):
        super().__init__()
        self.feature_dim  = feature_dim
        self.bucket_num   = bucket_num
        self.bucket_start = bucket_start
        self.half_ks      = (ks - 1) // 2
        self.momentum     = momentum
        self.start_update = start_update
        self.start_smooth = start_smooth

        n = bucket_num - bucket_start
        self.register_buffer('epoch',                    torch.zeros(1, dtype=torch.long).fill_(start_update))
        self.register_buffer('running_mean',             torch.zeros(n, feature_dim))
        self.register_buffer('running_var',              torch.ones(n, feature_dim))
        self.register_buffer('running_mean_last_epoch',  torch.zeros(n, feature_dim))
        self.register_buffer('running_var_last_epoch',   torch.ones(n, feature_dim))
        self.register_buffer('smoothed_mean_last_epoch', torch.zeros(n, feature_dim))
        self.register_buffer('smoothed_var_last_epoch',  torch.ones(n, feature_dim))
        self.register_buffer('num_samples_tracked',      torch.zeros(n))
        self.register_buffer('kernel_window',            self._make_kernel(kernel, ks, sigma))

    @staticmethod
    def _make_kernel(kernel, ks, sigma):
        # ouput (ks, )
        assert kernel in ('gaussian', 'triang', 'laplace')
        half = (ks - 1) // 2
        x = torch.arange(-half, half + 1).float()
        if kernel == 'gaussian':
            w = torch.exp(-0.5 * (x / sigma) ** 2)
        elif kernel == 'triang':
            w = (half + 1 - x.abs()) / (half + 1)
        else:
            w = torch.exp(-x.abs() / sigma)
        return (w / w.sum()).float()

    def _bucket_idx(self, labels_np: np.ndarray) -> np.ndarray:
        """Map labels in [0,1] to bucket indices."""
        norm = np.clip(labels_np.astype(np.float32), 0., 1.)
        idx  = np.clip((norm * self.bucket_num).astype(np.int64),
                       self.bucket_start, self.bucket_num - 1)
        return idx

    def _smooth_stats(self):
        """1-D convolution across bucket axis for mean and var."""
        w = self.kernel_window.view(1, 1, -1)
        for src, dst in [(self.running_mean_last_epoch, self.smoothed_mean_last_epoch),
                         (self.running_var_last_epoch,  self.smoothed_var_last_epoch)]:
            # src: (n_buckets, D) → treat D as batch, n_buckets as sequence
            x = src.unsqueeze(1).permute(2, 1, 0)              # (D, 1, n_buckets)
            x = F.pad(x, (self.half_ks, self.half_ks), mode='reflect')
            dst.copy_(F.conv1d(x, w).permute(2, 1, 0).squeeze(1))

    def update_last_epoch_stats(self, epoch: int):
        """Call once after each training epoch to snapshot and smooth running stats."""
        if epoch == int(self.epoch.item()) + 1:
            self.epoch += 1
            self.running_mean_last_epoch.copy_(self.running_mean)
            self.running_var_last_epoch.copy_(self.running_var)
            self._smooth_stats()

    @torch.no_grad()
    def update_running_stats_from_moments(self, count, feat_sum, feat_sumsq, epoch: int):
        """
        Update running (mean, var) from pre-accumulated per-bucket moments.

        Args:
            count:      (bucket_num,)           — number of samples per bucket
            feat_sum:   (bucket_num, feature_dim)
            feat_sumsq: (bucket_num, feature_dim)
            epoch:      0-indexed current epoch
        """
        if epoch < int(self.epoch.item()):
            return
        b0, b1 = self.bucket_start, self.bucket_num
        count     = count.long()
        feat_sum  = feat_sum.float()
        feat_sumsq = feat_sumsq.float()
        dev = self.running_mean.device

        for b in range(b0, b1):
            n = int(count[b].item())
            if n <= 0:
                continue
            curr_mean = feat_sum[b] / n
            if n > 1:
                curr_var = ((feat_sumsq[b] - feat_sum[b] ** 2 / n) / (n - 1)).clamp(min=0.)
            else:
                curr_var = torch.zeros_like(curr_mean)
            self.num_samples_tracked[b - b0] += n
            factor = 0.0 if epoch == self.start_update else self.momentum
            self.running_mean[b - b0] = (1 - factor) * curr_mean.to(dev) + factor * self.running_mean[b - b0]
            self.running_var[b - b0]  = (1 - factor) * curr_var.to(dev)  + factor * self.running_var[b - b0]

        # Interpolate empty buckets that have never been seen
        present = {int(b) for b in range(b0, b1) if count[b] > 0}
        for b in range(b0, b1):
            if b in present or self.num_samples_tracked[b - b0] > 0:
                continue
            if b == b0:
                self.running_mean[0] = self.running_mean[1]
                self.running_var[0]  = self.running_var[1]
            elif b == b1 - 1:
                self.running_mean[b - b0] = self.running_mean[b - b0 - 1]
                self.running_var[b - b0]  = self.running_var[b - b0 - 1]
            else:
                self.running_mean[b - b0] = (self.running_mean[b - b0 - 1] + self.running_mean[b - b0 + 1]) / 2.
                self.running_var[b - b0]  = (self.running_var[b - b0 - 1] + self.running_var[b - b0 + 1]) / 2.

    def smooth(self, features: torch.Tensor, labels_np: np.ndarray, epoch: int) -> torch.Tensor:
        """
        Calibrate features to the smoothed distribution.

        Args:
            features:  (N, feature_dim) — penultimate layer activations (will be cloned)
            labels_np: (N,) numpy array in [0, 1]
            epoch:     0-indexed epoch
        Returns:
            Calibrated features, same shape and device as input.
        """
        if epoch < self.start_smooth:
            return features
        orig_dtype = features.dtype
        feat = features.float()
        buckets = self._bucket_idx(labels_np)
        for b in np.unique(buckets):
            mask = torch.from_numpy((buckets == b).astype(bool)).to(feat.device)
            bi = b - self.bucket_start
            feat[mask] = _calibrate_mean_var(
                feat[mask],
                self.running_mean_last_epoch[bi],
                self.running_var_last_epoch[bi],
                self.smoothed_mean_last_epoch[bi],
                self.smoothed_var_last_epoch[bi],
            )
        return feat.to(orig_dtype)

# model image -> 1536
class Model(nn.Module):
    def __init__(self, base_model_name, hidden_dim, num_outputs, freeze=True):
        super().__init__()
        self.base_model_name = base_model_name
        self.encoder = self.foundation_model(base_model_name)
        self.hidden_dim = hidden_dim
        self.regression_head = nn.Sequential(
                nn.Linear(hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(128, num_outputs)
            )
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

    def forward(self, image):
        # image should have shape (batch_size, 3, 224, 224)
        # output should be (batch_size, num_outputs)
        assert(self.hidden_dim == 1536)
        if self.base_model_name == 'UNI2':
            return self.regression_head(self.encoder(image))
        elif self.base_model_name == 'H0-mini':
            output = self.encoder(image) # (batch_size, 261, 768)
            cls_features = output[:, 0] # (batch_size, 768)
            patch_token_features = output[:, self.encoder.num_prefix_tokens :] # Patch token features (batch_size, 256, 768):
            concatenated_features = torch.cat(
                [cls_features, patch_token_features.mean(1)], dim=-1
            ) # (batch_size, 1536)
            return self.regression_head(concatenated_features)

    def foundation_model(self, base_model_name):
        return foundation_model(base_model_name)


class CLSFiLM(nn.Module):
    """
    Inject patch-level context into the per-token features via FiLM.

    The CLS token (B, D) — UNI2's global summary of the whole patch — is mapped to
    a per-channel scale (gamma) and shift (beta), which modulate every token's
    128-dim features:  h ← gamma ⊙ h + beta.  All 256 tokens of a patch share that
    patch's (gamma, beta), so the CLS re-weights features by tissue context instead
    of adding a constant (as concatenation would).

    Zero-initialised → starts at gamma=1, beta=0 (identity); FiLM only grows as it
    proves useful, so adding it can't hurt early training.
    """
    def __init__(self, in_dim: int, feat_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * feat_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, cls: torch.Tensor, tokens_per_patch: int) -> torch.Tensor:
        gb          = self.net(cls)                                   # (B, 2*feat_dim)
        gamma, beta = gb[:, :self.feat_dim], gb[:, self.feat_dim:]    # each (B, feat_dim)
        gamma = (1.0 + gamma).repeat_interleave(tokens_per_patch, dim=0)   # (N, feat_dim)
        beta  =         beta.repeat_interleave(tokens_per_patch, dim=0)
        return gamma * h + beta


def inject_lora(encoder, last_n: int, rank: int, alpha: float, dropout: float,
                suffixes=("attn.qkv", "attn.proj")) -> int:
    """
    Inject LoRA adapters (via peft) into the `suffixes` Linear sub-modules of the
    last `last_n` transformer blocks of `encoder`, in place — the encoder keeps its
    original interface (forward_features still works). The wrapped base weights stay
    frozen. Returns the number of adapter parameters added.

    Explicit fully-qualified target names (blocks.{i}.attn.qkv …) are used rather
    than peft's layers_to_transform/layers_pattern, which is brittle on this ViT.
    """

    """
    last 4 layers
    - qkv adapter changes what the attention attends to and reads — it perturbs Q, K (the attention pattern / which tokens look at which) and V (what content is
    pulled).
    - proj adapter changes how the gathered per-head outputs are recombined before they re-enter the residual stream — it re-mixes/re-weights the head outputs without
    touching the attention pattern itself.
    """
    
    from peft import LoraConfig, inject_adapter_in_model
    depth   = len(encoder.blocks)
    layers  = range(max(0, depth - last_n), depth)   # clamp so last_n>=depth = all blocks
    targets = [f"blocks.{i}.{s}" for i in layers for s in suffixes]
    cfg     = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout,
                         target_modules=targets, bias="none")
    inject_adapter_in_model(cfg, encoder)
    return sum(p.numel() for nm, p in encoder.named_parameters() if "lora_" in nm)


def enable_grad_checkpointing(encoder) -> None:
    """
    Turn on activation checkpointing for the timm ViT blocks to cut backward-pass
    memory (recompute block activations instead of storing all of them) — needed
    when adapters live in every block, so gradients flow back to block 0 and the
    full forward activation stack would otherwise be retained.

    Forces use_reentrant=False: the encoder is frozen, so the input tensor does NOT
    require grad, and reentrant checkpointing would then drop gradients to the LoRA
    adapters entirely (verified). Non-reentrant tracks grad via the params instead.
    """
    import functools
    import timm.models.vision_transformer as _vit
    from timm.models._manipulate import checkpoint_seq as _cs
    _vit.checkpoint_seq = functools.partial(_cs, use_reentrant=False)
    encoder.set_grad_checkpointing(True)


class SpatialModel(nn.Module):
    """
    Predicts a (C, token_grid, token_grid) expression map per patch.

    UNI2 with patch_size=14 on 224×224 produces 224/14 = 16 tokens per side
    (256 patch tokens total).  A shared MLP is applied to each token independently
    to predict expression for the corresponding spatial cell.

    When fds_cfg is provided, FDS is applied at the 128-dim penultimate layer.
    forward() returns (preds, h) where h is the raw (pre-smoothing) penultimate
    features needed to accumulate FDS statistics in the training loop.

    Two mutually-exclusive ways to adapt UNI2 (the rest stays frozen):
      • unfreeze_last_n > 0 : fine-tune the last N transformer blocks + final norm
        (full weights — ~28M params/block).
      • lora_last_n   > 0 : inject LoRA adapters (peft) into the last N blocks'
        attention projections (low-rank — ~0.6M params at r=16). Base weights frozen.
    With LoRA, the adapter params are set frozen at construction so a head-only
    warmup phase stays head-only; the training loop activates them (requires_grad
    = True) when it reaches the encoder-tuning phase — parallel to unfreezing blocks.
    """
    def __init__(self, base_model_name: str, num_outputs: int,
                 token_grid: int = 16, freeze: bool = True,
                 fds_cfg: dict = None, unfreeze_last_n: int = 0,
                 conch_text_embs: torch.Tensor | None = None,
                 cls_film: bool = False,
                 lora_last_n: int = 0, lora_rank: int = 16,
                 lora_alpha: float = 32.0, lora_dropout: float = 0.0,
                 lora_suffixes: tuple = ("attn.qkv", "attn.proj"),
                 grad_checkpoint: bool = False):
        super().__init__()
        self.token_grid  = token_grid
        self.num_outputs = num_outputs
        self.encoder     = foundation_model(base_model_name)
        embed_dim        = self.encoder.embed_dim

        self.feature_extractor = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(p=0.2),
        )

        self.predictor = nn.Linear(128, num_outputs)
        self.predictors = nn.ModuleList([nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)) for _ in range(num_outputs)])

        if fds_cfg is not None:
            self.fds = nn.ModuleList([
                FDS(feature_dim=128, **fds_cfg) for _ in range(num_outputs)
            ])
        else:
            self.fds = None

        for p in self.encoder.parameters():
            p.requires_grad = False
        if unfreeze_last_n > 0:
            assert lora_last_n == 0, "use either block-unfreeze or LoRA, not both"
            for block in self.encoder.blocks[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.encoder.norm.parameters():
                p.requires_grad = True
            trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
            print(f"Encoder: last {unfreeze_last_n} blocks unfrozen ({trainable:,} trainable params)")
        elif lora_last_n > 0:
            n_lora = inject_lora(self.encoder, lora_last_n, lora_rank,
                                 lora_alpha, lora_dropout, suffixes=lora_suffixes)
            # peft marks adapters trainable; re-freeze so a head-only warmup phase
            # stays head-only. The training loop activates them at the encoder phase.
            for nm, p in self.encoder.named_parameters():
                if "lora_" in nm:
                    p.requires_grad = False
            print(f"Encoder: LoRA r={lora_rank} (α={lora_alpha:g}) on last "
                  f"{lora_last_n} blocks · {n_lora:,} adapter params "
                  f"(frozen until encoder phase)")

        if grad_checkpoint:
            enable_grad_checkpointing(self.encoder)
            print("Encoder: gradient checkpointing ON (use_reentrant=False)")

    def forward(self, image: torch.Tensor,
                labels: torch.Tensor = None,
                epoch: int = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image:            (B, 3, 224, 224)
            labels:           (B, C, G, G) — IF targets, only used with FDS
            epoch:            0-indexed epoch, only used with FDS
        Returns:
            preds: (B, C, G, G)
            h:     (B*G*G, 128) penultimate features (detached, for FDS)
        """
        features     = self.encoder.forward_features(image)           # (B, T, D)
        patch_tokens = features[:, self.encoder.num_prefix_tokens:]   # (B, G*G, D)
        assert patch_tokens.shape[1] == 256
        B, NUM_TOKENS, D = patch_tokens.shape
        G = self.token_grid
        N = B * NUM_TOKENS

        h_tokens = patch_tokens.reshape(N, D)                         # (N, D)
        h = self.feature_extractor(h_tokens)                          # (N, 128)

        use_fds = (self.training and self.fds is not None
                   and labels is not None and epoch is not None)

        if use_fds:
            labels_flat = labels.permute(0, 2, 3, 1).reshape(N, self.num_outputs)
            pred_cols = []
            for j, fds_j in enumerate(self.fds):
                labels_np = labels_flat[:, j].detach().cpu().numpy()
                h_j = fds_j.smooth(h.clone(), labels_np, epoch)
                pred_cols.append(self.predictors[j](h_j))             # (N, 1)
            preds = torch.cat(pred_cols, dim=1)                        # (N, C)
        else:
            preds = self.predictor(h)                                  # (N, C)

        preds_spatial = preds.reshape(B, G, G, self.num_outputs).permute(0, 3, 1, 2)  # (B,C,G,G)
        return preds_spatial, h.detach()


class NeighbourhoodSpatialModel(nn.Module):
    """
    Token-level expression prediction with explicit neighbour-patch context.

    The center patch and its 8 grid-neighbours are each encoded independently at
    native 224 (→ 16×16 UNI2 tokens, in-distribution).  The nine 16×16 token maps
    are assembled into their true 3×3 spatial layout — a 48×48 token grid — and a
    Transformer is run over the whole neighbourhood, so every center token can
    attend, directionally and selectively, to the surrounding tissue.  The center
    16×16 block is then read out and decoded to a (C, 16, 16) expression map.

    The 9 CLS tokens (one per patch) are appended as extra global-summary tokens.
    Missing neighbours (tissue edges) are masked out of attention.

    forward returns (preds, None) — the second slot keeps the training-loop API
    identical to SpatialModel (no FDS here).
    """
    # 3×3 block (row, col) for each of the 8 neighbour deltas, in the dataset's
    # canonical order: deltas = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    # with (dx, dy) → block (row = dy+1, col = dx+1); center is (1, 1).
    _NEIGHBOUR_BLOCK = [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)]

    def __init__(self, base_model_name: str, num_outputs: int,
                 token_grid: int = 16, d_model: int = 512,
                 n_layers: int = 4, n_heads: int = 8,
                 unfreeze_last_n: int = 0, detach_neighbours: bool = True,
                 use_cls: bool = True):
        super().__init__()
        self.token_grid        = token_grid
        self.num_outputs       = num_outputs
        self.detach_neighbours = detach_neighbours
        self.use_cls           = use_cls
        self.fds               = None          # training-loop compatibility
        self.encoder           = foundation_model(base_model_name)
        self.embed_dim         = self.encoder.embed_dim
        self.grid              = 3 * token_grid    # 48

        self.proj     = nn.Linear(self.embed_dim, d_model)
        self.grid_pos = nn.Parameter(torch.zeros(1, self.grid * self.grid, d_model))
        nn.init.trunc_normal_(self.grid_pos, std=0.02)
        if use_cls:
            self.cls_pos = nn.Parameter(torch.zeros(1, 9, d_model))   # per-patch id for the 9 CLS
            nn.init.trunc_normal_(self.cls_pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=0.1,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

        self.head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),     nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_outputs),
        )

        for p in self.encoder.parameters():
            p.requires_grad = False
        if unfreeze_last_n > 0:
            for block in self.encoder.blocks[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.encoder.norm.parameters():
                p.requires_grad = True
            trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
            print(f"Encoder: last {unfreeze_last_n} blocks unfrozen ({trainable:,} trainable params)")

    def _encode(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats  = self.encoder.forward_features(imgs)              # (M, prefix+256, D)
        prefix = self.encoder.num_prefix_tokens
        return feats[:, 0], feats[:, prefix:]                     # cls (M, D), tokens (M, 256, D)

    def forward(self, image: torch.Tensor,
                neighbours: torch.Tensor = None,
                neighbour_present: torch.Tensor = None,
                **kwargs) -> tuple[torch.Tensor, None]:
        """
        image:             (B, 3, 224, 224)             — center patch
        neighbours:        (B, 8, 3, 224, 224)          — grid-neighbours (zeros where missing)
        neighbour_present: (B, 8) bool                  — True where a neighbour exists
        """
        B   = image.shape[0]
        G   = self.token_grid
        D   = self.embed_dim
        dev = image.device

        cls_c, tok_c = self._encode(image)                        # (B, D), (B, 256, D)

        K = 0 if neighbours is None else neighbours.shape[1]
        if K:
            flat = neighbours.reshape(B * K, *neighbours.shape[2:])
            if self.detach_neighbours:
                with torch.no_grad():
                    cls_n, tok_n = self._encode(flat)
                cls_n, tok_n = cls_n.detach(), tok_n.detach()
            else:
                cls_n, tok_n = self._encode(flat)
            cls_n = cls_n.reshape(B, K, D)
            tok_n = tok_n.reshape(B, K, G * G, D)

        # ── Assemble the 48×48 neighbourhood token grid ─────────────────────────
        grid  = torch.zeros(B, self.grid, self.grid, D, device=dev, dtype=tok_c.dtype)
        valid = torch.zeros(B, self.grid, self.grid, device=dev, dtype=torch.bool)
        grid[:,  G:2*G, G:2*G] = tok_c.reshape(B, G, G, D)        # center block (1,1)
        valid[:, G:2*G, G:2*G] = True
        if K:
            for k, (br, bc) in enumerate(self._NEIGHBOUR_BLOCK):
                grid[:,  br*G:(br+1)*G, bc*G:(bc+1)*G] = tok_n[:, k].reshape(B, G, G, D)
                valid[:, br*G:(br+1)*G, bc*G:(bc+1)*G] = neighbour_present[:, k][:, None, None]

        seq       = self.proj(grid.reshape(B, self.grid * self.grid, D)) + self.grid_pos
        seq_valid = valid.reshape(B, self.grid * self.grid)

        # ── Append the 9 CLS tokens as global-summary tokens ────────────────────
        if self.use_cls:
            cls_stack = cls_c[:, None] if not K else torch.cat([cls_c[:, None], cls_n], dim=1)  # (B,1+K,D)
            cls_x     = self.proj(cls_stack) + self.cls_pos[:, :cls_stack.shape[1]]
            seq       = torch.cat([seq, cls_x], dim=1)
            cls_valid = torch.ones(B, 1, device=dev, dtype=torch.bool)
            if K:
                cls_valid = torch.cat([cls_valid, neighbour_present], dim=1)
            seq_valid = torch.cat([seq_valid, cls_valid], dim=1)

        seq = self.transformer(seq, src_key_padding_mask=~seq_valid)

        # ── Read out the center block and decode ────────────────────────────────
        x_grid = seq[:, :self.grid * self.grid].reshape(B, self.grid, self.grid, -1)
        center = x_grid[:, G:2*G, G:2*G]                          # (B, G, G, d_model)
        preds  = self.head(center).permute(0, 3, 1, 2)           # (B, C, G, G)
        return preds, None


class NeighbourCLSModel(nn.Module):
    """
    Neighbour context via self-attention over a 256 + 9 = 265-token sequence:

        [ 256 center patch tokens | 9 CLS tokens (center + 8 neighbours) ]

    A Transformer self-attends over these 265 tokens, so each center token sees its
    in-patch neighbours AND the 9 patch summaries.  Neighbours contribute ONLY their
    CLS token (cheap: 265 tokens, not the full 48×48 grid's ~2300).  The 256 center
    tokens are read out and decoded per-token to a (C, 16, 16) map.

    The 256 center tokens + the center CLS are always valid keys; absent neighbours
    (edge patches) are masked, so there are never all-masked attention rows.

    forward returns (preds, None) — same training-loop API as the other models.
    """
    def __init__(self, base_model_name: str, num_outputs: int,
                 token_grid: int = 16, d_model: int = 512,
                 n_layers: int = 4, n_heads: int = 8,
                 unfreeze_last_n: int = 0, detach_neighbours: bool = True):
        super().__init__()
        self.token_grid        = token_grid
        self.num_outputs       = num_outputs
        self.detach_neighbours = detach_neighbours
        self.fds               = None
        self.encoder           = foundation_model(base_model_name)
        self.embed_dim         = self.encoder.embed_dim
        self.n_center          = token_grid * token_grid           # 256

        self.proj    = nn.Linear(self.embed_dim, d_model)          # tokens & CLS share the projection
        self.tok_pos = nn.Parameter(torch.zeros(1, self.n_center, d_model))   # center 16×16 grid pos
        self.cls_pos = nn.Parameter(torch.zeros(1, 9, d_model))    # id: 0 = center, 1-8 = neighbour dirs
        nn.init.trunc_normal_(self.tok_pos, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=0.1,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

        self.head = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),     nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_outputs),
        )

        for p in self.encoder.parameters():
            p.requires_grad = False
        if unfreeze_last_n > 0:
            for block in self.encoder.blocks[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.encoder.norm.parameters():
                p.requires_grad = True

    def _encode(self, imgs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats  = self.encoder.forward_features(imgs)
        prefix = self.encoder.num_prefix_tokens
        return feats[:, 0], feats[:, prefix:]                      # cls, tokens

    def forward(self, image: torch.Tensor,
                neighbours: torch.Tensor = None,
                neighbour_present: torch.Tensor = None,
                neighbour_cls: torch.Tensor = None,
                **kwargs) -> tuple[torch.Tensor, None]:
        """
        neighbour_cls : (B, K, D) precomputed neighbour CLS (from cache) — preferred.
        neighbours    : (B, K, 3, 224, 224) raw neighbour patches — fallback, encoded
                        on the fly (only present ones).
        """
        B   = image.shape[0]
        G   = self.token_grid
        D   = self.embed_dim
        dev = image.device

        cls_c, tok_c = self._encode(image)                         # (B, D), (B, 256, D)

        K = 0
        if neighbour_cls is not None:                              # cached path — no neighbour encode
            cls_n = neighbour_cls.to(cls_c.dtype)                  # (B, K, D)
            K     = cls_n.shape[1]
        elif neighbours is not None:                               # on-the-fly fallback
            K            = neighbours.shape[1]
            flat         = neighbours.reshape(B * K, *neighbours.shape[2:])
            present_flat = neighbour_present.reshape(B * K)
            if present_flat.any():                                 # encode only existing neighbours
                if self.detach_neighbours:
                    with torch.no_grad():
                        c, _ = self._encode(flat[present_flat])
                    c = c.detach()
                else:
                    c, _ = self._encode(flat[present_flat])
                cls_n = torch.zeros(B * K, D, device=dev, dtype=c.dtype)
                cls_n[present_flat] = c
            else:
                cls_n = torch.zeros(B * K, D, device=dev, dtype=cls_c.dtype)
            cls_n = cls_n.reshape(B, K, D)                         # (B, K, D)

        tok       = self.proj(tok_c) + self.tok_pos               # (B, 256, d)
        cls_stack = cls_c[:, None] if not K else torch.cat([cls_c[:, None], cls_n], dim=1)  # (B, 1+K, D)
        cls       = self.proj(cls_stack) + self.cls_pos[:, :cls_stack.shape[1]]             # (B, 1+K, d)
        seq       = torch.cat([tok, cls], dim=1)                  # (B, 256 + 1 + K, d)

        valid = torch.ones(B, self.n_center + 1, device=dev, dtype=torch.bool)   # tokens + center CLS
        if K:
            valid = torch.cat([valid, neighbour_present], dim=1)
        seq = self.transformer(seq, src_key_padding_mask=~valid)

        center = seq[:, :self.n_center]                           # (B, 256, d)
        preds  = self.head(center).reshape(B, G, G, self.num_outputs).permute(0, 3, 1, 2)
        return preds, None


def foundation_model(base_model_name):
    match base_model_name:
        case "UNI2":
            timm_kwargs = {
                'img_size': 224,
                'patch_size': 14,
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5,
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0,
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked,
                'act_layer': torch.nn.SiLU,
                'reg_tokens': 8,
                'dynamic_img_size': True,
            }
            return timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)

        case "H0-mini":
            timm_kwargs = {
                'mlp_layer': timm.layers.SwiGLUPacked,
                'act_layer': torch.nn.SiLU,
            }
            return timm.create_model("hf_hub:bioptimus/H0-mini", pretrained=True, **timm_kwargs)

        case "PubMedBert":
            return SentenceTransformer("NeuML/pubmedbert-base-embeddings")

        case _:
            raise ValueError('unknown model')