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


class SpatialModel(nn.Module):
    """
    Predicts a (C, token_grid, token_grid) expression map per patch.

    UNI2 with patch_size=14 on 224×224 produces 224/14 = 16 tokens per side
    (256 patch tokens total).  A shared MLP is applied to each token independently
    to predict expression for the corresponding spatial cell.

    When fds_cfg is provided, FDS is applied at the 128-dim penultimate layer.
    forward() returns (preds, h) where h is the raw (pre-smoothing) penultimate
    features needed to accumulate FDS statistics in the training loop.

    When unfreeze_last_n > 0, the last N transformer blocks and the final norm
    are fine-tuned; the rest of the encoder stays frozen.
    """
    def __init__(self, base_model_name: str, num_outputs: int,
                 token_grid: int = 16, freeze: bool = True,
                 fds_cfg: dict = None, unfreeze_last_n: int = 0,
                 conch_text_embs: torch.Tensor | None = None):
        super().__init__()
        self.token_grid  = token_grid
        self.num_outputs = num_outputs
        self.encoder     = foundation_model(base_model_name)
        embed_dim        = self.encoder.embed_dim

        # Split into feature extractor (→ 128) and final predictor (128 → C)
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
            for block in self.encoder.blocks[-unfreeze_last_n:]:
                for p in block.parameters():
                    p.requires_grad = True
            for p in self.encoder.norm.parameters():
                p.requires_grad = True
            trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
            print(f"Encoder: last {unfreeze_last_n} blocks unfrozen ({trainable:,} trainable params)")

    def forward(self, image: torch.Tensor,
                labels: torch.Tensor = None,
                epoch: int = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image:  (B, 3, 224, 224)
            labels: (B, C, G, G) — IF targets in [0,1], only used during training
            epoch:  0-indexed epoch number, only used during training
        Returns:
            preds: (B, C, G, G)
            h:     (B*G*G, 128) raw penultimate features (detached, for FDS accumulation)
        """
        features     = self.encoder.forward_features(image)           # (B, T, D)
        patch_tokens = features[:, self.encoder.num_prefix_tokens:]   # (B, G*G, D)
        assert(patch_tokens.shape[1] == 256)
        B, NUM_TOKENS, D = patch_tokens.shape
        G = self.token_grid
        N = B * NUM_TOKENS

        h = self.feature_extractor(patch_tokens.reshape(N, D))        # (N, 128)

        use_fds = (self.training and self.fds is not None
                   and labels is not None and epoch is not None)

        if use_fds:
            # Per-marker FDS: smooth h with each marker's label distribution,
            # then apply the corresponding row of the weight matrix.
            labels_flat = labels.permute(0, 2, 3, 1).reshape(N, self.num_outputs)  # (N, C)
            pred_cols = []
            for j, fds_j in enumerate(self.fds):
                labels_np = labels_flat[:, j].detach().cpu().numpy()   # (N,)
                h_j = fds_j.smooth(h.clone(), labels_np, epoch)        # (N, 128)
                pred_cols.append(self.predictors[j](h_j))   # (N, 1)
            preds = torch.cat(pred_cols, dim=1)                        # (N, C)
        else:
            preds = self.predictor(h)                                  # (N, C)

        preds_spatial = preds.reshape(B, G, G, self.num_outputs).permute(0, 3, 1, 2)  # (B,C,G,G)
        return preds_spatial, h.detach()


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