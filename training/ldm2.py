"""
Diffusers-based latent diffusion for H&E -> protein expression.

Conditioning:
  - H&E VAE latent: concatenated with the noisy target latent (Marigold-style).
  - Frozen CONCH text embeddings from curated marker descriptions, projected to
    the UNet cross-attention dimension and used as cross-attention keys/values.
  - (optional) Frozen UNI2 global H&E embedding added to the timestep embedding
    via the UNet class-conditioning path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
import timm

MARKER_FIELDS: dict[str, dict[str, str]] = {
    "Hoechst": {
        "name": "Hoechst fluorescent nuclear stain bright glowing spots on every cell nucleus",
        "he_anchor": (
            "Every round or oval dark-stained hematoxylin nucleus on H&E corresponds to one bright glowing Hoechst spot, "
            "so the fluorescence pattern is a complete spatial map of all cell positions in the image."
        ),
        "cell_morphology": (
            "Small lymphocyte nuclei appear as tiny tight bright dots, larger epithelial nuclei as broader bright ovals, "
            "elongated spindle-cell nuclei as short bright rods, and granulocyte nuclei as multilobed bright clusters."
        ),
        "if_pattern": (
            "Dense field of bright round-to-oval glowing spots of varying sizes filling all cellular regions, each spot "
            "sharply bounded with no cytoplasmic halo, and completely absent in acellular lumens and extracellular matrix."
        ),
        "micro_context": (
            "Bright nuclear spots distributed uniformly across every cellular area including tumour nests, stroma, and "
            "immune aggregates; only vessel lumens, gland lumens, and acellular matrix appear completely dark and empty."
        ),
    },
    "SMA": {
        "name": "SMA smooth muscle actin bright fibrillar streaks and concentric vessel rings",
        "he_anchor": (
            "Bright SMA signal traces the elongated pale spindle-shaped cells forming fibrous stromal bands and outlines "
            "the circular smooth muscle layers surrounding round vessel lumens visible on H&E."
        ),
        "cell_morphology": (
            "Long bright linear streaks following the axis of spindle-shaped myofibroblast bodies arranged in parallel "
            "bundles, and bright annular rings tightly encircling circular vessel cross-sections with dark central lumens."
        ),
        "if_pattern": (
            "Bright parallel linear or gently curved fibrillar streaks in stromal regions plus bright concentric ring "
            "outlines around circular vessel profiles, with no filled blob shapes and a mostly dark background."
        ),
        "micro_context": (
            "Bright fibrillar and ring-shaped signal confined to reactive stroma between tumour cell groups and around "
            "vessels; cohesive epithelial cell clusters, fat cells, and immune cell aggregates are fully dark."
        ),
    },
    "Pan-CK": {
        "name": "Pan-CK cytokeratin bright solid cytoplasmic fill in epithelial tumour cell clusters",
        "he_anchor": (
            "Bright Pan-CK signal solidly fills the cytoplasm of cohesive polygonal cell clusters and gland-forming "
            "structures that show clear sharp epithelial-stromal boundaries on H&E."
        ),
        "cell_morphology": (
            "Solid bright cytoplasmic fill within tightly packed polygonal cells arranged in cohesive sheets, each cell "
            "body uniformly glowing with round dark unstained nuclear holes punched into the bright cytoplasmic mass."
        ),
        "if_pattern": (
            "Large contiguous bright filled islands or sheets with sharp high-contrast edges at the tumour-stroma border, "
            "round dark nuclear holes within the bright mass, and completely dark surrounding spindle stroma and immune cells."
        ),
        "micro_context": (
            "Bright solid islands surrounded by dark negative stroma; glandular structures show bright ring-like walls "
            "enclosing dark lumens; individual spindle cells, lymphocytes, and fat cells remain fully dark."
        ),
    },
    "CD3e": {
        "name": "CD3 T cell thin bright membrane rings small scattered lymphocytes",
        "he_anchor": (
            "Bright CD3 rings appear on the very small round dark lymphocytes scattered individually or in loose clusters "
            "within the stroma and at tumour margins on H&E."
        ),
        "cell_morphology": (
            "Numerous very small bright thin circular rings or halos each outlining a tiny round cell with a dark "
            "unstained centre, scattered individually or in loose groups across infiltrated regions."
        ),
        "if_pattern": (
            "Field of small bright punctate rings or thin circular halos each surrounding a dark nucleus, appearing as "
            "numerous fine glowing circles of uniform small size against a dark background."
        ),
        "micro_context": (
            "Small bright rings scattered in immune-infiltrated stroma and tumour margins with density increasing near "
            "lymphoid aggregates; epithelial nests, fat cells, fibroblasts, and vessel walls are fully dark."
        ),
    },
    "CD4": {
        "name": "CD4 helper T cell and macrophage bright membrane rings of mixed sizes",
        "he_anchor": (
            "Bright CD4 signal appears on both small compact lymphocytes and larger irregular pale macrophage-like cells "
            "in immune-rich stromal areas, producing a mix of ring sizes in the same region."
        ),
        "cell_morphology": (
            "Mixed population of small tight bright rings on compact lymphocytes alongside larger broader bright rings or "
            "peripheral halos on bigger irregular macrophage-like cells, creating visibly variable ring diameters."
        ),
        "if_pattern": (
            "Mixture of small tight bright rings and larger broader bright rings or peripheral signal patches against a "
            "dark background, with more size heterogeneity and larger maximum ring diameter than CD3 staining."
        ),
        "micro_context": (
            "Bright mixed-size rings in inflammatory stroma and immune aggregates; overlaps spatially with CD3 regions "
            "but shows larger rings; epithelial nests, fat tissue, and acellular matrix remain fully dark."
        ),
    },
}


_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[None, :, None, None]
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[None, :, None, None]


def _as_list(marker_names: str | Sequence[str], batch_size: int | None = None) -> list[str]:
    if isinstance(marker_names, str):
        if batch_size is None:
            return [marker_names]
        return [marker_names] * batch_size
    names = list(marker_names)
    if batch_size is not None and len(names) != batch_size:
        raise ValueError(f"Expected {batch_size} marker names, got {len(names)}")
    return names



def _protein_to_sd_range(protein_map: torch.Tensor) -> torch.Tensor:
    if protein_map.ndim != 4:
        raise ValueError(f"Expected protein_map with shape (B, C, H, W), got {tuple(protein_map.shape)}")
    if protein_map.shape[1] == 1:
        protein_map = protein_map.repeat(1, 3, 1, 1)
    elif protein_map.shape[1] != 3:
        raise ValueError(f"Expected 1 or 3 protein channels, got {protein_map.shape[1]}")

    if protein_map.min() < 0:
        return protein_map.clamp(-1.0, 1.0)
    return (protein_map.clamp(0.0, 1.0) * 2.0) - 1.0


def _imagenet_to_sd_range(he_patch: torch.Tensor) -> torch.Tensor:
    if he_patch.ndim != 4 or he_patch.shape[1] != 3:
        raise ValueError(f"Expected H&E patch with shape (B, 3, H, W), got {tuple(he_patch.shape)}")
    mean = _IMAGENET_MEAN.to(device=he_patch.device, dtype=he_patch.dtype)
    std = _IMAGENET_STD.to(device=he_patch.device, dtype=he_patch.dtype)
    rgb = he_patch * std + mean
    return (rgb.clamp(0.0, 1.0) * 2.0) - 1.0


def replace_unet_conv_in(unet: UNet2DConditionModel, in_channels: int = 8) -> UNet2DConditionModel:
    old_conv = unet.conv_in
    if old_conv.in_channels == in_channels:
        return unet
    if in_channels % old_conv.in_channels != 0:
        raise ValueError(
            f"Cannot expand conv_in from {old_conv.in_channels} to {in_channels}: "
            "the requested channel count is not an integer multiple."
        )

    repeat = in_channels // old_conv.in_channels
    weight = old_conv.weight.detach().clone().repeat(1, repeat, 1, 1) / repeat
    bias = None if old_conv.bias is None else old_conv.bias.detach().clone()

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
    )
    new_conv.weight = nn.Parameter(weight)
    if bias is not None:
        new_conv.bias = nn.Parameter(bias)

    unet.conv_in = new_conv
    unet.config["in_channels"] = in_channels
    return unet


def pyramid_noise_like(x: torch.Tensor, discount: float = 0.9) -> torch.Tensor:
    b, c, h, w = x.shape
    upsample = nn.Upsample(size=(h, w), mode="bilinear")
    noise = torch.randn_like(x)
    cur_h, cur_w = h, w
    for scale in range(1, 10):
        cur_h = max(1, int(cur_h / 2))
        cur_w = max(1, int(cur_w / 2))
        noise = noise + upsample(torch.randn(b, c, cur_h, cur_w, device=x.device, dtype=x.dtype)) * (discount**scale)
        if cur_h == 1 or cur_w == 1:
            break
    return noise / noise.std().clamp_min(1e-6)


@dataclass
class LDM2Output:
    loss: torch.Tensor
    model_pred: torch.Tensor
    target: torch.Tensor
    noisy_latents: torch.Tensor
    clean_latents: torch.Tensor
    timesteps: torch.Tensor


class OpenCLIPTextEncoder(nn.Module):
    """
    Frozen OpenCLIP ViT-H/14 text encoder — the exact model used by SD 2.1.

    Output dim (1024) matches SD 2.1 cross_attention_dim directly, so no
    projection layer is needed and the UNet cross-attention weights work
    out-of-the-box from the first fine-tuning step.

    Input : list of text strings
    Output: (N, 1024) L2-normalised embeddings, one per string
    """

    DIM = 1024

    def __init__(self) -> None:
        super().__init__()
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(
            "ViT-H-14", pretrained="laion2b_s32b_b79k"
        )
        self.model = model
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        tokens = self.tokenizer(texts).to(device)
        return self.model.encode_text(tokens)  # (N, 1024)


class CONCHTextEncoder(nn.Module):
    """
    Frozen CONCH text encoder (ViT-B/16, pathology-grounded CLIP from Mahmood Lab).

    Requires: pip install git+https://github.com/mahmoodlab/CONCH.git
    Output dim is 512 — BiomarkerConditioner adds a trainable projection to
    cross_attention_dim when this encoder is selected.

    Input : list of text strings
    Output: (N, 512) L2-normalised embeddings, one per string
    """

    DIM = 512

    def __init__(self) -> None:
        super().__init__()
        from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer
        model, _ = create_model_from_pretrained("conch_ViT-B-16", "hf_hub:MahmoodLab/CONCH")
        self.model = model
        self.tokenizer = get_tokenizer()
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        from conch.open_clip_custom import tokenize
        device = next(self.parameters()).device
        tokens = tokenize(texts=texts, tokenizer=self.tokenizer).to(device)
        return self.model.encode_text(tokens)  # (N, 512)


class UNI2GlobalEncoder(nn.Module):
    """
    Frozen UNI2 patch encoder.

    Input:  H&E patch, ImageNet-normalised, shape (B, 3, 224, 224)
    Output: global UNI2 embedding, shape (B, 1536)
    """

    def __init__(self) -> None:
        super().__init__()
        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        self.model = timm.create_model("hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def enable_unet_vector_conditioning(unet: UNet2DConditionModel) -> UNet2DConditionModel:
    """
    Set class_embedding = Identity so a pre-projected vector passed as
    class_labels is added directly to the timestep embedding.
    """
    unet.class_embedding = nn.Identity()
    unet.config["class_embed_type"] = None
    unet.config["num_class_embeds"] = None
    return unet


class BiomarkerConditioner(nn.Module):
    """
    Biomarker conditioning module.

    Text branch (always active):
      A frozen text encoder encodes each TEXT_FIELD for every marker into one
      vector.  Two encoders are supported via text_encoder="openclip"|"conch":

        openclip (default): OpenCLIP ViT-H/14 — the exact model used by SD 2.1.
          Output dim 1024 matches cross_attention_dim directly, so no projection
          is needed and the UNet cross-attention works from the first step.

        conch: CONCH ViT-B/16 — pathology-grounded but output dim 512, so a
          trainable linear projection (512 → cross_attention_dim) is added.

      The result is a sequence of n_fields tokens per sample passed as
      cross-attention keys/values to the UNet.

    UNI2 branch (optional, use_uni2=True):
      Frozen UNI2 encodes the H&E patch into a 1536-dim global vector.  A
      trainable MLP projects it to time_embed_dim and the result is added to
      the timestep embedding via the UNet class-conditioning path.

    forward() returns:
      encoder_hidden_states : (B, n_fields, cross_attention_dim)
      class_labels          : (B, time_embed_dim) or None
    """

    TEXT_FIELDS = ("name", "he_anchor", "cell_morphology", "if_pattern", "micro_context")
    UNI2_DIM = 1536

    def __init__(
        self,
        marker_names: Sequence[str],
        time_embed_dim: int,
        cross_attention_dim: int,
        use_uni2: bool = True,
        text_encoder: str = "openclip",
    ) -> None:
        super().__init__()
        if not marker_names:
            raise ValueError("marker_names must be non-empty")
        if text_encoder not in ("openclip", "conch"):
            raise ValueError(f"text_encoder must be 'openclip' or 'conch', got {text_encoder!r}")

        self.marker_names = marker_names
        self.marker_to_idx = {n: i for i, n in enumerate(self.marker_names)}
        self.n_fields = len(self.TEXT_FIELDS)
        self.use_uni2 = use_uni2

        if text_encoder == "openclip":
            self.text_encoder = OpenCLIPTextEncoder()
        else:
            self.text_encoder = CONCHTextEncoder()

        # Only needed when text encoder dim < cross_attention_dim (i.e. CONCH)
        text_dim = self.text_encoder.DIM
        if text_dim != cross_attention_dim:
            self.text_proj: nn.Module = nn.Sequential(
                nn.Linear(text_dim, cross_attention_dim),
                nn.LayerNorm(cross_attention_dim),
            )
        else:
            self.text_proj = nn.Identity()

        if use_uni2:
            self.uni2 = UNI2GlobalEncoder()
            self.uni2_proj = nn.Sequential(
                nn.Linear(self.UNI2_DIM, time_embed_dim * 2),
                nn.SiLU(),
                nn.LayerNorm(time_embed_dim * 2),
                nn.Linear(time_embed_dim * 2, time_embed_dim),
            )
            self.time_norm = nn.LayerNorm(time_embed_dim)

        text_table = self._build_text_feature_table(self.marker_names)
        self.register_buffer("text_feature_table", text_table, persistent=True)

    def _marker_fields(self, marker_name: str) -> dict[str, str]:
        if marker_name not in MARKER_FIELDS:
            raise ValueError(f"No description fields for marker: {marker_name}")
        return MARKER_FIELDS[marker_name]

    def _build_text_feature_table(self, marker_names: list[str]) -> torch.Tensor:
        rows = []
        for name in marker_names:
            fields = self._marker_fields(name)
            texts = [fields[f] for f in self.TEXT_FIELDS]
            emb = self.text_encoder.encode(texts)  # (n_fields, text_dim)
            rows.append(emb)
        return torch.stack(rows, dim=0)  # (n_markers, n_fields, text_dim)

    def marker_indices(self, marker_names: Sequence[str], device: torch.device) -> torch.Tensor:
        try:
            idx = [self.marker_to_idx[n] for n in marker_names]
        except KeyError as exc:
            raise KeyError(f"Unknown biomarker: {exc.args[0]}") from exc
        return torch.tensor(idx, device=device, dtype=torch.long)

    def forward(
        self,
        marker_names: Sequence[str],
        he_patches: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = he_patches.device
        dtype = he_patches.dtype

        # Text → cross-attention tokens: (B, n_fields, cross_attention_dim)
        marker_idx = self.marker_indices(marker_names, device)
        raw = self.text_feature_table[marker_idx].to(dtype)
        encoder_hidden_states = self.text_proj(raw)

        # UNI2 → time embedding: (B, time_embed_dim) or None
        class_labels = None
        if self.use_uni2:
            with torch.no_grad():
                uni2_global = self.uni2(he_patches)
            class_labels = self.time_norm(self.uni2_proj(uni2_global.to(dtype)))

        return encoder_hidden_states, class_labels


class LDM2(nn.Module):
    """
    Marigold-style image-conditional latent diffusion model for protein maps.

    Training:
      1. encode H&E and protein targets with the same SD VAE,
      2. add scheduler noise to the protein latent,
      3. concatenate [he_latent, noisy_target_latent],
      4. predict the scheduler target with a diffusers UNet conditioned on
         CONCH text embeddings (cross-attention) and optionally UNI2
         (time embedding).

    Sampling:
      1. start from Gaussian noise in target latent space,
      2. repeatedly denoise while keeping the H&E latent fixed,
      3. decode the final latent back into a single-channel protein map.
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet2DConditionModel,
        train_scheduler: DDPMScheduler,
        infer_scheduler: DDIMScheduler,
        conditioner: BiomarkerConditioner,
        *,
        noise_type: str = "gaussian",
    ) -> None:
        super().__init__()
        self.vae = vae
        self.unet = unet
        self.train_scheduler = train_scheduler
        self.infer_scheduler = infer_scheduler
        self.conditioner = conditioner
        self.noise_type = noise_type

        self.vae.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        marker_names: Sequence[str],
        use_uni2: bool = True,
        text_encoder: str = "openclip",
        noise_type: str = "gaussian",
        revision: str | None = None,
        variant: str | None = None,
    ) -> "LDM2":
        vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            revision=revision,
            variant=variant,
        )
        unet = UNet2DConditionModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="unet",
            revision=revision,
            variant=variant,
        )
        train_scheduler = DDPMScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
        )
        infer_scheduler = DDIMScheduler.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="scheduler",
            timestep_spacing="trailing",
        )

        replace_unet_conv_in(unet, in_channels=8)
        if use_uni2:
            enable_unet_vector_conditioning(unet)

        time_embed_dim = unet.time_embedding.linear_2.out_features
        cross_attention_dim = unet.config.cross_attention_dim

        conditioner = BiomarkerConditioner(
            marker_names=marker_names,
            time_embed_dim=time_embed_dim,
            cross_attention_dim=cross_attention_dim,
            use_uni2=use_uni2,
            text_encoder=text_encoder,
        )

        return cls(
            vae=vae,
            unet=unet,
            train_scheduler=train_scheduler,
            infer_scheduler=infer_scheduler,
            conditioner=conditioner,
            noise_type=noise_type,
        )

    @property
    def latent_scale(self) -> float:
        return float(self.vae.config.scaling_factor)

    def trainable_parameters(self):
        for p in self.unet.parameters():
            if p.requires_grad:
                yield p
        for p in self.conditioner.parameters():
            if p.requires_grad:
                yield p

    def encode_protein(self, protein_map: torch.Tensor) -> torch.Tensor:
        x = _protein_to_sd_range(protein_map)
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.mode()
        return latent * self.latent_scale

    def encode_he(self, he_patch: torch.Tensor) -> torch.Tensor:
        x = _imagenet_to_sd_range(he_patch)
        with torch.no_grad():
            latent = self.vae.encode(x).latent_dist.mode()
        return latent * self.latent_scale

    def decode_protein(self, latents: torch.Tensor, output_range: str = "zero_one") -> torch.Tensor:
        with torch.no_grad():
            decoded = self.vae.decode(latents / self.latent_scale).sample
        decoded = decoded.mean(dim=1, keepdim=True)
        if output_range == "minus_one_one":
            return decoded.clamp(-1.0, 1.0)
        if output_range == "zero_one":
            return ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
        raise ValueError(f"Unknown output_range: {output_range}")

    def _sample_noise(self, clean_latents: torch.Tensor) -> torch.Tensor:
        if self.noise_type == "gaussian":
            return torch.randn_like(clean_latents)
        if self.noise_type == "zeros":
            return torch.zeros_like(clean_latents)
        if self.noise_type == "pyramid":
            return pyramid_noise_like(clean_latents)
        raise ValueError(f"Unknown noise_type: {self.noise_type}")

    def _scheduler_target(
        self,
        clean_latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        prediction_type = self.train_scheduler.config.prediction_type
        if prediction_type == "epsilon":
            return noise
        if prediction_type == "v_prediction":
            return self.train_scheduler.get_velocity(clean_latents, noise, timesteps)
        if prediction_type == "sample":
            return clean_latents
        raise ValueError(f"Unsupported prediction_type: {prediction_type}")

    def forward(
        self,
        he_patches: torch.Tensor,
        protein_maps: torch.Tensor,
        marker_names: str | Sequence[str],
        *,
        marker_weights: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        return_dict: bool = True,
    ) -> LDM2Output | torch.Tensor:
        batch_size = he_patches.shape[0]
        names = [n for n in _as_list(marker_names, batch_size=batch_size)]

        clean_latents = self.encode_protein(protein_maps)
        he_latents = self.encode_he(he_patches)

        if timesteps is None:
            timesteps = torch.randint(
                0,
                self.train_scheduler.config.num_train_timesteps,
                (batch_size,),
                device=he_patches.device,
            ).long()
        else:
            timesteps = timesteps.to(device=he_patches.device, dtype=torch.long)

        noise = self._sample_noise(clean_latents)
        noisy_latents = self.train_scheduler.add_noise(clean_latents, noise, timesteps)

        encoder_hidden_states, class_labels = self.conditioner(names, he_patches)

        unet_input = torch.cat([he_latents, noisy_latents], dim=1)
        model_pred = self.unet(
            unet_input,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            class_labels=class_labels,
            return_dict=False,
        )[0]
        target = self._scheduler_target(clean_latents, noise, timesteps)

        per_sample = F.mse_loss(model_pred.float(), target.float(), reduction="none").mean(dim=(1, 2, 3))
        if marker_weights is not None:
            per_sample = per_sample * marker_weights.to(device=per_sample.device, dtype=per_sample.dtype)
        loss = per_sample.mean()

        if not return_dict:
            return loss
        return LDM2Output(
            loss=loss,
            model_pred=model_pred,
            target=target,
            noisy_latents=noisy_latents,
            clean_latents=clean_latents,
            timesteps=timesteps,
        )

    @torch.no_grad()
    def generate(
        self,
        he_patches: torch.Tensor,
        marker_names: str | Sequence[str],
        *,
        num_inference_steps: int = 25,
        output_range: str = "zero_one",
    ) -> torch.Tensor:
        batch_size = he_patches.shape[0]
        names = [n for n in _as_list(marker_names, batch_size=batch_size)]
        he_latents = self.encode_he(he_patches)

        latents = torch.randn_like(he_latents)
        self.infer_scheduler.set_timesteps(num_inference_steps, device=he_patches.device)

        encoder_hidden_states, class_labels = self.conditioner(names, he_patches)

        for timestep in self.infer_scheduler.timesteps:
            model_input = torch.cat([he_latents, latents], dim=1)
            timestep_batch = torch.full((batch_size,), int(timestep), device=he_patches.device, dtype=torch.long)
            noise_pred = self.unet(
                model_input,
                timestep_batch,
                encoder_hidden_states=encoder_hidden_states,
                class_labels=class_labels,
                return_dict=False,
            )[0]
            latents = self.infer_scheduler.step(noise_pred, timestep, latents).prev_sample

        return self.decode_protein(latents, output_range=output_range)