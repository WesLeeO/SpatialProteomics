"""
LDM for H&E → protein expression.

Conditioning paths:
  - H&E VAE latent: channel-concat with noisy protein latent (Marigold-style).
  - Learned per-marker embedding (nn.Embedding → t_dim): added to t_emb.
    Stochastically replaced with a learned null vector for CFG training.
  - Marker concept phrases (B, N_CONCEPTS, 768) via BiomedBERT: projected
    to ctx_dim and used as cross-attention K/V in every ResBlock.

CFG drops only the per-marker learned embedding; text cross-attention and
H&E latent stay on for both conditional and unconditional passes.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"  # use local cache only — models are in foundation_models/hub/
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from diffusers import AutoencoderKL
import timm
from model import foundation_model
import matplotlib.pyplot as plt

C_LAT     = 4        # SD VAE always outputs 4 latent channels
VAE_SCALE = 0.18215  # standard SD latent scaling factor
SLIDE_SPLIT = False

# Each marker → N_CONCEPTS phrases (H&E anchor, cell morphology, IF pattern).
# Unknown markers fall back to N_CONCEPTS copies of the raw marker name.
N_CONCEPTS = 3

# (h_e_anchor, cell_morphology, if_pattern, absence)
MARKER_CONCEPTS: dict[str, tuple[str, str, str, str]] = {
    # ── Nuclear markers ───────────────────────────────────────────────────────
    "Hoechst": (
        "nuclear DNA directly colocalising with the basophilic hematoxylin-stained "
        "nuclei visible as dark blue-purple round-to-oval features",
        "all nucleated cells: small dense round lymphocyte nuclei, large open "
        "vesicular epithelial nuclei with prominent chromatin, elongated thin spindle "
        "stromal nuclei, and multi-lobed granulocyte nuclei",
        "dense bright round-to-oval blob pattern reproducing the spatial layout of "
        "every nucleus in the tissue, with bright intensity concentrated at compact "
        "nuclear positions"#,
   #     "extracellular matrix, vessel lumens, gland lumens, adipose "
   #     "spaces, and every non-nuclear region of cytoplasm or intercellular space "
   #     "between cells"
    ),

    "Pan-CK": (
        "pan-cytokeratin cytoskeletal filaments filling the cytoplasm of cohesive "
        "epithelial tumour cells forming nests, solid sheets, and gland structures "
        "bounded by stroma",
        "medium-to-large epithelial cells with abundant eosinophilic cytoplasm, "
        "distinct visible cell borders, and round basally polarised nuclei, forming "
        "cohesive clusters sharply demarcated from spindle stroma",
        "contiguous bright regions fully filling epithelial cell nests with sharp "
        "high-contrast edges at the stromal interface, appearing as solid filled "
        "clusters rather than punctate spots",
      #  "spindle stromal cells, myofibroblasts, fibroblasts, "
      #  "lymphocytes, macrophages, vessel walls, vessel lumens, adipocytes, and "
      #  "acellular matrix between epithelial nests"
    ),

    "DAPI": (
        "nuclear DNA directly colocalising with the basophilic hematoxylin-stained "
        "nuclei visible as dark blue-purple round-to-oval features throughout the "
        "paired H&E input image",
        "all nucleated cells: small dense round lymphocyte nuclei, large open "
        "vesicular epithelial nuclei with prominent chromatin, elongated thin spindle "
        "stromal nuclei, and multi-lobed granulocyte nuclei",
        "dense bright round-to-oval blob pattern reproducing the spatial layout of "
        "every nucleus in the tissue, with bright intensity concentrated at compact "
        "nuclear positions",
        "extracellular matrix, vessel lumens, gland lumens, adipose "
        "spaces, and every non-nuclear region of cytoplasm or intercellular space "
        "between cells"
    ),
    "Ki-67": (
        "nuclei of actively proliferating cells within tumour epithelial nests and "
        "reactive stromal regions, concentrated where H&E shows enlarged "
        "hyperchromatic nuclei and occasional mitotic figures",
        "cycling tumour and stromal cells with enlarged open-chromatin nuclei and "
        "prominent nucleoli, appearing as scattered or clustered positive cells "
        "intermixed with a majority of non-proliferating cells",
        "sparse bright punctate-to-diffuse nuclear dots scattered within epithelial "
        "cell clusters, appearing as isolated small nuclear spots rather than "
        "contiguous filled regions",
        "mature quiescent lymphocytes, adipocytes, mature differentiated "
        "stroma, acellular extracellular matrix, vessel lumens, gland lumens, and "
        "all interphase non-proliferating nuclei"
    ),
    "pHH3": (
        "condensed mitotic chromosomes inside nuclei undergoing cell division, "
        "localised to the scarce mitotic figures visible as dark irregular chromosome "
        "clumps within tumour regions in H&E",
        "cells caught in mitosis with condensed metaphase or anaphase chromosomes and "
        "absent nuclear envelope, appearing as dark compact irregular shapes among "
        "otherwise interphase nuclear populations",
        "very sparse bright punctate signal restricted to rare mitotic figure "
        "locations, appearing as isolated bright spots on an otherwise empty "
        "background",
        "interphase nuclei, stromal cells, lymphocytes, "
        "vessel lumens, gland lumens, adipose tissue, and every non-dividing cell "
        "in the tissue"
    ),
    "p53": (
        "nuclei of tumour cells harbouring TP53 mutation, concentrated within cohesive "
        "epithelial tumour nests showing abnormal enlarged hyperchromatic nuclei in "
        "H&E",
        "tumour epithelial cells with enlarged irregular hyperchromatic nuclei and "
        "polymorphic nuclear appearance, clustered as cohesive epithelial populations "
        "rather than scattered single cells",
        "strong homogeneous bright nuclear signal filling tumour cell nests uniformly, "
        "appearing as contiguous bright clustered nuclear regions within cohesive "
        "epithelial groups",
        "stromal cells, lymphocytes, endothelium, macrophages, adipocytes, "
        "normal non-neoplastic epithelium, and all non-mutant cells outside "
        "tumour nests"
    ),
    "FoxP3": (
        "nuclei of regulatory T lymphocytes within tumour stroma and peri-tumoural "
        "immune infiltrates, anchored to small round basophilic lymphocyte nuclei "
        "visible in H&E",
        "small round lymphocytes with scant cytoplasm and dense dark nuclei, scattered "
        "sparsely among tumour cells or clustered within lymphoid aggregates in the "
        "stroma",
        "very sparse punctate bright nuclear dots in small lymphocyte distributions, "
        "appearing as isolated small bright spots within lymphocyte populations",
        "epithelial tumour nests, large macrophages, vessel walls, "
        "fibroblasts, adipocytes, acellular matrix, and all non-regulatory T cell "
        "populations"
    ),
    "FOXP3": (
        "nuclei of regulatory T lymphocytes within tumour stroma and peri-tumoural "
        "immune infiltrates, anchored to small round basophilic lymphocyte nuclei "
        "visible in H&E",
        "small round lymphocytes with scant cytoplasm and dense dark nuclei, scattered "
        "sparsely among tumour cells or clustered within lymphoid aggregates in the "
        "stroma",
        "very sparse punctate bright nuclear dots in small lymphocyte distributions, "
        "appearing as isolated small bright spots within lymphocyte populations",
        "epithelial tumour nests, large macrophages, vessel walls, "
        "fibroblasts, adipocytes, acellular matrix, and all non-regulatory T cell "
        "populations"
    ),

    # ── Cytoplasmic membrane markers ──────────────────────────────────────────
    "CD3": (
        "T-cell receptor complex on the membrane of T lymphocytes, anchored to small "
        "basophilic lymphocyte clusters scattered within tumour stroma and "
        "concentrated in lymphoid aggregates in H&E",
        "small round dense lymphocytes with minimal cytoplasm and compact "
        "hyperchromatic nuclei, arranged as scattered infiltrating cells among tumour "
        "cells or packed in follicle-like clusters",
        "thin bright ring pattern tracing the membrane outline of small lymphocyte "
        "cell bodies, appearing as small hollow circular signals with bright edges "
        "and darker interior",
        "large epithelial tumour cells, fibroblasts, myofibroblasts, "
        "endothelial vessel walls, adipocytes, acellular extracellular matrix, and "
        "non-T-cell immune populations"
    ),
    "CD3e": (
        "T-cell receptor complex on the membrane of T lymphocytes, anchored to small "
        "basophilic lymphocyte clusters scattered within tumour stroma and "
        "concentrated in lymphoid aggregates in H&E",
        "small round dense lymphocytes with minimal cytoplasm and compact "
        "hyperchromatic nuclei, arranged as scattered infiltrating cells among tumour "
        "cells or packed in follicle-like clusters",
        "thin bright ring pattern tracing the membrane outline of small lymphocyte "
        "cell bodies, appearing as small hollow circular signals with bright edges "
        "and darker interior",
        "large epithelial tumour cells, fibroblasts, myofibroblasts, "
        "endothelial vessel walls, adipocytes, acellular extracellular matrix, and "
        "non-T-cell immune populations"
    ),
    "CD4": (
        "membrane of helper T lymphocytes and some tissue macrophages, anchored to "
        "small lymphocytes in stromal regions alongside larger pale macrophages "
        "visible in H&E",
        "small round helper T lymphocytes with scant cytoplasm alongside larger "
        "irregularly shaped macrophages with abundant pale eosinophilic cytoplasm, "
        "distributed across tumour stroma and immune infiltrates",
        "thin bright ring signal on small lymphocyte membranes together with broader "
        "membrane signal on larger macrophage bodies, appearing as heterogeneously "
        "sized ring-like signals",
        "epithelial tumour nests, fibroblasts, myofibroblasts, endothelial "
        "vessel walls, adipocytes, acellular matrix, and cytotoxic-only lymphocyte "
        "populations"
    ),
    "CD8a": (
        "membrane of cytotoxic T lymphocytes infiltrating tumour nests and "
        "stromal-epithelial interfaces, anchored to small basophilic lymphocytes "
        "penetrating between epithelial cells in H&E",
        "small round cytotoxic T cells with compact dense nuclei and scant cytoplasm, "
        "distributed as scattered tumour-infiltrating lymphocytes within and around "
        "epithelial nests",
        "thin bright ring membrane pattern on small lymphocyte bodies located inside "
        "and at the edges of tumour nests, appearing as small hollow circular "
        "membrane signals",
        "fibroblasts, myofibroblasts, adipocytes, mature vessel walls, "
        "acellular matrix, epithelial cell cytoplasm interiors, and helper-only "
        "lymphocyte populations"
    ),
    "CD20": (
        "membrane of mature B lymphocytes concentrated in germinal centres and "
        "lymphoid follicles, anchored to rounded lymphoid aggregates visible as dense "
        "basophilic cell clusters in H&E",
        "small round B lymphocytes with minimal cytoplasm and compact dark nuclei, "
        "packed tightly into follicular structures with visible mantle and germinal "
        "centre organisation",
        "tight cluster of bright ring-like membrane signals in rounded follicular "
        "patterns, appearing as densely packed circular signals forming a clear "
        "aggregate boundary",
        "epithelial nests, spindle stromal cells, vessels, acellular "
        "matrix, adipocytes, and scattered T-lymphocyte populations outside "
        "B-cell follicles"
    ),
    "CD45": (
        "membrane of all haematopoietic leukocytes broadly distributed across tumour "
        "stroma and infiltrates, anchored to diverse basophilic immune cell "
        "populations visible in H&E",
        "diverse immune cells: small lymphocytes, larger macrophages with pale "
        "cytoplasm, and multi-lobed granulocytes, scattered throughout stroma and "
        "infiltrating tumour regions",
        "broadly distributed thin bright membrane rings on heterogeneous immune cell "
        "shapes across the tissue, appearing as scattered ring-like signals of "
        "varied sizes",
        "epithelial tumour cells, fibroblasts, myofibroblasts, endothelium, "
        "vessel lumens, adipose tissue, and acellular extracellular matrix"
    ),
    "CD45RO": (
        "membrane of antigen-experienced memory T lymphocytes, anchored to small "
        "basophilic lymphocytes accumulating within chronic inflammatory stroma "
        "visible in H&E",
        "small round memory T cells with compact dense nuclei and scant cytoplasm, "
        "scattered within tumour stroma and peri-tumoural immune infiltrates",
        "thin bright ring membrane pattern on small lymphocyte bodies located in "
        "stromal regions, appearing as small hollow circular signals",
        "epithelial tumour cells, large macrophages, vessel walls, "
        "adipocytes, acellular matrix, and naive-phenotype T cell populations "
        "outside memory compartments"
    ),
    "CD163": (
        "membrane and cytoplasmic vesicles of M2-polarised macrophages, anchored to "
        "large pale eosinophilic cells with abundant cytoplasm visible within tumour "
        "stroma in H&E",
        "large irregularly shaped macrophages with abundant pale eosinophilic "
        "cytoplasm, reniform or oval vesicular nuclei, and indistinct cell borders, "
        "scattered within stromal tissue",
        "broad granular cytoplasmic and membrane signal on large cell bodies with "
        "indistinct edges, appearing as diffuse irregular bright patches rather "
        "than sharp ring signals",
        "small lymphocytes, epithelial tumour nests, spindle stromal "
        "cells, vessel lumens, adipocytes, and M1-polarised macrophage populations"
    ),
    "PD-1": (
        "membrane checkpoint receptor on exhausted tumour-infiltrating T lymphocytes, "
        "anchored to small basophilic lymphocytes clustered within immune-infiltrated "
        "tumour regions in H&E",
        "small round T cells with compact dense nuclei and minimal cytoplasm, "
        "concentrated in immune-rich tumour-infiltrating zones alongside cytotoxic "
        "lymphocyte populations",
        "thin bright ring membrane signal on small lymphocytes within tumour "
        "infiltrates, appearing as small hollow circular signals clustered in "
        "immune-infiltrated zones",
        "epithelial tumour cell interiors, fibroblasts, myofibroblasts, "
        "vessel walls, adipocytes, acellular regions, and non-exhausted T cell "
        "populations outside tumour zones"
    ),
    "PD-L1": (
        "membrane and cytoplasm checkpoint ligand on tumour epithelial cells and "
        "tumour-associated macrophages, anchored to cohesive epithelial nests and "
        "pale macrophages visible in H&E",
        "medium-to-large tumour epithelial cells with eosinophilic cytoplasm "
        "alongside large macrophages with abundant pale cytoplasm, distributed at "
        "the tumour-stroma interface",
        "membrane and cytoplasmic bright signal on epithelial tumour cell surfaces "
        "and macrophage bodies, appearing as contiguous bright regions at the "
        "tumour-stroma boundary",
        "small lymphocytes, fibroblasts, myofibroblasts, vessel lumens, "
        "adipocytes, acellular matrix, and normal non-neoplastic epithelium without "
        "immune evasion context"
    ),
    "CD31": (
        "endothelial cell-cell junction membrane lining blood vessel walls, anchored "
        "to the thin single-cell endothelial layer outlining round or oval vessel "
        "lumens filled with red blood cells in H&E",
        "flattened elongated endothelial cells forming a continuous thin layer around "
        "vessel lumens, with compressed nuclei bulging into the lumen space and very "
        "little visible cytoplasm",
        "continuous thin bright linear signal outlining vascular lumen walls in ring "
        "or branching patterns, appearing as hollow circular or tubular outlines "
        "around empty luminal spaces",
        "epithelial nests, stromal spindle cells, fibroblasts, "
        "myofibroblasts, lymphoid aggregates, solid tissue regions, acellular "
        "matrix, and adipocytes"
    ),
    "CD11b": (
        "membrane integrin on myeloid cells including monocytes, macrophages, and "
        "granulocytes, anchored to varied immune cell morphologies scattered through "
        "stromal and inflammatory regions in H&E",
        "myeloid cells of varied morphology: small monocytes, large macrophages with "
        "abundant pale cytoplasm, and multi-lobed granulocytes, broadly distributed "
        "in stroma and inflammatory foci",
        "bright ring membrane signal on heterogeneous myeloid cell shapes across "
        "stromal regions, appearing as scattered ring-like signals of varied "
        "sizes and cell outlines",
        "epithelial tumour cells, mature lymphocytes with dense round "
        "nuclei, fibroblasts, myofibroblasts, vessel lumens, adipocytes, and "
        "acellular matrix"
    ),
    "CD138": (
        "cell surface membrane of terminally differentiated plasma B cells, anchored "
        "to round cells with eccentric clock-face nuclei visible within inflammatory "
        "stromal regions in H&E",
        "round plasma cells with eccentrically placed nuclei, clock-face chromatin "
        "pattern, and abundant basophilic cytoplasm surrounding the nucleus, "
        "clustered within chronic inflammatory stromal infiltrates",
        "strong bright ring membrane signal on round cells with clock-face nuclear "
        "pattern, appearing as medium-sized distinct circular signals within "
        "inflammatory clusters",
        "small lymphocytes, epithelial tumour nests, spindle stromal "
        "cells, vessel lumens, adipocytes, acellular matrix, and non-plasma "
        "B-cell populations"
    ),
    "E-cadherin": (
        "lateral cell membrane adherens junctions between adjacent epithelial cells, "
        "anchored to the continuous cell-cell contact lines within cohesive "
        "epithelial nests visible in H&E",
        "cohesive epithelial cells with distinct visible cell borders, abundant "
        "eosinophilic cytoplasm, and polarised round nuclei, forming glandular "
        "clusters and sheets sharply separated from stroma",
        "continuous thin bright ring pattern outlining epithelial cell-cell contact "
        "borders within tumour nests, appearing as honeycomb-like network of "
        "adjoining cell boundaries",
        "spindle stromal cells, fibroblasts, myofibroblasts, immune "
        "infiltrates, vessel walls, adipocytes, acellular matrix, and dissociated "
        "single cells without epithelial contacts"
    ),
    "E-Cadherin": (
        "lateral cell membrane adherens junctions between adjacent epithelial cells, "
        "anchored to the continuous cell-cell contact lines within cohesive "
        "epithelial nests visible in H&E",
        "cohesive epithelial cells with distinct visible cell borders, abundant "
        "eosinophilic cytoplasm, and polarised round nuclei, forming glandular "
        "clusters and sheets sharply separated from stroma",
        "continuous thin bright ring pattern outlining epithelial cell-cell contact "
        "borders within tumour nests, appearing as honeycomb-like network of "
        "adjoining cell boundaries",
        "spindle stromal cells, fibroblasts, myofibroblasts, immune "
        "infiltrates, vessel walls, adipocytes, acellular matrix, and dissociated "
        "single cells without epithelial contacts"
    ),
    "Podoplanin": (
        "membrane of lymphatic endothelial cells and some basal tumour cells, "
        "anchored to thin-walled lymphatic vessels with collapsed or slit-like lumens "
        "in peri-tumoural stroma visible in H&E",
        "flattened thin endothelial cells forming slit-like lymphatic channels with "
        "irregular collapsed lumen shapes, distributed in peri-tumoural stromal "
        "regions",
        "continuous thin bright linear signal outlining thin-walled lymphatic "
        "vessels in irregular slit-like patterns, appearing as elongated narrow "
        "hollow outlines",
        "solid tumour nests, blood vessels filled with red blood cells, "
        "acellular matrix, adipose tissue, fibroblasts, myofibroblasts, and "
        "lymphoid aggregates"
    ),

    # ── Cytoplasmic / structural markers ──────────────────────────────────────
    "CD68": (
        "lysosomal granules in the cytoplasm of tissue macrophages, anchored to large "
        "pale eosinophilic cells with abundant cytoplasm scattered through tumour "
        "stroma and inflammatory regions in H&E",
        "large irregularly shaped macrophages with abundant pale eosinophilic "
        "cytoplasm, reniform or oval vesicular nuclei, and indistinct cell borders, "
        "distributed within stromal tissue",
        "granular bright cytoplasmic signal filling large cell bodies with "
        "indistinct edges, appearing as irregular diffuse bright patches rather "
        "than sharp ring signals",
        "small lymphocytes, epithelial tumour nests, spindle fibroblasts, "
        "myofibroblasts, adipocytes, vessel lumens, and acellular extracellular "
        "matrix"
    ),
    "Keratin": (
        "cytoskeletal intermediate filaments filling epithelial cytoplasm, anchored "
        "to cohesive epithelial tumour nests and gland-forming clusters bounded by "
        "surrounding stroma in H&E",
        "cohesive epithelial cells with abundant eosinophilic cytoplasm, distinct "
        "visible cell borders, and round polarised nuclei, forming solid nests or "
        "gland structures with central luminal spaces",
        "diffuse bright filamentous cytoplasmic signal filling epithelial cell "
        "nests with sharp boundaries at the stromal interface, appearing as "
        "contiguous filled regions with fibrillar texture",
        "spindle stromal cells, fibroblasts, myofibroblasts, lymphocytes, "
        "macrophages, vessel walls, adipocytes, and acellular matrix between "
        "epithelial nests"
    ),
    "Vimentin": (
        "mesenchymal cytoskeletal intermediate filaments in cytoplasm of stromal and "
        "endothelial cells, anchored to broad spindle-cell stromal regions between "
        "tumour nests visible in H&E",
        "spindle-shaped fibroblasts, endothelial cells, and scattered stromal cells "
        "with elongated thin cytoplasm and oval nuclei, broadly distributed across "
        "stromal bands and vessel walls",
        "diffuse bright cytoplasmic signal covering broad spindle-cell stromal "
        "regions and vessel linings, appearing as contiguous bright background "
        "across stromal bands",
        "cohesive epithelial tumour nests, gland structures with clear "
        "epithelial borders, adipose tissue, and dense lymphoid follicular "
        "aggregates"
    ),
    "SMA": (
        "smooth muscle actin in concentric muscular layers surrounding blood vessel "
        "lumens and in elongated myofibroblasts within reactive desmoplastic stroma "
        "between tumour nests",
        "spindle cells with long thin eosinophilic cytoplasm and elongated "
        "cigar-shaped nuclei, arranged in parallel stromal streams and in "
        "circumferential rings around vessel walls",
        "linear fibrillar bright streaks tracing spindle cell bodies plus "
        "concentric ring patterns encircling vessel lumens, appearing as "
        "elongated streaks and tubular outlines",
       # "epithelial tumour nests, lymphoid aggregates, acellular matrix, "
       # "adipocytes, gland lumens, vessel lumen spaces, and non-myofibroblast "
       # "stromal regions"
    ),
    "Ly6G": (
        "membrane and cytoplasm of neutrophils and granulocytes, anchored to "
        "multi-lobed nuclear cells scattered within inflammatory foci and "
        "peri-tumoural stromal infiltrates visible in H&E",
        "granulocytes with characteristic multi-lobed segmented nuclei and finely "
        "granular pale cytoplasm, distributed within inflammatory infiltrates and "
        "stromal immune aggregates",
        "bright membrane and cytoplasmic signal on small cells with multi-lobed "
        "nuclear morphology, appearing as small irregular signals scattered in "
        "inflammatory foci",
        "epithelial tumour nests, spindle fibroblasts, myofibroblasts, "
        "round-nucleus lymphocyte populations, adipocytes, vessel lumens, and "
        "acellular matrix"
    ),
    "CollagenIV": (
        "extracellular basement membrane matrix outlining glandular epithelium and "
        "vessel walls, anchored to the thin pericellular boundary around epithelial "
        "nests and vessels visible in H&E",
        "acellular extracellular matrix in thin continuous bands at the "
        "epithelial-stromal interface and around vessel walls, separating parenchymal "
        "structures from surrounding tissue",
        "continuous thin bright linear signal outlining the outer boundaries of "
        "glands and vessels in a pericellular pattern, appearing as thin "
        "tubular or ring-shaped outlines",
        "cytoplasm of all cells, epithelial nest interiors, gland "
        "lumens, vessel lumens, bulk stromal space between structures, and "
        "adipocyte interiors"
    ),
}

# ── Pretrained VAE helpers ────────────────────────────────────────────────────

def load_vae(repo="stabilityai/sd-vae-ft-mse", device="cuda") -> AutoencoderKL:
    """Load and freeze the pretrained SD VAE."""
    vae = AutoencoderKL.from_pretrained(repo).to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def vae_encode(vae: AutoencoderKL, protein_map: torch.Tensor) -> torch.Tensor:
    """
    protein_map : (B, 1, H, W)  float in [0, 1]
    Returns     : (B, 4, H/8, W/8)  scaled latents  (deterministic — posterior mode)
    """
    x = protein_map.repeat(1, 3, 1, 1)
    x = (x * 2.0) - 1.0
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode()       # deterministic, no posterior noise
    return z * VAE_SCALE


# ImageNet stats used by the dataset's normalize_he_patch()
_IN_MEAN = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
_IN_STD  = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

def vae_encode_he(vae: AutoencoderKL, he_patch: torch.Tensor) -> torch.Tensor:
    """
    Encode an ImageNet-normalised H&E patch into the SD latent space.
    he_patch : (B, 3, H, W)  ImageNet-normalised
    Returns  : (B, 4, H/8, W/8)  scaled latents  (deterministic — posterior mode)

    Reverses ImageNet normalisation → [0,1] → [-1,1] so the SD VAE
    receives the same input distribution it was trained on.
    """
    mean = _IN_MEAN.to(he_patch.device)
    std  = _IN_STD.to(he_patch.device)
    x = he_patch * std + mean        # ImageNet-norm → [0, 1]
    x = x * 2.0 - 1.0               # [0, 1] → [-1, 1]
    with torch.no_grad():
        z = vae.encode(x).latent_dist.mode()
    return z * VAE_SCALE


def vae_decode(vae: AutoencoderKL, z: torch.Tensor) -> torch.Tensor:
    """
    z       : (B, 4, h, w)  scaled latents
    Returns : (B, 1, H, W)  protein map in [-1, 1]
    """
    with torch.no_grad():
        rec = vae.decode(z / VAE_SCALE).sample     # (B, 3, H, W) in [-1, 1]
    rec = (rec + 1.0) / 2.0
    rec_mean = rec.mean(dim=1, keepdim=True)
    return rec_mean  # (B, 1, H, W)

# ── H&E encoders ──────────────────────────────────────────────────────────────

class HEDetailCNN(nn.Module):
    """
    Lightweight CNN that extracts multiscale spatial detail from H&E.

    Outputs feature maps at two pyramid levels that match the U-Net latent
    spatial dimensions (H/8 and H/16 of the 224×224 input → 28×28 and 14×14).
    Injected additively into the U-Net decoder as skip connections.

    base_ch should match ConditionalUNet.base_ch so channel dims align.
    """
    def __init__(self, base_ch: int = 128):
        super().__init__()
        # group normalization over mini-batch of inputs
        # num groups, num_channels, the mean and standard-deviation are calculated separately over each group. 

        self.stem = nn.Sequential(
            # input channels, output channels, size, stride, padding 
            nn.Conv2d(3,  32,      3, 2, 1), nn.GroupNorm(4,  32),      nn.SiLU(),  # /2 →112
            nn.Conv2d(32, 64,      3, 2, 1), nn.GroupNorm(8,  64),      nn.SiLU(),  # /4 → 56
        )
        self.level1 = nn.Sequential(
            nn.Conv2d(64,     base_ch,   3, 2, 1), nn.GroupNorm(8, base_ch),   nn.SiLU(),  # /8 → 28
        )
        self.level2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch*2, 3, 2, 1), nn.GroupNorm(8, base_ch*2), nn.SiLU(), # /16→ 14
        )

    def forward(self, x: torch.Tensor):
        s  = self.stem(x)
        f1 = self.level1(s)     # (B, base_ch,   H/8,  W/8)   matches U-Net enc1
        f2 = self.level2(f1)    # (B, base_ch*2, H/16, W/16)  matches U-Net enc2
        return f1, f2


class UNI2Encoder(nn.Module):
    """
    Frozen UNI2 — H&E (B, 3, 224, 224) → spatial patch tokens (B, 256, 1536).

    UNI2 is built with `no_embed_class=True` and `reg_tokens=8`, so
    `forward_features` returns 1 + 8 + 256 = 265 tokens in this exact order:
        [CLS, reg_1..reg_8, patch_1..patch_256]
    Register tokens carry no spatial position — they must be skipped.  We use
    `num_prefix_tokens` from timm so this stays correct even if the model is
    rebuilt with different prefix-token settings.

    Spatial contract: patches are laid out row-major on a 16×16 grid covering
    the 224×224 input (patch_size=14).  Token index `i` ↔ patch (i//16, i%16),
    so `reshape(..., 16, 16)` places patch (y, x) at spatial position (y, x)
    — the same physical region as VAE latent position (y * 28/16, x * 28/16).
    """
    def __init__(self, model_name="UNI2"):
        super().__init__()
        self.model = foundation_model(model_name)
        self.num_prefix = self.model.num_prefix_tokens   # 9 for UNI2 (1 CLS + 8 regs)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(x)            # (B, 265, 1536)
        return feats[:, self.num_prefix : self.num_prefix + 256, :]   # (B, 256, 1536)


class UNI2ContextProj(nn.Module):
    """UNI2 tokens (B, 256, 1536) → cross-attention context (B, 256, ctx_dim)."""
    def __init__(self, uni2_dim: int = 1536, ctx_dim: int = 512):
        super().__init__()
        self.proj = nn.Linear(uni2_dim, ctx_dim)
        self.norm = nn.LayerNorm(ctx_dim)

    def forward(self, uni2_tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(uni2_tokens))


class MarkerTextEncoder(nn.Module):
    """Frozen BiomedBERT → (B, N_CONCEPTS, 768) mean-pooled phrase embeddings."""
    MAX_LEN  = 64
    HF_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(self.HF_MODEL)
        self.model     = AutoModel.from_pretrained(self.HF_MODEL)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _mean_pool(self, enc) -> torch.Tensor:
        """Mean-pool last hidden state over real (non-padding) tokens → (N, 768)."""
        hidden = self.model(**enc).last_hidden_state          # (N, seq_len, 768)
        mask   = enc["attention_mask"].float().unsqueeze(-1)  # (N, seq_len, 1)
        return (hidden * mask).sum(1) / mask.sum(1)           # (N, 768)

    @torch.no_grad()
    def forward(self, marker_names: list[str]) -> torch.Tensor:
        """marker_names → (B, N_CONCEPTS, 768)"""
        device = next(self.model.parameters()).device
        phrases = []
        for name in marker_names:
            concepts = MARKER_CONCEPTS.get(name, (name,) * N_CONCEPTS)
            phrases.extend(concepts)

        enc = self.tokenizer(
            phrases,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.MAX_LEN,
        ).to(device)

        pooled = self._mean_pool(enc)                          # (B*N_CONCEPTS, 768)
        return pooled.reshape(len(marker_names), N_CONCEPTS, -1)  # (B, N_CONCEPTS, 768)


# ── Diffusion utilities ───────────────────────────────────────────────────────

class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        emb  = t[:, None].float() * freq[None]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)   # (B, dim)

class GaussianDiffusion:
    """Linear noise schedule, T steps."""
    def __init__(self, T=1000, beta_start=1e-4, beta_end=2e-2, device="cuda",
                 snr_gamma: float | None = 5.0):
        """
        "If the image is already 83% or more pure, start lowering the weight because the model has already learned the easy stuff. 
        If it's noisier than that (less than 83% signal), give it full importance (weight 1.0)."
        """
        self.T        = T
        betas         = torch.linspace(beta_start, beta_end, T, device=device)
        alpha_bar     = torch.cumprod(1 - betas, dim=0)
        self.betas         = betas
        self.alphas        = 1 - betas                    # α_t  (per-step, not cumulative)
        self.alpha_bar     = alpha_bar
        self.sqrt_ab       = alpha_bar.sqrt()
        self.sqrt_one_mab  = (1 - alpha_bar).sqrt()

        # min-SNR-γ weights (Hang et al. 2023): SNR(t) = ᾱ_t / (1-ᾱ_t)
        # weight(t) = min(SNR(t), γ) / SNR(t)
        # → high-SNR (easy, low-noise) steps are downweighted toward γ/SNR
        # → low-SNR  (hard, high-noise) steps get weight ≈ 1
        # set snr_gamma=None to disable and use plain MSE
        if snr_gamma is not None:
            snr              = alpha_bar / (1 - alpha_bar)          # (T,)
            self.snr_weights = (torch.clamp(snr, max=snr_gamma) / snr)  # (T,)
        else:
            self.snr_weights = None

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise=None):
        """z_t = sqrt(ᾱ_t)·z0 + sqrt(1-ᾱ_t)·ε"""
        if noise is None:
            noise = torch.randn_like(z0)
        s_ab  = self.sqrt_ab[t][:, None, None, None]
        s_mab = self.sqrt_one_mab[t][:, None, None, None]
        # (B, 4, H/8 , W/8)
        return s_ab * z0 + s_mab * noise, noise

    @torch.no_grad()
    def p_sample_loop_cfg(self, unet, shape,
                          marker_emb_c, marker_emb_u, text_ctx, he_latent,
                          guidance_scale: float, device):
        """DDPM sampling with CFG on the per-marker learned embedding.

        ε_c uses the real marker embedding; ε_u uses the learned null.
        Text cross-attention and H&E latent stay on for both passes.
        Combined as ε = ε_u + w·(ε_c - ε_u).
        """
        z = torch.randn(shape, device=device)
        for t_val in reversed(range(self.T)):
            t    = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            z_in = torch.cat([z, he_latent], dim=1)
            if marker_emb_u is not None:
                eps_c = unet(z_in, t, marker_emb_c, text_ctx)
                eps_u = unet(z_in, t, marker_emb_u, text_ctx)
                eps   = eps_u + guidance_scale * (eps_c - eps_u)
            else:
                eps = unet(z_in, t, marker_emb_c, text_ctx)

            ab       = self.alpha_bar[t_val]
            ab_prev  = self.alpha_bar[t_val - 1] if t_val > 0 else torch.tensor(1.0, device=device)
            beta_t   = self.betas[t_val]
            alpha_t  = self.alphas[t_val]

            mean = (1 / alpha_t.sqrt()) * (z - beta_t / (1 - ab).sqrt() * eps)
            var  = beta_t * (1 - ab_prev) / (1 - ab)
            z    = mean + var.sqrt() * torch.randn_like(z) * (t_val > 0)
        return z


# ── Weighted loss ─────────────────────────────────────────────────────────────

def weighted_noise_loss(noise_pred: torch.Tensor, noise: torch.Tensor,
                        weights: torch.Tensor | None = None,
                        snr_weights: torch.Tensor | None = None,
                        t: torch.Tensor | None = None) -> torch.Tensor:
    """
    Per-sample MSE between predicted and actual noise, with two optional weightings
    that are multiplied together:

    weights     : (B,) — 1/σ_marker, upweights rare markers 
    snr_weights : (T,) — precomputed min-SNR-γ table from GaussianDiffusion
    t           : (B,) — sampled timesteps, used to index snr_weights
    """
    per_sample = (noise_pred - noise).pow(2).mean(dim=(1, 2, 3))   # (B,)
    if snr_weights is not None and t is not None:
        per_sample = per_sample * snr_weights[t]                    # reweight by timestep
    # based on std (increase weight of sparse marker low std)
    if weights is not None:
        per_sample = per_sample * weights
    return per_sample.mean()


# ── Conditioning U-Net ────────────────────────────────────────────────────────

class CrossAttention(nn.Module):
    """Multi-head cross-attention with per-head QK-norm."""
    def __init__(self, query_dim: int, context_dim: int, n_heads: int = 8):
        super().__init__()
        self.n_heads  = n_heads
        self.head_dim = query_dim // n_heads
        self.q = nn.Linear(query_dim,   query_dim, bias=False)
        self.k = nn.Linear(context_dim, query_dim, bias=False)
        self.v = nn.Linear(context_dim, query_dim, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.out = nn.Linear(query_dim, query_dim)
        self.debug        = False
        self.last_weights = None

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(context).reshape(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(context).reshape(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.debug:
            scale = self.head_dim ** -0.5
            w = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)
            self.last_weights = w.detach()
            attn = torch.matmul(w, v)
        else:
            attn = F.scaled_dot_product_attention(q, k, v)
        return self.out(attn.transpose(1, 2).reshape(B, N, C))


class ResBlock(nn.Module):
    """Conv block with AdaGN timestep conditioning and optional gated cross-attention."""
    def __init__(self, ch: int, t_dim: int, ctx_dim: int, use_xattn: bool = True):
        super().__init__()
        self.norm1  = nn.GroupNorm(8, ch)
        self.conv1  = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm2  = nn.GroupNorm(8, ch)
        self.conv2  = nn.Conv2d(ch, ch, 3, padding=1)
        self.t_proj = nn.Linear(t_dim, ch * 2)
        self.use_xattn = use_xattn
        if use_xattn:
            self.xattn      = CrossAttention(ch, ctx_dim)
            self.xattn_gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor,
                context: torch.Tensor | None) -> torch.Tensor:
        scale, shift = self.t_proj(F.silu(t_emb)).chunk(2, dim=-1)
        h = self.conv1(F.silu(self.norm1(x)))
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        x = x + h

        if not self.use_xattn or context is None:
            return x

        B, C, H, W = x.shape
        xf = x.flatten(2).transpose(1, 2)
        xf = xf + torch.tanh(self.xattn_gate) * self.xattn(xf, context)
        return xf.transpose(1, 2).reshape(B, C, H, W)


class ConditionalUNet(nn.Module):
    """
    Denoising U-Net.

    Conditioning:
      - timestep → AdaGN (every ResBlock).
      - learned per-marker embedding (nn.Embedding, indexed by marker id) →
        added to t_emb. Stochastically replaced with `null_marker` for CFG.
      - BiomedBERT marker phrases (B, N_CONCEPTS, 768) → projected to ctx_dim →
        cross-attention K/V in every ResBlock (always on, shared across blocks).
    """
    def __init__(self, c_lat: int = C_LAT, base_ch: int = 128, t_dim: int = 256,
                 text_dim: int = 768, ctx_dim: int = 512,
                 num_markers: int = 1, use_xattn: bool = True):
        super().__init__()
        ch = base_ch
        self.use_xattn = use_xattn

        self.t_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(t_dim),
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        # Per-marker learned embedding; added to t_emb for discriminative identity.
        self.marker_embed = nn.Embedding(num_markers, t_dim)
        # Learned null vector used when the marker embedding is dropped for CFG.
        self.null_marker = nn.Parameter(torch.zeros(t_dim))

        # Text phrases → cross-attention K/V (B, N_CONCEPTS, ctx_dim).
        self.text_ctx_proj = nn.Sequential(
            nn.Linear(text_dim, ctx_dim),
            nn.LayerNorm(ctx_dim),
        ) if use_xattn else None

        # 8 channels: 4 noisy protein latent + 4 H&E latent (channel-concat).
        self.input_conv = nn.Conv2d(c_lat * 2, ch, 3, padding=1)

        rb = lambda c: ResBlock(c, t_dim, ctx_dim, use_xattn=use_xattn)
        self.enc1  = rb(ch)
        self.down1 = nn.Conv2d(ch,   ch*2,  4, 2, 1)
        self.enc2  = rb(ch*2)
        self.down2 = nn.Conv2d(ch*2, ch*4,  4, 2, 1)

        self.bottleneck = rb(ch*4)

        self.up2  = nn.ConvTranspose2d(ch*4, ch*2, 4, 2, 1)
        self.dec2 = rb(ch*4)
        self.up1  = nn.ConvTranspose2d(ch*4, ch,   4, 2, 1)
        self.dec1 = rb(ch*2)

        self.out = nn.Conv2d(ch*2, c_lat, 1)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor,
                marker_emb: torch.Tensor,
                text_ctx: torch.Tensor | None,
                cnn_feats=None) -> torch.Tensor:
        """
        z_t        : (B, 2*c_lat, h, w) noisy protein latent ‖ H&E latent
        t          : (B,)
        marker_emb : (B, t_dim) — embed(marker_idx) or broadcast null_marker
        text_ctx   : (B, N_CONCEPTS, ctx_dim) K/V for cross-attention (or None)
        cnn_feats  : optional (f1, f2) from HEDetailCNN
        """
        t_emb = self.t_embed(t) + marker_emb

        x  = self.input_conv(z_t)
        e1 = self.enc1(x,              t_emb, text_ctx)
        e2 = self.enc2(self.down1(e1), t_emb, text_ctx)
        m  = self.bottleneck(self.down2(e2), t_emb, text_ctx)

        skip2 = e2 + cnn_feats[1] if cnn_feats is not None else e2
        skip1 = e1 + cnn_feats[0] if cnn_feats is not None else e1

        d2 = self.dec2(torch.cat([self.up2(m),  skip2], dim=1), t_emb, text_ctx)
        d1 = self.dec1(torch.cat([self.up1(d2), skip1], dim=1), t_emb, text_ctx)
        return self.out(d1)


# ── Full LDM pipeline ─────────────────────────────────────────────────────────

class LDM(nn.Module):
    def __init__(self, vae: AutoencoderKL, unet: ConditionalUNet,
                 he_cnn: HEDetailCNN, text_encoder: MarkerTextEncoder,
                 diffusion: GaussianDiffusion,
                 marker_names: list[str],
                 cfg_drop_prob: float = 0.1,
                 guidance_scale: float = 3.0):
        super().__init__()
        self.vae            = vae
        self.he_cnn         = he_cnn
        self.text_encoder   = text_encoder
        self.unet           = unet
        self.diffusion      = diffusion
        self.cfg_drop_prob  = cfg_drop_prob
        self.guidance_scale = guidance_scale
        self.marker_names   = list(marker_names)
        self.marker_to_idx  = {m: i for i, m in enumerate(self.marker_names)}

    def _marker_indices(self, marker_names: list[str], device) -> torch.Tensor:
        return torch.tensor([self.marker_to_idx[m] for m in marker_names],
                            device=device, dtype=torch.long)

    def _marker_emb_with_cfg_drop(self, marker_idx: torch.Tensor) -> torch.Tensor:
        """Embed marker ids, stochastically replacing each row with `null_marker`."""
        emb = self.unet.marker_embed(marker_idx)                         # (B, t_dim)
        if self.cfg_drop_prob <= 0.0:
            return emb
        B = emb.size(0)
        drop = (torch.rand(B, device=emb.device) < self.cfg_drop_prob)   # (B,)
        null = self.unet.null_marker.unsqueeze(0).expand(B, -1)          # (B, t_dim)
        return torch.where(drop[:, None], null, emb)

    def _text_ctx(self, marker_names: list[str]) -> torch.Tensor | None:
        """Compute cross-attention context from BiomedBERT marker phrases."""
        if not self.unet.use_xattn:
            return None
        text_emb = self.text_encoder(marker_names)          # (B, N_CONCEPTS, 768)
        return self.unet.text_ctx_proj(text_emb)            # (B, N_CONCEPTS, ctx_dim)

    def training_step(self, he_patches: torch.Tensor, protein_maps: torch.Tensor,
                      marker_names: list[str],
                      marker_weights: torch.Tensor | None = None) -> torch.Tensor:
        device = he_patches.device
        B = he_patches.size(0)

        z0         = vae_encode(self.vae, protein_maps)
        he_latent  = vae_encode_he(self.vae, he_patches)
        text_ctx   = self._text_ctx(marker_names)
        m_idx      = self._marker_indices(marker_names, device)
        m_emb      = self._marker_emb_with_cfg_drop(m_idx)

        t = torch.randint(0, self.diffusion.T, (B,), device=device)
        z_t, noise = self.diffusion.q_sample(z0, t)

        z_in       = torch.cat([z_t, he_latent], dim=1)

        noise_pred = self.unet(z_in, t, m_emb, text_ctx)

        return weighted_noise_loss(noise_pred, noise, marker_weights,
                                   snr_weights=self.diffusion.snr_weights, t=t)

    @torch.no_grad()
    def generate(self, he_patches: torch.Tensor, marker_name: str,
                 guidance_scale: float | None = None) -> torch.Tensor:
        """CFG-guided sampling. guidance_scale None → use self.guidance_scale."""
        device    = he_patches.device
        B         = he_patches.shape[0]
        w         = guidance_scale if guidance_scale is not None else self.guidance_scale

        he_latent = vae_encode_he(self.vae, he_patches)
        text_ctx  = self._text_ctx([marker_name] * B)
        idx       = self._marker_indices([marker_name] * B, device)
        m_emb_c   = self.unet.marker_embed(idx)                         # conditional
        # CFG off → pass None; the loop skips the second forward pass.
        m_emb_u   = (self.unet.null_marker.unsqueeze(0).expand(B, -1)
                     if self.cfg_drop_prob > 0 else None)

        h = w_ = he_patches.shape[-1] // 8
        z = self.diffusion.p_sample_loop_cfg(
            self.unet, (B, C_LAT, h, w_),
            m_emb_c, m_emb_u, text_ctx, he_latent,
            guidance_scale=w, device=device,
        )
        return vae_decode(self.vae, z)


import os
import sys
import csv
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import DataLoader, Subset, random_split
from skimage.metrics import structural_similarity as skimage_ssim
from scipy.stats import pearsonr
from dataset_orion_ldm import OrionLDMDataset

ORION_DATA_DIR = Path("/mnt/ssd1/virtual_proteomics/data/ORION_CRC")
H5_DIR         = Path("orion_crc_patch_dataset")
P99_FILE       = H5_DIR / "p99s_slide.txt"
LAMBDA_JSON    = Path("MIPHEI-ViT/preprocessings/mif_cleaning/lambda_settings/orion.json")


VAL_FRAC    = 0.2
BATCH_SIZE  = 512
NUM_EPOCHS  = 20
LR          = 1e-4
NUM_WORKERS = 8
SEED        = 42
DISPLAY_SIZE = 224
TIMESTEPS = 1000

OUTPUT_DIR   = Path(f"training_outputs/outputs_ldm_T{TIMESTEPS}_LR{LR}_EPOCHS{NUM_EPOCHS}-no0shot")

def plot_loss(train_losses, val_losses=None, path=OUTPUT_DIR / 'plot_loss.png'):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train", color='red')
    if val_losses:
        plt.plot(epochs, val_losses, label="Val", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Noise MSE (latent space)")
    plt.title("Train vs Val Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

 
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

VAL_VIZ_BATCHES = 1   # number of batches used for expensive generate() metrics each epoch


def compute_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict:
    """
    pred, gt : (B, 1, H, W) float tensors in [0, 1]
    Returns dict with mean SSIM, PSNR, PCC over the batch.
    """
    pred_np = pred.squeeze(1).cpu().numpy()   # (B, H, W)
    gt_np   = gt.squeeze(1).cpu().numpy()

    ssims, psnrs, pccs = [], [], []
    for p, g in zip(pred_np, gt_np):
        ssims.append(skimage_ssim(p, g, data_range=1.0))

        mse = np.mean((p - g) ** 2)
        psnrs.append(float('inf') if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse)))

        pf, gf = p.ravel(), g.ravel()
        if gf.std() < 1e-6 or pf.std() < 1e-6:
            pccs.append(0.0)   # constant map → undefined correlation
        else:
            pccs.append(pearsonr(pf, gf).statistic)

    return {
        "ssim": float(np.mean(ssims)),
        "psnr": float(np.mean(psnrs)),
        "pcc":  float(np.mean(pccs)),
    }


GEN_BATCH_SIZE  = 64    # sub-batch size for generate(); lower if OOM during validation
N_VIZ           = 8     # samples saved per marker in the per-epoch figure
N_EVAL_PATCHES  = 256   # minimum patches per marker before metrics are finalised

def validate_epoch(ldm_model, val_loader, device, marker_names: list[str]):
    """
    Iterate val_loader until every marker has accumulated at least N_EVAL_PATCHES
    evaluated patches, guaranteeing equal sample counts across markers.

    Samples within each batch are grouped by marker so generate() is always
    called with a single uniform marker string.  Generation is chunked into
    GEN_BATCH_SIZE sub-batches to avoid OOM.

    Returns:
        mean_metrics : dict(ssim, psnr, pcc) — mean over all markers
        per_marker   : dict  marker → dict(ssim, psnr, pcc, he, pred, gt)
                       he/pred/gt hold up to N_VIZ samples for visualisation
    """
    ldm_model.he_cnn.eval()
    ldm_model.unet.eval()

    acc   = {m: {"ssim": [], "psnr": [], "pcc": [],
                 "he": [], "pred": [], "gt": [],
                 "n": 0} for m in marker_names}

    with torch.no_grad():
        for he_patches, protein_maps, batch_markers, _ in val_loader:
            # stop as soon as every marker has enough patches
            if all(acc[m]["n"] >= N_EVAL_PATCHES for m in marker_names):
                break

            he_patches   = he_patches.to(device)
            protein_maps = protein_maps.to(device)

            for marker in marker_names:
                if acc[marker]["n"] >= N_EVAL_PATCHES:
                    continue   # this marker is done — skip to save time

                idx = [i for i, m in enumerate(batch_markers) if m == marker]
                if not idx:
                    continue

                idx_t = torch.tensor(idx, device=device)
                he_m  = he_patches[idx_t]
                gt_m  = protein_maps[idx_t]

                pred_chunks = []
                for start in range(0, len(he_m), GEN_BATCH_SIZE):
                    pred_chunks.append(
                        ldm_model.generate(he_m[start : start + GEN_BATCH_SIZE],
                                           marker_name=marker))
                preds = torch.cat(pred_chunks, dim=0)

                m_dict = compute_metrics(preds, gt_m)
                acc[marker]["ssim"].append(m_dict["ssim"])
                acc[marker]["psnr"].append(m_dict["psnr"])
                acc[marker]["pcc"].append(m_dict["pcc"])
                acc[marker]["n"] += len(he_m)

                # collect up to N_VIZ samples for visualisation
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
            "pcc":  float(np.mean(a["pcc"])),
            "n":    a["n"],
            "he":   torch.cat(a["he"],   dim=0) if a["he"]   else None,
            "pred": torch.cat(a["pred"], dim=0) if a["pred"] else None,
            "gt":   torch.cat(a["gt"],   dim=0) if a["gt"]   else None,
        }

    mean_metrics = {
        "ssim": float(np.mean([v["ssim"] for v in per_marker.values()])),
        "psnr": float(np.mean([v["psnr"] for v in per_marker.values()])),
        "pcc":  float(np.mean([v["pcc"]  for v in per_marker.values()])),
    }
    return mean_metrics, per_marker


def plot_metrics(history: list[dict], path, marker_names: list[str] | None = None):
    """
    history     : list of dicts, one per epoch.
                  Each dict has mean keys (ssim, psnr, pcc) and optionally
                  per-marker keys ({marker}_ssim, {marker}_psnr, {marker}_pcc).
    marker_names: if provided, one coloured line per marker is drawn in addition
                  to the bold mean line.
    """
    epochs = range(1, len(history) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, label in zip(axes,
                               ["ssim",  "psnr",      "pcc"],
                               ["SSIM",  "PSNR (dB)", "Pearson CC"]):
        # per-marker lines (thin, coloured)
        if marker_names:
            for marker in marker_names:
                mk_key = f"{marker}_{key}"
                vals = [h.get(mk_key) for h in history]
                if any(v is not None for v in vals):
                    ax.plot(epochs, vals, linewidth=1, alpha=0.6, label=marker)
        # mean line (bold)
        ax.plot(epochs, [h[key] for h in history],
                color="black", linewidth=2, marker='o', label="mean")
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.grid(True)
        if marker_names:
            ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def log_attn_stats(ldm_model, val_loader, device, marker_name: str) -> dict:
    """Per-layer cross-attention stats over the N_CONCEPTS text context tokens.

    Reports Shannon entropy (nats) averaged over heads & spatial queries,
    and the peak attention weight. Max entropy with N_CONCEPTS tokens = log(N_CONCEPTS).
    """
    ldm_model.he_cnn.eval()
    ldm_model.unet.eval()

    he_patch, _, _, _ = next(iter(val_loader))
    he_patch = he_patch[:1].to(device)

    xattn_layers = {
        name: mod
        for name, mod in ldm_model.unet.named_modules()
        if isinstance(mod, CrossAttention)
    }
    for mod in xattn_layers.values():
        mod.debug = True

    with torch.no_grad():
        he_latent = vae_encode_he(ldm_model.vae, he_patch)
        text_ctx  = ldm_model._text_ctx([marker_name])
        m_idx     = ldm_model._marker_indices([marker_name], device)
        m_emb     = ldm_model.unet.marker_embed(m_idx)
        h = w     = he_patch.shape[-1] // 8
        z_t       = torch.randn(1, C_LAT, h, w, device=device)
        z_in      = torch.cat([z_t, he_latent], dim=1)
        t_mid     = torch.full((1,), ldm_model.diffusion.T // 2,
                               device=device, dtype=torch.long)
        ldm_model.unet(z_in, t_mid, m_emb, text_ctx)

    stats = {}
    for name, mod in xattn_layers.items():
        w_attn = mod.last_weights[0]            # (n_heads, N_pixels, N_CONCEPTS)
        eps = 1e-9
        H    = -(w_attn * (w_attn + eps).log()).sum(dim=-1).mean().item()
        peak = w_attn.max(dim=-1).values.mean().item()
        stats[name] = {"entropy": H, "peak": peak}
        mod.debug        = False
        mod.last_weights = None

    return stats


def train_ldm():
    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset + slide-level split ──────────────────────────────────────────
    dataset = OrionLDMDataset(
        h5_dir      = H5_DIR,
        tiff_dir    = ORION_DATA_DIR,
        p99_file    = P99_FILE,
        lambda_json = LAMBDA_JSON,
        markers     = ["Hoechst", "SMA", "Pan-CK"],   # single-marker first run; None = all 16
        num_slides  = 1,
    )

    if SLIDE_SPLIT:
        n_slides = len(dataset.slide_names)
        rng      = np.random.default_rng(SEED)
        order    = rng.permutation(n_slides).tolist()
        n_val    = max(1, int(n_slides * VAL_FRAC))
        val_set  = set(order[:n_val])
        train_idx, val_idx = [], []
        for i, (slide_idx, *_) in enumerate(dataset.patch_map):
            (val_idx if slide_idx in val_set else train_idx).append(i)

        train_ds = Subset(dataset, train_idx)
        val_ds   = Subset(dataset, val_idx)
        print(f"Slide-level split: val slides {sorted(val_set)} "
            f"| train {len(train_ds)}  val {len(val_ds)}")

    else:
        n_val   = int(len(dataset) * VAL_FRAC)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(SEED))


    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True) # Transfers data to GPU faster, # Keeps handles open between epochs
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    val_metric_patches = min(VAL_VIZ_BATCHES * val_loader.batch_size, len(val_ds))
    print(f"\n{'─'*55}")
    print(f"  Dataset       : {len(dataset)} patches total")
    print(f"  Train patches : {len(train_ds)}  ({len(train_loader)} batches × bs={BATCH_SIZE})")
    print(f"  Val patches   : {len(val_ds)}  ({len(val_loader)} batches × bs={val_loader.batch_size})")
    print(f"  Val noise loss: all {len(val_ds)} val patches  (fast forward pass)")
    print(f"  Val metrics   : {val_metric_patches} patches  ({VAL_VIZ_BATCHES} batches, runs generate())")
    print(f"{'─'*55}\n")

    marker_names = dataset.marker_names   # e.g. ["Hoechst", "SMA", "Pan-CK", ...]

    # ── Model ────────────────────────────────────────────────────────────────
    diffusion    = GaussianDiffusion(T=TIMESTEPS, device=device)
    vae          = load_vae(device=device)
    he_cnn       = HEDetailCNN(base_ch=128).to(device)
    text_encoder = MarkerTextEncoder().to(device)
    unet         = ConditionalUNet(num_markers=len(marker_names),
                                   use_xattn=False).to(device)
    ldm          = LDM(vae, unet, he_cnn, text_encoder, diffusion,
                       marker_names=marker_names,
                       cfg_drop_prob=0.0, guidance_scale=3.0)

    optimizer = torch.optim.AdamW(
        list(unet.parameters()), lr=LR 
    )

    num_parameters = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    print(f'Num parameters = {num_parameters}')

    log_path             = OUTPUT_DIR / "training_log.csv"
    best_mean_ssim       = -1.0
    train_losses         = []
    val_losses           = []
    val_metrics_history  = []

    # ── CSV header: mean cols + per-marker SSIM, PSNR, PCC ──────────────────
    per_marker_cols = ([f"{m}_ssim" for m in marker_names] +
                       [f"{m}_psnr" for m in marker_names] +
                       [f"{m}_pcc"  for m in marker_names])
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_loss", "mean_ssim", "mean_psnr", "mean_pcc"]
            + per_marker_cols
        )

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f'Training epoch {epoch} ...')
        ldm.he_cnn.train()
        ldm.unet.train()
        loss_epoch = 0.0
        for i, (he_patches, protein_maps, batch_marker_names, marker_weights) in enumerate(train_loader):
            he_patches     = he_patches.to(device)
            protein_maps   = protein_maps.to(device)
            marker_weights = marker_weights.to(device)

            loss = ldm.training_step(he_patches, protein_maps,
                                     list(batch_marker_names), marker_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item() * len(he_patches)
            print(f'End training batch {i + 1}/{len(train_loader)}')

        train_loss = loss_epoch / len(train_loader.dataset)
        train_losses.append(train_loss)

        # ── Val noise loss (fast — no generate(), just forward pass) ─────────
        ldm.he_cnn.eval()
        ldm.unet.eval()
        val_loss_epoch = 0.0
        with torch.no_grad():
            print('Validation ...')
            for i, (he_patches, protein_maps, batch_marker_names, marker_weights) in enumerate(val_loader):
                he_patches     = he_patches.to(device)
                protein_maps   = protein_maps.to(device)
                marker_weights = marker_weights.to(device)
                val_loss_epoch += ldm.training_step(
                    he_patches, protein_maps, list(batch_marker_names), marker_weights
                ).item() * len(he_patches)
                print(f'End validation batch {i + 1}/{len(val_loader)}')
        val_loss = val_loss_epoch / len(val_loader.dataset)
        val_losses.append(val_loss)

        plot_loss(train_losses, val_losses)
        print(f"Epoch {epoch}/{NUM_EPOCHS}  train={train_loss:.5f}  val={val_loss:.5f}  — running image metrics...")

        # ── Val image metrics (expensive — runs generate()) ──────────────────
        # Fix RNG so every epoch samples the same starting noise → comparable metrics
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        mean_metrics, per_marker = validate_epoch(
            ldm, val_loader, device, marker_names=marker_names)

        # merge per-marker metrics into the history entry for plotting
        row_metrics = dict(mean_metrics)
        for m, v in per_marker.items():
            row_metrics[f"{m}_ssim"] = v["ssim"]
            row_metrics[f"{m}_psnr"] = v["psnr"]
            row_metrics[f"{m}_pcc"]  = v["pcc"]
        val_metrics_history.append(row_metrics)
        plot_metrics(val_metrics_history, OUTPUT_DIR / "val_metrics.png",
                     marker_names=marker_names)

        # log to CSV
        with open(log_path, "a", newline="") as f:
            def _fmt(marker, key):
                return f"{per_marker.get(marker, {}).get(key, float('nan')):.4f}"
            pm_vals = ([_fmt(m, "ssim") for m in marker_names] +
                       [_fmt(m, "psnr") for m in marker_names] +
                       [_fmt(m, "pcc")  for m in marker_names])
            csv.writer(f).writerow(
                [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                 f"{mean_metrics['ssim']:.4f}",
                 f"{mean_metrics['psnr']:.4f}",
                 f"{mean_metrics['pcc']:.4f}"]
                + pm_vals
            )

        print(f"           mean metrics  SSIM={mean_metrics['ssim']:.4f}  "
              f"PSNR={mean_metrics['psnr']:.2f}dB  PCC={mean_metrics['pcc']:.4f}")
        for m, v in per_marker.items():
            print(f"             {m:15s}  SSIM={v['ssim']:.4f}  "
                  f"PSNR={v['psnr']:.2f}dB  PCC={v['pcc']:.4f}  "
                  f"(n={v['n']})")

        # ── Save best checkpoint (by mean SSIM across markers) ───────────────
        if mean_metrics["ssim"] > best_mean_ssim:
            best_mean_ssim = mean_metrics["ssim"]
            torch.save({
                "epoch":        epoch,
                "unet":         unet.state_dict(),
                "he_cnn":       he_cnn.state_dict(),
                "metrics":      mean_metrics,
                "per_marker":   {m: {k: v[k] for k in ("ssim", "psnr", "pcc")}
                                 for m, v in per_marker.items()},
                "marker_names": marker_names,
            }, OUTPUT_DIR / "best_model.pt")
            print(f"           -> best model saved (mean SSIM={best_mean_ssim:.4f})")

        # ── Per-epoch visualisation: one PNG per marker ───────────────────────
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

    # ── Final checkpoint ─────────────────────────────────────────────────────
    torch.save({"unet": unet.state_dict(), "he_cnn": he_cnn.state_dict()},
               OUTPUT_DIR / "last_model.pt")


def save_prediction_figure(he_patches,
                            predictions: list,
                            protein_maps,
                            out_path,
                            marker_name: str = "Hoechst"):
    """Save rows of (H&E | prediction | GT) to out_path. One row per sample."""
    num_pred = len(predictions)
    fig, axes = plt.subplots(num_pred, 3,
                             figsize=(3 * DISPLAY_SIZE / 72, num_pred * DISPLAY_SIZE / 72),
                             squeeze=False)  # always (N, 3) even when num_pred=1

    for i in range(num_pred):
        he_display = he_patches[i].permute(1, 2, 0).cpu().numpy()
        he_display = (he_display * IMAGENET_STD) + IMAGENET_MEAN
        he_display = np.clip(he_display, 0, 1)

        axes[i, 0].imshow(he_display)
        axes[i, 0].set_title("H&E Patch", fontsize=9)

        axes[i, 1].imshow(predictions[i].squeeze().cpu().numpy(), cmap='gray')
        axes[i, 1].set_title(f"Pred: {marker_name}", fontsize=9)

        axes[i, 2].imshow(protein_maps[i].squeeze().cpu().numpy(), cmap='gray')
        axes[i, 2].set_title("GT Protein", fontsize=9)

    for row in axes:
        for ax in row:
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved figure → {out_path}")


def predict_from_checkpoint(checkpoint: str = "model.pt", marker_name: str = "Hoechst", num: int = 1):
    """Load saved weights and run prediction on one validation sample."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = OrionLDMDataset(
        h5_dir      = H5_DIR,
        tiff_dir    = ORION_DATA_DIR,
        p99_file    = P99_FILE,
        lambda_json = LAMBDA_JSON,
        markers     = [marker_name],
        num_slides  = 3,
    )

    n_val   = max(1, int(len(dataset) * VAL_FRAC))
    n_train = len(dataset) - n_val
    _, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED))

    val_loader = DataLoader(val_ds, batch_size=num, shuffle=False, num_workers=0)

    marker_names = dataset.marker_names

    diffusion    = GaussianDiffusion(T=100, device=device)
    vae          = load_vae(device=device)
    he_cnn       = HEDetailCNN(base_ch=128).to(device)
    text_encoder = MarkerTextEncoder().to(device)
    unet         = ConditionalUNet(num_markers=len(marker_names),
                                   use_xattn=True).to(device)
    ldm          = LDM(vae, unet, he_cnn, text_encoder, diffusion,
                       marker_names=marker_names, cfg_drop_prob=0.0)

    ckpt_path = OUTPUT_DIR / checkpoint
    print(f"Loading weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(ckpt["unet"])
    he_cnn.load_state_dict(ckpt["he_cnn"])
    print("Weights loaded.")

    ldm.he_cnn.eval()
    ldm.unet.eval()

    batch = next(iter(val_loader))
    he_patches, protein_maps, _, _ = batch


    predictions = []
    for i in range(num):
        predictions.append(ldm.generate(he_patches[i].unsqueeze(0).to(device), marker_name=marker_name))
    save_prediction_figure(he_patches, predictions, protein_maps,
                           OUTPUT_DIR / 'prediction.png', marker_name=marker_name)


def attn_stats_from_checkpoint(checkpoint: str = "best_model.pt", marker_name: str = "Hoechst"):
    """Load saved weights and print per-layer cross-attention stats.

    Context is now N_CONCEPTS=3 tokens only (localisation / cell type / staining).
    Reports per-layer entropy and mean attention weight per concept token.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = OrionLDMDataset(
        h5_dir      = H5_DIR,
        tiff_dir    = ORION_DATA_DIR,
        p99_file    = P99_FILE,
        lambda_json = LAMBDA_JSON,
        markers     = [marker_name],
        num_slides  = 2,
    )
    _, val_ds = random_split(
        dataset,
        [len(dataset) - max(1, int(len(dataset) * VAL_FRAC)),
         max(1, int(len(dataset) * VAL_FRAC))],
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    marker_names = dataset.marker_names

    diffusion    = GaussianDiffusion(T=TIMESTEPS, device=device)
    vae          = load_vae(device=device)
    he_cnn       = HEDetailCNN(base_ch=128).to(device)
    text_encoder = MarkerTextEncoder().to(device)
    unet         = ConditionalUNet(num_markers=len(marker_names),
                                   use_xattn=True).to(device)
    ldm          = LDM(vae, unet, he_cnn, text_encoder, diffusion,
                       marker_names=marker_names, cfg_drop_prob=0.0)

    ckpt_path = OUTPUT_DIR / checkpoint
    print(f"Loading weights from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(ckpt["unet"])
    he_cnn.load_state_dict(ckpt["he_cnn"])
    print("Weights loaded.\n")

    n_ctx  = N_CONCEPTS                   # text concept tokens per marker
    max_H  = math.log(n_ctx)              # uniform baseline
    unif_w = 1.0 / n_ctx                  # uniform per-token weight
    stats  = log_attn_stats(ldm, val_loader, device, marker_name=marker_name)

    print(f"{'Layer':<35} {'Entropy':>10} {'H/H_max':>10} {'Peak':>10}")
    print("─" * 70)
    for layer, s in stats.items():
        print(f"{layer:<35} {s['entropy']:>10.3f} "
              f"{s['entropy']/max_H:>10.3f} {s['peak']:>10.3f}")
    print("─" * 70)
    print(f"{'(uniform baseline)':35} {max_H:>10.3f} {'1.000':>10} {unif_w:>10.4f}")
    print(f"\nH/H_max → 1.0 : uniform over {N_CONCEPTS} concept tokens (not routing).")
    print(f"H/H_max → 0   : focused on a specific concept token.")
    print(f"Peak          : mean of max attention weight across queries.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", action="store_true",
                        help="Load checkpoint and display prediction instead of training")
    parser.add_argument("--attn_stats", action="store_true",
                        help="Print cross-attention stats for each ResBlock layer")
    parser.add_argument("--checkpoint", default="best_model.pt",
                        help="Checkpoint filename in OUTPUT_DIR (default: best_model.pt)")
    parser.add_argument("--marker", default="Hoechst",
                        help="Marker name for prediction (default: Hoechst)")
    parser.add_argument("--num", type=int, default=4,
                        help="Number of samples to predict (default: 4)")
    args = parser.parse_args()

    if args.predict:
        predict_from_checkpoint(checkpoint=args.checkpoint, marker_name=args.marker, num=args.num)
    elif args.attn_stats:
        attn_stats_from_checkpoint(checkpoint=args.checkpoint, marker_name=args.marker)
    else:
        train_ldm()

"""
  - SSIM (Structural Similarity) — measures perceptual similarity in terms of luminance, contrast, and structure. Range [0,1], higher is better. More sensitive to spatial
  structure than pixel-wise errors, which makes it a good proxy for whether nuclear shapes and positions look right.                                                       
  - PSNR (Peak Signal-to-Noise Ratio) — log-scale ratio of peak signal power to noise power (dB). Higher is better, no fixed upper bound. Directly tied to MSE, so it      
  penalises any pixel deviation equally regardless of location. Less meaningful than SSIM for sparse signals like Hoechst (most pixels are near-zero).                     
  - PCC (Pearson Correlation Coefficient) — linear correlation between predicted and GT pixel intensities. Range [-1,1], higher is better. Invariant to global             
  brightness/contrast shifts, so it captures whether bright regions land in the right places even if the absolute intensities are off. Most commonly reported metric in    
  virtual proteomics papers.   
"""