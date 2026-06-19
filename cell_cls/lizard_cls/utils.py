"""
Shared paths / geometry / loaders for the Lizard cell-TYPE benchmark.

Lizard (colon H&E, 6 nuclei classes) is scored as cell-type classification, same
protocol as cell_cls/pathocell_cls — the ORION 16-marker model is applied zero-shot
to the H&E, a logreg maps predicted marker expression to the nucleus class on
MIPHEI's fixed 20%/80% split.

Source (Kaggle aadimator/lizard-dataset, unzipped):
  /mnt/ssd/virtual_proteomics/data/lizard/
    lizard_images{1,2}/Lizard_Images{1,2}/<stem>.png   variable-size RGB H&E @ 20x (0.5 µm/px)
    lizard_labels/Lizard_Labels/Labels/<stem>.mat      inst_map (==cell_id), id, class, ...
  slide_name = image stem (e.g. consep_1), matching the parquet.

We read the .mat directly (scipy) — `inst_map` labels == MIPHEI cell_id (verified),
so no pyvips preprocessing is needed. GT cell-type one-hot + `split` come from
MIPHEI's lizard_cell_dataframe_logreg.parquet, joined by (slide_name, cell_id).

Geometry: Lizard is already 20x (0.5 µm/px) = OUR training scale, so PATCH_SIZE_LEVEL0
= round(224·0.5/0.5) = 224 — a native 224 crop IS the model input (no rescale). Images
are larger/variable, so each is tiled into a non-overlapping 224 grid (edges padded).
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Paths ───────────────────────────────────────────────────────────────────
FEAT_DIR           = Path(__file__).resolve().parent
LIZARD_DIR         = Path("/mnt/ssd/virtual_proteomics/data/lizard")
IMG_DIRS           = [LIZARD_DIR / "lizard_images1/Lizard_Images1",
                      LIZARD_DIR / "lizard_images2/Lizard_Images2"]
LABELS_DIR         = LIZARD_DIR / "lizard_labels/Lizard_Labels/Labels"
DEFAULT_GT_PARQUET = REPO_ROOT / "checkpoints/MIPHEI-vit/lizard_cell_dataframe_logreg.parquet"
DEFAULT_PRED_DIR   = REPO_ROOT / "outputs_orion_token_UNI2_baseline_bg0.2"

# ── Geometry ────────────────────────────────────────────────────────────────
MPP          = 0.5     # Lizard native (20x) == our training scale
MPP_TARGET   = 0.5
MODEL_INPUT  = 224
TOKEN_GRID   = 16
NTOK         = TOKEN_GRID * TOKEN_GRID
PATCH_SIZE_LEVEL0 = round(MODEL_INPUT * MPP_TARGET / MPP)      # = 224 (native crop == model input)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# MIPHEI's 6 Lizard nuclei classes (parquet one-hot column order)
NUCLEI_CLASSES = [
    "Neutrophil", "Epithelial", "Lymphocyte", "Plasma", "Eosinophil", "Connective_tissue",
]


# ── Image discovery / I/O ────────────────────────────────────────────────────

def list_images() -> list[Path]:
    """All Lizard H&E PNGs across both subsets."""
    return sorted(p for d in IMG_DIRS for p in d.glob("*.png"))


def slide_name_of(img_path: Path) -> str:
    return img_path.stem


def load_image_and_inst(img_path: Path):
    """Return (he uint8 (H,W,3), inst int32 (H,W)); inst label == cell_id (from .mat)."""
    he = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    mat = sio.loadmat(str(LABELS_DIR / f"{img_path.stem}.mat"))
    inst = mat["inst_map"].astype(np.int32)
    return he, inst


# ── Model ────────────────────────────────────────────────────────────────────

def load_model(pred_dir: Path, num_outputs: int = 16):
    import torch, yaml
    from model import SpatialModel
    pred_dir = Path(pred_dir)
    marker_names = [str(m) for m in np.load(pred_dir / "marker_names.npy")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LoRA checkpoints hold peft adapters (base_layer + lora_A/B); a plain
    # SpatialModel + strict=False would SILENTLY DROP them. Rebuild with the saved
    # LoRA config so the adapters load.
    cfg_path = pred_dir / "config.yaml"
    lora_kw = {}
    if cfg_path.exists():
        ft = yaml.safe_load(cfg_path.read_text()).get("finetune", {})
        if ft.get("mode") == "lora":
            lc = ft["lora"]
            lora_kw = dict(lora_last_n=lc["last_n"], lora_rank=lc["rank"],
                           lora_alpha=lc["alpha"], lora_dropout=lc["dropout"],
                           lora_suffixes=tuple(lc["suffixes"]))
    model = SpatialModel("UNI2", num_outputs=num_outputs,
                         token_grid=TOKEN_GRID, freeze=True, fds_cfg=None, **lora_kw)
    missing, unexpected = model.load_state_dict(
        torch.load(pred_dir / "best_model.pt", map_location=device), strict=False)
    if [k for k in missing if "lora_" in k] or unexpected:
        raise RuntimeError(f"bad load: lora_missing={[k for k in missing if 'lora_' in k][:3]} "
                           f"unexpected={unexpected[:3]}")
    model.to(device).eval()
    print(f"  model ← {pred_dir/'best_model.pt'}  ({len(marker_names)} markers)")
    return model, marker_names


def normalise_patch(crop_hw3: np.ndarray) -> np.ndarray:
    """uint8 (h,w,3) crop (≤224, edge-clamped) → pad white to 224 → ImageNet-norm (3,224,224)."""
    import cv2
    h, w = crop_hw3.shape[:2]
    if (h, w) != (MODEL_INPUT, MODEL_INPUT):
        crop_hw3 = cv2.copyMakeBorder(crop_hw3, 0, MODEL_INPUT - h, 0, MODEL_INPUT - w,
                                      cv2.BORDER_CONSTANT, value=255)
    crop = (crop_hw3.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return crop.transpose(2, 0, 1).astype(np.float32)
