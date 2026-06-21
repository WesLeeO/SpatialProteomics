"""
Shared paths / geometry / loaders for the PanNuke cell-TYPE benchmark.

PanNuke is scored as cell-type classification (5 classes), same protocol as
cell_cls/pathocell_cls — the ORION 16-marker model is applied zero-shot to the
H&E, and a logreg maps predicted marker expression to the nucleus class on
MIPHEI's fixed 20%/80% split.

Source (MIPHEI download + their pannuke_preprocess.convert_fold_npy_to_pngs):
  /mnt/ssd/virtual_proteomics/data/pannuke/process_data/<Fold>/images/img_<tissue>_<f>_<k>.png
                                                       /inst_masks/inst_<tissue>_<f>_<k>.png
  - images : 256×256 RGB H&E @ 40x (0.25 µm/px)
  - inst   : 256×256 uint16 instance mask; label == MIPHEI cell_id
  slide_name = the image stem (img_<tissue>_<f>_<k>), matching the parquet.

GT cell-type one-hot + the `split` come straight from MIPHEI's released
pannuke_cell_dataframe_logreg.parquet (joined by (slide_name, cell_id)),
exactly like PathoCell.

Run scripts from the repo root.
"""
import sys
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Paths ───────────────────────────────────────────────────────────────────
FEAT_DIR           = Path(__file__).resolve().parent
PROCESS_DIR        = Path("/mnt/ssd/virtual_proteomics/data/pannuke/process_data")
DEFAULT_GT_PARQUET = REPO_ROOT / "checkpoints/MIPHEI-vit/pannuke_cell_dataframe_logreg.parquet"
DEFAULT_PRED_DIR   = REPO_ROOT / "training_outputs/outputs_orion_token_UNI2_baseline_bg0.2"

# ── Geometry ────────────────────────────────────────────────────────────────
MPP          = 0.25    # PanNuke native (40x)
MPP_TARGET   = 0.5     # ORION training resolution
MODEL_INPUT  = 224
TOKEN_GRID   = 16
NTOK         = TOKEN_GRID * TOKEN_GRID
# native crop covering the same 112 µm FOV as 224 px @ 0.5 µm/px → 448 px.
# A PanNuke image is only 256 px, so it is one patch padded (white) up to 448.
PATCH_SIZE_LEVEL0 = round(MODEL_INPUT * MPP_TARGET / MPP)      # = 448

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# MIPHEI's 5 PanNuke nuclei classes (parquet one-hot column order)
NUCLEI_CLASSES = [
    "Neoplastic cells", "Inflammatory", "Connective/Soft tissue cells",
    "Dead Cells", "Epithelial",
]


# ── Image discovery / I/O ────────────────────────────────────────────────────

def list_images(process_dir: Path = PROCESS_DIR) -> list[Path]:
    """All H&E PNGs across folds, e.g. process_data/Fold 1/images/img_Breast_1_00000.png."""
    return sorted(process_dir.glob("*/images/img_*.png"))


def slide_name_of(img_path: Path) -> str:
    """img_<tissue>_<f>_<k>.png → 'img_<tissue>_<f>_<k>' (== MIPHEI slide_name)."""
    return img_path.stem


def inst_path_of(img_path: Path) -> Path:
    """images/img_<...>.png → inst_masks/inst_<...>.png (same fold dir)."""
    return img_path.parent.parent / "inst_masks" / img_path.name.replace("img_", "inst_", 1)


def load_image_and_mask(img_path: Path):
    """Return (he uint8 (256,256,3), inst int32 (256,256)); inst label == cell_id."""
    he = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)
    inst = np.asarray(Image.open(inst_path_of(img_path))).astype(np.int32)
    return he, inst


# ── Model ────────────────────────────────────────────────────────────────────

def load_model(pred_dir: Path, num_outputs: int = 16):
    """Load the trained SpatialModel (UNI2 token regressor) + its marker names."""
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


def crop_pad_resize(img: np.ndarray, pad_value: float,
                    ps0: int = PATCH_SIZE_LEVEL0) -> np.ndarray:
    """
    Pad the (small, 256) native image up to ps0 (bottom/right), then resize to
    MODEL_INPUT. Padding keeps the tissue at the correct 0.5 µm/px scale instead
    of stretching it; padded pixels carry no nuclei. (H, W, 3) -> (224, 224, 3).
    """
    import cv2
    H, W = img.shape[:2]
    if (H, W) != (ps0, ps0):
        img = cv2.copyMakeBorder(img, 0, max(0, ps0 - H), 0, max(0, ps0 - W),
                                 cv2.BORDER_CONSTANT, value=pad_value)[:ps0, :ps0]
    return cv2.resize(img, (MODEL_INPUT, MODEL_INPUT), interpolation=cv2.INTER_AREA)


def normalise_patch(he_uint8: np.ndarray, ps0: int = PATCH_SIZE_LEVEL0) -> np.ndarray:
    """uint8 (256,256,3) image → pad→resize→ImageNet-norm → (3,224,224) float32."""
    crop = crop_pad_resize(he_uint8.astype(np.float32), pad_value=255.0, ps0=ps0)
    crop = (crop / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return crop.transpose(2, 0, 1).astype(np.float32)
