"""
Shared paths / geometry / loaders for the PathoCell cell-TYPE benchmark.

PathoCell (CRC FFPE CODEX, Schürch et al.) is scored as **cell-type
classification**, NOT per-marker positivity — this is the difference from
`orion_cell_cls`/`hemit_cell_cls`. We reuse MIPHEI's processed data so our
cells are byte-aligned to theirs:

  /mnt/ssd/virtual_proteomics/data/pathocell/pathocell_hdf/<core>.hdf
     img          (3, H, W)  uint16  brightfield H&E (best-Z)         → model input
     gt_inst      (1, H, W)  uint16  nuclei instance mask; label == MIPHEI cell_id
     gt_ct/_coarse(1, H, W)          per-pixel cell-type GT (unused here)
     ifl          (58, H, W) uint16  CODEX IF (unused here)

The GT cell-type one-hot + the fixed 20%/80% train/test `split` come straight
from MIPHEI's released parquet (verified: gt_inst labels for a core == the
cell_ids in that parquet, exactly), so the only thing we swap in is OUR model's
predicted marker expression per cell.

Run scripts from the repo root, e.g.
    python cell_cls/pathocell_cls/build_cell_token_features.py ...
"""
import sys
from pathlib import Path

import numpy as np
import h5py

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Paths ───────────────────────────────────────────────────────────────────
FEAT_DIR         = Path(__file__).resolve().parent   # where cell_token_features_*/eval_* land
HDF_DIR          = Path("/mnt/ssd/virtual_proteomics/data/pathocell/pathocell_hdf")
# MIPHEI's released cell table: cell_id, slide_name, <marker>_pred, 15 celltype
# one-hots, split. We use it for GT celltype + split (model-independent) and, with
# --source parquet, also to reproduce MIPHEI's own pathocell number as a harness check.
DEFAULT_GT_PARQUET = REPO_ROOT / "checkpoints/MIPHEI-vit/pathocell_cell_dataframe_logreg.parquet"
DEFAULT_PRED_DIR   = REPO_ROOT / "training_outputs/outputs_orion_token_UNI2_baseline_bg0.2"

# ── Geometry ────────────────────────────────────────────────────────────────
MPP          = 0.377   # native CODEX resolution (raw bestFocus dims == HDF dims)
MPP_TARGET   = 0.5     # ORION training resolution our model expects
MODEL_INPUT  = 224
TOKEN_GRID   = 16
NTOK         = TOKEN_GRID * TOKEN_GRID   # 256 tokens per patch
# native crop covering the same 112 µm FOV as 224 px @ 0.5 µm/px  → 297 px
PATCH_SIZE_LEVEL0 = round(MODEL_INPUT * MPP_TARGET / MPP)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# MIPHEI's 15 coarse cell types (order matters: matches the parquet one-hot columns)
NUCLEI_CLASSES = [
    "Background", "B cells", "Macrophages/Monocytes", "Adipocytes",
    "Dendritic cells", "T cells", "Granulocytes", "NK cells", "Nerves",
    "Plasma cells", "Smooth muscle", "Stroma", "Tumor cells",
    "Vasculature/Lymphatics", "Other cells",
]


# ── Core discovery / HDF I/O ─────────────────────────────────────────────────

def list_cores(hdf_dir: Path = HDF_DIR) -> list[str]:
    """Sorted core stems, e.g. ['reg001_A', 'reg001_B', ...]."""
    return sorted(p.stem for p in hdf_dir.glob("*.hdf"))


def slide_name_of(core: str) -> str:
    """Core stem → MIPHEI slide_name (they suffix '.ome'): reg001_A → reg001_A.ome."""
    return f"{core}.ome"


def he_uint16_to_uint8(img_chw: np.ndarray) -> np.ndarray:
    """
    (3, H, W) uint16 brightfield → (H, W, 3) uint8, per-channel p99 stretch.

    Identical to build_patch_dataset_pancancer.export_he_rgb so the H&E our model
    sees here matches the H&E used to build the pancancer training targets
    (background → ~255 white, tissue darker).
    """
    rgb16 = img_chw.astype(np.float32)               # (3, H, W)
    rgb8  = np.empty_like(rgb16, dtype=np.uint8)
    for c in range(3):
        p99 = np.percentile(rgb16[c], 99)
        p99 = p99 if p99 > 0 else 1.0
        rgb8[c] = np.clip(rgb16[c] / p99 * 255, 0, 255).astype(np.uint8)
    return rgb8.transpose(1, 2, 0)                   # (H, W, 3)


def load_hdf_core(core: str, hdf_dir: Path = HDF_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Return (he_uint8 (H,W,3), inst (H,W) int32) for one core."""
    with h5py.File(hdf_dir / f"{core}.hdf", "r") as f:
        img  = f["img"][:]              # (3, H, W) uint16
        inst = f["gt_inst"][0].astype(np.int32)
    return he_uint16_to_uint8(img), inst


# ── Model ────────────────────────────────────────────────────────────────────

def load_model(pred_dir: Path, num_outputs: int = 16):
    """
    Load a trained SpatialModel (UNI2 token regressor) + its marker names.
    Returns (model.eval() on cuda/cpu, marker_names list).
    """
    import torch, yaml
    from model import SpatialModel

    pred_dir = Path(pred_dir)
    marker_names = list(np.load(pred_dir / "marker_names.npy"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If the checkpoint was a LoRA finetune, the state holds peft adapters
    # (base_layer + lora_A/B); a plain SpatialModel + strict=False would SILENTLY
    # DROP them → base-UNI2 preds. Rebuild the encoder with the saved LoRA config.
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
    state = torch.load(pred_dir / "best_model.pt", map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)  # FDS buffers ok
    lora_missing = [k for k in missing if "lora_" in k]
    if lora_missing or unexpected:
        raise RuntimeError(f"bad load: lora_missing={lora_missing[:3]} unexpected={unexpected[:3]}")
    model.to(device).eval()
    print(f"  model ← {pred_dir/'best_model.pt'}  ({len(marker_names)} markers)")
    return model, marker_names


def normalise_patch(crop_hw3: np.ndarray) -> np.ndarray:
    """uint8 (H,W,3) crop → resized, ImageNet-normalised (3,224,224) float32.

    Same path as visualize_orion_predictions.extract_patches.
    """
    import cv2
    crop = cv2.resize(crop_hw3.astype(np.float32), (MODEL_INPUT, MODEL_INPUT),
                      interpolation=cv2.INTER_LINEAR)
    crop = (crop / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return crop.transpose(2, 0, 1).astype(np.float32)