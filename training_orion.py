import os
import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from dotenv import load_dotenv
from huggingface_hub import login
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from cucim import CuImage
from PIL import Image
import sys                                                                                                                         
 
from dataset_orion import MultiImageOrionDataset
from dataset_singular_genomics import MultiImageSGDataset
from dataset_hemit import HEMITDataset
from dataset_pathocell import MultiImagePathoCellDataset
from model import Model
from math import sqrt

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # Ensures it writes even if the script crashes

    def flush(self):
        pass

MODEL_NAME  = 'UNI2'
H5_DIR      = Path("orion_crc_patch_dataset")
TIFF_DIR    = Path("/mnt/ssd1/virtual_proteomics/data/ORION_CRC/data")
SLIDE_SPLIT = False  # True → random slide-level split; False → random patch-level split
OUTPUT_DIR  = Path(f"outputs_orion_{MODEL_NAME}-{'full-slide-split' if SLIDE_SPLIT else 'full'}")
HIDDEN_DIM = 1536
VAL_FRAC   = 0.2   
BATCH_SIZE = 1024
NUM_EPOCHS = 10
BASE_LR = 1e-4
LR         = BASE_LR 
LOSS_FN    = "mse"
NUM_WORKERS = 4
SEED       = 42
NUM_OUTPUTS = 16


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pearson_per_marker(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Pearson r for each marker column independently. Returns shape (num_markers,)."""
    return np.array([
        pearsonr(preds[:, j], targets[:, j])[0] # 0 is the actual score 1 is the p-value
        for j in range(targets.shape[1])
    ])

def spearman_rank_per_marker(preds: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.array([
        spearmanr(preds[:, j], targets[:, j])[0]
        for j in range(targets.shape[1])
    ])



def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.set_grad_enabled(training):
        for patches, targets in loader:
            patches = patches.to(device)
            targets = targets.to(device)

            preds = model(patches)
            loss  = criterion(preds, targets)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(patches)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds   = np.concatenate(all_preds,   axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    mean_loss   = total_loss / len(loader.dataset)
    p_per_marker  = pearson_per_marker(all_preds, all_targets)  # (num_markers,)
    s_rank_per_marker = spearman_rank_per_marker(all_preds, all_targets) # (num_markers,)

    return mean_loss, p_per_marker, s_rank_per_marker

def plot_per_marker(train, val, epochs, title, marker_names):
    val_matrix = np.stack(val, axis=0)   # (epochs, num_markers)
    n_markers = val_matrix.shape[1]
    ncols = 6
    nrows = int(np.ceil(n_markers / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5), squeeze=False)

    for j in range(n_markers):
        ax = axes[j // ncols][j % ncols]
        ax.plot(epochs, [r[j] for r in train], label="Train")
        ax.plot(epochs, val_matrix[:, j], label="Val")
        ax.set_title(marker_names[j] if marker_names else f"Marker {j}", fontsize=8)
        ax.set_ylim(-1, 1)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.tick_params(labelsize=6)

    for j in range(n_markers, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8)
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{title.lower()}.png", dpi=150)
    plt.close()

def plot_curves(train_losses, val_losses, train_pearsons, val_pearsons, train_spearmans, val_spearmans, marker_names):
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))
    ax1.plot(epochs, train_losses, label="Train")
    ax1.plot(epochs, val_losses,   label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title(f"Loss ({LOSS_FN.upper()})"); ax1.legend()

    ax2.plot(epochs, [p.mean() for p in train_pearsons], label="Train")
    ax2.plot(epochs, [p.mean() for p in val_pearsons],   label="Val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Mean Pearson")
    ax2.set_title("Pearson (mean over markers)"); ax2.legend()

    ax3.plot(epochs, [s.mean() for s in train_spearmans], label="Train")
    ax3.plot(epochs, [s.mean() for s in val_spearmans],   label="Val")
    ax3.set_xlabel("Epoch"); ax2.set_ylabel("Mean Spearman")
    ax3.set_title("Spearman (mean over markers)"); ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close()

    plot_per_marker(train=train_pearsons, val=val_pearsons, epochs=epochs, title="Per_marker_Pearson", marker_names=marker_names)
    plot_per_marker(train=train_spearmans, val=val_spearmans, epochs=epochs, title="Per_marker_Spearman", marker_names=marker_names)


def train(model_name):



    torch.manual_seed(SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    log_file = OUTPUT_DIR / "training_log.txt"
    sys.stdout = Logger(log_file)

    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    print('Reading from:',  H5_DIR, TIFF_DIR)

    dataset      = MultiImageOrionDataset(str(H5_DIR), str(TIFF_DIR)) #, num_slides=10)
    marker_names = dataset.marker_names

    if SLIDE_SPLIT:
        n_slides = len(dataset.slides)
        rng = np.random.default_rng(SEED)
        slide_order = rng.permutation(n_slides).tolist()
        n_val_slides = max(1, int(n_slides * VAL_FRAC))
        val_slide_set = set(slide_order[:n_val_slides])
        train_idx, val_idx = [], []
        for i, (slide_idx, *_) in enumerate(dataset.patch_map):
            (val_idx if slide_idx in val_slide_set else train_idx).append(i)
        train_ds = Subset(dataset, train_idx)
        val_ds   = Subset(dataset, val_idx)
        print(f"Slide-level split: {n_slides} slides | val slides: {sorted(val_slide_set)}")
    else:
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

    model = Model(model_name, hidden_dim=HIDDEN_DIM, num_outputs=NUM_OUTPUTS).to(device)

    criterion = torch.nn.MSELoss() if LOSS_FN == "mse" else torch.nn.HuberLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    train_losses,  val_losses   = [], []
    train_pearsons, val_pearsons = [], []
    train_spearmans, val_spearmans = [], []
    best_val_pearson = -np.inf


    print(f'Start training for {NUM_EPOCHS} epochs...')

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f'Epoch {epoch}:')
        print('Training...')
        train_loss, train_p, train_s = run_epoch(model, train_loader, criterion, optimizer)
        print('Validating...')
        val_loss, val_p, val_s = run_epoch(model, val_loader, criterion)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_pearsons.append(train_p)
        val_pearsons.append(val_p)
        train_spearmans.append(train_s)
        val_spearmans.append(val_s)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | "
              f"Train  loss: {train_loss:.4f}  pearson: {train_p.mean():.4f} spearman: {train_s.mean():.4f} | "
              f"Val    loss: {val_loss:.4f}  pearson: {val_p.mean():.4f} spearman: {val_s.mean():.4f} ")

        # Per-marker breakdown
        names = marker_names or [f"M{j}" for j in range(len(val_p))]
        for j, (name, p, s) in enumerate(zip(names, val_p, val_s)):
            print(f"    {name:<20s}  val pearson: {p:.4f} val spearman: {s:.4f}")

        if val_p.mean() > best_val_pearson:
            best_val_pearson = val_p.mean()
            torch.save(model.state_dict(), OUTPUT_DIR / "best_model.pt")
            print(f"           -> best model saved (val r={best_val_pearson:.4f})")

        # Save all statistics as numpy arrays after every epoch so plots can
        # be regenerated even if training is interrupted before completion.
        np.save(OUTPUT_DIR / "train_losses.npy",   np.array(train_losses))
        np.save(OUTPUT_DIR / "val_losses.npy",     np.array(val_losses))
        np.save(OUTPUT_DIR / "train_pearsons.npy", np.stack(train_pearsons))   # (epoch, markers)
        np.save(OUTPUT_DIR / "val_pearsons.npy",   np.stack(val_pearsons))
        np.save(OUTPUT_DIR / "train_spearmans.npy",np.stack(train_spearmans))
        np.save(OUTPUT_DIR / "val_spearmans.npy",  np.stack(val_spearmans))
        np.save(OUTPUT_DIR / "marker_names.npy",   np.array(marker_names))

        plot_curves(train_losses, val_losses, train_pearsons, val_pearsons, train_spearmans, val_spearmans, marker_names)

    print(f"\nDone. Best mean val Pearson: {best_val_pearson:.4f}")


# Marker name mapping: ORION name → Singular Genomics name
SG_MARKER_MAP = {
    "CD3e":      "CD3",
    "CD4":       "CD4",
    "CD8a":      "CD8",
    "CD20":      "CD20",
    "CD31":      "CD31",
    "CD68":      "CD68",
    "Ki-67":     "KI67",
    "Pan-CK":    "PanCK",
    "SMA":       "aSMA",
}

# Marker name mapping: ORION name → HEMIT name
HEMIT_MARKER_MAP = {
    "Pan-CK": "Pan-CK",
    "CD3e":   "CD3",
}

# Marker name mapping: ORION name → PathoCell name
PATHOCELL_MARKER_MAP = {
    "CD31":   "CD31",
    "CD45":   "CD45",
    "CD68":   "CD68",
    "CD4":    "CD4",
    "FOXP3":  "FOXP3",
    "CD8a":   "CD8",
    "CD45RO": "CD45RO",
    "CD20":   "CD20",
    "PD-L1":  "PD-L1",
    "CD3e":   "CD3",
    "CD163":  "CD163",
    "Ki-67":  "Ki67",
    "Pan-CK": "Cytokeratin",
    "SMA":    "aSMA",
}



def _run_model_on_loader(model, loader, orion_indices, external_indices, device):
    """Run model inference on a DataLoader, return (preds, targets) for given marker indices."""
    all_preds, all_targets = [], []
    with torch.no_grad():
        for patches, targets, *_ in loader:
            patches = patches.to(device)
            preds   = (model(patches).cpu().numpy() + 0.9) / 1.8
            all_preds.append(preds[:, orion_indices])
            all_targets.append(targets.numpy()[:, external_indices])
    return np.concatenate(all_preds, axis=0), np.concatenate(all_targets, axis=0)


def test_generalization_sg(model_name, model_path, sg_dir):
    """
    Evaluate a model trained on ORION on Singular Genomics data.
    Per slide: only evaluate shared markers that are also valid (actually measured) for that slide.
    """
    log_file = OUTPUT_DIR / "training_log.txt"
    sys.stdout = Logger(log_file)

    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    model = Model(model_name, hidden_dim=HIDDEN_DIM, num_outputs=NUM_OUTPUTS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    sg_dataset      = MultiImageSGDataset(sg_dir)
    sg_marker_names = sg_dataset.marker_names

    h5_files = sorted(Path(H5_DIR).glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No ORION H5 files found in {H5_DIR}")
    with h5py.File(h5_files[0], "r") as f:
        orion_marker_names = list(f.attrs["marker_names"])

    # All shared markers between ORION and SG (superset; filtered per slide by valid_mask)
    orion_indices, sg_indices, shared_names = [], [], []
    for orion_name, sg_name in SG_MARKER_MAP.items():
        if orion_name in orion_marker_names and sg_name in sg_marker_names:
            orion_indices.append(orion_marker_names.index(orion_name))
            sg_indices.append(sg_marker_names.index(sg_name))
            shared_names.append(orion_name)

    print(f"\n  Shared markers ({len(shared_names)}): {shared_names}")

    slide_names = [
        Path(h).stem.replace("_patch_dataset", "")
        for h in sorted(Path(sg_dir).glob("*.h5"))
    ]

    slide_patch_indices = {name: [] for name in slide_names}
    for patch_idx, (slide_idx, *_) in enumerate(sg_dataset.patch_map):
        slide_patch_indices[slide_names[slide_idx]].append(patch_idx)

    all_results = {}   # slide_name → (preds, targets, s_names)

    for slide_name, indices in slide_patch_indices.items():
        if not indices:
            continue

        # Filter shared markers to those actually measured on this slide
        valid_mask = sg_dataset.slide_valid_masks[slide_names.index(slide_name)]
        keep       = [i for i, sg_i in enumerate(sg_indices) if valid_mask[sg_i]]
        s_orion    = [orion_indices[i] for i in keep]
        s_sg       = [sg_indices[i]    for i in keep]
        s_names    = [shared_names[i]  for i in keep]

        loader = DataLoader(
            Subset(sg_dataset, indices),
            batch_size=BATCH_SIZE, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
        )
        preds, targets = _run_model_on_loader(model, loader, s_orion, s_sg, device)
        all_results[slide_name] = (preds, targets, s_names)

        p = pearson_per_marker(preds, targets)
        s = spearman_rank_per_marker(preds, targets)
        print(f"\n{'='*60}")
        print(f"  {slide_name}  ({len(preds)} patches)  valid markers: {s_names}")
        print(f"  Mean Pearson:  {p.mean():.4f}   Mean Spearman: {s.mean():.4f}")
        print(f"{'='*60}")
        for name, pi, si in zip(s_names, p, s):
            print(f"    {name:<20s}  pearson: {pi:.4f}  spearman: {si:.4f}")

        # ── Save per-slide results ────────────────────────────────────────────────
        OUTPUT_DIR.mkdir(exist_ok=True)
        for slide_name, (preds, targets, s_names) in all_results.items():
            np.save(OUTPUT_DIR / f"sg_generalization_{slide_name}_preds.npy",   preds)
            np.save(OUTPUT_DIR / f"sg_generalization_{slide_name}_targets.npy", targets)
            np.save(OUTPUT_DIR / f"sg_generalization_{slide_name}_names.npy",   np.array(s_names))
            
    print(f"\n  Results saved to {OUTPUT_DIR}/")



def test_generalization_hemit(model_name, model_path, hemit_dir):
    """
    Evaluate a model trained on ORION on HEMIT data (train/val/test splits).
    """

    log_file = OUTPUT_DIR / "training_log.txt"
    sys.stdout = Logger(log_file)
    
    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    model = Model(model_name, hidden_dim=HIDDEN_DIM, num_outputs=NUM_OUTPUTS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    h5_files = sorted(Path(H5_DIR).glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No ORION H5 files found in {H5_DIR}")
    with h5py.File(h5_files[0], "r") as f:
        orion_marker_names = list(f.attrs["marker_names"])

    hemit_dir = Path(hemit_dir)
    OUTPUT_DIR.mkdir(exist_ok=True)

    for split in ["train", "val", "test"]:
        h5_path = hemit_dir / f"{split}.h5"
        if not h5_path.exists():
            continue

        dataset = HEMITDataset(h5_path)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

        # Resolve shared marker indices
        orion_indices, hemit_indices, shared_names = [], [], []
        for orion_name, hemit_name in HEMIT_MARKER_MAP.items():
            if orion_name in orion_marker_names and hemit_name in dataset.marker_names:
                orion_indices.append(orion_marker_names.index(orion_name))
                hemit_indices.append(dataset.marker_names.index(hemit_name))
                shared_names.append(orion_name)

        print(f"\n[HEMIT {split}]  Shared markers ({len(shared_names)}): {shared_names}")

        all_preds, all_targets = _run_model_on_loader(model, loader, orion_indices, hemit_indices, device)

        p = pearson_per_marker(all_preds, all_targets)
        s = spearman_rank_per_marker(all_preds, all_targets)

        print(f"\n{'='*60}")
        print(f"  HEMIT {split.upper()}  ({len(all_preds)} patches)")
        print(f"  Mean Pearson:  {p.mean():.4f}   Mean Spearman: {s.mean():.4f}")
        print(f"{'='*60}")
        for name, pi, si in zip(shared_names, p, s):
            print(f"    {name:<20s}  pearson: {pi:.4f}  spearman: {si:.4f}")

        np.save(OUTPUT_DIR / f"hemit_generalization_{split}_preds.npy",   all_preds)
        np.save(OUTPUT_DIR / f"hemit_generalization_{split}_targets.npy", all_targets)
        np.save(OUTPUT_DIR / f"hemit_generalization_{split}_names.npy",   np.array(shared_names))

    print(f"\n  Results saved to {OUTPUT_DIR}/")



def test_generalization_pathocell(model_name, model_path, h5_path, data_dir):
    """
    Evaluate a model trained on ORION on PathoCell data (all patches pooled).
    """

    log_file = OUTPUT_DIR / "training_log.txt"
    sys.stdout = Logger(log_file)

    load_dotenv()
    login(token=os.getenv("HF_TOKEN"))

    model = Model(model_name, hidden_dim=HIDDEN_DIM, num_outputs=NUM_OUTPUTS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    h5_files = sorted(Path(H5_DIR).glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No ORION H5 files found in {H5_DIR}")
    with h5py.File(h5_files[0], "r") as f:
        orion_marker_names = list(f.attrs["marker_names"])

    with h5py.File(h5_path, "r") as f:
        pathocell_marker_names = list(f.attrs["marker_names"])

    orion_indices, pc_indices, shared_names = [], [], []
    for orion_name, pc_name in PATHOCELL_MARKER_MAP.items():
        if orion_name in orion_marker_names and pc_name in pathocell_marker_names:
            orion_indices.append(orion_marker_names.index(orion_name))
            pc_indices.append(pathocell_marker_names.index(pc_name))
            shared_names.append(orion_name)

    print(f"\n  Shared markers ({len(shared_names)}): {shared_names}")

    dataset = MultiImagePathoCellDataset(h5_path, data_dir)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

    all_preds, all_targets = _run_model_on_loader(model, loader, orion_indices, pc_indices, device)

    p = pearson_per_marker(all_preds, all_targets)
    s = spearman_rank_per_marker(all_preds, all_targets)

    print(f"\n{'='*60}")
    print(f"  PATHOCELL  ({len(all_preds)} patches)")
    print(f"  Mean Pearson:  {p.mean():.4f}   Mean Spearman: {s.mean():.4f}")
    print(f"{'='*60}")
    for name, pi, si in zip(shared_names, p, s):
        print(f"    {name:<20s}  pearson: {pi:.4f}  spearman: {si:.4f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    np.save(OUTPUT_DIR / "pathocell_generalization_preds.npy",   all_preds)
    np.save(OUTPUT_DIR / "pathocell_generalization_targets.npy", all_targets)
    np.save(OUTPUT_DIR / "pathocell_generalization_names.npy",   np.array(shared_names))
    print(f"\n  Results saved to {OUTPUT_DIR}/")




if __name__ == "__main__":

    
    #train(MODEL_NAME)


    test_generalization_sg(
        model_name = MODEL_NAME,
        model_path  = OUTPUT_DIR / "best_model.pt",
        sg_dir  = Path("singular_genomics")
    )



    test_generalization_pathocell(
        model_name = MODEL_NAME,
        model_path = OUTPUT_DIR / "best_model.pt",
        h5_path    = Path("pathocell_patch_dataset/pathocell_hdf.h5"),
        data_dir  = Path("/mnt/ssd1/virtual_proteomics/data/pathocell/pathocell/pathocell_hdf")
    )



    test_generalization_hemit(
        model_name = MODEL_NAME,
        model_path = OUTPUT_DIR / "best_model.pt",
        hemit_dir    = Path("hemit_patch_dataset"),
    )

