import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Channel indices — sequential model output order (0-15) ────────────────────
CH = {
    "Hoechst":  0,
    "CD31":     1,
    "CD45":     2,
    "CD68":     3,
    "CD4":      4,
    "FOXP3":    5,
    "CD8a":     6,
    "CD45RO":   7,
    "CD20":     8,
    "PDL1":     9,
    "CD3e":    10,
    "CD163":   11,
    "ECad":    12,
    "Ki67":    13,
    "PanCK":   14,
    "SMA":     15,
}

# ── Constraint tables ─────────────────────────────────────────────────────────
# Each tuple: (idx_a, idx_b, weight)

# Hard exclusion: penalize p_a * p_b (both high simultaneously = bad)
HARD_EXCL = [
    # Epithelial vs immune / endothelial / stromal
    *[(CH["PanCK"], CH[b], 10.0) for b in
      ["CD3e","CD4","CD8a","FOXP3","CD45RO","CD20","CD68","CD163","CD45","CD31","SMA"]],
    *[(CH["ECad"],  CH[b], 10.0) for b in
      ["CD3e","CD4","CD8a","FOXP3","CD45RO","CD20","CD68","CD163","CD45","CD31","SMA"]],
    # Endothelial vs immune / epithelial
    *[(CH["CD31"], CH[b], 8.0) for b in
      ["CD3e","CD4","CD8a","FOXP3","CD20","CD68","CD163","CD45RO","PanCK","ECad","CD45"]],
    # Stromal vs immune / epithelial
    *[(CH["SMA"], CH[b], 8.0) for b in
      ["CD3e","CD4","CD8a","FOXP3","CD20","CD68","CD163","CD45RO","PanCK","ECad","CD45"]],
    # CD45 absent from non-immune
    *[(CH["CD45"], CH[b], 8.0) for b in ["PanCK","ECad","CD31","SMA"]],
    # T / B / macrophage lineage cross-exclusions
    *[(CH["CD20"], CH[b], 9.0) for b in ["CD3e","CD4","CD8a","FOXP3","CD68","CD163"]],
    *[(CH["CD68"], CH[b], 9.0) for b in ["CD3e","CD4","CD8a","FOXP3","CD20"]],
    # FOXP3 is CD4-lineage only
    (CH["FOXP3"], CH["CD8a"], 9.0),
    (CH["FOXP3"], CH["CD20"], 9.0),
    # CD45RO absent from non-immune
    *[(CH["CD45RO"], CH[b], 8.0) for b in ["PanCK","ECad","CD31","SMA"]],
]

# Deduplicate symmetric pairs
_seen = set()
_dedup = []
for a, b, w in HARD_EXCL:
    key = (min(a, b), max(a, b))
    if key not in _seen:
        _seen.add(key)
        _dedup.append((a, b, w))
HARD_EXCL = _dedup

# Soft exclusion: penalize p_a * p_b with lower weight
SOFT_EXCL = [
    (CH["CD4"], CH["CD8a"], 3.0),   # CD4+CD8a+ double-positive essentially absent in tissue
]

# Co-expression directed: if marker a is expressed, b must also be expressed
# Penalty = w * ReLU(p_a - p_b)^2, gated on p_a > threshold
COEXPR_DIRECTED = [
    # Leukocyte gate: every immune marker requires CD45
    *[(CH[a], CH["CD45"], 6.0) for a in
      ["CD3e","CD4","CD8a","FOXP3","CD20","CD68","CD163","CD45RO"]],
    # T cell hierarchy: CD4/CD8a/FOXP3 require CD3e
    (CH["CD4"],    CH["CD3e"], 8.0),
    (CH["CD8a"],   CH["CD3e"], 9.0),
    (CH["FOXP3"],  CH["CD4"],  9.0),
    (CH["FOXP3"],  CH["CD3e"], 9.0),
    # Macrophage hierarchy: CD163 is a strict subset of CD68
    (CH["CD163"],  CH["CD68"], 10.0),
    # Memory T cells: CD45RO enriched in CD3e+ cells (soft — can appear on B cells too)
    (CH["CD45RO"], CH["CD3e"], 4.0),
]

# E-Cadherin / Pan-CK asymmetric penalty (encodes EMT biology)
#   E-Cad+ without Pan-CK: biologically invalid → strong penalty
#   Pan-CK+ without E-Cad: valid (tumour invasion / EMT) → mild penalty
ECAD_PANCK_WEIGHTS = (5.0, 1.0)   # (w_ecad_without_panck, w_panck_without_ecad)


# ── Loss class ────────────────────────────────────────────────────────────────

class BiologicalConstraintLoss(nn.Module):
    """
    Biological constraint loss operating on (B, C, G, G) predictions in [0, 1].
    Flattens spatial dims to (B*G*G, C) — each token treated as an independent cell.

    Called via BioLoss.compute_bio_only() in training_orion_reg.py; the MSE loss
    is handled separately by the training loop using a masked criterion.
    """

    def __init__(
        self,
        lambda_max:        float = 0.3,
        warmup_steps:      int   = 5_000,
        w_hard:            float = 1.0,
        w_soft:            float = 1.0,
        w_coexpr:          float = 1.0,
        w_ecad:            float = 1.0,
        coexpr_threshold:  float = 0.2,
    ):
        super().__init__()
        self.lambda_max       = lambda_max
        self.warmup_steps     = warmup_steps
        self.w_hard           = w_hard
        self.w_soft           = w_soft
        self.w_coexpr         = w_coexpr
        self.w_ecad           = w_ecad
        self.coexpr_threshold = coexpr_threshold

        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))

        ha = torch.tensor([x[0] for x in HARD_EXCL], dtype=torch.long)
        hb = torch.tensor([x[1] for x in HARD_EXCL], dtype=torch.long)
        hw = torch.tensor([x[2] for x in HARD_EXCL], dtype=torch.float)
        self.register_buffer("hard_a", ha)
        self.register_buffer("hard_b", hb)
        self.register_buffer("hard_w", hw)

        sa = torch.tensor([x[0] for x in SOFT_EXCL], dtype=torch.long)
        sb = torch.tensor([x[1] for x in SOFT_EXCL], dtype=torch.long)
        sw = torch.tensor([x[2] for x in SOFT_EXCL], dtype=torch.float)
        self.register_buffer("soft_a", sa)
        self.register_buffer("soft_b", sb)
        self.register_buffer("soft_w", sw)

        ca = torch.tensor([x[0] for x in COEXPR_DIRECTED], dtype=torch.long)
        cb = torch.tensor([x[1] for x in COEXPR_DIRECTED], dtype=torch.long)
        cw = torch.tensor([x[2] for x in COEXPR_DIRECTED], dtype=torch.float)
        self.register_buffer("coexpr_a", ca)
        self.register_buffer("coexpr_b", cb)
        self.register_buffer("coexpr_w", cw)

    def _lambda(self) -> float:
        return self.lambda_max * min(1.0, self._step.item() / self.warmup_steps)

    def _hard_excl_loss(self, p: torch.Tensor) -> torch.Tensor:
        """Mean violation only over tokens where at least one marker is expressed."""
        pa   = p[:, self.hard_a]                                      # (N, pairs)
        pb   = p[:, self.hard_b]
        gate = (torch.maximum(pa, pb) >= self.coexpr_threshold).float()
        return (self.hard_w * gate * pa * pb).sum() / gate.sum().clamp(min=1)

    def _soft_excl_loss(self, p: torch.Tensor) -> torch.Tensor:
        pa   = p[:, self.soft_a]
        pb   = p[:, self.soft_b]
        gate = (torch.maximum(pa, pb) >= self.coexpr_threshold).float()
        return (self.soft_w * gate * pa * pb).sum() / gate.sum().clamp(min=1)

    def _coexpr_loss(self, p: torch.Tensor) -> torch.Tensor:
        """Mean violation only over tokens where the requiring marker is expressed."""
        pa   = p[:, self.coexpr_a]
        pb   = p[:, self.coexpr_b]
        gate = (pa >= self.coexpr_threshold).float()
        return (self.coexpr_w * gate * F.relu(pa - pb) ** 2).sum() / gate.sum().clamp(min=1)

    def _ecad_panck_loss(self, p: torch.Tensor) -> torch.Tensor:
        """Asymmetric ECad/PanCK penalty, mean over tokens where either is expressed."""
        ecad  = p[:, CH["ECad"]]
        panck = p[:, CH["PanCK"]]
        w_ecad_no_panck, w_panck_no_ecad = ECAD_PANCK_WEIGHTS
        gate  = (torch.maximum(ecad, panck) >= self.coexpr_threshold).float()
        loss  = (w_ecad_no_panck * F.relu(ecad  - panck) ** 2 +
                 w_panck_no_ecad * F.relu(panck - ecad)  ** 2)
        return (gate * loss).sum() / gate.sum().clamp(min=1)