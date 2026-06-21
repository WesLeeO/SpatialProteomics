"""IF token-artifact filters for the ORION_CRCv2 token targets.

Two independent passes. Only the lineage one is implemented for now.

1. LINEAGE GATE (this module). Pure per-token rule on the immune subset hierarchy:
   a child marker cannot be positive where its parent lineage is absent, so for each
   token, if the parent marker is ~0 there, the child marker is set to 0. Operates
   directly on the (N, C, G, G) target tensor channel-against-channel (parent and
   child are channels of the SAME token) -- no spatial assembly, no shapes. Cascades
   in topological order so e.g. CD4 is zeroed wherever CD3e was just zeroed (because
   CD45 was absent), giving CD4 <= CD3e <= CD45.

       Hoechst -> CD45                    (no nucleus -> no leukocyte; nuclear root)
       CD45 -> CD20, CD3e, CD68, CD45RO   (CD45+ leukocytes; CD45RO broadly leukocyte)
       CD3e -> CD4, CD8a                  (CD4 / CD8 T cells are CD3+)
       CD4  -> FOXP3                      (Tregs are CD4+)
       CD68 -> CD163                      (CD163 macrophages are CD68+)

2. SHAPE GATE (apply_shape_gate). For point-wise markers (PD-L1), a large UNIFORM disc --
   a flat-intensity circle on an ~0 background -- is "too regular" to be biology; it is an
   imaging artifact (dust / autofluorescent dot), often only faintly bright (~0.15). Found
   by multi-scale Laplacian-of-Gaussian blob detection (any radius), then kept only if the
   interior is uniform (low CV) and high-contrast vs background. Real PD-L1 is heterogeneous
   and tissue-textured, so it is spared. Independent of Hoechst.

3. BLOCK GATE (apply_block_gate). A corrupted 20x tile lights up EVERY channel, so
   mutually-exclusive markers (PD-L1 + CD31) end up CO-ACTIVE in the same coherent patch
   (e.g. CRC13's acellular corner) -- implausible biology. A patch is confirmed an artifact
   only where it is also ACELLULAR (Hoechst ~ 0), which spares genuine co-occurrence with
   nuclei (a vascularised tumour). This is NOT a Hoechst gate on the markers: lone CD31 /
   PD-L1 at low Hoechst is never co-active, so always spared. A mop-up then clears each
   marker's faint skirt around the confirmed patch.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

TOKEN_GRID = 16

# child -> list of OR-groups of parents. The child is supported at a token where EVERY
# group has at least one parent present; a single-parent edge is a one-element group.
# Topological order (CD45 root first) so zeroing cascades down the lineage.
HIERARCHY: list[tuple[str, list[list[str]]]] = [
    ("CD45",   [["Hoechst"]]),       # nuclear root: no nucleus -> no leukocyte
    ("CD20",   [["CD45"]]),
    ("CD3e",   [["CD45"]]),
    ("CD68",   [["CD45"]]),
    ("CD4",    [["CD3e"]]),
    ("CD8a",   [["CD3e"]]),
    ("CD163",  [["CD68"]]),
    ("FOXP3",  [["CD4"]]),
    ("CD45RO", [["CD45"]]),          # CD45RO is broadly leukocyte -> gate on CD45 only
]

# Per-edge "parent present" threshold overrides. Most edges use tau=0 (parent exactly 0).
EDGE_TAU: dict[str, float] = {}


def apply_lineage_gate(targets, marker_names, tau: float = 0.0):
    """Zero each child token where its parent lineage is absent (<= the edge threshold).

    targets: (N, C, G, G) float array, modified IN PLACE.
    marker_names: length-C list aligning channels to markers.
    tau: default "parent present" threshold. Default 0.0 -> a child is zeroed only where
         the parent is exactly 0 (strict). Per-edge overrides live in EDGE_TAU (e.g.
         PD-L1 uses a small Hoechst floor). For an OR-group (CD45RO) the child is zeroed
         only where ALL parents are <= the threshold.

    Returns dict child -> number of (token) entries zeroed.
    """
    idx = {m: i for i, m in enumerate(marker_names)}
    stats: dict[str, int] = {}
    for child, groups in HIERARCHY:
        if child not in idx or not all(p in idx for g in groups for p in g):
            continue
        t_edge = EDGE_TAU.get(child, tau)
        ch = targets[:, idx[child]]
        # absent = some required group has NO parent present (all parents <= t_edge)
        absent = np.zeros(ch.shape, dtype=bool)
        for group in groups:
            present = np.zeros(ch.shape, dtype=bool)
            for p in group:
                present |= targets[:, idx[p]] > t_edge
            absent |= ~present
        kill = absent & (ch > 0)
        stats[child] = int(kill.sum())
        ch[kill] = 0.0
    return stats


# ── shape gate (uniform-disc artifacts on point-wise markers) ─────────────────────

# point-wise markers that should never form a large, uniform, solid disc. PD-L1 dots are
# "too regular": a flat-intensity circle of varying size sitting on an ~0 background. They
# are NOT necessarily bright (often only ~0.15-0.2), so intensity thresholds miss them --
# the signal is SHAPE + UNIFORMITY, found by multi-scale blob detection. (CD31 rectangles
# are handled separately.)
SHAPE_MARKERS = ("PD-L1",)


@dataclass
class ShapeParams:
    disp_scale: float = 0.30   # normalise the canvas by this before blob detection (the
                               # discs are faint; this sets the working contrast)
    min_sigma: float = 5.0     # smallest disc radius ~= sigma*sqrt(2)
    max_sigma: float = 16.0    # largest disc (CRC15-scale discs are bigger than CRC01's)
    num_sigma: int = 12
    log_thr: float = 0.02      # Laplacian-of-Gaussian response threshold (on disp_scale)
    overlap: float = 0.5
    cv_max: float = 0.50       # interior uniformity: std/mean <= this ("too regular")
    min_contrast: float = 2.5  # interior mean / surrounding-ring mean (island on ~0 bg)
    min_mean: float = 0.04     # interior must reach at least this (skip pure noise)
    zero_scale: float = 1.30   # zero a disc of radius r*zero_scale (clears the soft edge)


def _assemble_canvas(coords, vals, s, G, H, W):
    cv = np.full((H, W), np.nan, np.float32)
    for i, (x, y) in enumerate(coords):
        r0 = round(int(y) * s); c0 = round(int(x) * s)
        r1, c1 = min(r0 + G, H), min(c0 + G, W)
        cv[r0:r1, c0:c1] = vals[i, :r1 - r0, :c1 - c0]
    return cv


def _shape_mask(canvas, vis, p: ShapeParams):
    """Bool canvas of artifact tokens. Multi-scale Laplacian-of-Gaussian finds round blobs
    at every radius; a blob is flagged an artifact when its interior is UNIFORM (low CV =
    too regular) and HIGH-CONTRAST against an ~0 background (an island). Real point-wise
    signal is heterogeneous (high CV) and tissue-textured, so it is left alone. Flagged
    discs are zeroed out to radius r*zero_scale."""
    from skimage.feature import blob_log
    c = np.nan_to_num(canvas)
    img = np.clip(c / p.disp_scale, 0.0, 1.0).astype(np.float32)
    blobs = blob_log(img, min_sigma=p.min_sigma, max_sigma=p.max_sigma,
                     num_sigma=p.num_sigma, threshold=p.log_thr, overlap=p.overlap)
    mask = np.zeros(vis.shape, np.uint8)
    yy, xx = np.ogrid[:c.shape[0], :c.shape[1]]
    for y, x, sigma in blobs:
        y, x = int(y), int(x)
        r = sigma * 1.41421356
        d2 = (yy - y) ** 2 + (xx - x) ** 2
        inside = (d2 <= r * r) & vis
        ann = (d2 > r * r) & (d2 <= (r + 3) ** 2) & vis
        if not inside.any():
            continue
        vi = c[inside]; m = float(vi.mean())
        cv_in = float(vi.std() / (m + 1e-9))
        ring = float(c[ann].mean()) if ann.any() else 0.0
        if m >= p.min_mean and cv_in <= p.cv_max and m / (ring + 1e-6) >= p.min_contrast:
            cv2.circle(mask, (x, y), int(r * p.zero_scale) + 1, 1, -1)
    return (mask > 0) & vis


def apply_shape_gate(targets, coords, ps0, marker_names, markers=SHAPE_MARKERS,
                     p: ShapeParams | None = None, G=TOKEN_GRID):
    """Zero isolated saturated (round) specks of point-wise markers in `targets`.

    targets: (N, C, G, G) float, modified IN PLACE. coords: level-0 (x, y) per patch.
    Returns dict marker -> number of tokens zeroed.
    """
    p = p or ShapeParams()
    s = G / ps0
    rs = [round(int(y) * s) for _, y in coords]
    cs = [round(int(x) * s) for x, _ in coords]
    H, W = max(rs) + G, max(cs) + G
    names = list(marker_names)
    idx = {m: i for i, m in enumerate(names)}
    vis = ~np.isnan(_assemble_canvas(coords, targets[:, 0], s, G, H, W))

    stats = {}
    for m in markers:
        if m not in idx:
            continue
        ch = idx[m]
        mask = _shape_mask(_assemble_canvas(coords, targets[:, ch], s, G, H, W), vis, p)
        zeroed = 0
        if mask.any():
            for i, (x, y) in enumerate(coords):
                r0 = round(int(y) * s); c0 = round(int(x) * s)
                r1, c1 = min(r0 + G, H), min(c0 + G, W)
                sub = mask[r0:r1, c0:c1]
                if sub.any():
                    tile = targets[i, ch, :r1 - r0, :c1 - c0]
                    zeroed += int((sub & (tile > 0)).sum())
                    tile[sub] = 0.0
        stats[m] = zeroed
    return stats


# ── block gate (acellular co-activation of exclusive markers = tile artifact) ─────

# A corrupted 20x tile lights up EVERY channel, so PD-L1 (tumour/immune membrane) and
# CD31 (endothelium) -- different, mutually exclusive cell types -- end up CO-ACTIVE in
# the same coherent patch, which is biologically implausible. To avoid removing genuine
# co-occurrence (a vascularised tumour where a 14px token holds both a PD-L1+ cell and a
# CD31+ vessel), a patch is confirmed an artifact ONLY where it is also ACELLULAR
# (Hoechst ~ 0). This is NOT a Hoechst gate on the markers themselves -- a lone CD31
# vessel or PD-L1 cell at low Hoechst is never co-active, so it is always spared; only
# the exclusive-marker COMBINATION in dead tissue is removed.
BLOCK_MARKERS = ("PD-L1", "CD31")
NUCLEAR = "Hoechst"


@dataclass
class BlockParams:
    co_thr: float = 0.20       # a marker is "active" at a token when >= this
    min_patch: int = 8         # min connected co-activation patch (tokens)
    close: int = 3             # morph-close to bridge the textured patch interior
    max_hoechst: float = 0.03  # patch confirmed artifact only if it sits in dead tissue
    hoechst_dilate: int = 6    # measure Hoechst over the patch dilated by this many tokens
                               # (a real perivascular patch has nuclei in its neighbourhood)
    mop_margin: int = 10       # also zero each marker's faint residual within this many
                               # tokens of a confirmed patch (cleans the soft skirt)
    mop_floor: float = 0.05    # residual is anything >= this in the mop-up neighbourhood


def _zero_back(targets, ch, coords, s, G, H, W, mask):
    """Zero, in targets[:, ch], every token whose canvas pixel is in `mask`."""
    zeroed = 0
    for i, (x, y) in enumerate(coords):
        r0 = round(int(y) * s); c0 = round(int(x) * s)
        r1, c1 = min(r0 + G, H), min(c0 + G, W)
        sub = mask[r0:r1, c0:c1]
        if sub.any():
            tile = targets[i, ch, :r1 - r0, :c1 - c0]
            zeroed += int((sub & (tile > 0)).sum())
            tile[sub] = 0.0
    return zeroed


def apply_block_gate(targets, coords, ps0, marker_names, markers=BLOCK_MARKERS,
                     p: BlockParams | None = None, G=TOKEN_GRID):
    """Zero corrupted-tile artifacts: patches where mutually-exclusive markers (PD-L1 +
    CD31) are CO-ACTIVE in acellular (Hoechst ~ 0) tissue. Genuine co-occurrence (with
    nuclei) and lone single-marker signal are left alone.

    targets: (N, C, G, G) float, modified IN PLACE. Returns dict marker -> tokens zeroed.
    """
    p = p or BlockParams()
    s = G / ps0
    rs = [round(int(y) * s) for _, y in coords]
    cs = [round(int(x) * s) for x, _ in coords]
    H, W = max(rs) + G, max(cs) + G
    idx = {m: i for i, m in enumerate(marker_names)}
    vis = ~np.isnan(_assemble_canvas(coords, targets[:, 0], s, G, H, W))

    present = [m for m in markers if m in idx]
    if len(present) < 2:
        return {m: 0 for m in present}
    canv = {m: np.nan_to_num(_assemble_canvas(coords, targets[:, idx[m]], s, G, H, W))
            for m in present}
    ho = (np.nan_to_num(_assemble_canvas(coords, targets[:, idx[NUCLEAR]], s, G, H, W))
          if NUCLEAR in idx else None)

    # co-activation: every block marker active at the token
    co = vis.copy()
    for m in present:
        co &= canv[m] >= p.co_thr
    co = (cv2.morphologyEx(co.astype(np.uint8), cv2.MORPH_CLOSE,
                           np.ones((p.close, p.close), np.uint8)) > 0)

    # keep coherent patches that are acellular (Hoechst ~ 0)
    n, lab, stats_cc, _ = cv2.connectedComponentsWithStats(co.astype(np.uint8), 8)
    hk = np.ones((2 * p.hoechst_dilate + 1, 2 * p.hoechst_dilate + 1), np.uint8)
    combined = np.zeros(vis.shape, bool)
    for i in range(1, n):
        if int(stats_cc[i, cv2.CC_STAT_AREA]) < p.min_patch:
            continue
        comp = lab == i
        if ho is None:
            combined |= comp
        else:                                          # acellular in a dilated neighbourhood
            nbr = (cv2.dilate(comp.astype(np.uint8), hk) > 0) & vis
            if float(ho[nbr].mean()) <= p.max_hoechst:
                combined |= comp

    # mop-up: within mop_margin tokens of a confirmed patch, zero each marker's faint
    # residual too (the textured skirt of the artifact). Scoped to where the gate fired,
    # so slides with no acellular co-activation patch are untouched.
    region = None
    if p.mop_margin > 0 and combined.any():
        mk = np.ones((2 * p.mop_margin + 1, 2 * p.mop_margin + 1), np.uint8)
        region = (cv2.dilate(combined.astype(np.uint8), mk) > 0) & vis

    stats = {}
    for m in present:
        mask = combined
        if region is not None:
            mask = combined | (region & (canv[m] >= p.mop_floor))
        stats[m] = _zero_back(targets, idx[m], coords, s, G, H, W, mask) if mask.any() else 0
    return stats