"""Visualize the artifact_filter passes on the ORION_CRCv2 token targets.

Two gates, selectable with --gate:

  lineage  (apply_lineage_gate)  -- per child marker, a 4-panel strip:
       [ PARENT before ] [ CHILD before ] [ CHILD before (red=zeroed) ] [ CHILD after ]
     Markers NOT in the hierarchy are written as single panels noh_<marker>.png.

  shape    (apply_shape_gate)    -- per point-wise marker (PD-L1), a 2-panel strip:
       [ <marker> before (red=dropped circles) ] [ <marker> after ]

  both (default) runs lineage then shape on the SAME targets, so the shape "before"
  already reflects the lineage gate. Output: artifact_report/<gate>_viz/<slide>/.

    python visualize_artifact_filter.py                       # both gates, ALL slides
    python visualize_artifact_filter.py --gate shape --slides CRC01,CRC15
    python visualize_artifact_filter.py --gate lineage --tau 0.02
    python visualize_artifact_filter.py --gate shape --set sat=0.6 --set max_area=30
"""
import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

import artifact_filter_orion as af

DATASET_DIR = Path("datasets/orion_crcv2_patch_dataset")
OUT_ROOT = Path("artifact_report")
G = af.TOKEN_GRID


def all_slides():
    return sorted((p.name[:-len("_patch_dataset.h5")]
                   for p in DATASET_DIR.glob("*_patch_dataset.h5")),
                  key=lambda s: (len(s), s))


def canvas_geom(coords, ps0):
    s = G / ps0
    H = max(round(int(y) * s) for _, y in coords) + G
    W = max(round(int(x) * s) for x, _ in coords) + G
    return s, H, W


def gray(canvas, vis):
    hi = max(float(np.nanpercentile(canvas[vis], 99.5)), 1e-6)
    g = (np.clip(np.nan_to_num(canvas) / hi, 0, 1) * 255).astype(np.uint8)
    return np.stack([g, g, g], -1)


def label(img, text):
    cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                cv2.LINE_AA)
    return img


def load(slide):
    with h5py.File(DATASET_DIR / f"{slide}_patch_dataset.h5") as f:
        names = list(f.attrs["marker_names"]); coords = f["coords"][:]
        targets = f["targets"][:]; ps0 = int(f.attrs["patch_size_level0"])
    return names, coords, targets, ps0


# ── lineage gate viz ──────────────────────────────────────────────────────────────

def viz_lineage(slide, names, coords, targets, ps0, tau):
    """targets is mutated in place by the gate (= the 'after' / shape input)."""
    idx = {m: i for i, m in enumerate(names)}
    s, H, W = canvas_geom(coords, ps0)
    asm = lambda v: af._assemble_canvas(coords, v, s, G, H, W)

    before = targets.copy()
    af.apply_lineage_gate(targets, names, tau)

    vis = ~np.isnan(asm(before[:, 0]))
    sdir = OUT_ROOT / "lineage_viz" / slide; sdir.mkdir(parents=True, exist_ok=True)
    sep = np.full((H, 3, 3), 60, np.uint8)
    child_names = {c for c, _ in af.HIERARCHY}
    for child, groups in af.HIERARCHY:
        if child not in idx or not all(p in idx for g in groups for p in g):
            continue
        parents = [p for g in groups for p in g]
        pcanv = np.maximum.reduce([asm(before[:, idx[p]]) for p in parents])
        b = asm(before[:, idx[child]]); a = asm(targets[:, idx[child]])
        removed = (np.nan_to_num(b) > 0) & (np.nan_to_num(a) == 0)
        pimg = label(gray(pcanv, vis), "parent " + "|".join(parents) + " (before)")
        cbimg = label(gray(b, vis), f"{child} before")
        rimg = gray(b, vis); rimg[removed] = (255, 40, 40)
        rimg = label(rimg, f"{child} red=zeroed {int(removed.sum())}")
        aimg = label(gray(a, vis), f"{child} after")
        strip = np.concatenate([pimg, sep, cbimg, sep, rimg, sep, aimg], axis=1)
        cv2.imwrite(str(sdir / f"{child.replace('/', '_')}.png"),
                    cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    noh = [m for m in names if m not in child_names]
    for m in noh:
        img = label(gray(asm(before[:, idx[m]]), vis), f"{m} (no hierarchy, unchanged)")
        cv2.imwrite(str(sdir / f"noh_{m.replace('/', '_')}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"  {slide} [lineage]: {len(child_names)} child strips + {len(noh)} unaffected")


# ── shape gate viz ────────────────────────────────────────────────────────────────

def viz_shape(slide, names, coords, targets, ps0, markers, p):
    idx = {m: i for i, m in enumerate(names)}
    s, H, W = canvas_geom(coords, ps0)
    asm = lambda v: af._assemble_canvas(coords, v, s, G, H, W)

    before = targets.copy()
    stats = af.apply_shape_gate(targets, coords, ps0, names, markers=markers, p=p)

    vis = ~np.isnan(asm(before[:, 0]))
    sdir = OUT_ROOT / "shape_viz" / slide; sdir.mkdir(parents=True, exist_ok=True)
    sep = np.full((H, 3, 3), 60, np.uint8)
    for m in markers:
        if m not in idx:
            continue
        b = asm(before[:, idx[m]]); a = asm(targets[:, idx[m]])
        removed = (np.nan_to_num(b) > 0) & (np.nan_to_num(a) == 0)
        bimg = gray(b, vis); bimg[removed] = (255, 40, 40)
        bimg = label(bimg, f"{m} before (red=dropped {stats.get(m, 0)})")
        aimg = label(gray(a, vis), f"{m} after")
        strip = np.concatenate([bimg, sep, aimg], axis=1)
        cv2.imwrite(str(sdir / f"{m.replace('/', '_')}.png"),
                    cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print(f"  {slide} [shape]: dropped {stats}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", choices=["lineage", "shape", "both"], default="both")
    ap.add_argument("--slides", default="", help="comma list (default: ALL)")
    ap.add_argument("--tau", type=float, default=0.0, help="lineage: parent-present floor")
    ap.add_argument("--markers", default="PD-L1", help="shape: comma list (default PD-L1)")
    ap.add_argument("--set", action="append", default=[], metavar="k=v",
                    help="override a ShapeParams field, e.g. --set sat=0.6")
    args = ap.parse_args()

    sp = af.ShapeParams()
    for kv in args.set:
        k, v = kv.split("=", 1)
        cur = getattr(sp, k)
        setattr(sp, k, type(cur)(v))
    markers = tuple(m for m in args.markers.split(",") if m)
    slides = [x for x in args.slides.split(",") if x] or all_slides()
    print(f"artifact-filter viz: gate={args.gate} slides={len(slides)} "
          f"(lineage tau={args.tau}; shape markers={markers} sigma=[{sp.min_sigma},{sp.max_sigma}] "
          f"cv_max={sp.cv_max} min_contrast={sp.min_contrast} min_mean={sp.min_mean})")

    for slide in slides:
        if not (DATASET_DIR / f"{slide}_patch_dataset.h5").exists():
            print(f"  {slide}: missing h5"); continue
        names, coords, targets, ps0 = load(slide)
        # both: lineage first (mutates targets), then shape on the gated targets
        if args.gate in ("lineage", "both"):
            viz_lineage(slide, names, coords, targets, ps0, args.tau)
        if args.gate in ("shape", "both"):
            viz_shape(slide, names, coords, targets, ps0, markers, sp)
    print(f"\nDone -> {OUT_ROOT}/{{lineage_viz,shape_viz}}/<slide>/")


if __name__ == "__main__":
    main()
