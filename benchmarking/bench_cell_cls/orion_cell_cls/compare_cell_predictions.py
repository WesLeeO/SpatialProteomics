"""
Per-patch CELL-LEVEL comparison: where does OUR model beat MIPHEI (and vice versa)
on ORION-CRC for one marker? The global AUPRC is on-par, so this localises the
disagreement.

Source = the benchmark's own per-cell predictions (cell_predictions_<tag>.csv from
eval_cell_auprc.py: logreg head, TEST slides CRC02/CRC11, 0.5 threshold) — same
labels and protocol that produce the headline numbers, both models scored on the
SAME cells. Columns per selected patch:

  1. H&E (native 20x)
  2. H&E + nuclei outlines + GT positivity   (filled = <marker>+ cell)
  3. MIPHEI call vs GT   (TP green | FP red | FN orange | TN faint grey)
  4. ours   call vs GT
with the 16x16 token grid on the overlay columns.

--select diff (default): rank patches by (#cells WE get right & MIPHEI wrong)
  minus (#cells MIPHEI right & WE wrong) → top `n` our-wins then top `n` our-losses.

  python cell_cls/orion_cell_cls/compare_cell_predictions.py --sample CRC02 --marker CD8a --n 4
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage as ndi

from utils import (BENCH_DIR, TILE_DIR, sample_rows, index_nuclei_tiles,
                   load_patch_labels, TILE_L0, TILE_MASK)
from visualize_orion_cell_overlay import (read_he_patch, expand_cells,
                                          draw_token_grid, to_uint8)
from eval_cell_auprc import SLIDE_KEY, MARKERS, norm
from build_patch_dataset_orion_crc_reg import crc_paths, open_zarr_level0

HERE = Path(__file__).resolve().parent
# outcome code → (RGB, label).  0=TN 1=TP 2=FP 3=FN
OUTCOME = {1: ((0, 180, 0), "TP"), 2: ((220, 30, 30), "FP"),
           3: ((255, 165, 0), "FN"), 0: ((150, 150, 150), "TN")}


def load_joined(marker: str, ours_tag: str, miphei_tag: str) -> pd.DataFrame:
    """Join the two per-cell prediction CSVs on (slide_name, cell_id) for one marker.
    Returns cell_id, slide_name, x, y, pos, mine, miphei (binary calls)."""
    def col(tag):
        d = pd.read_csv(HERE / f"cell_predictions_{tag}.csv",
                        usecols=["cell_id", "slide_name", "x", "y",
                                 f"{marker}_binary", f"{marker}_pos"])
        return d.rename(columns={f"{marker}_binary": "call", f"{marker}_pos": "pos"})
    o = col(ours_tag); m = col(miphei_tag)
    df = o.merge(m[["cell_id", "slide_name", "call"]], on=["cell_id", "slide_name"],
                 suffixes=("_mine", "_miphei"))
    df["mine_correct"]   = df["call_mine"]   == df["pos"]
    df["miphei_correct"] = df["call_miphei"] == df["pos"]
    return df


def outcome_code(call: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """TP=1, FP=2, FN=3, TN=0 from boolean call/pos."""
    call, pos = call.astype(bool), pos.astype(bool)
    out = np.zeros(len(call), np.int8)
    out[call & pos] = 1
    out[call & ~pos] = 2
    out[~call & pos] = 3
    return out


def pick_patches(df_s, coords, ps0, n, min_cells, min_pos):
    """Rank patches by our-vs-MIPHEI cell-correctness difference; top n wins + n losses."""
    cx, cy = df_s["x"].values, df_s["y"].values
    mc, pc = df_s["mine_correct"].values, df_s["miphei_correct"].values
    pos = df_s["pos"].values.astype(bool)
    rows = []
    for i, (px, py) in enumerate(coords):
        px, py = int(px), int(py)
        sel = (cx >= px) & (cx < px + ps0) & (cy >= py) & (cy < py + ps0)
        nsel = int(sel.sum())
        if nsel < min_cells or int(pos[sel].sum()) < min_pos:
            continue
        win = int((mc[sel] & ~pc[sel]).sum())
        loss = int((pc[sel] & ~mc[sel]).sum())
        rows.append((i, win - loss, win, loss, nsel, int(pos[sel].sum())))
    if not rows:
        return []
    rows.sort(key=lambda r: r[1])
    losers = rows[:n]                       # most negative (MIPHEI wins)
    winners = rows[::-1][:n]                # most positive (we win)
    sel = winners + [r for r in losers if r not in winners]
    return sel


def fill_overlay(cell_lab, code_lut, max_lab, mh, mw, alpha=0.6):
    """RGBA overlay coloring each cell footprint by its outcome code."""
    codes = code_lut[np.clip(cell_lab, 0, max_lab)]
    ov = np.zeros((mh, mw, 4), np.float32)
    for c, ((r, g, b), _) in OUTCOME.items():
        if c == 0:                       # leave TN unfilled (the vast majority) → declutter
            continue
        m = (codes == c) & (cell_lab > 0)
        ov[m] = (r / 255, g / 255, b / 255, alpha)
    return ov


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="CRC02", choices=["CRC02", "CRC11"],
                    help="test slide (only CRC02/CRC11 have per-cell predictions)")
    ap.add_argument("--marker", default="CD8a")
    ap.add_argument("--n", type=int, default=4, help="top-n our-wins + top-n our-losses")
    ap.add_argument("--ours_tag", default="ours")
    ap.add_argument("--miphei_tag", default="MIPHEI-vit")
    ap.add_argument("--cell_radius_um", type=float, default=2.0)
    ap.add_argument("--min_cells", type=int, default=15)
    ap.add_argument("--min_pos", type=int, default=3, help="min GT+ cells in FOV to qualify")
    ap.add_argument("--out_dir", default=str(HERE / "compare_cell_out"))
    args = ap.parse_args()
    marker = args.marker
    if norm(marker) not in {norm(m) for m in MARKERS}:
        raise SystemExit(f"{marker} not in benchmark markers {MARKERS}")
    marker = next(m for m in MARKERS if norm(m) == norm(marker))

    df = load_joined(marker, args.ours_tag, args.miphei_tag)
    key = SLIDE_KEY[args.sample]
    df_s = df[df["slide_name"].str.contains(key)].reset_index(drop=True)
    print(f"[{args.sample}] {marker}: {len(df_s)} cells  "
          f"({df_s['pos'].mean()*100:.1f}% GT+)  "
          f"ours acc={df_s['mine_correct'].mean():.3f}  MIPHEI acc={df_s['miphei_correct'].mean():.3f}")

    with h5py.File(BENCH_DIR / f"{args.sample}_patch_dataset.h5", "r") as f:
        coords = f["coords"][:]
        ps0 = int(f.attrs["patch_size_level0"])

    picks = pick_patches(df_s, coords, ps0, args.n, args.min_cells, args.min_pos)
    if not picks:
        raise SystemExit("no qualifying patches (loosen --min_cells/--min_pos)")

    # masks + H&E
    row = sample_rows(args.sample)
    cells = pd.read_csv(TILE_DIR / row.nuclei_csv_path, usecols=["label", "x", "y"])
    max_lab = int(cells.label.max())
    tile_index = index_nuclei_tiles(Path(row.nuclei_slide_path).stem)
    cell_scale = TILE_MASK / TILE_L0
    rad_px = round(args.cell_radius_um / 0.5)
    he_path, _ = crc_paths(args.sample)
    arr, c_ax, h_ax, w_ax = open_zarr_level0(he_path)
    H, W = arr.shape[h_ax], arr.shape[w_ax]

    # per-cell outcome LUTs (label-indexed), restricted to this slide
    pos_lut = np.zeros(max_lab + 1, np.int8)
    mine_lut = np.zeros(max_lab + 1, np.int8)
    mph_lut = np.zeros(max_lab + 1, np.int8)
    cid = df_s["cell_id"].values.astype(np.int64)
    keep = cid <= max_lab
    pos_lut[cid[keep]] = df_s["pos"].values[keep].astype(np.int8) + 1   # 1=neg,2=pos for "present"
    mine_lut[cid[keep]] = outcome_code(df_s["call_mine"].values, df_s["pos"].values)[keep]
    mph_lut[cid[keep]] = outcome_code(df_s["call_miphei"].values, df_s["pos"].values)[keep]

    nrow = len(picks)
    fig, axes = plt.subplots(nrow, 4, figsize=(4 * 4, 4 * nrow), squeeze=False)
    for r, (pi, score, win, loss, nsel, npos) in enumerate(picks):
        px, py = int(coords[pi, 0]), int(coords[pi, 1])
        lab = load_patch_labels(tile_index, px, py, ps0)
        mh, mw = lab.shape
        cell = expand_cells(lab, rad_px)
        m0, n0 = round(px * cell_scale), round(py * cell_scale)

        he = read_he_patch(arr, c_ax, h_ax, w_ax, px, py, ps0, H, W)
        import cv2
        disp = cv2.resize(to_uint8(np.asarray(he)), (mw, mh), interpolation=cv2.INTER_AREA)

        # col0: H&E
        axes[r][0].imshow(disp)
        axes[r][0].set_title(f"patch {pi}  ({px},{py})\n"
                             f"{'WE win' if score>0 else 'MIPHEI win'}  Δ={score:+d}  "
                             f"(win {win}/loss {loss}, {npos}+ /{nsel})", fontsize=8)

        # col1: H&E + nuclei outline + GT positive fill
        axes[r][1].imshow(disp)
        gt = pos_lut[np.clip(cell, 0, max_lab)]
        gt_pos = (gt == 2) & (cell > 0)
        ov = np.zeros((mh, mw, 4), np.float32); ov[gt_pos] = (0.1, 0.4, 1.0, 0.55)
        axes[r][1].imshow(ov)
        outline = (lab > 0) & (ndi.minimum_filter(lab, size=3) != lab)
        oy, ox = np.where(outline); axes[r][1].scatter(ox, oy, s=0.4, c="cyan", marker=".", alpha=0.5)
        axes[r][1].set_title(f"GT {marker}+ (blue)", fontsize=8)

        # col2/3: MIPHEI / ours confusion
        for c, (lut, ttl) in enumerate([(mph_lut, "MIPHEI"), (mine_lut, "ours")], start=2):
            axes[r][c].imshow(disp)
            axes[r][c].imshow(fill_overlay(cell, lut, max_lab, mh, mw))
            axes[r][c].set_title(f"{ttl} {marker}", fontsize=8)

        for c in range(4):
            if c > 0:
                draw_token_grid(axes[r][c], mh, mw)
            axes[r][c].set_xlim(0, mw); axes[r][c].set_ylim(mh, 0)
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])

    handles = [mpatches.Patch(color=np.array(rgb) / 255, label=lab)
               for _, (rgb, lab) in sorted(OUTCOME.items())]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(f"{args.sample} — {marker}: per-cell calls vs GT  "
                 f"(top {args.n} our-wins + {args.n} our-losses)", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.sample}_{marker}_compare.png"
    fig.savefig(out, dpi=135, bbox_inches="tight")
    print(f"saved → {out}")


if __name__ == "__main__":
    main()