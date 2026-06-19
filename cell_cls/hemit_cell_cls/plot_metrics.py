"""
Cell-level metric plots for HEMIT (AUPRC or F1), reading the per-marker + MEAN rows
from eval_cell_auprc_<tag>.csv. Mirror of pathocell_cls/plot_metrics.py. HEMIT has
2 markers (Pan-CK, CD3).

Encoding: COLOUR = model identity (a `-tokenagg` variant shares its native model's
colour), HATCH (//) = token aggregation (ours + any -tokenagg competitor), solid =
native pixel-in-nucleus, ours = blue. Same for auprc/f1.

  python cell_cls/hemit_cell_cls/plot_metrics.py --metric auprc
  python cell_cls/hemit_cell_cls/plot_metrics.py --metric auprc --compare MIPHEI-vit
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = Path(__file__).resolve().parent
NICE = {"auprc": "AUPRC", "f1": "F1"}
MEAN_ROW = "MEAN"
OURS_TAG = "ours"
OURS_BLUE = "#1f77b4"
COMP_COLOR = "#9467bd"


def load_csv(d: Path, tag: str) -> pd.DataFrame:
    csv = d / f"eval_cell_auprc_{tag}.csv"
    if not csv.exists():
        raise SystemExit(f"{csv.name} not found — run eval_cell_auprc.py --tag {tag} first")
    return pd.read_csv(csv)


def base_model(t: str) -> str:
    return t[:-len("-tokenagg")] if t.endswith("-tokenagg") else t


def compare_markers(d: Path, comp_tag: str, metric: str, out: str | None,
                    ours_tag: str = OURS_TAG) -> None:
    """Per-marker comparison: competitor `comp_tag` vs ours (2 bars/marker). Colour = model
    (ours blue, competitor purple); ours is hatched (token-agg), competitor solid unless -tokenagg."""
    want = [comp_tag, ours_tag]
    loaded = [(t, load_csv(d, t).query("marker != @MEAN_ROW").set_index("marker")) for t in want]
    common = set.intersection(*[set(df.index) for _, df in loaded])
    ref = dict(loaded)[ours_tag]
    markers = sorted([m for m in ref.index if m in common],
                     key=lambda m: ref.loc[m, f"{metric}_boot_mean"], reverse=True)

    n = len(loaded); w = 0.8 / n
    x = np.arange(len(markers))
    color = lambda t: OURS_BLUE if t == ours_tag else COMP_COLOR
    hatch = lambda t: "//" if (t == ours_tag or t.endswith("-tokenagg")) else ""

    fig, ax = plt.subplots(figsize=(max(6, len(markers) * 2.2), 5.5))
    top_max = 0.0
    for j, (t, df) in enumerate(loaded):
        v  = df.loc[markers, f"{metric}_boot_mean"].values
        lo = df.loc[markers, f"{metric}_ci_lo"].values
        hi = df.loc[markers, f"{metric}_ci_hi"].values
        top_max = max(top_max, float(hi.max()))
        bars = ax.bar(x + (j - (n - 1) / 2) * w, v, w, yerr=np.vstack([v - lo, hi - v]),
                      capsize=2, color=color(t), edgecolor="black", linewidth=0.5,
                      error_kw=dict(lw=0.8, ecolor="black"))
        for bar, val, hh in zip(bars, v, hi):
            if hatch(t):
                bar.set_hatch(hatch(t))
            ax.text(bar.get_x() + bar.get_width()/2, hh + 0.012, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=8, rotation=0)

    ax.set_xticks(x); ax.set_xticklabels(markers, fontsize=10)
    ax.set_ylabel(f"{NICE[metric]} (bootstrap mean, 95% CI)")
    ax.set_ylim(0, min(1.08, top_max + 0.12))
    means = {t: df.loc[markers, f"{metric}_boot_mean"].mean() for t, df in loaded}
    ax.set_title(f"HEMIT {NICE[metric]} per marker — ours vs {comp_tag}\n"
                 + "  ·  ".join(f"{t}={m:.3f}" for t, m in means.items()), fontsize=10)
    ax.grid(axis="y", lw=0.4, alpha=0.4)

    lbl = lambda t: f"{t} (token-agg)" if t.endswith("-tokenagg") else t
    handles = [mpatches.Patch(facecolor=OURS_BLUE, edgecolor="black", hatch="//",
                              label=f"{ours_tag} (token-agg)"),
               mpatches.Patch(facecolor=COMP_COLOR, edgecolor="black",
                              hatch=hatch(comp_tag) or None, label=lbl(comp_tag))]
    ax.legend(handles=handles, title="model", fontsize=8,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()

    out = out or str(d / f"compare_ours_vs_{comp_tag}_{metric}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", choices=["auprc", "f1"], default="auprc")
    ap.add_argument("--dir", default=str(HERE), help="dir with eval_cell_auprc_<tag>.csv")
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--compare", nargs="+", metavar="COMP", default=None,
                    help="per-marker competitor-vs-ours plot, e.g. --compare MIPHEI-vit")
    ap.add_argument("--out", default=None, help="default: <dir>/model_comparison_<metric>.png")
    args = ap.parse_args()
    metric = args.metric
    d = Path(args.dir)

    if args.compare:
        comp = [t for t in args.compare if t != OURS_TAG]
        if not comp:
            raise SystemExit("--compare needs a competitor tag, e.g. --compare MIPHEI-vit")
        compare_markers(d, comp[0], metric, args.out); return

    tags = args.tags or sorted(p.stem.replace("eval_cell_auprc_", "")
                               for p in d.glob("eval_cell_auprc_*.csv"))
    rows = []
    for t in tags:
        csv = d / f"eval_cell_auprc_{t}.csv"
        if not csv.exists():
            print(f"  skip {t}: {csv.name} not found"); continue
        h = pd.read_csv(csv).query("marker == @MEAN_ROW")
        if h.empty:
            print(f"  skip {t}: no {MEAN_ROW} row"); continue
        r = h.iloc[0]
        rows.append({"tag": t, "val": r[f"{metric}_boot_mean"],
                     "lo": r[f"{metric}_ci_lo"], "hi": r[f"{metric}_ci_hi"]})
    if not rows:
        raise SystemExit("no MEAN rows found — run eval_cell_auprc.py first")

    df = pd.DataFrame(rows)
    if not args.tags:
        df = df.sort_values("val", ascending=False).reset_index(drop=True)

    labels = df["tag"].tolist()
    x = np.arange(len(labels))
    yerr = np.vstack([df["val"] - df["lo"], df["hi"] - df["val"]])

    bases   = sorted({base_model(t) for t in labels})
    palette = [c for c in (list(plt.cm.tab10.colors) + list(plt.cm.tab20b.colors))
               if matplotlib.colors.to_hex(c) != OURS_BLUE]
    bcolor, ci = {}, 0
    for b in bases:
        if b == OURS_TAG:
            bcolor[b] = OURS_BLUE
        else:
            bcolor[b] = palette[ci % len(palette)]; ci += 1
    colors  = [bcolor[base_model(t)] for t in labels]
    hatches = ["//" if (t == OURS_TAG or t.endswith("-tokenagg")) else "" for t in labels]

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 5.5))
    bars = ax.bar(x, df["val"], 0.6, yerr=yerr, capsize=4, color=colors,
                  edgecolor="black", linewidth=0.6, error_kw=dict(lw=1.2, ecolor="black"))
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
    for bar, v, hh in zip(bars, df["val"], df["hi"]):
        ax.text(bar.get_x() + bar.get_width()/2, hh + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel(f"{NICE[metric]} (bootstrap mean, 95% CI)")
    ax.set_ylim(0, min(1.0, float(df["hi"].max()) + 0.12))
    ax.set_title(f"HEMIT cell-level mean-2 {NICE[metric]} by model (MIPHEI protocol)", fontsize=12)
    ax.grid(axis="y", lw=0.4, alpha=0.4)

    leg1 = ax.legend(handles=[mpatches.Patch(facecolor=bcolor[b], edgecolor="black", label=b)
                              for b in bases],
                     title="model", fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(leg1)
    if any(hatches):
        ax.legend(handles=[mpatches.Patch(facecolor="white", edgecolor="black",
                                          label="native (pixel mean-in-nucleus)"),
                           mpatches.Patch(facecolor="white", edgecolor="black", hatch="//",
                                          label="token-agg (ours)")],
                  title="aggregation", fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 0.45))
    fig.tight_layout()
    out = args.out or str(d / f"model_comparison_{metric}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved -> {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()