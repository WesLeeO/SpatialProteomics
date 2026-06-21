"""
Cell-level metric plots (one metric at a time: AUPRC or F1), reading the per-marker
+ MEAN15 rows from eval_cell_auprc_<tag>.csv (written by eval_cell_auprc.py). Bars are
bootstrapped means with 95%-CI error bars and the score printed on top.

Two modes:
  default        one bar per MODEL for the mean-15 headline:
    python cell_cls/orion_cell_cls/plot_metrics.py --metric auprc
    python cell_cls/orion_cell_cls/plot_metrics.py --metric f1 --tags ours MIPHEI-vit

  --compare A B  two chosen models side-by-side across all 15 markers:
    python cell_cls/orion_cell_cls/plot_metrics.py --metric auprc --compare ours MIPHEI-vit
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
COLOR = {"auprc": "#4363d8", "f1": "#f58231"}
NICE = {"auprc": "AUPRC", "f1": "F1"}


def load_csv(d: Path, tag: str) -> pd.DataFrame:
    csv = d / f"eval_cell_auprc_{tag}.csv"
    if not csv.exists():
        raise SystemExit(f"{csv.name} not found — run eval_cell_auprc.py --tag {tag} first")
    return pd.read_csv(csv)


OURS_BLUE = "#1f77b4"
COMP_COLOR = "#9467bd"   # competitor model colour (both aggregations share it)


def compare_markers(d: Path, comp_base: str, metric: str, out: str | None,
                    ours_tag: str = "ours") -> None:
    """Per-marker FAIR comparison: competitor token-agg (hatched) vs ours, 2 bars/marker.
    Colour = model (ours blue, competitor `comp_base` purple); the competitor bar keeps the
    token-agg hatch to mark it as matched-aggregation."""
    # bar order per marker: competitor token-agg (hatched) | ours (solid blue)
    want = [f"{comp_base}-tokenagg", ours_tag]
    loaded = []
    for t in want:
        csv = d / f"eval_cell_auprc_{t}.csv"
        if csv.exists():
            loaded.append((t, pd.read_csv(csv).query("marker != 'MEAN15'").set_index("marker")))
        else:
            print(f"  skip {t}: {csv.name} not found")
    if len(loaded) < 2:
        raise SystemExit(f"need >=2 of {want} present (have {[t for t,_ in loaded]})")

    common  = set.intersection(*[set(df.index) for _, df in loaded])
    ref     = dict(loaded).get(ours_tag, loaded[0][1])           # order markers by ours (or 1st)
    markers = sorted([m for m in ref.index if m in common],
                     key=lambda m: ref.loc[m, f"{metric}_boot_mean"], reverse=True)

    n = len(loaded); w = 0.8 / n
    x = np.arange(len(markers))
    color = lambda t: OURS_BLUE if t == ours_tag else COMP_COLOR
    # hatch = token aggregation: ours AND the -tokenagg competitor (both matched here)
    hatch = lambda t: "//" if (t == ours_tag or t.endswith("-tokenagg")) else ""

    fig, ax = plt.subplots(figsize=(max(11, len(markers) * 1.05), 5.5))
    top_max = 0.0
    for j, (t, df) in enumerate(loaded):
        v  = df.loc[markers, f"{metric}_boot_mean"].values
        lo = df.loc[markers, f"{metric}_ci_lo"].values
        hi = df.loc[markers, f"{metric}_ci_hi"].values
        top_max = max(top_max, float(hi.max()))
        off  = (j - (n - 1) / 2) * w
        bars = ax.bar(x + off, v, w, yerr=np.vstack([v - lo, hi - v]), capsize=2,
                      color=color(t), edgecolor="black", linewidth=0.5,
                      error_kw=dict(lw=0.8, ecolor="black"))
        for bar, val, h in zip(bars, v, hi):
            if hatch(t):
                bar.set_hatch(hatch(t))
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.008, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=6, rotation=0)

    ax.set_xticks(x); ax.set_xticklabels(markers, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel(f"{NICE[metric]} (bootstrap mean, 95% CI)")
    ax.set_ylim(0, min(1.08, top_max + 0.12))
    means = {t: df.loc[markers, f"{metric}_boot_mean"].mean() for t, df in loaded}
    ax.set_title(f"{NICE[metric]} per marker — ours vs {comp_base} (token-agg, matched)\n"
                 + "  ·  ".join(f"{t}={m:.3f}" for t, m in means.items()), fontsize=10)
    ax.grid(axis="y", lw=0.4, alpha=0.4)

    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(facecolor=OURS_BLUE, edgecolor="black", hatch="//",
                                      label=f"{ours_tag} (token-agg)"),
                       mpatches.Patch(facecolor=COMP_COLOR, edgecolor="black", hatch="//",
                                      label=f"{comp_base} (token-agg)")],
              title="model", fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()

    out = out or str(d / f"compare_ours_vs_{comp_base}_{metric}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metric", choices=["auprc", "f1"], default="auprc",
                    help="which metric to plot (one at a time)")
    ap.add_argument("--dir", default=str(HERE), help="dir with eval_cell_auprc_<tag>.csv")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="model tags to include, in this order (default: all found, sorted by score)")
    ap.add_argument("--compare", nargs="+", metavar="COMP", default=None,
                    help="per-marker 3-bar plot: competitor base name shown in native + token-agg "
                         "vs ours, e.g. --compare MIPHEI-vit (or 'ours MIPHEI-vit')")
    ap.add_argument("--out", default=None, help="default: <dir>/model_comparison_<metric>.png")
    args = ap.parse_args()
    metric = args.metric
    d = Path(args.dir)

    if args.compare:
        comp = [t for t in args.compare if t != "ours"]   # ours is always included
        if not comp:
            raise SystemExit("--compare needs a competitor base name, e.g. --compare MIPHEI-vit")
        compare_markers(d, comp[0], metric, args.out)
        return

    if args.tags:
        tags = args.tags
    else:
        tags = sorted(p.stem.replace("eval_cell_auprc_", "")
                      for p in d.glob("eval_cell_auprc_*.csv"))

    rows = []
    for t in tags:
        csv = d / f"eval_cell_auprc_{t}.csv"
        if not csv.exists():
            print(f"  skip {t}: {csv.name} not found")
            continue
        h = pd.read_csv(csv).query("marker == 'MEAN15'")
        if h.empty:
            print(f"  skip {t}: no MEAN15 row (re-run eval_cell_auprc.py)")
            continue
        r = h.iloc[0]
        rows.append({"tag": t, "val": r[f"{metric}_boot_mean"],
                     "lo": r[f"{metric}_ci_lo"], "hi": r[f"{metric}_ci_hi"]})
    if not rows:
        raise SystemExit("no MEAN15 rows found — run eval_cell_auprc.py first")

    df = pd.DataFrame(rows)
    if not args.tags:  # default ordering: best score first
        df = df.sort_values("val", ascending=False).reset_index(drop=True)

    labels = df["tag"].tolist()
    x = np.arange(len(labels))
    yerr = np.vstack([df["val"] - df["lo"], df["hi"] - df["val"]])

    # COLOUR = model identity (a token-agg variant shares its native model's colour);
    # HATCH (diagonal) = the token-agg / matched-aggregation variant. So you read model
    # from colour and aggregation (native vs token-agg) from the hatch. Same for auprc/f1.
    import matplotlib.patches as mpatches
    def base_model(t):
        return t[:-len("-tokenagg")] if t.endswith("-tokenagg") else t
    bases   = sorted({base_model(t) for t in labels})
    OURS_BLUE = "#1f77b4"
    # competitor palette excludes the ours-blue (tab10[0]) so no model collides with ours
    palette = [c for c in (list(plt.cm.tab10.colors) + list(plt.cm.tab20b.colors))
               if matplotlib.colors.to_hex(c) != OURS_BLUE]
    bcolor, ci = {}, 0
    for b in bases:
        if b == "ours":
            bcolor[b] = OURS_BLUE                        # ours fixed blue
        else:
            bcolor[b] = palette[ci % len(palette)]; ci += 1
    colors  = [bcolor[base_model(t)] for t in labels]
    hatches = ["//" if (t == "ours" or t.endswith("-tokenagg")) else "" for t in labels]

    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.1), 5.5))
    bars = ax.bar(x, df["val"], 0.6, yerr=yerr, capsize=4, color=colors,
                  edgecolor="black", linewidth=0.6, error_kw=dict(lw=1.2, ecolor="black"))
    for bar, h in zip(bars, hatches):
        if h:
            bar.set_hatch(h)
    # both legends OUTSIDE the axes (right side) so they never overlap bars/CIs
    leg1 = ax.legend(handles=[mpatches.Patch(facecolor=bcolor[b], edgecolor="black", label=b)
                              for b in bases],
                     fontsize=8, framealpha=0.9, title="model",
                     loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(leg1)
    ax.legend(handles=[mpatches.Patch(facecolor="white", edgecolor="black",
                                      label="native (pixel mean-in-nucleus)"),
                       mpatches.Patch(facecolor="white", edgecolor="black", hatch="//",
                                      label="token-agg (matched/fair)")],
              fontsize=8, framealpha=0.9, title="aggregation",
              loc="upper left", bbox_to_anchor=(1.01, 0.42))

    for bar, v, top in zip(bars, df["val"], df["hi"]):
        ax.text(bar.get_x() + bar.get_width()/2, top + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel(f"{NICE[metric]} (bootstrap mean, 95% CI)")
    ax.set_ylim(0, min(1.0, float(df["hi"].max()) + 0.12))
    ax.set_title(f"Cell-level mean-15 {NICE[metric]} by model (MIPHEI protocol)", fontsize=12)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    fig.tight_layout()

    out = args.out or str(d / f"model_comparison_{metric}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved -> {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()