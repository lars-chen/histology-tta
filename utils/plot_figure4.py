"""
Figure 4: Selective TTA Pareto curves — TCGA-UT and NCT-CRC-100K side by side.
Reads results/selective_tta_results.csv (linear head, frozen backbone).
Shows per-model thin curves + family-average thick curve for histology FMs
and general-purpose models. Annotates t=0.3 and t=0.5 for both families.
Output: figures/figure4_selective_pareto.pdf/.png  (figure*, full two-column width)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils.plot_analysis import SAVEKW

SEL_CSV  = Path("results/selective_tta_results.csv")
DATASETS = ["tcga-ut", "nct-crc-100k"]
DS_LABELS = {"tcga-ut": "TCGA-UT (31 classes)", "nct-crc-100k": "NCT-CRC-100K (9 classes)"}
HEAD     = "linear"

C_HISTO   = "#2a5fa5"
C_GENERAL = "#e07b39"

FM_HIST = {
    "phikon", "phikon2", "uni", "uni2",
    "virchow", "virchow2", "gigapath", "hoptimus", "ctranspath",
}

# xytext positions keyed by (dataset, family, target_idx) where 0=~50% gain, 1=~90% gain
ANNOT_POS = {
    ("tcga-ut",      "histology", 0): (30,  0.22),
    ("tcga-ut",      "histology", 1): (30,  0.25),
    ("tcga-ut",      "general",   0): (32,  0.90),
    ("tcga-ut",      "general",   1): (65,  1.65),
    ("nct-crc-100k", "histology", 0): (8,   0.12),
    ("nct-crc-100k", "histology", 1): (19,  0.16),
    ("nct-crc-100k", "general",   0): (12,  0.50),
    ("nct-crc-100k", "general",   1): (16,  0.82),
}
TARGET_PCTS       = [50, 90]
SKIP_ANNOTS: set  = {  # (dataset, family, target_idx) to skip
    ("tcga-ut",      "histology", 0),
    ("nct-crc-100k", "histology", 0),
}


def _load():
    df = pd.read_csv(SEL_CSV)
    df = df[df.head_type == HEAD]
    df["family"]   = df.model.apply(lambda x: "histology" if x in FM_HIST else "general")
    df["delta_pp"] = df["delta_vs_baseline"] * 100
    return df


def _per_model_curves(df: pd.DataFrame, dataset: str):
    curves: dict[str, list] = {"histology": [], "general": []}
    sub = df[df.dataset == dataset]
    thresholds = np.sort(sub.threshold.unique())
    for model, gm in sub.groupby("model"):
        family = gm["family"].iloc[0]
        agg = (gm.groupby("threshold")
                  .agg(coverage=("coverage_pct", "mean"),
                       delta=("delta_pp", "mean"))
                  .reindex(thresholds))
        f = agg["coverage"].values
        d = agg["delta"].values
        order = np.argsort(f)
        curves[family].append((f[order], d[order]))
    return curves


def _family_mean(runs):
    all_f = np.array([r[0] for r in runs])
    all_d = np.array([r[1] for r in runs])
    return all_f.mean(0), all_d.mean(0), all_d.std(0)


def _find_operating_point(df, dataset, family, target_pct):
    """Return (threshold, coverage, delta, actual_pct) closest to target_pct of full gain."""
    sub  = df[(df.dataset == dataset) & (df.family == family)]
    full = float(sub[sub.threshold == 0.0]["delta_pp"].mean())
    if full <= 0:
        return None
    thr = (sub.groupby("threshold")
              .agg(cov=("coverage_pct", "mean"), delta=("delta_pp", "mean"))
              .reset_index())
    thr["pct"] = thr["delta"] / full * 100
    idx = (thr["pct"] - target_pct).abs().idxmin()
    row = thr.loc[idx]
    return float(row["threshold"]), float(row["cov"]), float(row["delta"]), float(row["pct"])


def _annotate_operating_points(ax, df, dataset, family, color):
    for i, target in enumerate(TARGET_PCTS):
        key = (dataset, family, i)
        if key in SKIP_ANNOTS or key not in ANNOT_POS:
            continue
        xytext = ANNOT_POS[key]
        result = _find_operating_point(df, dataset, family, target)
        if result is None:
            continue
        t, cov, delt, pct = result
        ax.plot(cov, delt, "o", color=color, ms=6, zorder=6, mew=1.2, mec="white")
        ax.annotate(
            f"{cov:.0f}% samples,  {pct:.0f}% gain",
            xy=(cov, delt), xytext=xytext,
            fontsize=6, color=color, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.6),
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"),
            zorder=7,
        )


def _draw_panel(ax, df, dataset, show_ylabel=True, show_legend=True):
    curves = _per_model_curves(df, dataset)

    for family, color in [("histology", C_HISTO), ("general", C_GENERAL)]:
        runs = curves[family]
        if not runs:
            continue
        # thin per-model curves
        for f, d in runs:
            ax.plot(f, d, color=color, lw=0.6, alpha=0.22, zorder=2)
        # thick family mean ± std
        f_m, d_m, d_s = _family_mean(runs)
        label = "Histology FMs" if family == "histology" else "General-purpose"
        ax.fill_between(f_m, d_m - d_s, d_m + d_s, color=color, alpha=0.12, lw=0, zorder=2)
        ax.plot(f_m, d_m, color=color, lw=2.2, zorder=4, label=label)
        # annotations at ~50% and ~90% of full gain
        _annotate_operating_points(ax, df, dataset, family, color)

    ax.axhline(0, color="#888888", lw=0.8, ls="--", zorder=1)
    ax.set_xlim(-2, 105)
    # ymax: seed-averaged per-model delta at full coverage, then take max
    ymax = (df[(df.dataset == dataset) & (df.threshold == 0.0)]
            .groupby("model")["delta_pp"].mean().max())
    ax.set_ylim(-0.05, ymax * 1.25)
    ax.set_xlabel("Samples receiving $D_4$ TTA (%)", fontsize=8.5)
    if show_ylabel:
        ax.set_ylabel(r"$\Delta$ Balanced Accuracy (pp)", fontsize=8.5)
    ax.text(0.98, 0.97, DS_LABELS[dataset], transform=ax.transAxes,
            fontsize=8.5, va="top", ha="right", color="#555555",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.75,
                      edgecolor="none"))
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(labelsize=7.5)
    if show_legend:
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.grid(False)


def plot(out_dir: Path = Path("figures"), sel_csv: Path = SEL_CSV):
    df = _load()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=False)
    fig.patch.set_facecolor("white")

    _draw_panel(axes[0], df, "tcga-ut",      show_ylabel=True,  show_legend=True)
    _draw_panel(axes[1], df, "nct-crc-100k", show_ylabel=False, show_legend=False)

    fig.tight_layout(pad=0.6, w_pad=1.2)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"figure4_selective_pareto.{ext}", **SAVEKW)
    plt.close(fig)
    print(f"Saved → {out_dir}/figure4_selective_pareto.png/.pdf")

    # Print operating points
    for ds in DATASETS:
        print(f"\n{ds}:")
        for fam in ["histology", "general"]:
            sub = df[(df.dataset == ds) & (df.family == fam)]
            full = sub[sub.threshold == 0.0]["delta_pp"].mean()
            for t in [0.3, 0.5]:
                r = sub[sub.threshold == t]
                cov = r["coverage_pct"].mean()
                d   = r["delta_pp"].mean()
                print(f"  {fam} t={t}: {cov:.1f}% cov, {d:+.3f}pp ({d/full*100:.0f}% of {full:+.2f}pp)")


if __name__ == "__main__":
    plot()
