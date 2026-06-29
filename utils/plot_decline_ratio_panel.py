"""
Two-panel figure (TCGA-UT, frozen linear probe, per model):
  (a) Δ balanced accuracy vs baseline  — the net TTA benefit declines with model quality
  (b) corrected:corrupted ratio vs baseline — correction *quality* also declines
      (so the decline is not purely headroom / fewer errors to fix)

TCGA-UT is used because it has the widest baseline range across models; the NCT
datasets are ceiling-bunched and lack the dynamic range to test this.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from utils.plot_graphabs_inset import MODEL_TYPE as _MT, C_HISTO, C_GENERAL
MODEL_TYPE = {**_MT, "resnet18": "general", "resnet50": "general"}

DATASET = "tcga-ut"


def build_data(csv="results/canonical_results.csv", head_type="linear", dataset=DATASET):
    df = pd.read_csv(csv)
    d = df[(df.backbone_mode == "frozen") & (df.head_type == head_type) &
           (df.aggregation == "mean") & (df.dataset == dataset)]
    base = d[d.strategy == "none"].groupby(["model", "seed"]).balanced_acc.mean()
    tta  = d[d.strategy == "d4"].groupby(["model", "seed"]).agg(
        bacc=("balanced_acc", "mean"), nc=("n_corrected", "mean"), nk=("n_corrupted", "mean"))
    m = base.rename("bacc_base").reset_index().merge(tta.reset_index(), on=["model", "seed"])
    m["delta"] = (m.bacc - m.bacc_base) * 100
    m["base"] = m.bacc_base * 100
    m["ratio"] = m.nc / m.nk.replace(0, np.nan)
    g = m.groupby("model").agg(base=("base", "mean"), delta=("delta", "mean"),
                               ratio=("ratio", "mean")).reset_index()
    g = g[g.model.isin(MODEL_TYPE)].copy()
    g["fam"] = g.model.map(MODEL_TYPE)
    g["color"] = g.fam.map({"histology": C_HISTO, "general": C_GENERAL})
    return g


def _scatter_trend(ax, g, ycol, hline, ylabel):
    ax.axhline(hline, color="#999999", lw=0.9, ls="--", zorder=1)
    x = g["base"].values; y = g[ycol].values
    ax.scatter(x, y, color=g["color"], s=34, zorder=3, alpha=0.9, linewidths=0)
    b1, b0 = np.polyfit(x, y, 1)
    xl = np.linspace(x.min(), x.max(), 100)
    ax.plot(xl, b1 * xl + b0, color="#444444", lw=1.4, alpha=0.85, zorder=2)
    r = stats.pearsonr(x, y)[0]
    ax.text(0.96, 0.95, f"r = {r:+.2f}", transform=ax.transAxes, ha="right",
            va="top", fontsize=8.5, color="#333333")
    ax.set_xlabel("Baseline balanced accuracy (%)", fontsize=8.5)
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#cccccc"); ax.spines["bottom"].set_color("#cccccc")


def plot(out_dir: Path = Path("figures")):
    g = build_data()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    fig.patch.set_facecolor("white")
    _scatter_trend(axes[0], g, "delta", 0.0, "Δ balanced accuracy (pp)")
    _scatter_trend(axes[1], g, "ratio", 1.0, "corrected : corrupted")
    axes[0].set_title("(a) Net TTA benefit", fontsize=9)
    axes[1].set_title("(b) Correction quality", fontsize=9)
    axes[1].legend(handles=[mpatches.Patch(color=C_GENERAL, label="General-purpose"),
                            mpatches.Patch(color=C_HISTO,   label="Histology FMs")],
                   fontsize=7, framealpha=0.0, loc="upper right",
                   bbox_to_anchor=(1.0, 0.86), handlelength=1.0)
    fig.tight_layout(pad=0.5)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"decline_ratio_panel.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_dir}/decline_ratio_panel.png/.pdf  (n={len(g)} models, {DATASET})")
    print(g.sort_values("base")[["model", "fam", "base", "delta", "ratio"]].round(3).to_string(index=False))


if __name__ == "__main__":
    plot()
