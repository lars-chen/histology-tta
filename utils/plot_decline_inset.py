"""
Graphical-abstract inset (hero thesis panel):
TTA gain (Δ balanced accuracy) vs baseline balanced accuracy, one point per
frozen model averaged across the classification datasets. The downward trend
shows TTA's benefit shrinks as the backbone gets stronger.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Reuse the family maps / palette from the existing abstract inset
from utils.plot_graphabs_inset import MODEL_TYPE as _MT, MODEL_FULL as _MF, C_HISTO, C_GENERAL

# Extend with the general-purpose ResNets (omitted from the abstract-inset map)
MODEL_TYPE = {**_MT, "resnet18": "general", "resnet50": "general"}
MODEL_FULL = {**_MF, "resnet18": "ResNet-18", "resnet50": "ResNet-50"}


def build_data(results_csv="results/canonical_results.csv", head_type="linear"):
    df = pd.read_csv(results_csv)
    df = df[
        df.strategy.isin(["none", "d4"]) &
        (df.backbone_mode == "frozen") &
        (df.head_type == head_type) &
        (df.aggregation == "mean")
    ]   # all 4 datasets (incl. MHIST) — overall, across-everything statement
    base = df[df.strategy == "none"].groupby(["model", "dataset", "seed"])["balanced_acc"].mean()
    tta  = df[df.strategy == "d4"  ].groupby(["model", "dataset", "seed"])["balanced_acc"].mean()
    m = base.rename("base").reset_index().merge(
        tta.rename("tta").reset_index(), on=["model", "dataset", "seed"])
    m["delta"] = (m.tta - m.base) * 100
    m["base_pct"] = m.base * 100
    g = m.groupby("model").agg(base_mean=("base_pct", "mean"),
                               delta_mean=("delta", "mean")).reset_index()
    g["type"]  = g.model.map(MODEL_TYPE)
    g["label"] = g.model.map(MODEL_FULL)
    return g.dropna(subset=["type"])


def plot(out_dir: Path = Path("figures"),
         results_csv: str = "results/canonical_results.csv", head_type: str = "linear"):
    g = build_data(results_csv, head_type)

    # dimensions/fonts matched to utils/plot_combined_inset.py (single panel)
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.axhline(0, color="#999999", lw=1.0, ls="--", zorder=1)

    xs, ys = g.base_mean.values, g.delta_mean.values
    # trend line only (no correlation annotation — see stats discussion)
    if len(xs) >= 3:
        b1, b0 = np.polyfit(xs, ys, 1)
        xl = np.linspace(xs.min(), xs.max(), 100)
        ax.plot(xl, b1 * xl + b0, color="#444444", lw=2.0, zorder=2, alpha=0.8)

    for t, c in [("general", C_GENERAL), ("histology", C_HISTO)]:
        s = g[g.type == t]
        ax.scatter(s.base_mean, s.delta_mean, color=c, s=90, zorder=3,
                   alpha=0.9, linewidths=0, clip_on=False)

    ax.set_xlabel("Baseline balanced accuracy (%)", fontsize=16)
    ax.set_ylabel("Δ Balanced Accuracy (pp)", fontsize=16)
    ax.tick_params(axis="both", labelsize=15, length=5, width=1.0, color="#666666")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#cccccc"); ax.spines["bottom"].set_color("#cccccc")

    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=C_GENERAL, label="General-purpose"),
                       mpatches.Patch(color=C_HISTO,   label="Histology FMs")],
              fontsize=14, framealpha=0.0, loc="upper right",
              bbox_to_anchor=(1.0, 1.0), handlelength=1.2)

    fig.tight_layout(pad=0.4)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"decline_inset.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_dir}/decline_inset.png/.pdf")
    print(g[["label", "type", "base_mean", "delta_mean"]].round(2).to_string(index=False))


if __name__ == "__main__":
    plot()
