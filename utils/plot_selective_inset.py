"""
Selective TTA inset: fraction of full D4 TTA gain recovered vs average views per sample.
Histology FMs only (cleanest signal). Shows that ~2 views per sample captures the full gain.
Output: figures/selective_inset.png/.pdf
"""
from __future__ import annotations
import json
import glob
from pathlib import Path

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils.plot_analysis import MODEL_TYPE_MAP

C_HISTO   = "#2a5fa5"
C_GENERAL = "#e07b39"

KEEP_GENERAL = ["convnextv2_tiny", "convnextv2_base", "dinov2_s", "dinov2_b"]


def build_data(raw_dir: str = "results/raw") -> dict:
    rows = []
    for f in glob.glob(f"{raw_dir}/selective_tta_*.json"):
        for r in json.load(open(f)):
            rows.append(r)
    df = pd.DataFrame(rows)
    df["model_type"] = df["model"].map(MODEL_TYPE_MAP)

    key = ["model", "dataset", "seed", "backbone_mode"]
    base = df[df.strategy == "none"].set_index(key)["balanced_acc"].rename("base")
    full_d4 = df[df.strategy == "d4"].set_index(key)["balanced_acc"].rename("full")

    selective = df[df.strategy.str.startswith("selective_", na=False)].copy()
    selective["threshold_frac"] = selective["threshold_frac"].astype(float)
    selective = selective.join(base, on=key).join(full_d4, on=key)
    selective = selective[selective["backbone_mode"] == "frozen"]
    selective = selective[selective["full"] - selective["base"] > 0.001]
    selective["frac_gain"] = (selective["balanced_acc"] - selective["base"]) / (selective["full"] - selective["base"])

    results = {}
    for mt, mask in [("histology", selective.model_type == "histology"),
                     ("general",   selective.model.isin(KEEP_GENERAL))]:
        sub = selective[mask & (selective.dataset == "tcga-ut") & (selective["full"] - selective["base"] > 0.001)]
        curve = sub.groupby("threshold_frac").agg(
            frac_gain=("frac_gain", "mean"),
            coverage=("coverage", "mean"),
        ).reset_index().sort_values("coverage")
        results[mt] = curve
    return results


def _plot_curve(ax, curve, color, label, annot_x=20):
    x = curve["coverage"].values * 100
    y = curve["frac_gain"].values * 100
    x_full = np.append(x, 100.0)
    y_full = np.append(y, 100.0)
    ax.plot(x_full, y_full, color=color, lw=2.2, marker="o", markersize=4, zorder=3, label=label)
    ax.scatter([x_full[-1]], [y_full[-1]], color=color, s=35, zorder=5, marker="s")
    idx = np.argmin(np.abs(x - annot_x))
    ax.scatter([x[idx]], [y[idx]], color=color, s=50, zorder=6)
    return x[idx], y[idx]


def plot(out_dir: Path = Path("figures")):
    curves = build_data()

    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(False)

    xh, yh = _plot_curve(ax, curves["histology"], C_HISTO,   "Histology FMs", annot_x=20)
    xg, yg = _plot_curve(ax, curves["general"],   C_GENERAL, "General-purpose", annot_x=20)

    # 100% reference
    ax.axhline(100, color="#999999", lw=0.9, ls="--", zorder=1)
    ax.text(98, 101.5, "all re-processed", ha="right", fontsize=6.5, color="#999999", va="bottom")

    # Annotations at 20%
    ax.annotate(f"{yh:.0f}%", xy=(xh, yh), xytext=(xh + 4, yh - 20),
                fontsize=7, color=C_HISTO, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C_HISTO, lw=0.8), ha="left")
    ax.annotate(f"{yg:.0f}%", xy=(xg, yg), xytext=(xg + 4, yg + 8),
                fontsize=7, color=C_GENERAL, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=C_GENERAL, lw=0.8), ha="left")


    ax.set_xlabel("% of samples re-processed with D₄ TTA", fontsize=8)
    ax.set_ylabel("% of full TTA gain", fontsize=8.5)
    ax.set_xlim(-2, 105)
    ax.set_ylim(-15, 130)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(axis="both", labelsize=7.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    fig.tight_layout(pad=0.4)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"selective_inset.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_dir}/selective_inset.png/.pdf")


if __name__ == "__main__":
    plot()
