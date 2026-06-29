"""
Data-ablation spatial analysis (Camelyon17, leave-one-patient-out patch probes).
Training probes on shrinking fractions of patches weakens the baseline in a
controlled way, exposing where and how much D4 TTA helps.

  (a) Baseline Dice vs training data: histology FM (UNI) stays strong even with
      2% of the data — its D4 invariance lives in the features, so the probe
      needs almost no data; the general-purpose backbone (ConvNeXtV2) degrades
      sharply without data.
  (b) ΔDice vs baseline Dice (controlled): within a single backbone, weaker
      baselines (less data) get larger TTA gains — the same inverse law, now
      causal rather than cross-model.
  (c) Net corrections per 1000 patches vs distance from the tumour margin, for
      ConvNeXtV2 at each training fraction: as the baseline weakens, corrections
      concentrate near the margin (peak at 0–0.5 mm), then flatten with full data.

Reads results/patch_dice_camelyon17_ablation.csv,
      results/camelyon17_margin_ablation.csv
Output: figures/figure_spatial_ablation.png/.pdf
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

from utils.plot_analysis import SAVEKW

DICE_CSV   = Path("results/patch_dice_camelyon17_ablation.csv")
MARGIN_CSV = Path("results/camelyon17_margin_ablation.csv")
MIN_POS    = 10

C_HISTO   = "#2a5fa5"   # UNI
C_GENERAL = "#e07b39"   # ConvNeXtV2
DISP = {"uni": "UNI (histology FM)", "convnextv2": "ConvNeXtV2 (general)"}
COL  = {"uni": C_HISTO, "convnextv2": C_GENERAL}


def _dim_grid(ax):
    for gl in ax.get_xgridlines() + ax.get_ygridlines():
        gl.set_alpha(0.15)


def _panel_a(ax, d):
    for m in ["uni", "convnextv2"]:
        g = (d[d.model == m].groupby("train_frac")
             .agg(base=("f1_base", lambda x: x.mean() * 100)).reset_index())
        ax.plot(g.train_frac * 100, g.base, "-o", color=COL[m], lw=2, ms=5,
                label=DISP[m])
    ax.set_xscale("log")
    ax.set_xlabel("Training patches used (%)", fontsize=8.5)
    ax.set_ylabel("Baseline Dice (%)", fontsize=8.5)
    ax.set_title("(a) Histology FM needs almost no data", fontsize=8.5,
                 fontweight="bold", pad=4, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.tick_params(labelsize=7.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")


def _panel_b(ax, d):
    for m in ["uni", "convnextv2"]:
        g = (d[d.model == m].groupby("train_frac")
             .agg(dD=("dDice", "mean")).reset_index())
        ax.plot(g.train_frac * 100, g.dD, "-o", color=COL[m], lw=1.8, ms=5,
                label=DISP[m], alpha=0.9)
    ax.axhline(0, color="#888888", lw=0.8, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Training patches used (%)", fontsize=8.5)
    ax.set_ylabel(r"$\Delta$ Dice (pp)", fontsize=8.5)
    ax.set_title("(b) General model gains more at every data level",
                 fontsize=8.5, fontweight="bold", pad=4, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="center right")
    ax.set_ylim(0, None)
    ax.tick_params(labelsize=7.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")


def _panel_c(ax, mg, model="convnextv2"):
    sub = mg[mg.model == model].copy()
    sub["net_per_1k"] = (sub.corrected - sub.corrupted) / sub.total.replace(0, np.nan) * 1000
    fracs = sorted(sub.train_frac.unique())
    cmap = plt.colormaps["YlOrRd"]
    for i, fr in enumerate(fracs):
        s = sub[(sub.train_frac == fr) & (sub.dist_um.between(-1000, 6000))].sort_values("dist_um")
        shade = 0.35 + 0.6 * (1 - i / max(len(fracs) - 1, 1))  # low frac = darker
        ax.plot(s.dist_um / 1000, s.net_per_1k, "-", color=cmap(shade), lw=1.8,
                label=f"{fr*100:.0f}%")
    ax.axhline(0, color="#888888", lw=0.8, zorder=1)
    ax.axvline(0, color="#bbbbbb", lw=0.8, ls=":", zorder=1)
    ax.text(0.0, ax.get_ylim()[1], " tumour\n margin", fontsize=6,
            color="#999999", va="top", ha="left")
    ax.set_xlabel("Distance from tumour margin (mm)", fontsize=8.5)
    ax.set_ylabel("Net corrections per 1000 patches", fontsize=8.5)
    ax.set_title("(c) ConvNeXtV2: corrections cluster at margin", fontsize=8.5,
                 fontweight="bold", pad=4, loc="left")
    leg = ax.legend(fontsize=6, frameon=False, loc="upper right",
                    title="train data", title_fontsize=6, ncol=2)
    ax.set_xlim(-1, 6)
    ax.tick_params(labelsize=7.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")


def plot(out_dir: Path = Path("figures")):
    d = pd.read_csv(DICE_CSV)
    d = d[d.n_pos >= MIN_POS].copy()
    d["dDice"] = (d.f1_tta - d.f1_base) * 100
    mg = pd.read_csv(MARGIN_CSV)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    fig.patch.set_facecolor("white")
    _panel_a(axes[0], d)
    _panel_b(axes[1], d)
    _panel_c(axes[2], mg)
    for ax in axes:
        _dim_grid(ax)
    fig.tight_layout(pad=0.6, w_pad=1.8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"figure_spatial_ablation.{ext}", **SAVEKW)
    plt.close(fig)
    print(f"Saved → {out_dir}/figure_spatial_ablation.png/.pdf")


if __name__ == "__main__":
    plot()
