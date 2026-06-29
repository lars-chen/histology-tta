"""
Out-of-sample patch-level segmentation analysis of D4 TTA on the 50
fully-annotated Camelyon17 slides, across 6 backbones. Predictions come from
leave-one-patient-out patch probes (camelyon17/lopo_patch_probe.py), so every
slide is scored by a probe that never saw that patient. Per-slide F1 == Dice
for binary tumour segmentation. Three panels (family contrast):
  (a) Per-model ΔDice: general-purpose backbones gain most, histology FMs least;
      every model shows per-slide heterogeneity.
  (b) ΔDice vs baseline Dice: the gain tracks baseline weakness — general-purpose
      probes start far lower (≈63% Dice) and benefit more, histology FMs start
      high (≈88%) with little headroom. (Spearman r reported.)
  (c) Δsensitivity vs Δspecificity: with single class-balancing the gains come
      from both fewer false positives and recovered true positives.

Reads results/patch_dice_camelyon17_lopo.csv (per-slide, per-model).
Output: figures/figure_spatial_analysis.png/.pdf
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from utils.plot_analysis import SAVEKW

DICE_CSV = Path("results/patch_dice_camelyon17_lopo.csv")
MIN_POS  = 10

C_HISTO   = "#2a5fa5"
C_GENERAL = "#e07b39"

FAMILY = {
    "uni": "histology", "phikon2": "histology", "virchow2": "histology",
    "gigapath": "histology", "convnextv2": "general", "dinov2-s": "general",
}
DISPLAY = {
    "uni": "UNI", "phikon2": "Phikon-v2", "virchow2": "Virchow-v2",
    "gigapath": "GigaPath", "convnextv2": "ConvNeXtV2", "dinov2-s": "DINOv2-S",
}


def _color(m):  return C_HISTO if FAMILY[m] == "histology" else C_GENERAL


def _dim_grid(ax):
    for gl in ax.get_xgridlines() + ax.get_ygridlines():
        gl.set_alpha(0.15)


def _load():
    d = pd.read_csv(DICE_CSV)
    d = d[d.n_pos >= MIN_POS].copy()
    d["dDice"] = (d.f1_tta   - d.f1_base)   * 100
    d["dSpec"] = (d.spec_tta - d.spec_base) * 100
    d["dSens"] = (d.sens_tta - d.sens_base) * 100
    d["base"]  = d.f1_base * 100
    d["fam"]   = d.model.map(FAMILY)
    return d


def _panel_a(ax, d):
    order = d.groupby("model")["dDice"].mean().sort_values().index.tolist()
    rng = np.random.RandomState(0)
    for i, m in enumerate(order):
        v = d[d.model == m]["dDice"].values
        c = _color(m)
        jit = (rng.rand(len(v)) - 0.5) * 0.55
        ax.scatter(v, np.full(len(v), i) + jit, s=10, color=c, alpha=0.35,
                   lw=0, zorder=2)
        mu = v.mean()
        ax.scatter(mu, i, s=58, color=c, zorder=4, edgecolor="white", lw=1.0)
        ax.annotate(f"{mu:+.2f}", xy=(mu, i), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=6.4,
                    color=c, fontweight="bold")
    ax.axvline(0, color="#888888", lw=0.8, ls="--", zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([DISPLAY[m] for m in order], fontsize=7.5)
    ax.set_xlabel(r"Per-slide $\Delta$ Dice (pp)", fontsize=8.5)
    ax.set_title("(a) General-purpose backbones gain most", fontsize=8.5,
                 fontweight="bold", pad=4, loc="left")
    ax.tick_params(labelsize=7.5)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_GENERAL,
               markersize=7, label="General-purpose"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_HISTO,
               markersize=7, label="Histology FM"),
    ]
    ax.legend(handles=handles, fontsize=6.8, frameon=False, loc="lower right")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")


def _panel_b(ax, d):
    for fam, c in [("general", C_GENERAL), ("histology", C_HISTO)]:
        sub = d[d.fam == fam]
        ax.scatter(sub.base, sub.dDice, s=24, color=c, alpha=0.5,
                   edgecolor="white", lw=0.5, zorder=3)
    ax.axhline(0, color="#888888", lw=0.8, ls="--", zorder=1)
    r, _ = spearmanr(d.base, d.dDice)
    ax.text(0.96, 0.96, f"Spearman $r={r:.2f}$\n(weaker baseline → larger gain)",
            transform=ax.transAxes, fontsize=7, va="top", ha="right",
            color="#555555")
    ax.set_xlabel("Baseline Dice (%)", fontsize=8.5)
    ax.set_ylabel(r"$\Delta$ Dice (pp)", fontsize=8.5)
    ax.set_title("(b) Gain tracks baseline weakness", fontsize=8.5,
                 fontweight="bold", pad=4, loc="left")
    ax.tick_params(labelsize=7.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")


def _panel_c(ax, d):
    for fam, c in [("general", C_GENERAL), ("histology", C_HISTO)]:
        sub = d[d.fam == fam]
        ax.scatter(sub.dSpec, sub.dSens, s=24, color=c, alpha=0.5,
                   edgecolor="white", lw=0.5, zorder=3)
    ax.axhline(0, color="#888888", lw=0.8, ls="--", zorder=1)
    ax.axvline(0, color="#888888", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel(r"$\Delta$ specificity (pp)", fontsize=8.5)
    ax.set_ylabel(r"$\Delta$ sensitivity (pp)", fontsize=8.5)
    ax.set_title("(c) Gains from both precision and recall", fontsize=8.5,
                 fontweight="bold", pad=4, loc="left")
    ax.tick_params(labelsize=7.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")


def plot(out_dir: Path = Path("figures")):
    d = _load()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    fig.patch.set_facecolor("white")
    _panel_a(axes[0], d)
    _panel_b(axes[1], d)
    _panel_c(axes[2], d)
    for ax in axes:
        _dim_grid(ax)
    fig.tight_layout(pad=0.6, w_pad=1.8)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"figure_spatial_analysis.{ext}", **SAVEKW)
    plt.close(fig)
    print(f"Saved → {out_dir}/figure_spatial_analysis.png/.pdf")

    print(f"\nout-of-sample (LOPO), {len(d)} slide-model pairs >= {MIN_POS} tumour patches")
    for fam, g in d.groupby("fam"):
        print(f"  {fam:10s} baseDice {g.base.mean():.1f}  ΔDice {g.dDice.mean():+.2f}")
    print(f"  slide-level ΔDice vs baseline: r={spearmanr(d.base, d.dDice)[0]:.2f}")


if __name__ == "__main__":
    plot()
