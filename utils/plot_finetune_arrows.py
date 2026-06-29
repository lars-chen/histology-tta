"""
Frozen → finetuned arrows (TCGA-UT, linear head): for each general-purpose model
trained both ways, an arrow from its frozen point to its finetuned point in
(baseline accuracy, Δ TTA balanced acc) space. Finetuning raises baseline and
shrinks the TTA gain — augmentation invariance gets baked into the weights.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

C_GENERAL = "#e07b39"
LABELS = {"resnet18": "RN18", "resnet50": "RN50", "convnextv2_tiny": "CNX-T",
          "convnextv2_base": "CNX-B", "dinov2_s": "DV2-S", "dinov2_b": "DV2-B"}


def build(dataset="tcga-ut"):
    df = pd.read_csv("results/canonical_results.csv")
    d = df[(df.head_type == "linear") & (df.aggregation == "mean") & (df.dataset == dataset)]
    base = d[d.strategy == "none"].groupby(["model", "backbone_mode", "seed"]).balanced_acc.mean()
    tta  = d[d.strategy == "d4"].groupby(["model", "backbone_mode", "seed"]).balanced_acc.mean()
    m = base.rename("b").reset_index().merge(tta.rename("t").reset_index(),
                                             on=["model", "backbone_mode", "seed"])
    m["delta"] = (m.t - m.b) * 100; m["base"] = m.b * 100
    g = m.groupby(["model", "backbone_mode"]).agg(base=("base", "mean"),
                                                  delta=("delta", "mean")).reset_index()
    return g


def plot(out_dir=Path("figures")):
    g = build()
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.axhline(0, color="#999999", lw=0.8, ls="--", zorder=1)
    for model in LABELS:
        fr = g[(g.model == model) & (g.backbone_mode == "frozen")]
        ft = g[(g.model == model) & (g.backbone_mode == "finetuned")]
        if fr.empty or ft.empty:
            continue
        x0, y0 = fr.base.iloc[0], fr.delta.iloc[0]
        x1, y1 = ft.base.iloc[0], ft.delta.iloc[0]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=C_GENERAL, lw=1.5, alpha=0.8))
        ax.scatter([x0], [y0], color=C_GENERAL, s=42, zorder=3, edgecolor="white", lw=0.8)   # frozen
        ax.scatter([x1], [y1], color="white", edgecolor=C_GENERAL, s=42, zorder=3, lw=1.6)    # finetuned
        ax.text(x0, y0 + 0.08, LABELS[model], fontsize=6.5, ha="center", color="#555555")
    ax.set_xlabel("Baseline balanced accuracy (%)", fontsize=8.5)
    ax.set_ylabel("Δ TTA balanced accuracy (pp)", fontsize=8.5)
    ax.tick_params(labelsize=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#cccccc"); ax.spines["bottom"].set_color("#cccccc")
    # legend for marker meaning
    import matplotlib.lines as mlines
    ax.legend(handles=[
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor=C_GENERAL,
                      markersize=7, label="frozen"),
        mlines.Line2D([], [], marker="o", color="w", markerfacecolor="white",
                      markeredgecolor=C_GENERAL, markersize=7, label="finetuned")],
        fontsize=7, framealpha=0.0, loc="upper right")
    fig.tight_layout(pad=0.5)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"finetune_arrows.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved → figures/finetune_arrows.png/.pdf")


if __name__ == "__main__":
    plot()
