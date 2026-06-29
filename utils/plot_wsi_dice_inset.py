"""
Small inset plot for the graphical abstract (WSI beat):
Δ Dice (D4 TTA vs baseline) per backbone on Camelyon17 patch tumor segmentation
(leave-one-patient-out probes, tumor slides). Bars colored by model family and
ordered high→low so the panel rhymes with the classification family scatter
(graphabs_inset): the TTA gain declines as the backbone gets stronger.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Same palette as utils/plot_graphabs_inset.py
C_HISTO   = "#2a5fa5"   # blue
C_GENERAL = "#e07b39"   # orange

MODEL_TYPE = {
    "convnextv2": "general", "dinov2-s": "general",
    "uni": "histology", "gigapath": "histology",
    "virchow2": "histology", "phikon2": "histology",
}
MODEL_FULL = {
    "convnextv2": "ConvNeXtV2", "dinov2-s": "DINOv2-S",
    "uni": "UNI", "gigapath": "GigaPath",
    "virchow2": "Virchow2", "phikon2": "Phikon-v2",
}


def build_data(csv="results/patch_dice_camelyon17_lopo.csv", min_pos=20):
    df = pd.read_csv(csv)
    df = df[df.n_pos >= min_pos]
    g = df.groupby("model")["delta_dice"].mean().reset_index()
    g = g[g.model.isin(MODEL_TYPE)]
    g["type"]  = g.model.map(MODEL_TYPE)
    g["label"] = g.model.map(MODEL_FULL)
    g["color"] = g.type.map({"histology": C_HISTO, "general": C_GENERAL})
    return g.sort_values("delta_dice", ascending=False).reset_index(drop=True)


def plot(out_dir: Path = Path("figures"),
         csv: str = "results/patch_dice_camelyon17_lopo.csv"):
    g = build_data(csv)

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(g))
    ax.bar(x, g.delta_dice, color=g.color, width=0.72, zorder=3, linewidth=0)

    ax.axhline(0, color="#999999", lw=0.8, ls="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(g.label, fontsize=7, rotation=35, ha="right")
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Δ Dice", fontsize=8)

    ax.set_ylim(0, max(0.03, g.delta_dice.max() * 1.18))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax.tick_params(axis="y", labelsize=7.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    # family legend
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=C_GENERAL, label="General-purpose"),
                       mpatches.Patch(color=C_HISTO,   label="Histology FMs")],
              fontsize=6.5, framealpha=0.0, loc="upper right", handlelength=1.0)

    fig.tight_layout(pad=0.4)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"wsi_dice_inset.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_dir}/wsi_dice_inset.png/.pdf")
    print(g[["label", "type", "delta_dice"]].to_string(index=False))


if __name__ == "__main__":
    plot()
