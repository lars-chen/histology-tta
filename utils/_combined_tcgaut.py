#!/usr/bin/env python3
"""1×3 scatter: both model families vs TTA delta, TCGA-UT only, coloured by family."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plot_analysis import FM_META, PUB_RCPARAMS, SAVEKW, CB_PALETTE

MODEL_LABELS = {
    "ctranspath":      "CTransPath",
    "phikon":          "Phikon",
    "phikon2":         "Phikon-2",
    "uni":             "UNI",
    "uni2":            "UNI-v2",
    "virchow":         "Virchow",
    "virchow2":        "Virchow2",
    "gigapath":        "GigaPath",
    "hoptimus":        "H-Opt-1",
    "resnet18":        "ResNet-18",
    "resnet50":        "ResNet-50",
    "dinov2_s":        "DINOv2-S",
    "dinov2_b":        "DINOv2-B",
    "convnextv2_tiny": "CNXv2-T",
    "convnextv2_base": "CNXv2-B",
}

GEN_META = {
    "resnet18":        {"params_m": 11,  "pretrain_tiles_m": 1.28, "year_frac": 2015 + 11/12},
    "resnet50":        {"params_m": 25,  "pretrain_tiles_m": 1.28, "year_frac": 2015 + 11/12},
    "dinov2_s":        {"params_m": 22,  "pretrain_tiles_m": 142,  "year_frac": 2023 +  3/12},
    "dinov2_b":        {"params_m": 86,  "pretrain_tiles_m": 142,  "year_frac": 2023 +  3/12},
    "convnextv2_tiny": {"params_m": 28,  "pretrain_tiles_m": 1.28, "year_frac": 2023 +  0/12},
    "convnextv2_base": {"params_m": 89,  "pretrain_tiles_m": 1.28, "year_frac": 2023 +  0/12},
}

PANELS = [
    ("year_frac",        "Release date",          False),
    ("params_m",         "Parameters (M)",        True),
    ("pretrain_tiles_m", "Pretraining tiles (M)", True),
]


def compute_delta(df, models, backbone_mode):
    sub = df[df["model"].isin(models) & (df["backbone_mode"] == backbone_mode) & (df["dataset"] == "tcga-ut")]
    base = sub[sub["strategy"] == "none"][["model", "seed", "balanced_acc"]].rename(columns={"balanced_acc": "base_acc"})
    best = sub[sub["strategy"] == "d4" ][["model", "seed", "balanced_acc"]].rename(columns={"balanced_acc": "d4_acc"})
    merged = base.merge(best, on=["model", "seed"])
    merged["delta"] = (merged["d4_acc"] - merged["base_acc"]) * 100
    return merged.groupby("model")["delta"].mean().reset_index(name="delta")


def draw(ax, agg, meta, xcol, logx, family_color, family_label):
    x = np.array([meta[m][xcol] for m in agg["model"]])
    y = agg["delta"].values
    ax.scatter(x, y, color=family_color, s=60, zorder=3, alpha=0.88, label=family_label)
    for xi, yi, m in zip(x, y, agg["model"]):
        ax.annotate(MODEL_LABELS.get(m, m), (xi, yi),
                    textcoords="offset points", xytext=(3, 3),
                    fontsize=7, color="#444444")
    if len(x) >= 3:
        xp = np.log10(x) if logx else x
        coef = np.polyfit(xp, y, 1)
        xs = np.linspace(xp.min(), xp.max(), 200)
        ys = np.polyval(coef, xs)
        ax.plot(10**xs if logx else xs, ys,
                color=family_color, linewidth=1.1, linestyle="--", alpha=0.6, zorder=2)


def main():
    df = pd.read_csv("results/canonical_results.csv")

    agg_fm  = compute_delta(df, list(FM_META.keys()),  "frozen")

    agg_gen_frz = compute_delta(df, list(GEN_META.keys()), "frozen")
    agg_gen_ft  = compute_delta(df, list(GEN_META.keys()), "finetuned")
    frozen_models = set(agg_gen_frz["model"])
    agg_gen = pd.concat([agg_gen_frz, agg_gen_ft[~agg_gen_ft["model"].isin(frozen_models)]], ignore_index=True)

    with plt.rc_context(PUB_RCPARAMS):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        fig.suptitle("TTA Δ Balanced Acc — TCGA-UT (d4 vs. no TTA)", fontsize=12, fontweight="bold", y=1.01)

        for ax, (xcol, xlabel, logx) in zip(axes, PANELS):
            draw(ax, agg_fm,  FM_META,  xcol, logx, CB_PALETTE["histology"], "Histology FM (frozen)")
            draw(ax, agg_gen, GEN_META, xcol, logx, CB_PALETTE["general"],   "General model")
            if logx:
                ax.set_xscale("log")
            ax.set_xlabel(xlabel, fontsize=10)
            ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle=":")
            if xcol == "year_frac":
                all_x = [FM_META[m][xcol] for m in agg_fm["model"]] + \
                        [GEN_META[m][xcol] for m in agg_gen["model"]]
                years = sorted(set(int(v) for v in all_x))
                ax.set_xticks(years)
                ax.set_xticklabels(years)

        axes[0].set_ylabel("TTA Δ Bal. Acc. (pp)", fontsize=10)

        handles = [
            mlines.Line2D([], [], marker="o", color=CB_PALETTE["histology"], linestyle="None", markersize=7, label="Histology FM (frozen)"),
            mlines.Line2D([], [], marker="o", color=CB_PALETTE["general"],   linestyle="None", markersize=7, label="General model"),
        ]
        fig.legend(handles=handles, loc="lower center", ncol=2,
                   fontsize=10, frameon=False, bbox_to_anchor=(0.5, -0.06))

        fig.tight_layout()
        out = Path("figures/plot_fm_scaling_tcgaut.png")
        fig.savefig(out, **SAVEKW)
        plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
