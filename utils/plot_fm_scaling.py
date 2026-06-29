#!/usr/bin/env python3
"""2×3 scatter: FM metadata vs TTA delta, coloured per dataset."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.plot_analysis import FM_META, PUB_RCPARAMS, SAVEKW, CB_PALETTE, DS_COLORS

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

DS_LABELS = {
    "tcga-ut":       "TCGA-UT",
    "nct-crc-100k":  "NCT-CRC-100K",
    "nct-crc-nonorm":"NCT-CRC-NoNorm",
    "mhist":         "MHIST",
}


def _compute_per_dataset(df, models, backbone_mode):
    """Return per-(model, dataset) mean delta (d4 - none)."""
    sub = df[df["model"].isin(models) & (df["backbone_mode"] == backbone_mode)]
    base = sub[sub["strategy"] == "none"][["model", "dataset", "seed", "balanced_acc"]].rename(columns={"balanced_acc": "base_acc"})
    best = sub[sub["strategy"] == "d4" ][["model", "dataset", "seed", "balanced_acc"]].rename(columns={"balanced_acc": "d4_acc"})
    merged = base.merge(best, on=["model", "dataset", "seed"])
    merged["delta"] = (merged["d4_acc"] - merged["base_acc"]) * 100
    return merged.groupby(["model", "dataset"])["delta"].mean().reset_index(name="delta")


def _draw_row(axes, agg, meta, panels, annotate=True):
    """Draw per-dataset scatter in each panel; regression line per dataset."""
    datasets = agg["dataset"].unique()

    for ax, (xcol, xlabel, logx) in zip(axes, panels):
        for ds in datasets:
            sub = agg[agg["dataset"] == ds].copy()
            x = np.array([meta[m][xcol] for m in sub["model"]])
            y = sub["delta"].values
            color = DS_COLORS.get(ds, "#888888")

            ax.scatter(x, y, color=color, s=50, zorder=3, alpha=0.85, label=DS_LABELS.get(ds, ds))

            if annotate:
                for xi, yi, m in zip(x, y, sub["model"]):
                    ax.annotate(MODEL_LABELS.get(m, m), (xi, yi),
                                textcoords="offset points", xytext=(3, 3),
                                fontsize=6.5, color="#444444")

            if len(x) >= 3:
                xp = np.log10(x) if logx else x
                coef = np.polyfit(xp, y, 1)
                xs = np.linspace(xp.min(), xp.max(), 200)
                ys = np.polyval(coef, xs)
                ax.plot(10**xs if logx else xs, ys,
                        color=color, linewidth=1.0, linestyle="--", alpha=0.6, zorder=2)

        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel, fontsize=10)
        ax.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle=":")

        if xcol == "year_frac":
            all_x = [meta[m][xcol] for m in agg["model"].unique()]
            years = sorted(set(int(v) for v in all_x))
            ax.set_xticks(years)
            ax.set_xticklabels(years)

    axes[0].set_ylabel("TTA Δ Bal. Acc. (pp)", fontsize=10)


def main():
    df = pd.read_csv("results/canonical_results.csv")

    panels = [
        ("year_frac",        "Release date",          False),
        ("params_m",         "Parameters (M)",        True),
        ("pretrain_tiles_m", "Pretraining tiles (M)", True),
    ]

    DATASETS_KEEP = {"tcga-ut", "nct-crc-100k"}

    # Histology FMs — frozen backbone
    agg_fm = _compute_per_dataset(df, list(FM_META.keys()), "frozen")
    agg_fm = agg_fm[agg_fm["dataset"].isin(DATASETS_KEEP)]

    # General models — frozen where available (DINOv2), finetuned otherwise
    agg_gen_frz = _compute_per_dataset(df, list(GEN_META.keys()), "frozen")
    agg_gen_ft  = _compute_per_dataset(df, list(GEN_META.keys()), "finetuned")
    frozen_models = set(agg_gen_frz["model"])
    agg_gen = pd.concat([
        agg_gen_frz,
        agg_gen_ft[~agg_gen_ft["model"].isin(frozen_models)],
    ], ignore_index=True)
    agg_gen = agg_gen[agg_gen["dataset"].isin(DATASETS_KEEP)]

    with plt.rc_context(PUB_RCPARAMS):
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))

        _draw_row(axes[0], agg_fm,  FM_META,  panels, annotate=True)
        _draw_row(axes[1], agg_gen, GEN_META, panels, annotate=True)

        for ax, title in zip([axes[0][0], axes[1][0]],
                              ["Histology FMs (frozen)", "General models"]):
            ax.set_title(title, fontsize=11, fontweight="bold", loc="left")

        # Shared legend (datasets) on bottom
        handles = [
            mlines.Line2D([], [], marker="o", color=DS_COLORS[ds], linestyle="None",
                          markersize=7, label=DS_LABELS[ds])
            for ds in ["tcga-ut", "nct-crc-100k"]
        ]
        fig.legend(handles=handles, loc="lower center", ncol=4,
                   fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.01))

        fig.tight_layout(rect=[0, 0.06, 1, 1])
        out = Path("figures/plot_fm_scaling.png")
        fig.savefig(out, **SAVEKW)
        fig.savefig(out.with_suffix(".pdf"), **SAVEKW)
        plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
