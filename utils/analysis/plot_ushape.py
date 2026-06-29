"""Plot TTA delta vs baseline accuracy for all models (2×2 grid, one panel per dataset).

Each point = one model × backbone_mode combination, averaged across seeds.
Color = model family. Marker = frozen (○) vs finetuned (□).
Quadratic fit + Pearson r in subtitle.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path("results/tta_results.csv")
OUT = Path("figures/plot_ushape_all.png")

DATASET_ORDER = ["tcga-ut", "nct-crc-100k", "nct-crc-nonorm", "mhist"]
DATASET_NCLASSES = {"tcga-ut": 31, "nct-crc-100k": 9, "nct-crc-nonorm": 9, "mhist": 2}
DATASET_LABELS = {
    "tcga-ut": "TCGA-UT",
    "nct-crc-100k": "NCT-CRC-100k",
    "nct-crc-nonorm": "NCT-CRC-NoNorm",
    "mhist": "MHIST",
}

CB_PALETTE = {
    "histology":   "#0072B2",
    "general":     "#D4772A",
    "equivariant": "#009E73",
}

MODEL_TYPE_MAP = {
    "gigapath": "histology", "hoptimus": "histology",
    "phikon": "histology", "phikon2": "histology",
    "uni": "histology", "uni2": "histology",
    "virchow": "histology", "virchow2": "histology",
    "ctranspath": "histology",
    "resnet18": "general", "resnet50": "general",
    "convnextv2_tiny": "general", "convnextv2_base": "general",
    "dinov2_s": "general", "dinov2_b": "general", "dinov2_base": "general",
    "d4wrn": "equivariant",
}

MODEL_LABELS = {
    "gigapath": "GigaPath", "hoptimus": "H-opt", "phikon": "Phikon",
    "phikon2": "Phikon2", "uni": "UNI", "uni2": "UNI2",
    "virchow": "Virchow", "virchow2": "Virchow2", "ctranspath": "CTransPath",
    "resnet18": "R18", "resnet50": "R50",
    "convnextv2_tiny": "CNX-T", "convnextv2_base": "CNX-B",
    "dinov2_s": "DV2-S", "dinov2_b": "DV2-B", "dinov2_base": "DV2-B",
    "d4wrn": "D4-WRN",
}


def load_data():
    df = pd.read_csv(RESULTS)
    if "model_type" not in df.columns:
        df["model_type"] = df["model"].map(MODEL_TYPE_MAP).fillna("general")
    return df


def compute_points(df):
    """For each (model, dataset, backbone_mode): mean baseline bacc and mean delta."""
    base = df[(df["strategy"] == "none") & (df["aggregation"] == "mean")].copy()
    tta  = df[(df["strategy"] == "d4")   & (df["aggregation"] == "mean")].copy()

    merge_on = ["model", "dataset", "backbone_mode", "train_augment", "seed"]
    merged = base[merge_on + ["balanced_acc", "model_type"]].merge(
        tta[merge_on + ["balanced_acc"]],
        on=merge_on, suffixes=("_base", "_tta"),
    )
    merged["delta"] = (merged["balanced_acc_tta"] - merged["balanced_acc_base"]) * 100
    merged["base_pct"] = merged["balanced_acc_base"] * 100

    agg = merged.groupby(["model", "dataset", "backbone_mode", "model_type"], observed=True).agg(
        base_mean=("base_pct", "mean"),
        delta_mean=("delta", "mean"),
        delta_sem=("delta", lambda x: x.sem() if len(x) > 1 else 0.0),
        n_seeds=("delta", "count"),
    ).reset_index()
    return agg


def plot_ushape(agg, out_path):
    datasets = [d for d in DATASET_ORDER if d in agg["dataset"].unique()]
    nds = len(datasets)
    ncols = 2
    nrows = (nds + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 4.5))
    axes = axes.flatten()

    for ax_idx, ds in enumerate(datasets):
        ax = axes[ax_idx]
        sub = agg[agg["dataset"] == ds].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        ncls = DATASET_NCLASSES.get(ds, "?")
        ax.axhline(0, color="#888888", lw=1.0, ls="--", alpha=0.6, zorder=1)

        xs = sub["base_mean"].values
        ys = sub["delta_mean"].values

        # Quadratic fit
        if len(xs) >= 3:
            coeffs = np.polyfit(xs, ys, 2)
            x_line = np.linspace(xs.min(), xs.max(), 200)
            y_line = np.polyval(coeffs, x_line)
            ax.plot(x_line, y_line, color="black", lw=1.5, zorder=2, alpha=0.7)

        # Pearson r
        r, _ = stats.pearsonr(xs, ys)

        # Scatter — frozen first (lower zorder), finetuned on top
        for mode, marker, msize, edgecolor, edgewidth, zorder in [
            ("frozen",    "o", 7,  "white", 0.6, 4),
            ("finetuned", "D", 9,  "black", 1.2, 5),
        ]:
            pts = sub[sub["backbone_mode"] == mode]
            for _, row in pts.iterrows():
                color = CB_PALETTE.get(row["model_type"], "gray")
                ax.errorbar(
                    row["base_mean"], row["delta_mean"],
                    yerr=row["delta_sem"],
                    fmt=marker, markersize=msize, alpha=0.92,
                    color=color, markerfacecolor=color,
                    markeredgecolor=edgecolor, markeredgewidth=edgewidth,
                    capsize=2, elinewidth=0.8, zorder=zorder,
                )
                lbl = MODEL_LABELS.get(row["model"], row["model"])
                if mode == "finetuned":
                    lbl += " ft"
                ax.annotate(
                    lbl,
                    xy=(row["base_mean"], row["delta_mean"]),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=6.5, color=color,
                    fontweight="bold" if mode == "finetuned" else "normal",
                    zorder=6,
                )

        ax.set_title(
            f"{DATASET_LABELS.get(ds, ds)} ({ncls} cls)  (r={r:.2f})",
            fontsize=10,
        )
        ax.set_xlabel("Baseline balanced accuracy (%)", fontsize=9)
        ax.set_ylabel("Δ balanced acc (pp)", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.15)

    # Hide unused panels
    for ax_idx in range(len(datasets), len(axes)):
        axes[ax_idx].set_visible(False)

    # Legend
    type_handles = [
        mpatches.Patch(facecolor=CB_PALETTE["histology"], label="histology"),
        mpatches.Patch(facecolor=CB_PALETTE["general"], label="general"),
        mpatches.Patch(facecolor=CB_PALETTE["equivariant"], label="equivariant"),
    ]
    proto_handles = [
        mlines.Line2D([], [], marker="o", color="gray", markerfacecolor="gray",
                      markersize=7, linestyle="None", label="frozen"),
        mlines.Line2D([], [], marker="D", color="gray", markerfacecolor="gray",
                      markeredgecolor="black", markeredgewidth=1.2,
                      markersize=9, linestyle="None", label="finetuned"),
    ]
    fig.legend(
        handles=type_handles + proto_handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=5, fontsize=9, frameon=True, framealpha=0.95,
    )
    fig.suptitle("TTA delta vs baseline — all datasets (per model, mean across seeds)", fontsize=11, y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    df = load_data()
    agg = compute_points(df)
    print(f"Points: {len(agg)} ({agg['model'].nunique()} models, {agg['dataset'].nunique()} datasets)")
    plot_ushape(agg, OUT)


if __name__ == "__main__":
    main()
