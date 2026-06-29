"""
Model × dataset heatmap of per-model TTA net correction rate (%), frozen linear.
Rows ordered by mean benefit; shows whether a model's TTA-responsiveness is
consistent across datasets (transferable) — i.e. rows that stay hot or cold.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.plot_correct_corrupt_inset import build_data, MODEL_TYPE
from utils.plot_graphabs_inset import C_HISTO, C_GENERAL

DSETS = ["tcga-ut", "nct-crc-100k", "nct-crc-nonorm", "mhist"]
DLABEL = {"tcga-ut": "TCGA-UT", "nct-crc-100k": "NCT-100k",
          "nct-crc-nonorm": "NCT-nonorm", "mhist": "MHIST"}
MODEL_FULL = {"resnet18": "ResNet-18", "resnet50": "ResNet-50",
              "convnextv2_tiny": "ConvNeXtV2-T", "convnextv2_base": "ConvNeXtV2-B",
              "dinov2_s": "DINOv2-S", "dinov2_b": "DINOv2-B", "ctranspath": "CTransPath",
              "phikon": "Phikon", "phikon2": "Phikon-v2", "uni": "UNI", "uni2": "UNI2-h",
              "virchow": "Virchow", "virchow2": "Virchow2", "gigapath": "GigaPath",
              "hoptimus": "H-optimus"}


def plot(out_dir=Path("figures")):
    g = build_data(include_mhist=True)
    piv = g.pivot_table("net", "model", "dataset").reindex(columns=DSETS)
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index]   # hot→cold rows

    vmax = float(np.ceil(np.nanmax(np.abs(piv.values)) * 10) / 10)
    fig, ax = plt.subplots(figsize=(4.2, 6.0))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(DSETS))); ax.set_xticklabels([DLABEL[d] for d in DSETS],
                                                         rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([MODEL_FULL.get(m, m) for m in piv.index], fontsize=8)
    # color model labels by family
    for tick, m in zip(ax.get_yticklabels(), piv.index):
        tick.set_color(C_GENERAL if MODEL_TYPE.get(m) == "general" else C_HISTO)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=7, color="#999")
            else:
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > vmax * 0.6 else "#222")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("net correction rate (%)", fontsize=8); cb.ax.tick_params(labelsize=7)
    ax.set_title("TTA net benefit by model × dataset\n(rows: orange=general, blue=histology FM)",
                 fontsize=8.5)
    fig.tight_layout()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"model_dataset_heatmap.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_dir}/model_dataset_heatmap.png/.pdf")


if __name__ == "__main__":
    plot()
