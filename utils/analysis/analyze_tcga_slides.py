"""Slide-level aggregation for TCGA-UT per-sample TTA results."""
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CSV = "results/per_sample/per_sample_tcga-ut_phikon_seed42.csv"
OUT_CSV = "results/tcga_slide_stats.csv"
OUT_FIG = "figures/tcga_slide_stats.png"

df = pd.read_csv(CSV)
df["slide_id"] = df["image_id"].str.split("/").str[0]

def agg(g):
    n = len(g)
    return pd.Series({
        "n_patches":        n,
        "dominant_class":   g["true_label"].mode().iloc[0],
        "baseline_acc":     g["baseline_correct"].sum() / n,
        "tta_acc":          g["tta_correct"].sum() / n,
        "corrected":        (g["outcome"] == "corrected").sum(),
        "corrupted":        (g["outcome"] == "corrupted").sum(),
        "both_correct":     (g["outcome"] == "both_correct").sum(),
        "both_wrong":       (g["outcome"] == "both_wrong").sum(),
    })

slides = df.groupby("slide_id").apply(agg, include_groups=False).reset_index()
slides["delta_acc"]        = slides["tta_acc"] - slides["baseline_acc"]
slides["corrected_frac"]   = slides["corrected"] / slides["n_patches"]
slides["corrupted_frac"]   = slides["corrupted"] / slides["n_patches"]
slides["net_frac"]         = slides["corrected_frac"] - slides["corrupted_frac"]

slides.to_csv(OUT_CSV, index=False)
print(f"Saved {OUT_CSV}  ({len(slides)} slides)")

# Summary stats
print(f"\nBaseline acc:   {slides.baseline_acc.mean():.3f}")
print(f"TTA acc:        {slides.tta_acc.mean():.3f}")
print(f"Mean Δacc:      {slides.delta_acc.mean()*100:+.2f} pp")
print(f"Slides improved: {(slides.delta_acc > 0).sum()} / {len(slides)}")
print(f"Slides hurt:     {(slides.delta_acc < 0).sum()} / {len(slides)}")
print(f"\nMean corrected frac: {slides.corrected_frac.mean()*100:.1f}%")
print(f"Mean corrupted frac: {slides.corrupted_frac.mean()*100:.1f}%")
print(f"Mean net frac:       {slides.net_frac.mean()*100:+.1f}%")

# Per-class delta
per_class = slides.groupby("dominant_class")["delta_acc"].mean().sort_values() * 100
print(f"\nPer-class Δacc (pp):\n{per_class.round(2).to_string()}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

# 1. Scatter: baseline vs TTA acc per slide
ax = axes[0]
ax.scatter(slides.baseline_acc, slides.tta_acc, s=6, alpha=0.3, color="#2a5fa5")
lim = [slides[["baseline_acc","tta_acc"]].min().min() - 0.02,
       slides[["baseline_acc","tta_acc"]].max().max() + 0.02]
ax.plot(lim, lim, "k--", lw=0.8)
ax.set_xlabel("Baseline patch acc")
ax.set_ylabel("TTA patch acc")
ax.set_title("Per-slide accuracy")
ax.set_xlim(lim); ax.set_ylim(lim)

# 2. Distribution of Δacc
ax = axes[1]
ax.hist(slides.delta_acc * 100, bins=40, color="#2a5fa5", alpha=0.8, edgecolor="none")
ax.axvline(0, color="k", lw=0.8, ls="--")
ax.axvline(slides.delta_acc.mean() * 100, color="#e07b39", lw=1.5, ls="-",
           label=f"mean {slides.delta_acc.mean()*100:+.2f} pp")
ax.set_xlabel("Δ patch acc (pp)")
ax.set_ylabel("# slides")
ax.set_title("TTA gain distribution")
ax.legend(fontsize=7)

# 3. Per-class Δacc
ax = axes[2]
colors = ["#d94f3d" if v < 0 else "#2a5fa5" for v in per_class]
ax.barh(range(len(per_class)), per_class.values, color=colors)
ax.set_yticks(range(len(per_class)))
ax.set_yticklabels(per_class.index, fontsize=6)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Δ patch acc (pp)")
ax.set_title("By tissue class")

fig.tight_layout(pad=0.5)
Path(OUT_FIG).parent.mkdir(exist_ok=True)
fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {OUT_FIG}")
