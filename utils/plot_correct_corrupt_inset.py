"""
Decompose the TTA decline vs baseline accuracy. One point per (model, dataset)
[averaged over seeds], frozen linear probe, classification datasets (no mhist).

Emits two figures sharing the same points:
  correct_corrupt_inset — per-sample correction & corruption rates vs baseline
  ratio_inset           — corrected:corrupted ratio vs baseline (break-even = 1)

Net Δaccuracy = correction_rate - corruption_rate; the ratio shows how close TTA
gets to break-even (a coin flip) as the backbone gets stronger.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

from utils.plot_graphabs_inset import MODEL_TYPE as _MT, C_HISTO, C_GENERAL

# Extend with the general-purpose ResNets (omitted from the abstract-inset map)
MODEL_TYPE = {**_MT, "resnet18": "general", "resnet50": "general"}

C_CORR    = "#2a9d3f"   # green — corrections
C_CORRUPT = "#d1495b"   # red   — corruptions


# class counts for chance-corrected skill = (bacc - 1/C)/(1 - 1/C)
N_CLASSES = {"tcga-ut": 31, "nct-crc-100k": 9, "nct-crc-nonorm": 9, "mhist": 2}


def build_data(csv="results/canonical_results.csv", head_type="linear",
               include_mhist=False, min_skill=0.25):
    nt = pd.read_csv("results/tta_results.csv").groupby("dataset").n_test.max().to_dict()
    df = pd.read_csv(csv)
    d = df[(df.backbone_mode == "frozen") & (df.head_type == head_type) &
           (df.aggregation == "mean")]
    if not include_mhist:
        d = d[d.dataset != "mhist"]
    base = d[d.strategy == "none"].groupby(["model", "dataset", "seed"]).agg(
        acc_base=("acc", "mean"), bacc_base=("balanced_acc", "mean"))
    tta  = d[d.strategy == "d4"].groupby(["model", "dataset", "seed"]).agg(
        nc=("n_corrected", "mean"), nk=("n_corrupted", "mean"))
    m = base.reset_index().merge(tta.reset_index(), on=["model", "dataset", "seed"])
    m["N"] = m.dataset.map(nt)
    m["base"]         = m.acc_base * 100
    m["corr_rate"]    = m.nc / m.N * 100
    m["corrupt_rate"] = m.nk / m.N * 100
    m["ratio"]        = m.nc / m.nk.replace(0, np.nan)
    m["net_rate"]     = m.corr_rate - m.corrupt_rate
    # aggregate over seeds → one point per (model, dataset), with SEM across seeds
    sem = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    g = m.groupby(["model", "dataset"]).agg(
        base=("base", "mean"), base_sem=("base", sem), bacc=("bacc_base", "mean"),
        corrected=("corr_rate", "mean"), corrupted=("corrupt_rate", "mean"),
        ratio=("ratio", "mean"), ratio_sem=("ratio", sem),
        net=("net_rate", "mean"), net_sem=("net_rate", sem)).reset_index()
    g = g[g.model.isin(MODEL_TYPE)].copy()
    g["fam"] = g.model.map(MODEL_TYPE)
    # chance-corrected competence filter: drop near-chance (model, dataset) cells
    C = g.dataset.map(N_CLASSES)
    g["skill"] = (g.bacc - 1 / C) / (1 - 1 / C)
    dropped = g[g.skill < min_skill]
    if len(dropped):
        print(f"[skill<{min_skill}] excluded: "
              + ", ".join(f"{r.model}/{r.dataset}(skill={r.skill:.2f})" for _, r in dropped.iterrows()))
    g = g[g.skill >= min_skill].copy()
    return g


def _style(ax):
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#cccccc"); ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(labelsize=7.5)


def plot(out_dir: Path = Path("figures")):
    g = build_data()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    xb = g["base"].values

    # ── Figure 1: correction & corruption rates ────────────────────────────────
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.axhline(0, color="#999999", lw=0.8, ls="--", zorder=1)
    for col, c, lbl in [("corrected", C_CORR, "corrected"),
                        ("corrupted", C_CORRUPT, "corrupted")]:
        ys = g[col].values
        ax.scatter(xb, ys, color=c, s=22, zorder=3, alpha=0.75, linewidths=0)
        b1, b0 = np.polyfit(xb, ys, 1)
        xl = np.linspace(xb.min(), xb.max(), 100)
        ax.plot(xl, b1 * xl + b0, color=c, lw=1.5, alpha=0.9, zorder=2)
        r = stats.pearsonr(xb, ys)[0]
        ax.text(0.02, 0.97 if lbl == "corrected" else 0.89,
                f"{lbl}: r={r:+.2f}", transform=ax.transAxes, fontsize=7,
                color=c, va="top", fontweight="bold")
    ax.set_xlabel("Baseline accuracy (%)", fontsize=8)
    ax.set_ylabel("% of test patches", fontsize=8)
    _style(ax)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"correct_corrupt_inset.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ── Figure 2: corrected:corrupted ratio ────────────────────────────────────
    gr = g.dropna(subset=["ratio"])
    xr = gr["base"].values; yr = gr["ratio"].values
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.axhline(1.0, color="#999999", lw=1.0, ls="--", zorder=1)  # break-even
    ax.text(xr.min(), 1.02, "break-even", fontsize=6.5, color="#888888", va="bottom")
    for t, c in [("general", C_GENERAL), ("histology", C_HISTO)]:
        s = gr[gr.fam == t]
        ax.scatter(s["base"], s["ratio"], color=c, s=22, zorder=3, alpha=0.8, linewidths=0)
    b1, b0 = np.polyfit(xr, yr, 1)
    xl = np.linspace(xr.min(), xr.max(), 100)
    ax.plot(xl, b1 * xl + b0, color="#444444", lw=1.4, alpha=0.85, zorder=2)
    r = stats.pearsonr(xr, yr)[0]
    ax.text(0.97, 0.95, f"r = {r:+.2f}", transform=ax.transAxes, ha="right",
            va="top", fontsize=8, color="#333333")
    ax.set_xlabel("Baseline accuracy (%)", fontsize=8)
    ax.set_ylabel("corrected : corrupted", fontsize=8)
    _style(ax)
    ax.legend(handles=[mpatches.Patch(color=C_GENERAL, label="General-purpose"),
                       mpatches.Patch(color=C_HISTO,   label="Histology FMs")],
              fontsize=6.5, framealpha=0.0, loc="upper right", handlelength=1.0)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"ratio_inset.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved → {out_dir}/correct_corrupt_inset.* and ratio_inset.*  ({len(g)} model×dataset points)")
    print(g.sort_values("base")[["model", "dataset", "base", "corrected", "corrupted", "ratio"]].round(2).to_string(index=False))


def plot_per_dataset(out_dir: Path = Path("figures")):
    """Facet by dataset so the x-axis (baseline) varies only by model quality."""
    g = build_data(include_mhist=True)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    order = [d for d in ["tcga-ut", "nct-crc-100k", "nct-crc-nonorm", "mhist"] if d in g.dataset.unique()]

    # ── correction & corruption rates, per dataset ─────────────────────────────
    fig, axes = plt.subplots(1, len(order), figsize=(3.0 * len(order), 2.7), squeeze=False)
    for ax, ds in zip(axes[0], order):
        s = g[g.dataset == ds]; xb = s["base"].values
        ax.axhline(0, color="#999999", lw=0.7, ls="--", zorder=1)
        for col, c in [("corrected", C_CORR), ("corrupted", C_CORRUPT)]:
            ys = s[col].values
            ax.scatter(xb, ys, color=c, s=24, alpha=0.8, linewidths=0, zorder=3)
            if len(xb) >= 3:
                b1, b0 = np.polyfit(xb, ys, 1)
                xl = np.linspace(xb.min(), xb.max(), 50)
                ax.plot(xl, b1 * xl + b0, color=c, lw=1.4, alpha=0.9, zorder=2)
        ax.set_title(ds, fontsize=8)
        ax.set_xlabel("Baseline acc (%)", fontsize=8)
        _style(ax)
    axes[0][0].set_ylabel("% of test patches", fontsize=8)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"correct_corrupt_by_dataset.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ── corrected:corrupted ratio, per dataset ─────────────────────────────────
    fig, axes = plt.subplots(1, len(order), figsize=(3.0 * len(order), 2.7), squeeze=False)
    for ax, ds in zip(axes[0], order):
        s = g[g.dataset == ds].dropna(subset=["ratio"]); xb = s["base"].values
        ax.axhline(1.0, color="#999999", lw=0.9, ls="--", zorder=1)
        for t, c in [("general", C_GENERAL), ("histology", C_HISTO)]:
            ss = s[s.fam == t]
            ax.errorbar(ss["base"], ss["ratio"], yerr=ss["ratio_sem"], xerr=ss["base_sem"],
                        fmt="o", ms=4.5, color=c, alpha=0.85, zorder=3,
                        elinewidth=0.8, capsize=1.5, markeredgewidth=0)
        if len(xb) >= 3:
            b1, b0 = np.polyfit(xb, s["ratio"].values, 1)
            xl = np.linspace(xb.min(), xb.max(), 50)
            ax.plot(xl, b1 * xl + b0, color="#444444", lw=1.4, alpha=0.85, zorder=2)
            r = stats.pearsonr(xb, s["ratio"].values)[0]
            ax.text(0.95, 0.93, f"r={r:+.2f}", transform=ax.transAxes, ha="right",
                    va="top", fontsize=7.5, color="#333333")
        ax.set_title(ds, fontsize=8)
        ax.set_xlabel("Baseline acc (%)", fontsize=8)
        _style(ax)
    axes[0][0].set_ylabel("corrected : corrupted", fontsize=8)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"ratio_by_dataset.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # ── NET correction rate (corrected - corrupted, % of test), per dataset ────
    fig, axes = plt.subplots(1, len(order), figsize=(3.0 * len(order), 2.7), squeeze=False)
    for ax, ds in zip(axes[0], order):
        s = g[g.dataset == ds]; xb = s["base"].values
        ax.axhline(0.0, color="#999999", lw=0.9, ls="--", zorder=1)
        for t, c in [("general", C_GENERAL), ("histology", C_HISTO)]:
            ss = s[s.fam == t]
            ax.errorbar(ss["base"], ss["net"], yerr=ss["net_sem"], xerr=ss["base_sem"],
                        fmt="o", ms=4.5, color=c, alpha=0.85, zorder=3,
                        elinewidth=0.8, capsize=1.5, markeredgewidth=0)
        if len(xb) >= 3:
            b1, b0 = np.polyfit(xb, s["net"].values, 1)
            xl = np.linspace(xb.min(), xb.max(), 50)
            ax.plot(xl, b1 * xl + b0, color="#bbbbbb", lw=1.0, alpha=0.7, zorder=2)
        ax.set_title(ds, fontsize=8)
        ax.set_xlabel("Baseline acc (%)", fontsize=8)
        _style(ax)
    axes[0][0].set_ylabel("net correction rate (%)", fontsize=8)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"net_by_dataset.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved → {out_dir}/correct_corrupt_by_dataset.*, ratio_by_dataset.*, net_by_dataset.*")
    for ds in order:
        s = g[g.dataset == ds]
        rr = stats.pearsonr(s["base"], s["ratio"].fillna(1))[0]
        rn = stats.pearsonr(s["base"], s["net"])[0]
        print(f"  {ds:16s} net-vs-baseline r={rn:+.2f}  mean net={s['net'].mean():.2f}%   (ratio r={rr:+.2f})  n={len(s)}")


if __name__ == "__main__":
    plot()
    plot_per_dataset()
