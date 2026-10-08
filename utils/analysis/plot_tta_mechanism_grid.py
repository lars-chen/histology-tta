"""
Grid of TTA-mechanism heatmaps (TCGA-UT, frozen linear, seed0).

For each model, a 5x5 heatmap of NET correction rate (corrected - corrupted, % of
cell) over baseline entropy (x) × D4 orbit variance (y), using per-model quintile
bins. Shows TTA's beneficial flips concentrate in the high-entropy × high-orbit-
variance corner — and that this "hot corner" shrinks as the backbone gets stronger
(fewer uncertain, view-unstable samples).
"""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EMB_ROOT = Path(os.environ.get("HISTO_EMB_DIR", "embeddings"))
DATASET, SEED = "tcga-ut", 0
MODELS = ["resnet18", "dinov2_s", "convnextv2_base", "ctranspath", "uni", "hoptimus"]
NBIN = 5


def orbit_variance(emb):
    v = emb / (np.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)
    c = v.mean(axis=1, keepdims=True)
    return ((v - c) ** 2).sum(-1).mean(1)


def load(model):
    d = EMB_ROOT / model / DATASET / "test"
    sid = np.load(d / "sample_ids.npy", allow_pickle=True)
    ov = orbit_variance(np.load(d / "embeddings.npy"))
    e = pd.DataFrame({"sample_id": sid, "orbit_var": ov})
    p = pd.read_parquet(f"results/raw/probe_persample_{DATASET}_{model}_linear_seed{SEED}.parquet")
    df = p.merge(e, on="sample_id", how="inner")
    df["net"] = ((~df.correct_0 & df.correct_d4).astype(int)
                 - (df.correct_0 & ~df.correct_d4).astype(int))
    df["acc"] = df.correct_0.mean()
    df["ov_q"]  = pd.qcut(df.orbit_var, NBIN, labels=False, duplicates="drop")
    df["ent_q"] = pd.qcut(df.entropy_0, NBIN, labels=False, duplicates="drop")
    grid = df.pivot_table("net", "ov_q", "ent_q", aggfunc="mean") * 100
    return grid.reindex(index=range(NBIN), columns=range(NBIN)), df.correct_0.mean(), df.net.mean() * 100


def main(out_dir=Path("figures")):
    results = [(m, *load(m)) for m in MODELS]
    results.sort(key=lambda r: r[2])   # by baseline accuracy
    vmax = max(np.nanmax(np.abs(r[1].values)) for r in results)
    vmax = float(np.ceil(vmax))

    ncol = 3
    nrow = (len(results) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 3.2 * nrow))
    fig.subplots_adjust(hspace=0.72, wspace=0.35)
    axes = np.atleast_1d(axes).flatten()
    im = None
    for ax, (m, grid, acc, net) in zip(axes, results):
        im = ax.imshow(grid.values, origin="lower", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{m}\nacc={acc*100:.0f}%  net={net:+.2f}pp", fontsize=8, pad=6)
        ax.set_xticks([0, NBIN - 1]); ax.set_xticklabels(["low", "high"], fontsize=7)
        ax.set_yticks([0, NBIN - 1]); ax.set_yticklabels(["low", "high"], fontsize=7)
        ax.set_xlabel("baseline entropy →", fontsize=7.5)
        ax.set_ylabel("orbit variance →", fontsize=7.5)
    for ax in axes[len(results):]:
        ax.set_visible(False)
    fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.02,
                 label="net correction rate (%)")
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"tta_mechanism_grid.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_dir}/tta_mechanism_grid.png/.pdf")
    for m, _, acc, net in results:
        print(f"  {m:16s} acc={acc*100:5.1f}%  net={net:+.2f}pp")


if __name__ == "__main__":
    main()
