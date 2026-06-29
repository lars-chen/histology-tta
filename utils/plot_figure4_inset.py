"""
Compact single-panel inset version of figure4_selective_pareto for graphical abstract.
Shows family-average Pareto curves (Δ balanced accuracy vs % samples receiving D4 TTA)
for histology FMs and general-purpose models on TCGA-UT.

Output: figures/figure4_inset.png/.pdf
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

from utils.plot_analysis import MODEL_TYPE_MAP, CB_PALETTE, SAVEKW

LOGITS_DIR = Path(__file__).parent.parent / "logits"
THRESHOLDS  = np.linspace(0, 1, 50)
ALL_DATASETS = ["tcga-ut", "nct-crc-100k", "nct-crc-nonorm", "mhist"]

C_HISTO   = CB_PALETTE["histology"]   # "#0072B2"
C_GENERAL = CB_PALETTE["general"]     # "#D4772A"


def _load_curves(dataset: str) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Return {model_type: [(frac_array, delta_array), ...]} for dataset."""
    curves: dict[str, list] = {"histology": [], "general": []}

    for pt_file in sorted(LOGITS_DIR.rglob("*.pt")):
        try:
            ck = torch.load(pt_file, map_location="cpu", weights_only=False)
        except Exception:
            continue

        if ck.get("backbone_mode") != "frozen":
            continue
        if ck.get("aug_tag", "aug") != "aug":
            continue
        if ck.get("dataset", "") != dataset:
            continue

        model = ck.get("model", "")
        mtype = MODEL_TYPE_MAP.get(model)
        if mtype not in curves:
            continue

        logits = ck["logits"].float()   # (n_views, N, C)
        labels = ck["labels"]
        _, N, C = logits.shape

        base_probs  = F.softmax(logits[0], dim=-1)
        base_ent    = (-(base_probs * (base_probs + 1e-12).log()).sum(-1) / np.log(C)).numpy()
        base_preds  = base_probs.argmax(-1).numpy()
        d4_preds    = F.softmax(logits, dim=-1).mean(0).argmax(-1).numpy()
        labels_np   = labels.numpy()
        baseline_ba = balanced_accuracy_score(labels_np, base_preds) * 100

        fracs  = np.empty(len(THRESHOLDS))
        deltas = np.empty(len(THRESHOLDS))
        for i, t in enumerate(THRESHOLDS):
            mask       = base_ent > t
            sel_preds  = np.where(mask, d4_preds, base_preds)
            fracs[i]   = mask.mean() * 100
            deltas[i]  = balanced_accuracy_score(labels_np, sel_preds) * 100 - baseline_ba

        curves[mtype].append((fracs, deltas))

    return curves


def _family_avg(runs: list[tuple[np.ndarray, np.ndarray]]):
    all_f = np.array([r[0] for r in runs])
    all_d = np.array([r[1] for r in runs])
    return all_f.mean(0), all_d.mean(0), all_d.std(0)


def plot_one(dataset: str, out_dir: Path):
    curves = _load_curves(dataset)

    if not curves["histology"] and not curves["general"]:
        print(f"  WARNING: no usable logit files for {dataset}, skipping")
        return

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for mtype, color in [("histology", C_HISTO), ("general", C_GENERAL)]:
        runs = curves[mtype]
        if not runs:
            continue
        f, d, s = _family_avg(runs)
        order = np.argsort(f)
        f, d, s = f[order], d[order], s[order]

        ax.fill_between(f, d - s, d + s, color=color, alpha=0.15, lw=0, zorder=2)
        ax.plot(f, d, color=color, lw=2.4, zorder=3)
        ax.plot(f[::5], d[::5], "o", color=color, ms=4, zorder=5)

    ax.axhline(0, color="#888888", lw=0.9, ls="--", zorder=1)

    ax.set_xlim(0, 102)
    ax.set_ylim(0, 1.6)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.4))
    ax.set_xlabel("Samples receiving D₄ TTA (%)", fontsize=8)
    ax.set_ylabel("Δ Balanced Accuracy (pp)", fontsize=8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.tick_params(axis="both", labelsize=7.5)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.grid(False)

    fig.tight_layout(pad=0.5)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"figure4_inset_{dataset}"
    for ext in ("png", "pdf"):
        fig.savefig(str(stem) + f".{ext}", **SAVEKW)
    plt.close(fig)
    print(f"Saved → {stem}.png/.pdf")


def plot(out_dir: Path = Path("figures"), dataset: str = None):
    datasets = [dataset] if dataset else ALL_DATASETS
    for ds in datasets:
        plot_one(ds, out_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="figures")
    p.add_argument("--dataset", default=None, choices=ALL_DATASETS + [None])
    args = p.parse_args()
    plot(Path(args.out_dir), args.dataset)
