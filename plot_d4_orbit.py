#!/usr/bin/env python3
"""
D4 Orbit Geometry Visualization

Loads pre-extracted D4 embeddings (N, 8, D) from embeddings/ and plots the orbit
of two TCGA-UT patches in a principled 2D projection subspace anchored to the
linear classifier's decision boundary.

Projection construction
-----------------------
Axis 1 (x-axis):  w = (W[i] - W[j]) / ||W[i] - W[j]||
                  where i, j are the two classes orbit_A's views disagree about
                  → decision boundary maps to a vertical line at
                    x_boundary = −(b[i]−b[j]) / ||W[i]−W[j]||

Axis 2 (y-axis):  leading D4-local-variance direction, Gram-Schmidt ⊥ to Axis 1

Patch selection
---------------
orbit_A : sample with the most view-disagreement (highest # unique predicted classes)
orbit_B : fully-agreeing sample (all 8 views same class) furthest from the boundary

Usage
-----
    python plot_d4_orbit.py \\
        [--embeddings embeddings/tcga-ut_dinov2_s_frozen_seed42] \\
        [--checkpoint checkpoints/tcga-ut_dinov2_s_frozen_aug_seed42_best.pt] \\
        [--out_dir figures/]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
import torch


# ---------------------------------------------------------------------------
# D4 metadata (mirrors utils/plot_analysis.py)
# ---------------------------------------------------------------------------

COLOR_ORIG = "#E24A33"   # red — original (0°) view
COLOR_AUG  = "#5B8DB8"   # steel blue — all 7 D4-augmented views

SAVEKW = dict(dpi=300, bbox_inches="tight", facecolor="white")

PUB_RC = {
    "font.family": "DejaVu Sans",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "savefig.dpi": 300,
}

# Short display names for TCGA-UT classes (indices 0–30)
TCGA_SHORT = [
    "ACC", "BLCA", "LGG", "BRCA", "CESC",
    "CHOL", "COADREAD", "ESCA", "GBM", "HNSC",
    "KICH", "KIRC", "KIRP", "LIHC", "LUAD",
    "LUSC", "DLBCL", "MESO", "OV", "PAAD",
    "PCPG", "PRAD", "SARC", "SKCM", "STAD",
    "TGCT", "THYM", "THCA", "UCS", "UCEC",
    "UVM",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings(emb_dir: str):
    """
    Load pre-extracted D4 embeddings from a directory produced by extract_embeddings.py.

    Returns
    -------
    embs    : (N, 8, D) float32
    labels  : (N,) int
    indices : (N,) int  (0..N-1 if no index file)
    meta    : dict
    """
    d = Path(emb_dir)
    embs   = np.load(d / "embeddings.npy").astype(np.float32)
    labels = np.load(d / "labels.npy").astype(int)
    idx_f  = d / "indices.npy"
    indices = np.load(idx_f).astype(int) if idx_f.exists() else np.arange(len(labels))
    with open(d / "meta.json") as f:
        meta = json.load(f)
    assert embs.ndim == 3 and embs.shape[1] == 8, (
        f"Expected (N, 8, D) embeddings, got {embs.shape}"
    )
    print(
        f"Loaded embeddings: {embs.shape}  "
        f"model={meta.get('model')}  dataset={meta.get('dataset')}"
    )
    return embs, labels, indices, meta


def load_classifier(checkpoint_path: str):
    """
    Load the final linear layer from a frozen head-only checkpoint.

    Returns W (C, D) and b (C,) as float32 numpy arrays.
    Supports both old format (classifier_state_dict with '1.weight') and
    new format (flat state dict with 'net.0.weight').
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # New format: flat state dict saved directly
    if "net.0.weight" in ckpt:
        W = ckpt["net.0.weight"].float().numpy()
        b = ckpt["net.0.bias"].float().numpy()
    else:
        sd = ckpt.get("classifier_state_dict", ckpt)
        if any(k.startswith("0.") for k in sd):
            raise ValueError(
                f"{checkpoint_path!r} has an MLP head.  "
                "This script requires a simple linear head."
            )
        W = sd["1.weight"].float().numpy()
        b = sd["1.bias"].float().numpy()

    val_acc = ckpt.get('val_acc', float('nan'))
    val_str = f"{val_acc:.4f}" if isinstance(val_acc, float) and not (val_acc != val_acc) else str(val_acc)
    print(f"Loaded classifier: W={W.shape}, b={b.shape}  val_acc={val_str}")
    return W, b


# ---------------------------------------------------------------------------
# Patch selection
# ---------------------------------------------------------------------------

def select_orbits(
    embs: np.ndarray,
    labels: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    force_class_i: int | None = None,
    force_class_j: int | None = None,
):
    """
    Select orbit_A (max view-disagreement) and orbit_B (original on wrong side,
    remaining D4 views on correct side — the canonical TTA-correction case).

    Returns
    -------
    orbit_A, orbit_B : (8, D) each
    class_i, class_j : int — the two competing classes
                       class_i = positive side (right of boundary)
                       class_j = negative side (left of boundary)
    axis1            : (D,) normalized decision normal
    b_diff           : scalar b[i] - b[j]
    w_norm           : ||W[i] - W[j]||
    label_A, label_B : ground-truth class index for each patch
    """
    N, V, D = embs.shape

    # Logits: (N, 8, C)
    logits   = embs @ W.T + b[None, None, :]
    preds    = logits.argmax(axis=-1)   # (N, 8)

    # Determine class pair for axis1
    if force_class_i is not None and force_class_j is not None:
        class_i, class_j = force_class_i, force_class_j
        print(f"Forcing class pair: {TCGA_SHORT[class_i]} vs {TCGA_SHORT[class_j]}")
    else:
        n_unique  = np.array([len(set(preds[i].tolist())) for i in range(N)])
        _idx_seed = int(n_unique.argmax())
        _votes    = Counter(preds[_idx_seed].tolist())
        top2      = [c for c, _ in _votes.most_common(2)]
        class_i, class_j = top2[0], top2[1]

    # Decision axis
    w_diff     = W[class_i] - W[class_j]
    b_diff     = float(b[class_i] - b[class_j])
    w_norm     = float(np.linalg.norm(w_diff))
    axis1      = w_diff / w_norm
    boundary_x = -b_diff / w_norm

    # x-projections for every sample × view
    xs_all     = embs @ axis1        # (N, 8)
    x_orig_all = xs_all[:, 0]        # identity-view projection
    x_mean_all = xs_all.mean(axis=1) # mean over all 8 views

    # When class pair is forced, restrict candidates to ground-truth samples from those classes
    if force_class_i is not None and force_class_j is not None:
        pool = np.where(np.isin(labels, [class_i, class_j]))[0]
        print(f"Candidate pool restricted to gt∈{{{TCGA_SHORT[class_i]},{TCGA_SHORT[class_j]}}}: {len(pool)} samples")
    else:
        pool = np.arange(N)

    n_unique = np.array([len(set(preds[i].tolist())) for i in range(N)])

    # orbit_A (right panel — deep territory): all 8 views agree, original sits far
    # from the boundary. Score purely by distance so the orbit is deep and tight.
    dist_orig  = np.abs(x_orig_all - boundary_x)
    dist_mean  = np.abs(x_mean_all - boundary_x)
    deep_mask  = (n_unique == 1) & (dist_orig > 0.20)
    score_A_full = np.where(deep_mask, dist_orig, -np.inf)
    score_A   = np.full(N, -np.inf)
    score_A[pool] = score_A_full[pool]
    idx_A     = int(np.argmax(score_A))

    vote_counts = Counter(preds[idx_A].tolist())
    print(
        f"orbit_A: sample {idx_A}  gt={TCGA_SHORT[labels[idx_A]]}  "
        f"view votes={dict(vote_counts)}  "
        f"x_orig={x_orig_all[idx_A]:.3f}  x_mean={x_mean_all[idx_A]:.3f}  "
        f"boundary_x={boundary_x:.3f}"
    )

    # orbit_B: original view on the wrong side, augmented views pull mean to correct side.
    x_aug        = xs_all[:, 1:].mean(axis=1)
    correct_mask = (x_orig_all < boundary_x) & (x_aug > boundary_x)
    score_B      = (x_aug - boundary_x) + (boundary_x - x_orig_all)
    score_B_full = np.where(correct_mask, score_B, -np.inf)
    score_B_mask = np.full(N, -np.inf)
    score_B_mask[pool] = score_B_full[pool]
    idx_B        = int(np.argmax(score_B_mask))

    if score_B_mask[idx_B] == -np.inf:
        fallback = (x_orig_all < boundary_x) ^ (x_aug < boundary_x)
        score_B_full = np.where(fallback, score_B, -np.inf)
        score_B_mask = np.full(N, -np.inf)
        score_B_mask[pool] = score_B_full[pool]
        idx_B = int(np.argmax(score_B_mask))

    print(
        f"orbit_B: sample {idx_B}  gt={TCGA_SHORT[labels[idx_B]]}  "
        f"x_orig={x_orig_all[idx_B]:.3f}  x_aug_mean={x_aug[idx_B]:.3f}  "
        f"boundary_x={boundary_x:.3f}"
    )

    return (
        embs[idx_A], embs[idx_B],
        class_i, class_j,
        axis1, b_diff, w_norm,
        int(labels[idx_A]), int(labels[idx_B]),
    )


# ---------------------------------------------------------------------------
# Subspace construction
# ---------------------------------------------------------------------------

def build_local_axis(
    axis1: np.ndarray,
    orbit_A: np.ndarray,
    orbit_B: np.ndarray,
) -> np.ndarray:
    """
    Compute Axis 2: leading D4-local-variance direction, Gram-Schmidt ⊥ axis1.

    Centers each orbit independently, stacks the 16 residuals, takes the
    leading right singular vector, then projects out the axis1 component.
    """
    centered = np.vstack([
        orbit_A - orbit_A.mean(0),
        orbit_B - orbit_B.mean(0),
    ])                                                  # (16, D)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)

    for v_raw in Vt:
        v_orth = v_raw - np.dot(v_raw, axis1) * axis1
        norm   = np.linalg.norm(v_orth)
        if norm > 1e-8:
            return v_orth / norm

    raise RuntimeError("All orbit PCs are parallel to axis1 — degenerate orbits.")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _draw_panel(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    boundary_x: float,
    label_neg: str,
    label_pos: str,
) -> None:
    # Symmetric xlim around boundary_x so both class regions are visible
    x_pad  = max(0.30, (xs.max() - xs.min()) * 0.45)
    half   = max(abs(xs - boundary_x).max() + x_pad, 0.6)
    xlim   = (boundary_x - half, boundary_x + half)
    y_pad  = max(0.25, (ys.max() - ys.min()) * 0.60)
    ylim   = (ys.min() - y_pad, ys.max() + y_pad)

    # Class region shading (neg = class_j side, pos = class_i side)
    ax.axvspan(xlim[0], boundary_x, alpha=0.065, color="#E24A33", zorder=0)
    ax.axvspan(boundary_x, xlim[1], alpha=0.065, color="#0072B2", zorder=0)

    # Decision boundary
    ax.axvline(
        boundary_x,
        color="#2d2d2d", lw=1.5, ls="--", alpha=0.85, zorder=3,
        label="Decision boundary",
    )

    # Closed orbit polygon (vertices sorted by polar angle around centroid)
    pts      = np.column_stack([xs, ys])
    centroid = pts.mean(0)
    order    = np.argsort(
        np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    )
    poly = MplPolygon(
        pts[order], closed=True,
        facecolor="#888888", edgecolor="#555555",
        alpha=0.10, linewidth=1.0, zorder=2,
    )
    ax.add_patch(poly)

    # Scatter: original in red, augmented views in blue
    colors = [COLOR_ORIG] + [COLOR_AUG] * 7
    for i, (x, y, color) in enumerate(zip(xs, ys, colors)):
        ax.scatter(
            x, y, c=color, s=50, zorder=5,
            edgecolors="white", linewidths=0.6,
        )

    # Arrow from original view to mean
    ax.annotate(
        "",
        xy=(xs.mean(), ys.mean()),
        xytext=(xs[0], ys[0]),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#2d2d2d",
            lw=1.2,
            mutation_scale=10,
        ),
        zorder=7,
    )

    # Mean marker — filled diamond
    ax.scatter(
        xs.mean(), ys.mean(),
        marker="D", s=40, color="#2d2d2d", edgecolors="white",
        linewidths=0.6, zorder=8,
    )

    # Class-region text
    txt_y = ylim[0] + (ylim[1] - ylim[0]) * 0.055
    ax.text(
        xlim[0] + (boundary_x - xlim[0]) * 0.5, txt_y,
        label_neg, color="#E24A33", alpha=0.6, fontsize=7,
        ha="center", va="bottom", style="italic",
    )
    ax.text(
        boundary_x + (xlim[1] - boundary_x) * 0.5, txt_y,
        label_pos, color="#0072B2", alpha=0.6, fontsize=7,
        ha="center", va="bottom", style="italic",
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(
        rf"$\mathbf{{z}}_{{p,g}} \cdot \hat{{\mathbf{{w}}}}_{{\mathrm{{{label_pos}\ vs\ {label_neg}}}}}$",
        labelpad=5,
    )
    ax.set_ylabel(r"$(\mathbf{z}_{p,g} - \boldsymbol{\mu}_p) \cdot \hat{\mathbf{w}}^\perp$", labelpad=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(length=0)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5, symmetric=True))
    ax.grid(False)



def plot_empirical_orbits(
    axis1: np.ndarray,
    b_diff: float,
    w_norm: float,
    orbit_A: np.ndarray,
    orbit_B: np.ndarray,
    class_i: int,
    class_j: int,
    label_A: int,
    label_B: int,
    save_path: str = "figures/plot_d4_orbit_geometry",
) -> None:
    """
    Generate the two-panel D4 orbit figure and save PNG + PDF.

    Parameters
    ----------
    axis1        : (D,) normalized classifier weight normal (class_i vs class_j)
    b_diff       : b[i] - b[j]
    w_norm       : ||W[i] - W[j]||
    orbit_A/B    : (8, D) embeddings
    class_i/j    : class indices (i = positive / right side)
    label_A/B    : ground-truth class index for subtitle
    save_path    : output path prefix (no extension)
    """
    boundary_x = -b_diff / w_norm
    axis2 = build_local_axis(axis1, orbit_A, orbit_B)

    xs_A = orbit_A @ axis1
    ys_A = (orbit_A - orbit_A.mean(0)) @ axis2
    xs_B = orbit_B @ axis1
    ys_B = (orbit_B - orbit_B.mean(0)) @ axis2

    name_i = TCGA_SHORT[class_i]
    name_j = TCGA_SHORT[class_j]

    with plt.rc_context(PUB_RC):
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(5.5, 2.5))

        _draw_panel(ax_left,  xs_B, ys_B, boundary_x, label_neg=name_j, label_pos=name_i)
        _draw_panel(ax_right, xs_A, ys_A, boundary_x, label_neg=name_j, label_pos=name_i)

        fig.tight_layout(w_pad=2.5)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        for ext in (".png", ".pdf"):
            fig.savefig(save_path + ext, **SAVEKW)
        print(f"Saved → {save_path}.png  +  {save_path}.pdf")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Single-panel variant
# ---------------------------------------------------------------------------

def _plot_left_only(
    axis1: np.ndarray,
    b_diff: float,
    w_norm: float,
    orbit: np.ndarray,
    class_i: int,
    class_j: int,
    save_path: str,
) -> None:
    """Output just the correction panel with y-axis ticks and label on the right."""
    boundary_x = -b_diff / w_norm
    axis2 = build_local_axis(axis1, orbit, orbit)

    xs = orbit @ axis1
    ys = (orbit - orbit.mean(0)) @ axis2

    name_i = TCGA_SHORT[class_i]
    name_j = TCGA_SHORT[class_j]

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(1, 1, figsize=(2.9, 2.5))

        _draw_panel(ax, xs, ys, boundary_x, label_neg=name_j, label_pos=name_i)

        # Move y-axis to the right
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

        fig.tight_layout()

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        for ext in (".png", ".pdf"):
            fig.savefig(save_path + ext, **SAVEKW)
        print(f"Saved → {save_path}.png  +  {save_path}.pdf")
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="D4 orbit geometry visualization using pre-extracted embeddings"
    )
    p.add_argument(
        "--embeddings",
        default="/gpfs/scratch/lpc8816/histology_embeddings/hoptimus/tcga-ut/test",
        help="Directory containing embeddings.npy, labels.npy, meta.json",
    )
    p.add_argument(
        "--checkpoint",
        default="checkpoints/frozen/tcga-ut_hoptimus_linear_seed0.pt",
        help="Matching frozen linear-head checkpoint",
    )
    p.add_argument("--class_i", type=int, default=14,
                   help="Positive class index (default: 14 = LUAD)")
    p.add_argument("--class_j", type=int, default=15,
                   help="Negative class index (default: 15 = LUSC)")
    p.add_argument(
        "--out_dir", default="figures",
        help="Output directory",
    )
    p.add_argument("--left_only", action="store_true",
                   help="Output just the left (correction) panel with y-axis on the right")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    embs, labels, indices, meta = load_embeddings(args.embeddings)
    W, b = load_classifier(args.checkpoint)

    assert W.shape[1] == embs.shape[2], (
        f"Embedding dim {embs.shape[2]} does not match classifier input dim {W.shape[1]}"
    )

    orbit_A, orbit_B, class_i, class_j, axis1, b_diff, w_norm, label_A, label_B = \
        select_orbits(embs, labels, W, b,
                      force_class_i=args.class_i,
                      force_class_j=args.class_j)

    save_path = str(Path(args.out_dir) / "plot_d4_orbit_geometry")

    if args.left_only:
        _plot_left_only(axis1, b_diff, w_norm, orbit_B, class_i, class_j, save_path + "_left")
        return

    plot_empirical_orbits(
        axis1, b_diff, w_norm,
        orbit_A, orbit_B,
        class_i, class_j,
        label_A, label_B,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
