"""
WSI TTA Heatmap for PANDA (Radboud subset).

Uses Radboud pixel-level mask annotations (GG0-5) to compute per-patch TTA
benefit: whether D4 TTA corrects or corrupts binary cancer predictions
(cancer = GG3/4/5) compared to the baseline (view 0 only).

Output: 5-panel figure
  1. WSI thumbnail + cancer outline
  2. Baseline P(cancer)   (view 0)
  3. TTA P(cancer)        (mean of 8 D4 views)
  4. Δ P(cancer)          (TTA − baseline, diverging)
  5. Per-patch outcome    (corrected / corrupted / both_correct / both_wrong)

Usage:
  python -m panda.wsi_tta_heatmap
  python -m panda.wsi_tta_heatmap --slide_id 0018ae58b01bdadc8e347995b69f99aa
  python -m panda.wsi_tta_heatmap --all   # all Radboud slides with features
  python -m panda.wsi_tta_heatmap --grade 4   # random slide per ISUP grade
"""

from __future__ import annotations
import os

import argparse
import math
import random
from pathlib import Path

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from panda.annot_utils import patch_labels

_TRAIN_CSV   = Path(os.path.join(os.environ.get("PANDA_DIR", "data/panda"), "train.csv"))
_SLIDES_DIR  = Path(os.path.join(os.environ.get("PANDA_DIR", "data/panda"), "train_images"))
_PATCHES_DIR = Path(os.path.join(os.environ.get("PANDA_DIR", "data/panda"), "clam_patches/patches"))
_FEAT_ROOT   = Path("panda_features/uni/d4_all")
_CKPT        = Path("checkpoints/panda/patch_probe_uni_seed42/best.pt")

_OUTCOME_COLORS = {
    "corrected":    "#2ecc71",
    "corrupted":    "#e74c3c",
    "both_correct": "#3498db",
    "both_wrong":   "#95a5a6",
}


# ---------------------------------------------------------------------------
# Grid helpers (identical to camelyon17 version)
# ---------------------------------------------------------------------------

def _build_grid(
    coords: np.ndarray,
    values: np.ndarray,
    thumb_hw: tuple[int, int],
    x_extent: int,
    y_extent: int,
    patch_size_lv0: int,
    fill: float = np.nan,
) -> np.ndarray:
    H, W = thumb_hw
    sx = W / x_extent
    sy = H / y_extent
    pw = max(1, math.ceil(patch_size_lv0 * sx))
    ph = max(1, math.ceil(patch_size_lv0 * sy))
    grid = np.full((H, W), fill, dtype=np.float32)
    for (cx, cy), v in zip(coords, values):
        x0, y0 = int(cx * sx), int(cy * sy)
        grid[y0 : y0 + ph, x0 : x0 + pw] = v
    return grid


def _heatmap_float(ax, grid, cmap, vmin, vmax, title, cbar_label):
    ax.set_facecolor("white")
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest",
                   extent=[0, grid.shape[1], grid.shape[0], 0])
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(cbar_label, fontsize=6)
    cb.ax.tick_params(labelsize=5)


def _heatmap_categorical(ax, grid, title):
    from matplotlib.colors import ListedColormap
    keys   = ["corrected", "corrupted", "both_correct", "both_wrong"]
    colors = [_OUTCOME_COLORS[k] for k in keys]
    cmap   = ListedColormap(colors)
    ax.set_facecolor("white")
    masked = np.ma.masked_where(np.isnan(grid), grid)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=len(keys) - 1,
              interpolation="nearest",
              extent=[0, grid.shape[1], grid.shape[0], 0])
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    patches = [mpatches.Patch(color=_OUTCOME_COLORS[k], label=k) for k in keys]
    ax.legend(handles=patches, loc="lower right", fontsize=5,
              framealpha=0.6, markerscale=0.7)


# ---------------------------------------------------------------------------
# Core per-slide function
# ---------------------------------------------------------------------------

@torch.no_grad()
def process_slide(
    slide_id: str,
    isup_grade: int,
    gleason_score: str,
    classifier: torch.nn.Module,
    device: torch.device,
    args,
) -> Path | None:
    import openslide as _osl

    feat_path  = args.features_dir / f"{slide_id}.pt"
    h5_path    = args.patches_dir  / f"{slide_id}.h5"
    slide_path = args.slides_dir   / f"{slide_id}.tiff"

    if not feat_path.exists():
        print(f"  skip {slide_id}: no features")
        return None

    # ── features & coords ───────────────────────────────────────────────────
    feat = torch.load(feat_path, weights_only=True)   # (N, 8, 1024)
    if feat.ndim == 2:
        feat = feat.unsqueeze(1)

    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]   # (N, 2)

    N = feat.shape[0]

    # ── patch-level cancer labels from Radboud mask ──────────────────────────
    gt_labels = patch_labels(slide_id, coords, args.patch_size_lv0)
    n_cancer  = gt_labels.sum()
    print(f"  {slide_id} ISUP={isup_grade} ({gleason_score}): "
          f"{N} patches, {n_cancer} cancer ({100*n_cancer/N:.1f}%)")

    # ── inference ────────────────────────────────────────────────────────────
    p_base_all, p_tta_all, p_var_all = [], [], []
    for i in range(0, N, args.batch_size):
        b     = feat[i : i + args.batch_size].to(device)  # (B, 8, d)
        f0    = b[:, 0, :]
        f_avg = b.mean(dim=1)
        p_base_all.append(F.softmax(classifier(f0),    dim=-1)[:, 1].cpu().numpy())
        p_tta_all.append( F.softmax(classifier(f_avg), dim=-1)[:, 1].cpu().numpy())
        B, V, d = b.shape
        p_views = F.softmax(classifier(b.reshape(B * V, d)), dim=-1)[:, 1]
        p_var_all.append(p_views.reshape(B, V).var(dim=1).cpu().numpy())

    p_base = np.concatenate(p_base_all)
    p_tta  = np.concatenate(p_tta_all)
    p_var  = np.concatenate(p_var_all)
    delta  = p_tta - p_base

    # ── per-patch outcome ────────────────────────────────────────────────────
    thr = 0.5
    pred_base = (p_base >= thr).astype(int)
    pred_tta  = (p_tta  >= thr).astype(int)
    outcome_idx = np.full(N, np.nan, dtype=np.float32)
    for i in range(N):
        bc = (pred_base[i] == gt_labels[i])
        tc = (pred_tta[i]  == gt_labels[i])
        if   not bc and tc:  outcome_idx[i] = 0
        elif bc and not tc:  outcome_idx[i] = 1
        elif bc and tc:      outcome_idx[i] = 2
        else:                outcome_idx[i] = 3

    corrected = (outcome_idx == 0).sum()
    corrupted = (outcome_idx == 1).sum()
    print(f"    corrected={corrected}  corrupted={corrupted}  "
          f"net={(corrected - corrupted):+d}")

    # ── thumbnail via openslide ───────────────────────────────────────────────
    if slide_path.exists():
        _sl      = _osl.OpenSlide(str(slide_path))
        x_ext, y_ext = _sl.level_dimensions[0]
        lv       = min(args.wsi_level, _sl.level_count - 1)
        lv_w, lv_h = _sl.level_dimensions[lv]
        thumb_img = np.array(_sl.read_region((0, 0), lv, (lv_w, lv_h)).convert("RGB"))
        _sl.close()
        thumb_img[thumb_img.max(axis=-1) < 20] = 255
    else:
        print(f"  WARNING: slide not found at {slide_path}")
        thumb_img = np.full((512, 512, 3), 255, dtype=np.uint8)
        x_ext = int(coords[:, 0].max()) + args.patch_size_lv0
        y_ext = int(coords[:, 1].max()) + args.patch_size_lv0

    H, W = thumb_img.shape[:2]
    grid_kw = dict(thumb_hw=(H, W), x_extent=x_ext, y_extent=y_ext,
                   patch_size_lv0=args.patch_size_lv0)

    grid_delta   = _build_grid(coords, delta,       **grid_kw)
    grid_outcome = _build_grid(coords, outcome_idx, **grid_kw)
    grid_var     = _build_grid(coords, p_var,       **grid_kw)

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 5),
                             constrained_layout=True)
    fig.suptitle(
        f"{slide_id}  |  ISUP {isup_grade} ({gleason_score})  |  "
        f"{n_cancer}/{N} patches cancer  |  "
        f"corrected={corrected}  corrupted={corrupted}  net={(corrected-corrupted):+d}",
        fontsize=9,
    )

    # Panel 0: thumbnail + cancer outline (rectangles avoid imshow/contour
    # axis-orientation conflicts)
    axes[0].imshow(thumb_img)
    if n_cancer > 0:
        sx = W / x_ext
        sy = H / y_ext
        pw_px = max(1, math.ceil(args.patch_size_lv0 * sx))
        ph_px = max(1, math.ceil(args.patch_size_lv0 * sy))
        for (cx, cy), lbl in zip(coords, gt_labels):
            if lbl == 1:
                axes[0].add_patch(mpatches.Rectangle(
                    (int(cx * sx), int(cy * sy)), pw_px, ph_px,
                    linewidth=1.2, edgecolor="#e74c3c",
                    facecolor="#e74c3c", alpha=0.35))
    axes[0].set_title("WSI + cancer outline", fontsize=8)
    axes[0].axis("off")

    max_abs = max(0.05, float(np.nanpercentile(np.abs(delta), 99)))
    _heatmap_float(axes[1], grid_delta,
                   "RdBu_r", -max_abs, max_abs,
                   "Δ P(cancer)\n(TTA − baseline)", "Δ P(cancer)")

    _heatmap_categorical(axes[2], grid_outcome,
                         "Per-patch outcome\n(TTA vs baseline)")

    max_var = max(1e-4, float(np.nanpercentile(p_var, 99)))
    _heatmap_float(axes[3], grid_var,
                   "YlOrRd", 0, max_var,
                   "D4 view variance\nof P(cancer)", "Var P(cancer)")
    out = args.out_dir / f"wsi_tta_heatmap_{slide_id}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--slide_id",       type=str,  default=None)
    p.add_argument("--all",            action="store_true",
                   help="Process all Radboud slides with features.")
    p.add_argument("--grade",          type=int,  default=None,
                   help="Pick one random slide per ISUP grade (or a specific grade 0-5).")
    p.add_argument("--n_per_grade",    type=int,  default=1,
                   help="How many slides per grade when using --grade.")
    p.add_argument("--checkpoint",     type=Path, default=_CKPT)
    p.add_argument("--features_dir",   type=Path, default=_FEAT_ROOT)
    p.add_argument("--patches_dir",    type=Path, default=_PATCHES_DIR)
    p.add_argument("--slides_dir",     type=Path, default=_SLIDES_DIR)
    p.add_argument("--train_csv",      type=Path, default=_TRAIN_CSV)
    p.add_argument("--out_dir",        type=Path, default=Path("figures/panda_wsi_heatmaps"))
    p.add_argument("--device",         type=str,  default="cuda")
    p.add_argument("--patch_size_lv0", type=int,  default=256)
    p.add_argument("--wsi_level",      type=int,  default=2)
    p.add_argument("--batch_size",     type=int,  default=1024)
    p.add_argument("--seed",           type=int,  default=42)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)

    # load probe
    ckpt       = torch.load(args.checkpoint, weights_only=False, map_location=device)
    embed_dim  = ckpt["model"]["weight"].shape[1]
    classifier = nn.Linear(embed_dim, 2).to(device)
    classifier.load_state_dict(ckpt["model"])
    classifier.eval()

    df       = pd.read_csv(args.train_csv)
    radboud  = df[df.data_provider == "radboud"].set_index("image_id")

    # build slide list
    if args.all:
        slides = [sid for sid in radboud.index
                  if (args.features_dir / f"{sid}.pt").exists()]
    elif args.slide_id:
        slides = [args.slide_id]
    elif args.grade is not None:
        grades = [args.grade] if args.grade >= 0 else list(range(6))
        slides = []
        for g in grades:
            candidates = [sid for sid in radboud[radboud.isup_grade == g].index
                          if (args.features_dir / f"{sid}.pt").exists()]
            random.shuffle(candidates)
            slides.extend(candidates[: args.n_per_grade])
    else:
        # default: one slide per grade
        slides = []
        for g in range(6):
            candidates = [sid for sid in radboud[radboud.isup_grade == g].index
                          if (args.features_dir / f"{sid}.pt").exists()]
            random.shuffle(candidates)
            if candidates:
                slides.append(candidates[0])

    print(f"Processing {len(slides)} slide(s)...")
    for slide_id in slides:
        row = radboud.loc[slide_id] if slide_id in radboud.index else None
        isup  = int(row["isup_grade"])  if row is not None else -1
        glee  = str(row["gleason_score"]) if row is not None else "unknown"
        process_slide(slide_id, isup, glee, classifier, device, args)


if __name__ == "__main__":
    main()
