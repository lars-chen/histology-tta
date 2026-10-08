"""
Export high-fidelity individual panels for two representative PANDA slides.

Slides selected by highest net TTA corrections (corrected − corrupted):
  a4d8550e3c9eeb147f16812c33fe594e  corrected=41 corrupted=9  net=+32
  b23fa1eb9b60a23d45c43df4c20e180e  corrected=18 corrupted=1  net=+17

Panel 0 — H&E thumbnail + cancer rectangles:  PNG (photograph, raster)
Panel 1 — ΔP(cancer) map:                     PDF + PNG
Panel 2 — Per-patch outcome:                  PDF + PNG

Outputs in figures/panels/:
  panda_{slide_id[:8]}_he_outline.png
  panda_{slide_id[:8]}_delta_p.pdf / .png
  panda_{slide_id[:8]}_outcome.pdf  / .png
"""
from __future__ import annotations
import os
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
import torch.nn as nn
import torch.nn.functional as F

from panda.annot_utils import patch_labels

_SLIDES_DIR  = Path(os.path.join(os.environ.get("PANDA_DIR", "data/panda"), "train_images"))
_PATCHES_DIR = Path(os.path.join(os.environ.get("PANDA_DIR", "data/panda"), "clam_patches/patches"))
_FEAT_DIR    = Path("panda_features/uni/d4_all")
_CKPT        = Path("checkpoints/panda/patch_probe_uni_seed42/best.pt")
_OUT_DIR     = Path("figures/panels")

PATCH_SIZE_LV0 = 256

SLIDES = [
    "a4d8550e3c9eeb147f16812c33fe594e",
    "b23fa1eb9b60a23d45c43df4c20e180e",
]

_OUTCOME_COLORS = {
    "corrected":    "#2ecc71",
    "corrupted":    "#e74c3c",
    "both_correct": "#3498db",
    "both_wrong":   "#95a5a6",
}


def _build_grid(coords, values, thumb_hw, x_extent, y_extent,
                patch_size_lv0=PATCH_SIZE_LV0, fill=np.nan):
    H, W = thumb_hw
    sx, sy = W / x_extent, H / y_extent
    pw = max(1, math.ceil(patch_size_lv0 * sx))
    ph = max(1, math.ceil(patch_size_lv0 * sy))
    grid = np.full((H, W), fill, dtype=np.float32)
    for (cx, cy), v in zip(coords, values):
        x0, y0 = int(cx * sx), int(cy * sy)
        grid[y0: y0+ph, x0: x0+pw] = v
    return grid


def _get_thumbnail(slide_id: str, max_side: int = 1200):
    slide_path = _SLIDES_DIR / f"{slide_id}.tiff"
    sl = openslide.OpenSlide(str(slide_path))
    x_ext, y_ext = sl.level_dimensions[0]
    # pick level closest to target size
    scale = min(max_side / max(x_ext, y_ext), 1.0)
    target_w = int(x_ext * scale)
    target_h = int(y_ext * scale)
    # use highest available level that is still >= target
    lv = 0
    for l in range(sl.level_count):
        lw, lh = sl.level_dimensions[l]
        if lw >= target_w and lh >= target_h:
            lv = l
    lv_w, lv_h = sl.level_dimensions[lv]
    thumb = np.array(sl.read_region((0, 0), lv, (lv_w, lv_h)).convert("RGB"))
    sl.close()
    # blank out near-black background
    thumb[thumb.max(axis=-1) < 20] = 255
    return thumb, x_ext, y_ext


@torch.no_grad()
def export_panels(slide_id: str, out_dir: Path, classifier, device):
    feat_path = _FEAT_DIR / f"{slide_id}.pt"
    h5_path   = _PATCHES_DIR / f"{slide_id}.h5"

    feat = torch.load(feat_path, weights_only=True)   # (N, 8, 1024)
    if feat.ndim == 2:
        feat = feat.unsqueeze(1)
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]   # (N, 2)
    N = feat.shape[0]

    gt_labels = patch_labels(slide_id, coords, PATCH_SIZE_LV0)

    p_base_all, p_tta_all = [], []
    for i in range(0, N, 512):
        b = feat[i:i+512].to(device)
        p_base_all.append(F.softmax(classifier(b[:, 0, :]),    dim=-1)[:, 1].cpu().numpy())
        p_tta_all.append( F.softmax(classifier(b.mean(dim=1)), dim=-1)[:, 1].cpu().numpy())
    p_base = np.concatenate(p_base_all)
    p_tta  = np.concatenate(p_tta_all)
    delta  = p_tta - p_base

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
    print(f"  {slide_id[:16]}: N={N}, cancer={gt_labels.sum()}, "
          f"corrected={corrected}, corrupted={corrupted}, net={corrected-corrupted:+.0f}")

    thumb, x_ext, y_ext = _get_thumbnail(slide_id)
    H, W = thumb.shape[:2]
    sx, sy = W / x_ext, H / y_ext
    pw_px = max(1, math.ceil(PATCH_SIZE_LV0 * sx))
    ph_px = max(1, math.ceil(PATCH_SIZE_LV0 * sy))

    grid_kw = dict(thumb_hw=(H, W), x_extent=x_ext, y_extent=y_ext)
    grid_delta   = _build_grid(coords, delta,       **grid_kw)
    grid_outcome = _build_grid(coords, outcome_idx, **grid_kw)

    # Crop grids to patch bounding box (skip isolated stray patches)
    valid = ~np.isnan(grid_outcome)
    col_density = valid.mean(axis=0)
    dense_cols  = np.where(col_density > 0.002)[0]
    rows_with   = np.where(valid.any(axis=1))[0]
    pad = 8
    r0 = max(0, rows_with[0]  - pad)
    r1 = min(H, rows_with[-1] + pad)
    c0 = max(0, dense_cols[0] - pad)
    c1 = min(W, dense_cols[-1]+ pad)

    thumb_crop   = thumb[r0:r1, c0:c1]
    grid_delta   = grid_delta[r0:r1, c0:c1]
    grid_outcome = grid_outcome[r0:r1, c0:c1]
    H_crop, W_crop = grid_delta.shape

    # Cancer patch rects in cropped coordinate space
    cancer_rects = []
    for (cx, cy), lbl in zip(coords, gt_labels):
        if lbl == 1:
            px, py = int(cx * sx) - c0, int(cy * sy) - r0
            if 0 <= px < W_crop and 0 <= py < H_crop:
                cancer_rects.append((px, py))

    pfx = f"panda_{slide_id[:8]}"

    # ── Panel 0: H&E thumbnail + cancer outlines ─────────────────────────────
    aspect = H_crop / W_crop
    fig, ax = plt.subplots(figsize=(5, 5 * aspect))
    ax.imshow(thumb_crop, aspect="auto")
    for (px, py) in cancer_rects:
        ax.add_patch(mpatches.Rectangle(
            (px, py), pw_px, ph_px,
            linewidth=0, facecolor="#e74c3c", alpha=0.35))
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    out = out_dir / f"{pfx}_he_outline.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {out.name}  ({thumb_crop.shape[1]}×{thumb_crop.shape[0]}px)")

    # ── Panel 1: ΔP(cancer) ──────────────────────────────────────────────────
    max_abs = max(0.05, float(np.nanpercentile(np.abs(delta), 99)))
    fig, ax = plt.subplots(figsize=(5, 5 * aspect))
    ax.set_facecolor("white")
    im = ax.imshow(np.ma.masked_invalid(grid_delta), cmap="RdBu_r",
                   vmin=-max_abs, vmax=max_abs, interpolation="nearest",
                   extent=[0, W_crop, H_crop, 0])
    for (px, py) in cancer_rects:
        ax.add_patch(mpatches.Rectangle(
            (px, py), pw_px, ph_px,
            linewidth=0.6, edgecolor="black", facecolor="none", alpha=0.5))
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.8)
    cb.set_label("Δ P(cancer)  (TTA − baseline)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{pfx}_delta_p.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {pfx}_delta_p.pdf/.png")

    # ── Panel 2: per-patch outcome ────────────────────────────────────────────
    from matplotlib.colors import ListedColormap
    keys   = ["corrected", "corrupted", "both_correct", "both_wrong"]
    colors = [_OUTCOME_COLORS[k] for k in keys]
    cmap   = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(5, 5 * aspect))
    ax.set_facecolor("white")
    masked = np.ma.masked_where(np.isnan(grid_outcome), grid_outcome)
    ax.imshow(masked, cmap=cmap, vmin=0, vmax=len(keys) - 1,
              interpolation="nearest", extent=[0, W_crop, H_crop, 0])
    for (px, py) in cancer_rects:
        ax.add_patch(mpatches.Rectangle(
            (px, py), pw_px, ph_px,
            linewidth=0.6, edgecolor="black", facecolor="none", alpha=0.5))
    legend_patches = [mpatches.Patch(color=_OUTCOME_COLORS[k], label=k) for k in keys]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
              framealpha=0.8, edgecolor="grey")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{pfx}_outcome.{ext}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {pfx}_outcome.pdf/.png")


def main():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(_CKPT, map_location=device, weights_only=False)
    model_state = ckpt["model"] if "model" in ckpt else ckpt
    in_features = model_state["weight"].shape[1]
    classifier = nn.Linear(in_features, 2).to(device)
    classifier.load_state_dict(model_state)
    classifier.eval()
    print(f"Loaded classifier from {_CKPT}  (device={device})")

    for slide_id in SLIDES:
        print(f"\n── {slide_id[:16]}... ──")
        export_panels(slide_id, _OUT_DIR, classifier, device)

    print(f"\nAll panels saved to {_OUT_DIR}/")


if __name__ == "__main__":
    main()
