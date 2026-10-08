"""
WSI TTA Heatmap for Camelyon17.

Uses 50 slides with pixel-level ASAP lesion annotations to compute *true*
per-patch TTA benefit: whether D4 TTA corrects or corrupts patch-level
tumor predictions compared to the baseline (view 0 only).

The patch-level linear probe (nn.Linear, trained by train_patch_probe.py) is
applied to each patch's view-0 features (baseline) and mean-of-8-views features
(TTA) to get per-patch tumor probabilities.

Output: 6-panel figure
  1. WSI thumbnail
  2. Ground-truth tumor mask (from annotations)
  3. Baseline P(tumor)  (view 0)
  4. TTA P(tumor)       (mean of 8 D4 views)
  5. Δ P(tumor)         (TTA − baseline, diverging)
  6. Per-patch outcome  (corrected / corrupted / both_correct / both_wrong)

Usage:
  python -m camelyon17.wsi_tta_heatmap
  python -m camelyon17.wsi_tta_heatmap --slide_id patient_039_node_1
  python -m camelyon17.wsi_tta_heatmap --all   # iterate over all 50 annotated slides
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from camelyon17.annot_utils import parse_lesion_polygons, patch_labels

_ANNOT_DIR   = Path("/gpfs/data/oermannlab/public_data/camelyon17/training/lesion_annotations")
_DATA_ROOT   = Path("/gpfs/data/mankowskilab/chen/camelyon17_patched")
_FEAT_ROOT   = Path("camelyon17_features/uni/d4_all")
_CKPT        = Path("checkpoints/camelyon17/patch_probe_uni_seed42/best.pt")

# Outcome colours for the categorical panel
_OUTCOME_COLORS = {
    "corrected":    "#2ecc71",   # green  — TTA fixes wrong baseline
    "corrupted":    "#e74c3c",   # red    — TTA breaks correct baseline
    "both_correct": "#3498db",   # blue   — both right
    "both_wrong":   "#95a5a6",   # grey   — both wrong
}


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _build_grid(
    coords: np.ndarray,
    values: np.ndarray,         # float or int (N,)
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


def _heatmap_float(
    ax, grid: np.ndarray,
    cmap: str, vmin: float, vmax: float,
    title: str, cbar_label: str,
):
    """Show float heatmap on a white background (no thumbnail)."""
    ax.set_facecolor("white")
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="nearest",
                   extent=[0, grid.shape[1], grid.shape[0], 0])
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(cbar_label, fontsize=6)
    cb.ax.tick_params(labelsize=5)


def _heatmap_categorical(
    ax, grid: np.ndarray,
    title: str,
):
    """Show categorical outcome heatmap on a white background (no thumbnail)."""
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
    classifier: torch.nn.Module,
    device: torch.device,
    args,
) -> Path | None:
    feat_path  = args.features_dir / f"{slide_id}.pt"
    h5_path    = args.patches_dir  / f"{slide_id}.h5"
    annot_path = _ANNOT_DIR / f"{slide_id}.xml"
    slide_path = args.slides_dir   / f"{slide_id}.tif"

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

    # ── annotations → patch-level ground truth ──────────────────────────────
    tumor_region = parse_lesion_polygons(annot_path) if annot_path.exists() else None
    gt_labels    = patch_labels(coords, args.patch_size_lv0, tumor_region)
    n_tumor      = gt_labels.sum()
    print(f"  {slide_id}: {N} patches, {n_tumor} tumor ({100*n_tumor/N:.1f}%)")

    # ── inference ────────────────────────────────────────────────────────────
    p_base_all, p_tta_all, p_var_all = [], [], []
    for i in range(0, N, args.batch_size):
        b     = feat[i : i + args.batch_size].to(device)  # (B, 8, d)
        f0    = b[:, 0, :]
        f_avg = b.mean(dim=1)
        p_base_all.append(F.softmax(classifier(f0),    dim=-1)[:, 1].cpu().numpy())
        p_tta_all.append( F.softmax(classifier(f_avg), dim=-1)[:, 1].cpu().numpy())
        # variance of P(tumor) across the 8 D4 views
        B, V, d = b.shape
        p_views = F.softmax(classifier(b.view(B * V, d)), dim=-1)[:, 1]
        p_var_all.append(p_views.view(B, V).var(dim=1).cpu().numpy())

    p_base = np.concatenate(p_base_all)
    p_tta  = np.concatenate(p_tta_all)
    p_var  = np.concatenate(p_var_all)
    delta  = p_tta - p_base

    # ── per-patch outcome ────────────────────────────────────────────────────
    thr = 0.5
    pred_base = (p_base >= thr).astype(int)
    pred_tta  = (p_tta  >= thr).astype(int)
    outcome_map = {"corrected": 0, "corrupted": 1, "both_correct": 2, "both_wrong": 3}
    outcome_idx = np.full(N, np.nan, dtype=np.float32)
    for i in range(N):
        bc = (pred_base[i] == gt_labels[i])
        tc = (pred_tta[i]  == gt_labels[i])
        if   not bc and tc:  outcome_idx[i] = 0  # corrected
        elif bc and not tc:  outcome_idx[i] = 1  # corrupted
        elif bc and tc:      outcome_idx[i] = 2  # both correct
        else:                outcome_idx[i] = 3  # both wrong

    corrected = (outcome_idx == 0).sum()
    corrupted = (outcome_idx == 1).sum()
    print(f"    corrected={corrected}  corrupted={corrupted}  "
          f"net={(corrected - corrupted):+d}")

    # ── thumbnail from openslide (proper white background + true dims) ─────────
    import openslide as _osl
    if slide_path.exists():
        _sl      = _osl.OpenSlide(str(slide_path))
        x_ext, y_ext = _sl.level_dimensions[0]          # true level-0 (W, H)
        lv       = min(args.wsi_level, _sl.level_count - 1)
        lv_w, lv_h = _sl.level_dimensions[lv]
        thumb_img = np.array(_sl.read_region((0, 0), lv, (lv_w, lv_h)).convert("RGB"))
        _sl.close()
        # make background white (near-black pixels → white)
        thumb_img[thumb_img.max(axis=-1) < 20] = 255
    else:
        thumb_img = np.full((512, 512, 3), 255, dtype=np.uint8)
        x_ext = int(coords[:, 0].max()) + args.patch_size_lv0
        y_ext = int(coords[:, 1].max()) + args.patch_size_lv0

    H, W = thumb_img.shape[:2]

    grid_kw   = dict(thumb_hw=(H, W), x_extent=x_ext, y_extent=y_ext,
                     patch_size_lv0=args.patch_size_lv0)

    grid_delta   = _build_grid(coords, delta,        **grid_kw)
    grid_outcome = _build_grid(coords, outcome_idx,  **grid_kw)
    grid_var     = _build_grid(coords, p_var,        **grid_kw)

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(16, 5),
                             constrained_layout=True)
    fig.suptitle(
        f"{slide_id}  |  {n_tumor}/{N} patches tumor  |  "
        f"corrected={corrected}  corrupted={corrupted}  net={(corrected-corrupted):+d}",
        fontsize=9,
    )

    # Panel 0: thumbnail + tumor outline (drawn directly from XML polygons)
    axes[0].imshow(thumb_img)
    if annot_path.exists():
        sx_ann = W / x_ext
        sy_ann = H / y_ext
        tree = ET.parse(annot_path)
        for ann in tree.getroot().findall(".//Annotation"):
            if ann.get("PartOfGroup", "").lower() != "metastases":
                continue
            pts = np.array([(float(c.get("X")) * sx_ann,
                             float(c.get("Y")) * sy_ann)
                            for c in ann.findall(".//Coordinate")])
            if len(pts) < 3:
                continue
            poly = mpatches.Polygon(pts, closed=True, fill=False,
                                    edgecolor="#e74c3c", linewidth=1.0)
            axes[0].add_patch(poly)
    axes[0].set_title("WSI + tumor outline", fontsize=8)
    axes[0].axis("off")

    max_abs = max(0.05, float(np.nanpercentile(np.abs(delta), 99)))
    _heatmap_float(axes[1], grid_delta,
                   "RdBu_r", -max_abs, max_abs,
                   "Δ P(tumor)\n(TTA − baseline)", "Δ P(tumor)")

    _heatmap_categorical(axes[2], grid_outcome,
                         "Per-patch outcome\n(TTA vs baseline)")

    max_var = max(1e-4, float(np.nanpercentile(p_var, 99)))
    _heatmap_float(axes[3], grid_var,
                   "YlOrRd", 0, max_var,
                   "D4 view variance\nof P(tumor)", "Var P(tumor)")

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
    p.add_argument("--slide_id",           type=str,  default=None,
                   help="Single slide to process. Default: auto-pick first annotated slide.")
    p.add_argument("--all",                action="store_true",
                   help="Process all 50 annotated slides.")
    p.add_argument("--checkpoint",         type=Path, default=_CKPT)
    p.add_argument("--features_dir",       type=Path, default=_FEAT_ROOT)
    p.add_argument("--patches_dir",        type=Path, default=_DATA_ROOT / "patches")
    p.add_argument("--stitches_dir",       type=Path, default=_DATA_ROOT / "stitches")
    p.add_argument("--slides_dir",         type=Path, default=_DATA_ROOT / "slides_symlinks")
    p.add_argument("--out_dir",            type=Path, default=Path("figures"))
    p.add_argument("--device",             type=str,  default="cuda")
    p.add_argument("--patch_size_lv0",     type=int,  default=512)
    p.add_argument("--alpha",              type=float, default=0.55)
    p.add_argument("--batch_size",         type=int,  default=1024)
    p.add_argument("--wsi_level",          type=int,  default=4)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load patch-level linear probe
    ckpt       = torch.load(args.checkpoint, weights_only=False, map_location=device)
    embed_dim  = ckpt["model"]["weight"].shape[1]   # infer from probe weights
    classifier = nn.Linear(embed_dim, 2).to(device)
    classifier.load_state_dict(ckpt["model"])
    classifier.eval()

    # select slides
    all_annotated = sorted(p.stem for p in _ANNOT_DIR.glob("*.xml"))

    if args.all:
        slides = all_annotated
    elif args.slide_id:
        slides = [args.slide_id]
    else:
        # default: pick a positive slide that has features
        for s in all_annotated:
            if (args.features_dir / f"{s}.pt").exists():
                slides = [s]
                break
        else:
            raise RuntimeError("No annotated slides with features found.")

    print(f"Processing {len(slides)} slide(s)...")
    for slide_id in slides:
        process_slide(slide_id, classifier, device, args)


if __name__ == "__main__":
    main()
