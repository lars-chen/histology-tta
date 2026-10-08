"""
Export high-fidelity individual panels for patient_096_node_0 and patient_052_node_1.

Panel 0 — H&E + tumor outline:  extracted from existing heatmap PNG (high-res crop)
Panel 1 — ΔP(tumor) map:        regenerated from raw features → PDF + PNG
Panel 2 — Per-patch outcome:    regenerated from raw features → PDF + PNG

Outputs in figures/panels/:
  p096_he_outline.png
  p096_delta_p.pdf / .png
  p096_outcome.pdf  / .png
  p052_he_outline.png
  p052_delta_p.pdf  / .png
  p052_outcome.pdf  / .png
"""
from __future__ import annotations
import math
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter, label as nd_label

# ── Paths (mirror wsi_tta_heatmap.py) ────────────────────────────────────────
_ANNOT_DIR   = Path(os.path.join(os.environ.get("CAMELYON17_DIR", "data/camelyon17"), "training/lesion_annotations"))
_PATCHES_DIR = Path(os.path.join(os.environ.get("CAMELYON17_PATCHED_DIR", "data/camelyon17_patched"), "patches"))
_SLIDES_DIR  = Path(os.path.join(os.environ.get("CAMELYON17_PATCHED_DIR", "data/camelyon17_patched"), "slides_symlinks"))
# Backbone to render, e.g. uni / convnextv2 / gigapath (set via PANEL_MODEL env var).
_MODEL       = os.environ.get("PANEL_MODEL", "uni")
_FEAT_DIR    = Path(f"camelyon17_features/{_MODEL}/d4_all")
_CKPT        = Path("checkpoints/camelyon17/patch_probe_uni_seed42/best.pt")
# Per-slide honest (leave-one-patient-out) probe for the chosen backbone —
# never trained on the slide shown.
_CKPTS = {
    "patient_096_node_0": Path(f"checkpoints/camelyon17/lopo_{_MODEL}/heldout_patient_096.pt"),
    "patient_052_node_1": Path(f"checkpoints/camelyon17/lopo_{_MODEL}/heldout_patient_052.pt"),
}
_HEATMAP_DIR = Path("figures/wsi_heatmaps")
_OUT_DIR     = Path("figures/panels")

# Slides to render (comma-separated PANEL_SLIDES env var); default p096.
SLIDES = os.environ.get("PANEL_SLIDES", "patient_096_node_0").split(",")
PATCH_SIZE_LV0 = 512

# Crop boxes for Panel 0 (H&E) within original 4-panel heatmap PNG
# (y0, y1, panel_x0, panel_x1) — coordinates within the per-panel slice
_HE_CROPS = {
    "patient_096_node_0": (340, 610, 222, 597),
    "patient_052_node_1": (270, 735,  15, 599),
}

_OUTCOME_COLORS = {
    "corrected":    "#00a000",   # deep saturated green — pops on pink H&E
    "corrupted":    "#e74c3c",
    "both_correct": "#a8cce8",
    "both_wrong":   "#9b9b9b",   # grey — still-missed by both
}
_TUMOR_OUTLINE = "#1a3a6b"       # dark blue

# Crop to a single tissue lobe (largest connected component) for tight abstract panels.
_SINGLE_LOBE = {"patient_096_node_0"}

# Optional tight crop for outcome/delta panels (r0, r1, c0, c1) in grid coords.
# None = no crop (show full slide).
_PANEL_CROPS = {
    "patient_052_node_1": (177, 716, 280, 960),
    "patient_096_node_0": None,
}

# Zoom crop for a separate tight delta-P panel (top-right annotated mass)
_ZOOM_CROPS = {
    "patient_052_node_1": (196, 491, 660, 960),
}


# ── Helpers (same as wsi_tta_heatmap.py) ─────────────────────────────────────

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


def _outline_polygons(annot_path, sx, sy):
    """Return list of polygon point arrays scaled by (sx, sy)."""
    polys = []
    if not annot_path.exists():
        return polys
    tree = ET.parse(annot_path)
    for ann in tree.getroot().findall(".//Annotation"):
        if ann.get("PartOfGroup", "").lower() != "metastases":
            continue
        pts = np.array([(float(c.get("X")) * sx, float(c.get("Y")) * sy)
                        for c in ann.findall(".//Coordinate")])
        if len(pts) >= 3:
            polys.append(pts)
    return polys


# ── Panel 0: extract H&E from existing heatmap PNG ───────────────────────────

def export_he_panel(slide_id: str, out_dir: Path):
    src = _HEATMAP_DIR / f"wsi_tta_heatmap_{slide_id}.png"
    arr = np.array(Image.open(src))
    pw = arr.shape[1] // 4
    y0, y1, x0, x1 = _HE_CROPS[slide_id]
    crop = arr[y0:y1, x0:x1]  # panel 0 starts at x=0

    # Save at native resolution as PNG
    out = out_dir / f"{slide_id.replace('patient_', 'p').replace('_node_', '_node')}_he_outline.png"
    Image.fromarray(crop).save(out, dpi=(300, 300))
    print(f"  Saved {out}  ({crop.shape[1]}×{crop.shape[0]}px)")


# ── Panels 1 & 2: regenerate from raw data ───────────────────────────────────

def _load_he_thumb(slide_id: str, thumb_hw: tuple[int, int]) -> np.ndarray | None:
    """Load H&E thumbnail at thumb_hw resolution via openslide. Returns None if unavailable."""
    slide_path = _SLIDES_DIR / f"{slide_id}.tif"
    if not slide_path.exists():
        return None
    try:
        import openslide as _osl
        sl  = _osl.OpenSlide(str(slide_path))
        lv  = min(4, sl.level_count - 1)
        lv_w, lv_h = sl.level_dimensions[lv]
        arr = np.array(sl.read_region((0, 0), lv, (lv_w, lv_h)).convert("RGB"))
        sl.close()
        arr[arr.max(axis=-1) < 20] = 255   # fix near-black artifacts → white
        H, W = thumb_hw
        thumb = np.array(Image.fromarray(arr).resize((W, H), Image.LANCZOS))
        # Destripe: remove horizontal banding by subtracting per-row mean deviation
        thumb_f = thumb.astype(np.float32)
        row_means = thumb_f.mean(axis=1, keepdims=True)   # (H, 1, 3)
        global_mean = thumb_f.mean(axis=(0, 1), keepdims=True)
        thumb_f = (thumb_f - row_means + global_mean).clip(0, 255)
        thumb = thumb_f.astype(np.uint8)
        return thumb
    except Exception as e:
        print(f"  [warn] could not load slide thumbnail: {e}")
        return None


def _hex_to_rgb(h: str) -> tuple[float, float, float]:
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (1, 3, 5))


@torch.no_grad()
def export_vector_panels(slide_id: str, out_dir: Path, classifier, device):
    feat_path  = _FEAT_DIR   / f"{slide_id}.pt"
    h5_path    = _PATCHES_DIR / f"{slide_id}.h5"
    annot_path = _ANNOT_DIR  / f"{slide_id}.xml"

    if not feat_path.exists():
        print(f"  skip {slide_id}: no features at {feat_path}")
        return

    # Load features and coords
    feat = torch.load(feat_path, weights_only=True)   # (N, 8, 1024)
    if feat.ndim == 2:
        feat = feat.unsqueeze(1)
    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]   # (N, 2)
    N = feat.shape[0]

    # Use actual slide dimensions so the thumbnail and grid share the same coordinate space
    slide_path = _SLIDES_DIR / f"{slide_id}.tif"
    if slide_path.exists():
        import openslide as _osl
        _sl = _osl.OpenSlide(str(slide_path))
        x_ext, y_ext = _sl.level_dimensions[0]
        _sl.close()
    else:
        x_ext = int(coords[:, 0].max()) + PATCH_SIZE_LV0
        y_ext = int(coords[:, 1].max()) + PATCH_SIZE_LV0

    # Determine thumbnail size (scale so longest side ≤ 1200px for speed)
    scale = min(1200 / max(x_ext, y_ext), 1.0)
    thumb_hw = (int(y_ext * scale), int(x_ext * scale))

    # Inference
    p_base_all, p_tta_all = [], []
    for i in range(0, N, 512):
        b = feat[i:i+512].to(device)
        p_base_all.append(F.softmax(classifier(b[:, 0, :]),    dim=-1)[:, 1].cpu().numpy())
        p_tta_all.append( F.softmax(classifier(b.mean(dim=1)), dim=-1)[:, 1].cpu().numpy())
    p_base = np.concatenate(p_base_all)
    p_tta  = np.concatenate(p_tta_all)
    delta  = p_tta - p_base

    # Outcomes
    thr = 0.5
    from camelyon17.annot_utils import parse_lesion_polygons, patch_labels
    tumor_region = parse_lesion_polygons(annot_path) if annot_path.exists() else None
    gt_labels = patch_labels(coords, PATCH_SIZE_LV0, tumor_region)
    pred_base = (p_base >= thr).astype(int)
    pred_tta  = (p_tta  >= thr).astype(int)
    outcome_idx = np.full(N, np.nan, dtype=np.float32)
    for i in range(N):
        bc = (pred_base[i] == gt_labels[i])
        tc = (pred_tta[i]  == gt_labels[i])
        if   not bc and tc:  outcome_idx[i] = 0  # corrected
        elif bc and not tc:  outcome_idx[i] = 1  # corrupted
        elif bc and tc:      outcome_idx[i] = 2  # both_correct
        else:                outcome_idx[i] = 3  # both_wrong

    corrected = (outcome_idx == 0).sum()
    corrupted = (outcome_idx == 1).sum()
    print(f"  {slide_id}: N={N}, corrected={corrected}, corrupted={corrupted}, net={corrected-corrupted:+d}")

    grid_kw = dict(thumb_hw=thumb_hw, x_extent=x_ext, y_extent=y_ext)
    grid_delta   = _build_grid(coords, delta,       **grid_kw)
    grid_outcome = _build_grid(coords, outcome_idx, **grid_kw)

    # Tight crop: keep all sizeable tissue masses (>=5% of the largest), then take
    # their joint bounding box. Keeps both lobes of p096 while dropping stray
    # outlier clusters.
    valid = ~np.isnan(grid_outcome)
    labeled, n_comp = nd_label(valid)
    if n_comp > 1:
        comp_sizes = np.bincount(labeled.ravel())
        comp_sizes[0] = 0   # ignore background label
        if slide_id in _SINGLE_LOBE:
            # leftmost sizeable lobe (smallest mean column among comps >=20% of largest)
            big = np.where(comp_sizes >= 0.20 * comp_sizes.max())[0]
            cols = {lab: np.where(labeled == lab)[1].mean() for lab in big}
            keep = [min(cols, key=cols.get)]
        else:
            keep = np.where(comp_sizes >= 0.05 * comp_sizes.max())[0]
        valid = np.isin(labeled, keep)
    rows_with = np.where(valid.any(axis=1))[0]
    cols_with = np.where(valid.any(axis=0))[0]
    pad = 8
    r0 = max(0, rows_with[0]  - pad)
    r1 = min(thumb_hw[0], rows_with[-1] + pad)
    c0 = max(0, cols_with[0]  - pad)
    c1 = min(thumb_hw[1], cols_with[-1] + pad)

    # Apply optional per-slide tight crop (overrides automatic bounding box)
    manual_crop = _PANEL_CROPS.get(slide_id)
    if manual_crop is not None:
        mr0, mr1, mc0, mc1 = manual_crop
        r0, r1, c0, c1 = mr0, mr1, mc0, mc1

    # Crop the bottom to 5 patches below the lowest tumour-boundary point
    polys = _outline_polygons(annot_path, thumb_hw[1] / x_ext, thumb_hw[0] / y_ext)
    if polys:
        patch_h = PATCH_SIZE_LV0 * (thumb_hw[0] / y_ext)
        tumour_bottom = max(pts[:, 1].max() for pts in polys)
        r1 = min(r1, int(tumour_bottom + 5 * patch_h))

    grid_delta   = grid_delta[r0:r1, c0:c1]
    grid_outcome = grid_outcome[r0:r1, c0:c1]

    # Outline polygons adjusted to cropped space
    polys_cropped = [pts - np.array([c0, r0]) for pts in polys]

    # H&E background thumbnail (cropped to same region)
    he_full = _load_he_thumb(slide_id, thumb_hw)
    he_bg   = he_full[r0:r1, c0:c1] if he_full is not None else None
    has_bg  = he_bg is not None

    pfx = slide_id.replace("patient_", "p").replace("_node_", "_node") + f"_{_MODEL}"
    H_crop, W_crop = grid_delta.shape

    # ── Panel 1: ΔP(tumor) ─────────────────────────────────────────────────
    # Raw per-patch delta (no smoothing) to match the outcome panel's blocks.
    max_abs = max(0.05, float(np.nanpercentile(np.abs(delta), 99)))
    fig, ax = plt.subplots(figsize=(5, 5 * H_crop / W_crop))
    ax.set_facecolor("white")
    if has_bg:
        ax.imshow(he_bg, extent=[0, W_crop, H_crop, 0], zorder=0)
    im = ax.imshow(np.ma.masked_invalid(grid_delta), cmap="RdBu_r",
                   vmin=-max_abs, vmax=max_abs, interpolation="nearest",
                   alpha=0.72 if has_bg else 1.0,
                   extent=[0, W_crop, H_crop, 0], zorder=1)
    outline_color = _TUMOR_OUTLINE if has_bg else "black"
    for pts in polys_cropped:
        ax.add_patch(mpatches.Polygon(pts, closed=True, fill=False,
                                      edgecolor=outline_color, linewidth=0.8, alpha=0.95, zorder=2))
    cb = plt.colorbar(im, ax=ax, fraction=0.018, pad=0.015, shrink=0.35,
                      ticks=[-max_abs, 0, max_abs])
    cb.set_label("Δ P(tumor)", fontsize=7)
    cb.ax.set_yticklabels([f"{-max_abs:.1f}", "0", f"{max_abs:.1f}"])
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.4)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        p = out_dir / f"{pfx}_delta_p.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {pfx}_delta_p.pdf/.png")

    # ── Panel 1b: zoomed delta-P on the annotated mass (if crop defined) ────
    zoom_crop = _ZOOM_CROPS.get(slide_id)
    if zoom_crop is not None:
        zr0, zr1, zc0, zc1 = zoom_crop
        # Build a fresh full-res delta grid (before main crop) for the zoom
        grid_delta_full = _build_grid(coords, delta, **dict(thumb_hw=thumb_hw, x_extent=x_ext, y_extent=y_ext))
        zd = grid_delta_full[zr0:zr1, zc0:zc1]
        zh, zw = zd.shape
        fig, ax = plt.subplots(figsize=(4, 4 * zh / zw))
        ax.set_facecolor("white")
        # H&E background if available
        if he_bg is not None:
            he_bg_full = _load_he_thumb(slide_id, thumb_hw)
            if he_bg_full is not None:
                ax.imshow(he_bg_full[zr0:zr1, zc0:zc1], extent=[0, zw, zh, 0], zorder=0)
        im = ax.imshow(np.ma.masked_invalid(zd), cmap="RdBu_r",
                       vmin=-0.4, vmax=0.4,
                       interpolation="nearest",
                       alpha=0.75 if he_bg is not None else 1.0,
                       extent=[0, zw, zh, 0], zorder=1)
        # polys scaled to full thumb space; shift to zoom crop coords
        for pts in polys:
            pts_zoom = pts - np.array([zc0, zr0])
            ax.add_patch(mpatches.Polygon(pts_zoom, closed=True, fill=False,
                                          edgecolor="#1a3a6b", linewidth=1.6, alpha=0.95, zorder=2))
        cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02, shrink=0.55)
        cb.set_label("Δ P(tumor)", fontsize=11)
        cb.ax.tick_params(labelsize=10)
        ax.axis("off")
        fig.tight_layout(pad=0.2)
        fig.savefig(out_dir / f"{pfx}_delta_p_zoom.png", dpi=300,
                    bbox_inches="tight", facecolor="white")
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(out_dir / f"{pfx}_delta_p_zoom.pdf") as pdf:
            pdf.savefig(fig, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved {pfx}_delta_p_zoom.pdf/.png")

    # ── Panel 2: per-patch outcome ─────────────────────────────────────────
    # Build per-pixel RGBA: both_correct nearly transparent so tissue shows through,
    # corrected/corrupted fully opaque so they pop against the H&E background.
    _CLASS_RGBA = [
        (0, _OUTCOME_COLORS["corrected"],    1.0),   # corrected    — green, pops
        (1, _OUTCOME_COLORS["corrupted"],    1.0),   # corrupted    — red, pops
        (3, _OUTCOME_COLORS["both_wrong"],   0.38),  # both_wrong   — grey, faint
        (2, _OUTCOME_COLORS["both_correct"], 0.0),   # both_correct — transparent (tissue shows)
    ]
    rgba = np.zeros((H_crop, W_crop, 4), dtype=np.float32)
    for idx, hex_color, alpha in _CLASS_RGBA:
        mask = (grid_outcome == idx) & ~np.isnan(grid_outcome)
        r_v, g_v, b_v = _hex_to_rgb(hex_color)
        rgba[mask] = [r_v, g_v, b_v, alpha]

    fig, ax = plt.subplots(figsize=(5, 5 * H_crop / W_crop))
    ax.set_facecolor("white")
    if has_bg:
        ax.imshow(he_bg, extent=[0, W_crop, H_crop, 0], zorder=0)
    ax.imshow(rgba, interpolation="nearest", extent=[0, W_crop, H_crop, 0], zorder=1)
    for pts in polys_cropped:
        ax.add_patch(mpatches.Polygon(pts, closed=True, fill=False,
                                      edgecolor=_TUMOR_OUTLINE, linewidth=0.8, alpha=0.95, zorder=2))
    legend_patches = [
        mpatches.Patch(color=_OUTCOME_COLORS["corrected"], label="corr."),
        mpatches.Patch(color=_OUTCOME_COLORS["both_wrong"], label="both wr."),
        mpatches.Patch(color=_OUTCOME_COLORS["corrupted"], label="corrupt."),
    ]
    # legend below the image (outside axes) so it doesn't shrink the tissue
    ax.legend(handles=legend_patches, loc="upper center",
              bbox_to_anchor=(0.5, -0.02), fontsize=13, ncol=3,
              framealpha=0.0, columnspacing=1.0, handletextpad=0.5)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    for ext in ("pdf", "png"):
        p = out_dir / f"{pfx}_outcome.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {pfx}_outcome.pdf/.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def _load_classifier(ckpt_path: Path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_state = ckpt["model"] if "model" in ckpt else ckpt
    in_features = model_state["weight"].shape[1]
    classifier = nn.Linear(in_features, 2).to(device)
    classifier.load_state_dict(model_state)
    classifier.eval()
    return classifier


def main():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for slide_id in SLIDES:
        print(f"\n── {slide_id} ──")
        ckpt_path = _CKPTS.get(slide_id, _CKPT)
        classifier = _load_classifier(ckpt_path, device)
        print(f"  probe: {ckpt_path}")
        src = _HEATMAP_DIR / f"wsi_tta_heatmap_{slide_id}.png"
        if src.exists():
            export_he_panel(slide_id, _OUT_DIR)
        else:
            print(f"  skip H&E panel (no source heatmap)")
        export_vector_panels(slide_id, _OUT_DIR, classifier, device)

    print(f"\nAll panels saved to {_OUT_DIR}/")


if __name__ == "__main__":
    main()
