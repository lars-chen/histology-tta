"""
Patch-level dice / sensitivity / specificity: baseline vs D4-TTA.

Compares binary predictions (view-0 = baseline, mean-of-8-views = TTA) against
ground-truth patch labels for PANDA (binary cancer) and Camelyon17 (binary tumor).

Outputs:
  results/patch_dice_panda.csv
  results/patch_dice_camelyon17.csv
  figures/plot_patch_dice.png

Usage:
  python eval_patch_dice.py                  # both datasets
  python eval_patch_dice.py --dataset panda
  python eval_patch_dice.py --dataset camelyon17
"""

from __future__ import annotations
import os

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── dataset-specific paths ────────────────────────────────────────────────────

PANDA = dict(
    features_dir = Path("panda_features/uni/d4_all"),
    patches_dir  = Path(os.path.join(os.environ.get("PANDA_DIR", "data/panda"), "clam_patches/patches")),
    checkpoint   = Path("checkpoints/panda/patch_probe_uni_seed42/best.pt"),
    patch_size   = 256,
    embed_dim    = 1024,
    n_classes    = 2,
    pos_label    = "cancer",
)

CAMELYON = dict(
    features_dir = Path("camelyon17_features/uni/d4_all"),
    patches_dir  = Path(os.path.join(os.environ.get("CAMELYON17_PATCHED_DIR", "data/camelyon17_patched"), "patches")),
    checkpoint   = Path("checkpoints/camelyon17/patch_probe_uni_seed42/best.pt"),
    annot_dir    = Path(os.path.join(os.environ.get("CAMELYON17_DIR", "data/camelyon17"), "training/lesion_annotations")),
    patch_size   = 512,
    embed_dim    = 1024,
    n_classes    = 2,
    pos_label    = "tumor",
)


def dice(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else float("nan")


@torch.no_grad()
def process_slide(
    slide_id: str,
    features_dir: Path,
    patches_dir: Path,
    gt_fn,            # callable(slide_id, coords) → np.ndarray of 0/1
    probe: nn.Module,
    device: torch.device,
    batch_size: int = 1024,
) -> dict | None:
    feat_path = features_dir / f"{slide_id}.pt"
    h5_path   = patches_dir  / f"{slide_id}.h5"
    if not feat_path.exists() or not h5_path.exists():
        return None

    feat = torch.load(feat_path, weights_only=True)
    if feat.ndim == 2:
        feat = feat.unsqueeze(1)

    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]

    gt = gt_fn(slide_id, coords).astype(int)
    N  = feat.shape[0]

    p_base_list, p_tta_list = [], []
    for i in range(0, N, batch_size):
        b = feat[i : i + batch_size].to(device)
        p_base_list.append(F.softmax(probe(b[:, 0, :]),    dim=-1)[:, 1].cpu().numpy())
        p_tta_list.append( F.softmax(probe(b.mean(dim=1)), dim=-1)[:, 1].cpu().numpy())

    p_base = np.concatenate(p_base_list)
    p_tta  = np.concatenate(p_tta_list)
    pred_b = (p_base >= 0.5).astype(int)
    pred_t = (p_tta  >= 0.5).astype(int)

    def _metrics(pred):
        tp = int(((pred == 1) & (gt == 1)).sum())
        fp = int(((pred == 1) & (gt == 0)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum())
        tn = int(((pred == 0) & (gt == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        return tp, fp, fn, tn, dice(tp, fp, fn), sens, spec

    tp_b, fp_b, fn_b, tn_b, d_b, sens_b, spec_b = _metrics(pred_b)
    tp_t, fp_t, fn_t, tn_t, d_t, sens_t, spec_t = _metrics(pred_t)

    n_pos = int(gt.sum())
    return dict(
        slide_id    = slide_id,
        n_patches   = N,
        n_pos       = n_pos,
        pos_frac    = n_pos / N if N > 0 else 0.0,
        # baseline
        dice_base   = d_b,
        sens_base   = sens_b,
        spec_base   = spec_b,
        tp_base     = tp_b, fp_base = fp_b, fn_base = fn_b,
        # tta
        dice_tta    = d_t,
        sens_tta    = sens_t,
        spec_tta    = spec_t,
        tp_tta      = tp_t, fp_tta  = fp_t, fn_tta  = fn_t,
        # deltas
        delta_dice  = d_t - d_b,
        delta_sens  = sens_t - sens_b,
        delta_spec  = spec_t - spec_b,
        corrected   = int(((pred_b != gt) & (pred_t == gt)).sum()),
        corrupted   = int(((pred_b == gt) & (pred_t != gt)).sum()),
    )


# ── per-dataset runners ───────────────────────────────────────────────────────

def run_panda(device, batch_size):
    from panda.annot_utils import patch_labels as _pl
    cfg = PANDA
    ckpt  = torch.load(cfg["checkpoint"], weights_only=False, map_location=device)
    probe = nn.Linear(cfg["embed_dim"], cfg["n_classes"]).to(device)
    probe.load_state_dict(ckpt["model"])
    probe.eval()

    ps = cfg["patch_size"]
    def gt_fn(sid, coords):
        return _pl(sid, coords, ps)

    feat_dir  = cfg["features_dir"]
    slide_ids = sorted(p.stem for p in feat_dir.glob("*.pt"))
    print(f"PANDA: {len(slide_ids)} slides with features")

    rows = []
    for i, sid in enumerate(slide_ids):
        r = process_slide(sid, feat_dir, cfg["patches_dir"], gt_fn,
                          probe, device, batch_size)
        if r is None:
            continue
        rows.append(r)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(slide_ids)}] processed {len(rows)} slides", flush=True)

    return pd.DataFrame(rows)


def run_camelyon(device, batch_size):
    from camelyon17.annot_utils import parse_lesion_polygons, patch_labels as _pl
    cfg = CAMELYON
    ckpt  = torch.load(cfg["checkpoint"], weights_only=False, map_location=device)
    probe = nn.Linear(cfg["embed_dim"], cfg["n_classes"]).to(device)
    probe.load_state_dict(ckpt["model"])
    probe.eval()

    annot_dir = cfg["annot_dir"]
    ps        = cfg["patch_size"]

    def gt_fn(sid, coords):
        xml = annot_dir / f"{sid}.xml"
        tumor = parse_lesion_polygons(xml) if xml.exists() else None
        return _pl(coords, ps, tumor)

    feat_dir  = cfg["features_dir"]
    slide_ids = sorted(p.stem for p in feat_dir.glob("*.pt"))
    print(f"Camelyon17: {len(slide_ids)} slides with features")

    rows = []
    for i, sid in enumerate(slide_ids):
        r = process_slide(sid, feat_dir, cfg["patches_dir"], gt_fn,
                          probe, device, batch_size)
        if r is None:
            continue
        rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(slide_ids)}] processed {len(rows)} slides", flush=True)

    return pd.DataFrame(rows)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot(df_panda, df_cam, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fig.suptitle("Patch-level binary classification: baseline vs D4-TTA", fontsize=11)

    datasets = [
        ("PANDA (cancer)", df_panda),
        ("Camelyon17 (tumor)", df_cam),
    ]

    for row_idx, (label, df) in enumerate(datasets):
        if df is None or len(df) == 0:
            continue

        # filter to slides that have at least 1 positive patch
        dfp = df[df.n_pos > 0].copy()

        ax = axes[row_idx]

        # ── Panel 0: per-slide dice, baseline vs TTA ──────────────────────────
        ax[0].scatter(dfp.dice_base, dfp.dice_tta,
                      c=dfp.delta_dice, cmap="RdBu_r",
                      vmin=-0.2, vmax=0.2, s=15, alpha=0.7, linewidths=0)
        lim = [0, 1]
        ax[0].plot(lim, lim, "k--", linewidth=0.8, alpha=0.5)
        ax[0].set_xlabel("Dice (baseline)", fontsize=9)
        ax[0].set_ylabel("Dice (TTA)", fontsize=9)
        ax[0].set_xlim(lim); ax[0].set_ylim(lim)
        ax[0].set_title(f"{label}\nDice: baseline vs TTA (n={len(dfp)} slides)", fontsize=9)
        # summary text
        mean_b = dfp.dice_base.mean()
        mean_t = dfp.dice_tta.mean()
        ax[0].text(0.05, 0.93, f"mean base={mean_b:.3f}\nmean TTA={mean_t:.3f}\nΔ={mean_t-mean_b:+.3f}",
                   transform=ax[0].transAxes, fontsize=7, va="top",
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        # ── Panel 1: Δdice vs positive fraction ──────────────────────────────
        ax[1].scatter(dfp.pos_frac * 100, dfp.delta_dice,
                      c=dfp.n_pos, cmap="viridis_r",
                      s=15, alpha=0.7, linewidths=0)
        ax[1].axhline(0, color="grey", linewidth=0.8, linestyle="--")
        ax[1].set_xlabel("Positive patch fraction (%)", fontsize=9)
        ax[1].set_ylabel("Δ Dice (TTA − baseline)", fontsize=9)
        ax[1].set_title("Δ Dice vs class imbalance", fontsize=9)

        # ── Panel 2: sensitivity and specificity ─────────────────────────────
        metrics = ["sens", "spec"]
        labels  = ["Sensitivity", "Specificity"]
        x = np.array([0, 1])
        width = 0.35
        for mi, (m, ml) in enumerate(zip(metrics, labels)):
            b_vals = dfp[f"{m}_base"].dropna()
            t_vals = dfp[f"{m}_tta"].dropna()
            ax[2].bar(x[mi] - width/2, b_vals.mean(), width, label=f"{ml} base" if row_idx == 0 else "",
                      color="#3498db", alpha=0.8, yerr=b_vals.std(), capsize=3,
                      error_kw=dict(elinewidth=0.8))
            ax[2].bar(x[mi] + width/2, t_vals.mean(), width, label=f"{ml} TTA" if row_idx == 0 else "",
                      color="#e74c3c", alpha=0.8, yerr=t_vals.std(), capsize=3,
                      error_kw=dict(elinewidth=0.8))
        ax[2].set_xticks(x)
        ax[2].set_xticklabels(labels, fontsize=9)
        ax[2].set_ylim(0, 1.05)
        ax[2].set_ylabel("Mean ± std across slides", fontsize=9)
        ax[2].set_title("Sensitivity & specificity", fontsize=9)
        ax[2].legend(fontsize=7)
        for xi, (m, ml) in enumerate(zip(metrics, labels)):
            b = dfp[f"{m}_base"].mean()
            t = dfp[f"{m}_tta"].mean()
            ax[2].text(xi, max(b, t) + 0.04, f"Δ{t-b:+.3f}", ha="center", fontsize=7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}")


def print_summary(name: str, df: pd.DataFrame):
    if df is None or len(df) == 0:
        return
    dfp = df[df.n_pos > 0]
    print(f"\n── {name} ({len(df)} slides, {len(dfp)} with positives) ──")
    print(f"  Patches total:    {df.n_patches.sum():,}  pos: {df.n_pos.sum():,}  ({100*df.n_pos.sum()/df.n_patches.sum():.1f}%)")
    for col, label in [("dice_base","Dice"), ("sens_base","Sensitivity"), ("spec_base","Specificity")]:
        col_t = col.replace("base", "tta")
        b = dfp[col].mean(); t = dfp[col_t].mean()
        print(f"  {label:14s}  base={b:.4f}  TTA={t:.4f}  Δ={t-b:+.4f}")
    n_improved = (dfp.delta_dice > 0).sum()
    n_hurt     = (dfp.delta_dice < 0).sum()
    print(f"  Slides improved:  {n_improved}/{len(dfp)}  ({100*n_improved/len(dfp):.0f}%)")
    print(f"  Slides hurt:      {n_hurt}/{len(dfp)}  ({100*n_hurt/len(dfp):.0f}%)")
    total_corr = df.corrected.sum(); total_corru = df.corrupted.sum()
    print(f"  Total corrected:  {total_corr:,}  corrupted: {total_corru:,}  net: {total_corr-total_corru:+,}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",    choices=["panda","camelyon17","both"], default="both")
    p.add_argument("--device",     default="cuda")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--out_dir",    type=Path, default=Path("figures"))
    p.add_argument("--results_dir",type=Path, default=Path("results"))
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.results_dir.mkdir(parents=True, exist_ok=True)

    df_panda = df_cam = None

    if args.dataset in ("panda", "both"):
        df_panda = run_panda(device, args.batch_size)
        csv = args.results_dir / "patch_dice_panda.csv"
        df_panda.to_csv(csv, index=False)
        print(f"Saved → {csv}")
        print_summary("PANDA", df_panda)

    if args.dataset in ("camelyon17", "both"):
        df_cam = run_camelyon(device, args.batch_size)
        csv = args.results_dir / "patch_dice_camelyon17.csv"
        df_cam.to_csv(csv, index=False)
        print(f"Saved → {csv}")
        print_summary("Camelyon17", df_cam)

    if df_panda is not None or df_cam is not None:
        plot(df_panda, df_cam, args.out_dir / "plot_patch_dice.png")


if __name__ == "__main__":
    main()
