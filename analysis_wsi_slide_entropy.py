"""
Does per-slide mean patch entropy predict which WSI slides benefit from D4 TTA?
Mirrors the §4.6 patch-level entropy-routing result at the slide level.

Uses ONLY pre-extracted features + saved probes (no WSI re-extraction):
  Camelyon17: checkpoints/camelyon17/lopo_uni/heldout_<patient>.pt  (out-of-sample,
              consistent with patch_dice_camelyon17_lopo.csv)
              features: camelyon17_features/uni/d4_all/<slide>.pt   (N,8,1024)
  PANDA:      checkpoints/panda/patch_probe_uni_seed42/best.pt
              features: panda_features/uni/d4_all/<slide>.pt

Per patch we take the BASE-view (view 0) softmax and its binary Shannon entropy,
normalised by log 2; slide entropy = mean over the slide's patches. We then merge
with the existing per-slide delta-Dice and ask whether high-entropy slides capture
the TTA benefit.

Output: results/wsi_slide_entropy.csv  (+ printed summary)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from scipy import stats

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG2 = np.log(2.0)


def binary_entropy(p1: np.ndarray) -> np.ndarray:
    p1 = np.clip(p1, 1e-12, 1 - 1e-12)
    return (-(p1 * np.log(p1) + (1 - p1) * np.log(1 - p1)) / LOG2)


@torch.no_grad()
def slide_mean_entropy(feat_path: Path, probe: nn.Linear, bs: int = 8192) -> float:
    feat = torch.load(feat_path, weights_only=True)          # (N,8,d) or (N,d)
    if feat.ndim == 2:
        feat = feat.unsqueeze(1)
    N = feat.shape[0]
    ent = []
    for i in range(0, N, bs):
        b = feat[i:i + bs, 0, :].to(device).float()          # base view only
        p1 = F.softmax(probe(b), dim=-1)[:, 1].cpu().numpy()
        ent.append(binary_entropy(p1))
    del feat
    return float(np.concatenate(ent).mean())


def load_linear(state: dict, dim: int = 1024) -> nn.Linear:
    probe = nn.Linear(dim, 2).to(device)
    probe.load_state_dict(state)
    probe.eval()
    return probe


def camelyon17_entropy() -> pd.DataFrame:
    feat_root = Path("camelyon17_features/uni/d4_all")
    probe_root = Path("checkpoints/camelyon17/lopo_uni")
    dice = pd.read_csv("results/patch_dice_camelyon17_lopo.csv")
    dice = dice[dice.n_pos > 0].copy()                       # cancer slides
    rows = []
    for sid in dice.slide_id:
        pat = "_".join(sid.split("_")[:2])                   # patient_XXX
        pf = probe_root / f"heldout_{pat}.pt"
        ff = feat_root / f"{sid}.pt"
        if not (pf.exists() and ff.exists()):
            continue
        probe = load_linear(torch.load(pf, weights_only=False)["model"])
        rows.append({"slide_id": sid, "mean_entropy": slide_mean_entropy(ff, probe)})
    ent = pd.DataFrame(rows)
    d = dice.merge(ent, on="slide_id")
    d["delta_dice"] = d.dice_tta - d.dice_base if "dice_tta" in d.columns else d.f1_tta - d.f1_base
    d["dataset"] = "camelyon17"
    return d[["dataset", "slide_id", "mean_entropy", "delta_dice"]]


def panda_entropy() -> pd.DataFrame:
    feat_root = Path("panda_features/uni/d4_all")
    ckpt = torch.load("checkpoints/panda/patch_probe_uni_seed42/best.pt",
                      weights_only=False, map_location=device)
    probe = load_linear(ckpt["model"])
    dice = pd.read_csv("results/patch_dice_panda.csv")
    dice = dice[dice.n_pos > 0].copy()
    rows = []
    for k, sid in enumerate(dice.slide_id):
        ff = feat_root / f"{sid}.pt"
        if not ff.exists():
            continue
        rows.append({"slide_id": sid, "mean_entropy": slide_mean_entropy(ff, probe)})
        if k % 500 == 0:
            print(f"  panda {k}/{len(dice)}", flush=True)
    ent = pd.DataFrame(rows)
    d = dice.merge(ent, on="slide_id")
    d["delta_dice"] = d.dice_tta - d.dice_base if "dice_tta" in d.columns else d.f1_tta - d.f1_base
    d["dataset"] = "panda"
    return d[["dataset", "slide_id", "mean_entropy", "delta_dice"]]


def summarize(d: pd.DataFrame, name: str):
    print(f"\n=== {name}  (n={len(d)} cancer slides) ===")
    r, p = stats.spearmanr(d.mean_entropy, d.delta_dice)
    print(f"  Spearman(slide mean entropy, ΔDice) = {r:+.3f}  (p={p:.2e})")
    d = d.copy()
    d["tert"] = pd.qcut(d.mean_entropy.rank(method="first"), 3, labels=["low", "mid", "high"])
    print("  mean ΔDice by entropy tertile:")
    for t in ["low", "mid", "high"]:
        print(f"    {t:4s} entropy: ΔDice = {d.delta_dice[d.tert == t].mean() * 100:+.2f}pp")
    # top-k% highest-entropy slides: fraction of total positive benefit captured
    tot = d.delta_dice.clip(lower=0).sum()
    s = d.sort_values("mean_entropy", ascending=False)
    print("  cumulative share of total (positive) ΔDice captured by top-k% entropy slides:")
    for k in [0.1, 0.2, 0.3, 0.5]:
        nk = max(1, int(round(k * len(s))))
        cap = s.head(nk).delta_dice.clip(lower=0).sum() / tot if tot > 0 else float("nan")
        print(f"    top {int(k*100):2d}%: {cap*100:4.0f}%")


def main():
    parts = []
    print("Camelyon17 (LOPO probes)...")
    cam = camelyon17_entropy(); parts.append(cam)
    print("PANDA...")
    pan = panda_entropy(); parts.append(pan)
    out = pd.concat(parts, ignore_index=True)
    out.to_csv("results/wsi_slide_entropy.csv", index=False)
    print(f"\nSaved results/wsi_slide_entropy.csv ({len(out)} slides)")
    summarize(cam, "Camelyon17")
    summarize(pan, "PANDA")


if __name__ == "__main__":
    main()
