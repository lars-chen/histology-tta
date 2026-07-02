"""
MIL-null mechanism: does ABMIL down-weight exactly the patches D4 TTA changes?

For each annotated Camelyon17 slide we compute, on cached UNI features (no
re-extraction):
  - ABMIL attention weight per patch  (trained ABMIL, view-0 features)
  - whether D4 TTA flips that patch's prediction  (out-of-sample LOPO patch probe:
    base = argmax probe(view0)  vs  d4 = argmax probe(mean of 8 views))

If the patches TTA flips carry little attention mass, their corrections cannot
move the bag logit -- the mechanism behind the slide-level MIL null.

Metric (pooled over slides):
  flip_rate              fraction of patches flipped by TTA
  attn_mass_flipped      share of total ABMIL attention on flipped patches
  under-weighting        attn_mass_flipped / flip_rate   (<1 => under-weighted)
  median attention percentile (within slide): flipped vs unflipped

Output: results/mil_attention_vs_flip.csv (+ printed summary)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from camelyon17.model_abmil import ABMIL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FEAT = Path("camelyon17_features/uni/d4_all")
PROBE = Path("checkpoints/camelyon17/lopo_uni")
ABMIL_SEEDS = [0, 1, 2, 3, 4]   # average attention over the trained bags


def load_abmils():
    models = []
    for s in ABMIL_SEEDS:
        f = Path(f"checkpoints/camelyon17/abmil_uni_none_seed{s}/best.pt")
        if not f.exists():
            continue
        ckpt = torch.load(f, weights_only=False, map_location=device)
        m = ABMIL(in_dim=1024).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        models.append(m)
    return models


def load_probe(patient):
    f = PROBE / f"heldout_{patient}.pt"
    if not f.exists():
        return None
    p = nn.Linear(1024, 2).to(device)
    p.load_state_dict(torch.load(f, weights_only=False)["model"])
    p.eval()
    return p


@torch.no_grad()
def process_slide(sid, abmils, probe):
    feat = torch.load(FEAT / f"{sid}.pt", weights_only=True)   # (N,8,1024)
    if feat.ndim == 2:
        feat = feat.unsqueeze(1)
    v0 = feat[:, 0, :].to(device).float()
    mean8 = feat.mean(1).to(device).float()

    # TTA flip from the out-of-sample patch probe
    base = probe(v0).argmax(1)
    d4 = probe(mean8).argmax(1)
    flipped = (base != d4).cpu().numpy()

    # ABMIL attention (mean over seeds), normalised per slide
    attn = np.zeros(len(v0))
    for m in abmils:
        _, a = m(v0)
        attn += a.cpu().numpy()
    attn /= max(len(abmils), 1)
    attn = attn / attn.sum()                       # attention mass, sums to 1
    pct = stats.rankdata(attn) / len(attn)         # within-slide percentile

    return pd.DataFrame({"slide_id": sid, "attn": attn, "attn_pct": pct,
                         "flipped": flipped})


def main():
    abmils = load_abmils()
    print(f"Loaded {len(abmils)} ABMIL seeds")
    slides = pd.read_csv("results/patch_dice_camelyon17_lopo.csv").slide_id.tolist()
    frames = []
    for sid in slides:
        pat = "_".join(sid.split("_")[:2])
        probe = load_probe(pat)
        if probe is None or not (FEAT / f"{sid}.pt").exists():
            continue
        frames.append(process_slide(sid, abmils, probe))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv("results/mil_attention_vs_flip.csv", index=False)
    print(f"Saved results/mil_attention_vs_flip.csv ({len(df)} patches, "
          f"{df.slide_id.nunique()} slides)\n")

    # --- pooled summary ---
    flip_rate = df.flipped.mean()
    attn_mass_flipped = df.attn[df.flipped].sum() / df.attn.sum()
    print("="*60)
    print(f"Patches flipped by TTA:            {flip_rate*100:.1f}%")
    print(f"ABMIL attention mass on flipped:   {attn_mass_flipped*100:.1f}%")
    print(f"Under-weighting ratio (mass/rate): {attn_mass_flipped/flip_rate:.2f}x  (<1 = ignored)")
    print(f"Median attention percentile  flipped:   {df.attn_pct[df.flipped].median():.3f}")
    print(f"Median attention percentile  unflipped: {df.attn_pct[~df.flipped].median():.3f}")
    # fraction of flipped patches that fall in the top attention decile
    top = df.attn_pct >= 0.9
    print(f"\nOf the top-attention decile, fraction flipped: {df.flipped[top].mean()*100:.1f}%"
          f"  (vs {flip_rate*100:.1f}% overall)")
    # per-slide: attention mass on flipped, to show consistency
    per = df.groupby("slide_id").apply(
        lambda g: g.attn[g.flipped].sum() / g.attn.sum() if g.flipped.any() else 0.0,
        include_groups=False)
    perr = df.groupby("slide_id").flipped.mean()
    print(f"\nPer-slide (median over {len(per)} slides): flip rate {perr.median()*100:.1f}%, "
          f"attention mass on flipped {per.median()*100:.1f}%")


if __name__ == "__main__":
    main()
