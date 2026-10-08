"""
Leave-one-patient-out (LOPO) patch-probe evaluation on the 50 fully-annotated
Camelyon17 slides. For each held-out patient, a fresh linear probe is trained on
every *other* patient's slides, then used to score the held-out patient's slides
— so every slide receives an out-of-sample prediction with no patient leakage.

Differences vs the original single-split protocol (train_patch_probe.py):
  * leave-one-patient-out instead of a single 40/10 split → every analysed slide
    is held out exactly once;
  * single class-balancing (WeightedRandomSampler only) instead of sampler + a
    ~300x cross-entropy weight, which made the base probe over-sensitive.

Outputs (model column included so 6 backbones can be concatenated):
  results/patch_dice_camelyon17_lopo.csv          — per-slide Dice / sens / spec
  results/camelyon17_boundary_dist_lopo.csv       — per-patch outcome + dist_um

Usage:
  python -m camelyon17.lopo_patch_probe --model uni
  python -m camelyon17.lopo_patch_probe --model uni phikon2 virchow2 gigapath ...
"""
from __future__ import annotations
import os

import argparse
import random
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import shapely
import torch
import torch.nn as nn
import torch.nn.functional as F
from camelyon17.annot_utils import parse_lesion_polygons, patch_labels

_ANNOT_DIR = Path(os.path.join(os.environ.get("CAMELYON17_DIR", "data/camelyon17"), "training/lesion_annotations"))
_DATA_ROOT = Path(os.environ.get("CAMELYON17_PATCHED_DIR", "data/camelyon17_patched"))
_PATCH_LV0 = 512
_MPP       = 0.243   # µm / pixel at level 0


def _patient(slide_id: str) -> str:
    return "_".join(slide_id.split("_")[:2])


# ---------------------------------------------------------------------------
# Preload: cache features, labels and (probe-independent) margin distances once
# ---------------------------------------------------------------------------

def preload(model: str, device: torch.device) -> dict[str, dict]:
    """Cache only view-0 features (for training) + labels + margin distances.
    The full 8-view tensor is loaded on demand in score_slide (each slide is
    scored exactly once), keeping peak RAM low enough for the largest backbones."""
    feat_root = Path(f"camelyon17_features/{model}/d4_all")
    patches   = _DATA_ROOT / "patches"
    ids = sorted(p.stem for p in _ANNOT_DIR.glob("*.xml")
                 if (feat_root / f"{p.stem}.pt").exists())

    cache: dict[str, dict] = {}
    for sid in ids:
        feat = torch.load(feat_root / f"{sid}.pt", weights_only=True)  # (N,8,d)
        if feat.ndim == 2:
            feat = feat.unsqueeze(1)
        v0 = feat[:, 0, :].float().contiguous()
        del feat
        with h5py.File(patches / f"{sid}.h5", "r") as f:
            coords = f["coords"][:]
        tumor = parse_lesion_polygons(_ANNOT_DIR / f"{sid}.xml")
        gt = patch_labels(coords, _PATCH_LV0, tumor)

        # signed distance (µm) from each patch centre to nearest tumour boundary
        half = _PATCH_LV0 / 2
        pts  = shapely.points(coords[:, 0] + half, coords[:, 1] + half)
        if tumor is not None:
            d_px   = shapely.distance(tumor.boundary, pts)
            inside = shapely.contains(tumor, pts)
            d_px   = np.where(inside, -d_px, d_px)
        else:
            d_px = np.full(len(coords), np.nan)

        cache[sid] = dict(
            v0=v0, gt=gt.astype(int), dist_um=d_px * _MPP,
            feat_path=feat_root / f"{sid}.pt",
        )
    print(f"  preloaded {len(cache)} slides "
          f"({sum(c['v0'].shape[0] for c in cache.values()):,} patches)", flush=True)
    return cache


# ---------------------------------------------------------------------------
# Train one probe on a set of slides (single balancing: sampler only)
# ---------------------------------------------------------------------------

def train_probe(train_ids, cache, embed_dim, device, *, epochs, lr, wd,
                batch_size, seed, train_frac=1.0):
    """GPU-resident balanced-minibatch training for the linear probe.
    The whole training matrix lives on the GPU; each step draws batch_size/2
    positive and batch_size/2 negative indices (class-balanced, equivalent to
    the old WeightedRandomSampler) without any DataLoader/host-transfer cost.
    train_frac<1.0 stratified-subsamples the *distinct* training patches per
    class (weakens the baseline without starving tumour) for data ablation."""
    g = torch.Generator(device=device).manual_seed(seed)
    X = torch.cat([cache[s]["v0"] for s in train_ids], dim=0).to(device)
    y = torch.from_numpy(np.concatenate([cache[s]["gt"] for s in train_ids])).long().to(device)

    pos_idx = (y == 1).nonzero(as_tuple=True)[0]
    neg_idx = (y == 0).nonzero(as_tuple=True)[0]
    if train_frac < 1.0:
        # keep a stratified subset of distinct patches per class
        kp = max(int(len(pos_idx) * train_frac), 1)
        kn = max(int(len(neg_idx) * train_frac), 1)
        pos_idx = pos_idx[torch.randperm(len(pos_idx), generator=g, device=device)[:kp]]
        neg_idx = neg_idx[torch.randperm(len(neg_idx), generator=g, device=device)[:kn]]
    half = max(batch_size // 2, 1)
    # scale passes with the effective (subsampled) data so "less data" is not
    # confounded with extra gradient steps on the same patches
    n_eff = len(pos_idx) + len(neg_idx)
    steps_per_epoch = max(n_eff // batch_size, 1)

    probe = nn.Linear(embed_dim, 2).to(device)
    opt   = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    probe.train()
    for _ in range(epochs):
        for _ in range(steps_per_epoch):
            pi = pos_idx[torch.randint(len(pos_idx), (half,), generator=g, device=device)]
            ni = neg_idx[torch.randint(len(neg_idx), (half,), generator=g, device=device)]
            idx = torch.cat([pi, ni])
            opt.zero_grad()
            F.cross_entropy(probe(X[idx]), y[idx]).backward()
            opt.step()
        sched.step()
    probe.eval()
    del X, y
    return probe


# ---------------------------------------------------------------------------
# Score one held-out slide → per-slide metrics + per-patch outcome rows
# ---------------------------------------------------------------------------

def _dice(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else float("nan")


@torch.no_grad()
def score_slide(sid, cache, probe, device, batch_size=8192):
    c = cache[sid]
    gt = c["gt"]
    feat = torch.load(c["feat_path"], weights_only=True)   # (N,8,d), on demand
    if feat.ndim == 2:
        feat = feat.unsqueeze(1)
    N = feat.shape[0]
    pb, pt = [], []
    for i in range(0, N, batch_size):
        b = feat[i:i + batch_size].to(device).float()
        pb.append(F.softmax(probe(b[:, 0, :]),    dim=-1)[:, 1].cpu().numpy())
        pt.append(F.softmax(probe(b.mean(dim=1)), dim=-1)[:, 1].cpu().numpy())
    del feat
    pred_b = (np.concatenate(pb) >= 0.5).astype(int)
    pred_t = (np.concatenate(pt) >= 0.5).astype(int)

    def m(pred):
        tp = int(((pred == 1) & (gt == 1)).sum())
        fp = int(((pred == 1) & (gt == 0)).sum())
        fn = int(((pred == 0) & (gt == 1)).sum())
        tn = int(((pred == 0) & (gt == 0)).sum())
        sens = tp / (tp + fn) if tp + fn else float("nan")
        spec = tn / (tn + fp) if tn + fp else float("nan")
        return tp, fp, fn, _dice(tp, fp, fn), sens, spec

    tp_b, fp_b, fn_b, d_b, se_b, sp_b = m(pred_b)
    tp_t, fp_t, fn_t, d_t, se_t, sp_t = m(pred_t)
    n_pos = int(gt.sum())
    slide_row = dict(
        slide_id=sid, n_patches=N, n_pos=n_pos,
        f1_base=d_b, sens_base=se_b, spec_base=sp_b,
        f1_tta=d_t,  sens_tta=se_t,  spec_tta=sp_t,
        dice_base=d_b, dice_tta=d_t,
        delta_dice=d_t - d_b, delta_sens=se_t - se_b, delta_spec=sp_t - sp_b,
        corrected=int(((pred_b != gt) & (pred_t == gt)).sum()),
        corrupted=int(((pred_b == gt) & (pred_t != gt)).sum()),
    )

    bc = pred_b == gt; tc = pred_t == gt
    outcome = np.where(~bc & tc, "corrected",
              np.where(bc & ~tc, "corrupted",
              np.where(bc & tc, "both_correct", "both_wrong")))
    patch_rows = dict(slide_id=sid, gt=gt, outcome=outcome, dist_um=c["dist_um"])
    return slide_row, patch_rows


# ---------------------------------------------------------------------------
# LOPO driver
# ---------------------------------------------------------------------------

def run_model(model, device, args):
    import pandas as pd
    print(f"\n=== {model} ===")
    cache = preload(model, device)
    embed_dim = next(iter(cache.values()))["v0"].shape[-1]

    by_patient = defaultdict(list)
    for sid in cache:
        by_patient[_patient(sid)].append(sid)
    patients = sorted(by_patient)

    save_root = None
    if args.save_probes:
        tag = f"lopo_{model}" if args.probe_frac >= 1.0 else f"lopo_{model}_frac{args.probe_frac}"
        save_root = Path("checkpoints/camelyon17") / tag
        save_root.mkdir(parents=True, exist_ok=True)

    slide_rows, patch_frames = [], []
    for k, pat in enumerate(patients, 1):
        held = by_patient[pat]
        train_ids = [s for s in cache if _patient(s) != pat]
        probe = train_probe(train_ids, cache, embed_dim, device,
                            epochs=args.epochs, lr=args.lr, wd=args.weight_decay,
                            batch_size=args.batch_size, seed=args.seed,
                            train_frac=args.probe_frac)
        if save_root is not None:
            # key by held-out patient so wsi_tta_heatmap can load the
            # out-of-sample probe for any of that patient's slides
            torch.save({"model": probe.state_dict(), "held_out_patient": pat,
                        "held_slides": held},
                       save_root / f"heldout_{pat}.pt")
        for sid in held:
            srow, prow = score_slide(sid, cache, probe, device)
            srow["model"] = model
            slide_rows.append(srow)
            pf = pd.DataFrame(prow); pf["model"] = model
            patch_frames.append(pf)
        if k % 10 == 0:
            print(f"  [{k}/{len(patients)}] patients done", flush=True)

    return pd.DataFrame(slide_rows), pd.concat(patch_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Data-ablation driver: sweep training-data fractions, accumulate per-slide
# Dice + a compact margin histogram (no per-patch dump) per fraction.
# ---------------------------------------------------------------------------

ABL_BINS = np.arange(-1000, 8500, 500)


def run_ablation(model, device, args):
    import pandas as pd
    print(f"\n=== {model} (data ablation) ===")
    cache = preload(model, device)
    embed_dim = next(iter(cache.values()))["v0"].shape[-1]

    by_patient = defaultdict(list)
    for sid in cache:
        by_patient[_patient(sid)].append(sid)
    patients = sorted(by_patient)

    slide_rows = []
    # margin histogram: (train_frac, bin) -> [corrected, corrupted, total]
    hist = defaultdict(lambda: np.zeros((len(ABL_BINS) - 1, 3)))

    for frac in args.train_frac:
        n_train_patches = None
        for k, pat in enumerate(patients, 1):
            held = by_patient[pat]
            train_ids = [s for s in cache if _patient(s) != pat]
            probe = train_probe(train_ids, cache, embed_dim, device,
                                epochs=args.epochs, lr=args.lr, wd=args.weight_decay,
                                batch_size=args.batch_size, seed=args.seed,
                                train_frac=frac)
            for sid in held:
                srow, prow = score_slide(sid, cache, probe, device)
                srow["model"] = model; srow["train_frac"] = frac
                slide_rows.append(srow)
                # bin per-patch outcomes by margin distance
                dist = prow["dist_um"]; out = prow["outcome"]
                bi = np.digitize(dist, ABL_BINS) - 1
                valid = (bi >= 0) & (bi < len(ABL_BINS) - 1)
                h = hist[frac]
                np.add.at(h[:, 0], bi[valid & (out == "corrected")], 1)
                np.add.at(h[:, 1], bi[valid & (out == "corrupted")], 1)
                np.add.at(h[:, 2], bi[valid], 1)
        sub = pd.DataFrame([r for r in slide_rows if r["train_frac"] == frac])
        t = sub[sub.n_pos >= 10]
        dd = (t.f1_tta - t.f1_base) * 100
        print(f"  frac={frac:5.2f}: baseDice {t.f1_base.mean()*100:5.1f}  "
              f"ΔDice {dd.mean():+5.2f}  improved {(dd>0.01).sum()}/{(dd<-0.01).sum()}",
              flush=True)

    # flatten margin histogram
    centers = (ABL_BINS[:-1] + ABL_BINS[1:]) / 2
    mrows = []
    for frac, h in hist.items():
        for bi in range(len(centers)):
            mrows.append(dict(model=model, train_frac=frac,
                              dist_um=centers[bi],
                              corrected=int(h[bi, 0]), corrupted=int(h[bi, 1]),
                              total=int(h[bi, 2])))
    return pd.DataFrame(slide_rows), pd.DataFrame(mrows)


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["uni"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_probes", action="store_true",
                    help="save each fold's probe keyed by held-out patient "
                         "(for out-of-sample WSI heatmap rendering)")
    ap.add_argument("--probe_frac", type=float, default=1.0,
                    help="train_frac for the non-ablation run_model path "
                         "(<1 trains a deliberately weaker probe for illustration)")
    ap.add_argument("--ablation", action="store_true",
                    help="sweep --train_frac levels; write ablation CSVs instead")
    ap.add_argument("--train_frac", nargs="+", type=float,
                    default=[0.02, 0.05, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dice_csv", type=Path,
                    default=Path("results/patch_dice_camelyon17_lopo.csv"))
    ap.add_argument("--bd_csv", type=Path,
                    default=Path("results/camelyon17_boundary_dist_lopo.csv"))
    ap.add_argument("--abl_dice_csv", type=Path,
                    default=Path("results/patch_dice_camelyon17_ablation.csv"))
    ap.add_argument("--abl_margin_csv", type=Path,
                    default=Path("results/camelyon17_margin_ablation.csv"))
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    if args.ablation:
        dice_all, marg_all = [], []
        for model in args.model:
            sdf, mdf = run_ablation(model, device, args)
            dice_all.append(sdf); marg_all.append(mdf)
        args.abl_dice_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(dice_all, ignore_index=True).to_csv(args.abl_dice_csv, index=False)
        pd.concat(marg_all, ignore_index=True).to_csv(args.abl_margin_csv, index=False)
        print(f"\nWrote {args.abl_dice_csv}\nWrote {args.abl_margin_csv}")
        return

    dice_all, bd_all = [], []
    for model in args.model:
        sdf, pdf = run_model(model, device, args)
        dice_all.append(sdf); bd_all.append(pdf)
        # per-model summary on tumour-bearing slides
        t = sdf[sdf.n_pos >= 10].copy()
        t["dD"] = (t.f1_tta - t.f1_base) * 100
        print(f"  {model}: {len(t)} tumour slides  meanΔDice {t.dD.mean():+.2f}  "
              f"max {t.dD.max():+.1f}  improved {(t.dD>0.01).sum()}/{(t.dD<-0.01).sum()}")

    args.dice_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(dice_all, ignore_index=True).to_csv(args.dice_csv, index=False)
    pd.concat(bd_all, ignore_index=True).to_csv(args.bd_csv, index=False)
    print(f"\nWrote {args.dice_csv}\nWrote {args.bd_csv}")


if __name__ == "__main__":
    main()
