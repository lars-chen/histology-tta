#!/usr/bin/env python3
"""
Embedding-space orbit tightness evaluation.

For each test sample, applies all 8 D4 transforms, extracts backbone embeddings
(before the classifier), L2-normalises them, and computes the orbit centroid norm:

    OCN(x) = || mean_g( normalise(phi(g·x)) ) ||_2  ∈ [0, 1]

OCN = 1  → all 8 views map to the same direction (perfect invariance)
OCN = 1/√8 ≈ 0.354 → views are orthogonal (random baseline in high-D)

Equivalently, mean pairwise cosine similarity:
    MPCS = (8·OCN² - 1) / 7  ∈ [-1/7, 1]

This metric is measured BEFORE the classifier, refuting the argument that
the linear probe collapses orientation-dependent structure.

Optionally saves raw subsampled embeddings for UMAP orbit visualisation.

Usage:
    python eval_orbit_tightness.py \\
        --model phikon \\
        --checkpoint checkpoints/tcga-ut_phikon_frozen_aug_seed0_best.pt \\
        --dataset tcga-ut \\
        --backbone_mode frozen \\
        --seed 0 \\
        --append

    # Also save embeddings for UMAP (2000-sample subsample):
    python eval_orbit_tightness.py \\
        --model gigapath \\
        --checkpoint checkpoints/tcga-ut_gigapath_frozen_aug_seed0_best.pt \\
        --dataset tcga-ut \\
        --backbone_mode frozen \\
        --save_embeddings embeddings/tcga-ut_gigapath_frozen_seed0
"""

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.data import HistoDataset, MHISTDataset
from data.transforms import get_tta_transforms
from models import get_model
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Model-type lookup (mirrors compile_results.py)
# ---------------------------------------------------------------------------

_HISTOLOGY_MODELS = {
    "phikon", "phikon2", "uni", "uni2",
    "virchow", "virchow2", "gigapath", "hoptimus",
}
_EQUIVARIANT_MODELS = {"d4wrn"}


def _model_type(model_name: str) -> str:
    if model_name in _HISTOLOGY_MODELS:
        return "histology"
    if model_name in _EQUIVARIANT_MODELS:
        return "equivariant"
    return "general"


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Core: compute orbit centroid norms from backbone embeddings
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_orbit_tightness(
    model,
    dataset,
    d4_transforms,
    device: str,
    batch_size: int,
    num_workers: int,
    amp: bool = True,
    save_embeddings_path: str | None = None,
    emb_subsample: int = 2000,
):
    """
    For every test sample, apply all 8 D4 transforms, extract L2-normalised
    backbone embeddings, and compute the orbit centroid norm per sample.

    Args:
        model             : model with .extract_features() method
        dataset           : HistoDataset / MHISTDataset (transform set to None internally)
        d4_transforms     : list of 8 TTATransform from get_tta_transforms("d4")
        device            : "cuda" / "cpu"
        batch_size        : per-transform batch size
        num_workers       : DataLoader workers
        amp               : use fp16 AMP
        save_embeddings_path : if set, save (N, 8, D) float16 embeddings to .npy
        emb_subsample     : how many samples to save for UMAP (randomly sampled)

    Returns:
        dict with keys: mean_ocn, std_ocn, mean_mpcs, std_mpcs, n_samples
        and optionally emb_array (N_sub, 8, D) if save_embeddings_path is set
    """
    model.eval()
    n_views = len(d4_transforms)
    use_amp = amp and device == "cuda"

    orig_transform = dataset.transform

    def _collate_pil(batch):
        imgs, labels = zip(*batch)
        return list(imgs), torch.tensor(labels)

    dataset.transform = None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_pil,
        pin_memory=(device == "cuda"),
    )

    all_ocn = []
    all_id_embs = []     # list of (B, D) float32 tensors; always accumulated for within-class MPCS
    all_embs = []        # list of (B, 8, D) float16 tensors; accumulated if saving
    all_labels = []      # list of (B,) int tensors
    all_logits = []      # list of (B, 8, C) float16 tensors
    save_embs = save_embeddings_path is not None

    pbar = tqdm(loader, desc=f"  orbit tightness ({n_views} D4 views)", leave=False,
                dynamic_ncols=True)

    for images, labels_batch in pbar:
        B = len(images)
        # Stack all views: (n_views * B, C, H, W)
        view_tensors = []
        for tf in d4_transforms:
            view_tensors.append(torch.stack([tf(img) for img in images]))
        mega_batch = torch.cat(view_tensors, dim=0).to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device, dtype=torch.float16, enabled=use_amp):
            emb = model.extract_features(mega_batch)  # (n_views * B, D)
            if save_embs:
                logits_flat = model.classifier(emb)   # (n_views * B, C)

        emb = emb.float()
        emb = emb.reshape(n_views, B, -1)          # (8, B, D)

        if save_embs:
            logits_batch = (
                logits_flat.float()
                .reshape(n_views, B, -1)
                .permute(1, 0, 2)               # (B, 8, C)
                .to(torch.float16).cpu()
            )
            all_logits.append(logits_batch)
            all_embs.append(emb.permute(1, 0, 2).to(torch.float16).cpu())

        emb = F.normalize(emb, dim=-1)              # unit-sphere normalise per embedding

        # Always keep identity view + labels for within-class MPCS
        all_id_embs.append(emb[0].cpu())            # (B, D) normalized identity view
        all_labels.append(labels_batch)             # (B,)

        # OCN: norm of the per-sample centroid (1 = perfect invariance)
        centroid = emb.mean(dim=0)                  # (B, D)
        ocn = centroid.norm(dim=-1).cpu()           # (B,)
        all_ocn.append(ocn)

    dataset.transform = orig_transform

    all_ocn = torch.cat(all_ocn)                   # (N,)
    mpcs = (8.0 * all_ocn**2 - 1.0) / 7.0         # mean pairwise cosine similarity

    # Within-class MPCS: mean pairwise cosine similarity between different samples
    # of the same class, using the identity-view (L2-normalised) embeddings.
    # For k unit vectors with centroid c: mean_pairwise_cos = (k*||c||^2 - 1) / (k-1)
    id_embs = torch.cat(all_id_embs, dim=0).float()   # (N, D), already L2-normalised
    labels_all = torch.cat(all_labels, dim=0)          # (N,)
    classes = labels_all.unique()
    within_class_mpcs_vals = []
    for cls in classes:
        idx = (labels_all == cls).nonzero(as_tuple=True)[0]
        if len(idx) < 2:
            continue
        c = id_embs[idx].mean(dim=0)                   # (D,)
        k = len(idx)
        wmpcs = (k * c.norm().item() ** 2 - 1.0) / (k - 1)
        within_class_mpcs_vals.append(wmpcs)
    mean_within_class_mpcs = float(np.mean(within_class_mpcs_vals))

    result = {
        "mean_ocn":               float(all_ocn.mean()),
        "std_ocn":                float(all_ocn.std()),
        "mean_mpcs":              float(mpcs.mean()),
        "std_mpcs":               float(mpcs.std()),
        "mean_within_class_mpcs": mean_within_class_mpcs,
        "n_samples":              len(all_ocn),
    }

    if save_embs:
        full_embs   = torch.cat(all_embs,   dim=0).numpy()   # (N, 8, D)
        full_labels = labels_all.numpy()                      # (N,)
        full_logits = torch.cat(all_logits, dim=0).numpy()   # (N, 8, C)
        N = len(full_embs)
        # Random subsample (consistent index across embs/labels/logits)
        rng = np.random.default_rng(seed=0)
        idx = rng.choice(N, size=min(emb_subsample, N), replace=False)
        idx = np.sort(idx)
        result["emb_array"]   = full_embs[idx]    # (N_sub, 8, D)
        result["label_array"] = full_labels[idx]  # (N_sub,)
        result["logit_array"] = full_logits[idx]  # (N_sub, 8, C)
        result["emb_indices"] = idx

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Embedding-space orbit tightness evaluation")
    p.add_argument("--model", required=True)
    p.add_argument("--checkpoint", required=True,
                   help="Trained checkpoint (.pt)")
    p.add_argument("--dataset", default="tcga-ut",
                   choices=["tcga-ut", "nct-crc-100k", "nct-crc-nonorm", "mhist"])
    p.add_argument("--backbone_mode", default=None,
                   choices=["frozen", "finetuned"],
                   help="If None, inferred from checkpoint filename")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", default=True,
                   help="Use fp16 AMP (default: on)")
    p.add_argument("--no_amp", action="store_false", dest="amp")
    p.add_argument("--output", default="results/orbit_tightness.csv",
                   help="Output CSV path")
    p.add_argument("--append", action="store_true",
                   help="Append to existing CSV instead of overwriting")
    p.add_argument("--save_embeddings", default=None, metavar="PATH",
                   help="If set, save subsampled embeddings to PATH.npy "
                        "and PATH_transform_names.json for UMAP visualisation")
    p.add_argument("--emb_subsample", type=int, default=2000,
                   help="Number of samples to save for UMAP (default: 2000)")
    return p.parse_args()


def _infer_backbone_mode(ckpt_path: str) -> str:
    stem = os.path.basename(ckpt_path)
    if "_frozen" in stem:
        return "frozen"
    if "_finetuned" in stem:
        return "finetuned"
    return "unknown"


def main():
    args = parse_args()
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    set_seed(args.seed)

    # --- Dataset ---
    if args.dataset == "mhist":
        test_set = MHISTDataset(split="test", transform=None)
    else:
        test_set = HistoDataset(
            args.dataset, split="test", transform=None, cache_dir=args.cache_dir
        )
    num_classes = test_set.num_classes
    print(f"Dataset: {args.dataset} | {num_classes} classes | {len(test_set)} samples")

    # --- Backbone mode ---
    backbone_mode = args.backbone_mode or _infer_backbone_mode(args.checkpoint)
    print(f"Backbone mode: {backbone_mode}")

    # --- Model ---
    freeze = backbone_mode == "frozen"
    model = get_model(args.model, num_classes=num_classes, freeze_backbone=freeze).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if ckpt.get("head_only"):
        model.classifier.load_state_dict(ckpt["classifier_state_dict"])
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ckpt_name = os.path.basename(args.checkpoint)
    print(f"Loaded {ckpt_name} (epoch {ckpt['epoch']})")

    # --- D4 transforms ---
    d4_transforms = get_tta_transforms("d4")   # 8 TTATransform objects
    transform_names = [tf.name for tf in d4_transforms]

    # --- Compute orbit tightness ---
    print(f"\nComputing orbit tightness for {args.model} / {args.dataset} / seed {args.seed}...")
    save_emb_path = args.save_embeddings
    results = compute_orbit_tightness(
        model=model,
        dataset=test_set,
        d4_transforms=d4_transforms,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
        save_embeddings_path=save_emb_path,
        emb_subsample=args.emb_subsample,
    )

    print(
        f"  mean OCN              = {results['mean_ocn']:.4f} ± {results['std_ocn']:.4f}\n"
        f"  mean MPCS (orbit)     = {results['mean_mpcs']:.4f} ± {results['std_mpcs']:.4f}\n"
        f"  mean MPCS (w/in-cls)  = {results['mean_within_class_mpcs']:.4f}\n"
        f"  n_samples             = {results['n_samples']}"
    )

    # --- Save embeddings for UMAP ---
    if save_emb_path is not None:
        Path(save_emb_path).parent.mkdir(parents=True, exist_ok=True)
        np.save(f"{save_emb_path}.npy",         results["emb_array"])
        np.save(f"{save_emb_path}_labels.npy",  results["label_array"])
        np.save(f"{save_emb_path}_logits.npy",  results["logit_array"])
        np.save(f"{save_emb_path}_indices.npy", results["emb_indices"])
        meta = {
            "model": args.model,
            "dataset": args.dataset,
            "backbone_mode": backbone_mode,
            "seed": args.seed,
            "transform_names": transform_names,
            "shape": list(results["emb_array"].shape),
            "num_classes": int(results["logit_array"].shape[-1]),
        }
        with open(f"{save_emb_path}_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  Embeddings saved to {save_emb_path}.npy "
              f"(shape {results['emb_array'].shape})")

    # --- Append to CSV ---
    row = {
        "model":                  args.model,
        "dataset":                args.dataset,
        "backbone_mode":          backbone_mode,
        "seed":                   args.seed,
        "model_type":             _model_type(args.model),
        "mean_ocn":               results["mean_ocn"],
        "std_ocn":                results["std_ocn"],
        "mean_mpcs":              results["mean_mpcs"],
        "std_mpcs":               results["std_mpcs"],
        "mean_within_class_mpcs": results["mean_within_class_mpcs"],
        "n_samples":              results["n_samples"],
        "checkpoint":             ckpt_name,
    }
    fieldnames = list(row.keys())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not (args.append and out_path.exists())
    mode = "a" if args.append else "w"
    with open(out_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"  Row {'appended' if args.append else 'saved'} to {out_path}")


if __name__ == "__main__":
    main()
