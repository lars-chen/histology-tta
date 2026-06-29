#!/usr/bin/env python3
"""
Per-sample TTA inference for qualitative Figure 5.

Runs baseline and D4 TTA on the test split of MHIST (and optionally NCT-CRC-100K),
saves a CSV with per-sample predictions and softmax confidences.

For MHIST, joins annotator agreement from data/mhist/annotations.csv.

Usage:
    python eval_per_sample.py --model phikon --seed 42
    python eval_per_sample.py --model phikon --seed 42 --dataset nct-crc-100k
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.data import MHISTDataset, get_dataloaders
from data.transforms import get_tta_transforms
from models import get_model
from torch.utils.data import DataLoader


DATASET_CLASSES = {
    "mhist":        ["HP", "SSA"],
    "nct-crc-100k": ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"],
    "tcga-ut":      None,  # resolved at runtime from data.data._TCGA_UT_CLASSES
}


def parse_args():
    p = argparse.ArgumentParser(description="Per-sample TTA inference for qualitative analysis")
    p.add_argument("--model", default="phikon")
    p.add_argument("--dataset", default="mhist", choices=["mhist", "nct-crc-100k", "tcga-ut"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoints_dir", default="checkpoints")
    p.add_argument("--out_dir", default="results/per_sample")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default=None)
    return p.parse_args()


def load_model(args, num_classes, device):
    model = get_model(args.model, num_classes=num_classes, pretrained=True, freeze_backbone=True)
    ckpt_name = f"{args.dataset}_{args.model}_frozen_aug_seed{args.seed}_best.pt"
    ckpt_path = Path(args.checkpoints_dir) / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ck.get("classifier_state_dict", ck)
    model.classifier.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"  Loaded: {ckpt_path}")
    return model


def run_inference_pil(model, pil_images, labels, tta_transforms, device, batch_size=32):
    """Run baseline and D4 TTA from PIL images. Each TTATransform handles resize+normalize.

    Args:
        pil_images: list of PIL Images
        labels: list/array of integer label indices
        tta_transforms: list of TTATransform objects (index 0 = identity/baseline)
    """
    records = []
    n = len(pil_images)
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Inference", leave=False):
            batch_pil = pil_images[start:start + batch_size]
            batch_labels = labels[start:start + batch_size]

            # Apply each TTA transform to the batch, stack → (B, C, H, W)
            all_logits = []
            for tf in tta_transforms:
                batch_t = torch.stack([tf(img) for img in batch_pil]).to(device)
                all_logits.append(model(batch_t).cpu())

            all_logits = torch.stack(all_logits, dim=1)   # (B, n_views, n_cls)
            probs_base = F.softmax(all_logits[:, 0, :], dim=-1).numpy()
            probs_tta  = F.softmax(all_logits, dim=-1).mean(dim=1).numpy()

            for i in range(len(batch_pil)):
                records.append({
                    "true_label_idx": int(batch_labels[i]),
                    "baseline_pred_idx": int(probs_base[i].argmax()),
                    "tta_pred_idx":      int(probs_tta[i].argmax()),
                    "baseline_softmax":  probs_base[i].tolist(),
                    "tta_softmax":       probs_tta[i].tolist(),
                    "baseline_conf":     float(probs_base[i].max()),
                    "tta_conf":          float(probs_tta[i].max()),
                })
    return records


def main():
    args = parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}, Model: {args.model}, Dataset: {args.dataset}, Seed: {args.seed}")

    if args.dataset == "tcga-ut":
        from data.data import _TCGA_UT_CLASSES
        DATASET_CLASSES["tcga-ut"] = _TCGA_UT_CLASSES

    classes = DATASET_CLASSES[args.dataset]
    num_classes = len(classes)

    # Load model
    model = load_model(args, num_classes, device)

    # Build D4 TTA transforms — all 8 views; index 0 = identity (baseline)
    tta_transforms = get_tta_transforms("d4")

    # Load PIL images and labels for the test split
    if args.dataset == "mhist":
        ds_raw = MHISTDataset(split="test", transform=None)   # no transform → PIL
        pil_images = [Image.open(
            Path("data/mhist/images") / fname).convert("RGB")
            for fname in ds_raw.filenames]
        labels_arr = [ds_raw.class_to_idx[lbl] for lbl in ds_raw.labels]
        image_ids  = ds_raw.filenames
    elif args.dataset == "tcga-ut":
        from data.data import DATASETS, _get_field
        from datasets import load_dataset
        cfg    = DATASETS["tcga-ut"]
        hf_ds  = load_dataset(cfg[0], cfg[1], split=cfg[3])
        pil_images = [hf_ds[i][cfg[4]].convert("RGB") for i in range(len(hf_ds))]
        cls2idx    = {c: i for i, c in enumerate(classes)}
        labels_arr = [cls2idx[str(_get_field(hf_ds[i], cfg[5]))] for i in range(len(hf_ds))]
        image_ids  = list(hf_ds["__key__"])
    else:
        # NCT-CRC: load via HuggingFace dataset (streaming=False for random access)
        from data.data import DATASETS, _get_field
        cfg = DATASETS[args.dataset]
        from datasets import load_dataset
        hf_ds = load_dataset(cfg[0], cfg[1], split=cfg[3])
        pil_images = [hf_ds[i][cfg[4]].convert("RGB") for i in range(len(hf_ds))]
        cls_list   = sorted({str(_get_field(s, cfg[5])) for s in hf_ds})
        cls2idx    = {c: i for i, c in enumerate(cls_list)}
        labels_arr = [cls2idx[str(_get_field(hf_ds[i], cfg[5]))] for i in range(len(hf_ds))]
        image_ids  = None

    records = run_inference_pil(model, pil_images, labels_arr, tta_transforms,
                                device, batch_size=args.batch_size)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df["true_label"] = [classes[i] for i in df["true_label_idx"]]
    df["baseline_pred"] = [classes[i] for i in df["baseline_pred_idx"]]
    df["tta_pred"] = [classes[i] for i in df["tta_pred_idx"]]
    df["baseline_correct"] = df["true_label_idx"] == df["baseline_pred_idx"]
    df["tta_correct"] = df["true_label_idx"] == df["tta_pred_idx"]
    df["outcome"] = "neutral"
    df.loc[~df["baseline_correct"] &  df["tta_correct"], "outcome"] = "corrected"
    df.loc[ df["baseline_correct"] & ~df["tta_correct"], "outcome"] = "corrupted"
    df.loc[ df["baseline_correct"] &  df["tta_correct"], "outcome"] = "both_correct"
    df.loc[~df["baseline_correct"] & ~df["tta_correct"], "outcome"] = "both_wrong"

    df.insert(0, "sample_idx", range(len(df)))
    if image_ids is not None:
        df.insert(0, "image_id", image_ids)

    # Join MHIST annotator agreement
    if args.dataset == "mhist":
        ann = pd.read_csv("data/mhist/annotations.csv")
        ann = ann.rename(columns={
            "Image Name": "image_id",
            "Number of Annotators who Selected SSA (Out of 7)": "n_ssa_annotators",
            "Majority Vote Label": "mv_label",
        })
        ann["test"] = ann["Partition"] == "test"
        df = df.merge(ann[["image_id", "n_ssa_annotators", "mv_label"]],
                      on="image_id", how="left")
        # Agreement: max(n_ssa, 7-n_ssa) / 7 → 1.0 = unanimous, 0.5 = maximally ambiguous
        df["agreement"] = df["n_ssa_annotators"].apply(lambda n: max(n, 7 - n) / 7)

    out_path = out_dir / f"per_sample_{args.dataset}_{args.model}_seed{args.seed}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df["outcome"].value_counts().to_string())


if __name__ == "__main__":
    main()
