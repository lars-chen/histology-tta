#!/usr/bin/env python3
"""
TTA evaluation entry point.

Loads a trained checkpoint and evaluates with multiple TTA strategies,
producing a comparison table.

Usage:
    python evaluate_tta.py \
        --model resnet50 \
        --checkpoint checkpoints/resnet50_best.pt \
        --num_classes 33 \
        --tta_strategies none flips d4 d4_color full \
        --aggregations mean vote confidence
"""

import argparse
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from models import get_model
from data.data import HistoDataset
from data.transforms import get_tta_transforms, get_val_transform
from tta.aggregator import aggregate_predictions
from utils.metrics import per_class_metrics, print_per_class_table
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="TTA evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="tcga-ut",
                        help="Dataset name (tcga-ut, nct-crc-100k, nct-crc-nonorm)")
    parser.add_argument("--num_classes", type=int, default=None,
                        help="If None, inferred from dataset")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument(
        "--tta_strategies", nargs="+",
        default=["none", "flips", "d4", "d4_color"],
        help="TTA strategies to evaluate"
    )
    parser.add_argument(
        "--aggregations", nargs="+",
        default=["mean", "vote", "confidence"],
        help="Aggregation strategies"
    )
    return parser.parse_args()


@torch.no_grad()
def run_tta_eval(model, dataset, tta_transforms, aggregation, device, batch_size, num_workers):
    """Run TTA over the full dataset and return accuracy metrics."""
    model.eval()

    # We need raw PIL images — disable the dataset's tensor transform temporarily
    orig_transform = dataset.transform

    all_probs = []
    all_labels = []
    val_transform = get_val_transform()

    def _collate_pil(batch):
        imgs, labels = zip(*batch)
        return list(imgs), torch.tensor(labels)

    dataset.transform = None  # return raw PIL images
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=_collate_pil,
    )

    for images, labels in loader:
        # images is a list of PIL images when transform=None
        logits_list = []
        for tta_tf in tta_transforms:
            batch = torch.stack([tta_tf(img) for img in images]).to(device)
            logits_list.append(model(batch))
        probs = aggregate_predictions(logits_list, aggregation)
        all_probs.append(probs.cpu())
        all_labels.append(labels)

    dataset.transform = orig_transform

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    preds = all_probs.argmax(dim=1).numpy()
    labels = all_labels.numpy()

    n_total = len(labels)
    n_correct = int((preds == labels).sum())
    n_wrong = n_total - n_correct

    return {
        "acc": accuracy_score(labels, preds),
        "balanced_acc": balanced_accuracy_score(labels, preds),
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "n_total": n_total,
        "preds": preds,
        "labels": labels,
    }


def main():
    args = parse_args()
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    # Load test dataset
    test_set = HistoDataset(args.dataset, split="test", transform=None, cache_dir=args.cache_dir)
    num_classes = args.num_classes or test_set.num_classes

    # Load model
    model = get_model(args.model, num_classes=num_classes).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")

    # --- Evaluate all strategies ---
    W = 115
    print(f"\n{'='*W}")
    print(
        f"{'Strategy':<20} {'Agg':<12} {'Acc':>8} {'BalAcc':>10} "
        f"{'Correct':>9} {'Wrong':>7} {'Total':>8} {'Views':>7} "
        f"{'Corrected':>11} {'Corrupted':>11}"
    )
    print(f"{'='*W}")

    baseline_preds = None  # predictions from strategy="none"
    results = []
    for strategy in args.tta_strategies:
        tta_transforms = get_tta_transforms(strategy)
        for agg in args.aggregations:
            if strategy == "none" and agg != "mean":
                continue  # no point aggregating a single view
            metrics = run_tta_eval(
                model=model,
                dataset=test_set,
                tta_transforms=tta_transforms,
                aggregation=agg,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            n_views = len(tta_transforms)

            # Compare to the no-TTA baseline
            if strategy == "none":
                baseline_preds = metrics["preds"]
                n_corrected = n_corrupted = None
            elif baseline_preds is not None:
                labels_arr = metrics["labels"]
                tta_preds   = metrics["preds"]
                baseline_wrong = baseline_preds != labels_arr
                baseline_right = ~baseline_wrong
                n_corrected = int((baseline_wrong & (tta_preds == labels_arr)).sum())
                n_corrupted = int((baseline_right & (tta_preds != labels_arr)).sum())
            else:
                n_corrected = n_corrupted = None

            row = {
                "strategy": strategy,
                "aggregation": agg,
                "acc": metrics["acc"],
                "balanced_acc": metrics["balanced_acc"],
                "n_correct": metrics["n_correct"],
                "n_wrong": metrics["n_wrong"],
                "n_total": metrics["n_total"],
                "n_views": n_views,
                "n_corrected": n_corrected,
                "n_corrupted": n_corrupted,
                "per_class": per_class_metrics(
                    metrics["labels"], metrics["preds"], test_set.classes
                ),
            }
            results.append(row)

            corrected_str = f"{n_corrected:>11,}" if n_corrected is not None else f"{'—':>11}"
            corrupted_str = f"{n_corrupted:>11,}" if n_corrupted is not None else f"{'—':>11}"
            print(
                f"{strategy:<20} {agg:<12} {metrics['acc']:>8.4f} "
                f"{metrics['balanced_acc']:>10.4f} {metrics['n_correct']:>9,} "
                f"{metrics['n_wrong']:>7,} {metrics['n_total']:>8,} {n_views:>7} "
                f"{corrected_str} {corrupted_str}"
            )

    print(f"{'='*W}")

    # Per-class breakdown for every strategy / aggregation
    for row in results:
        print_per_class_table(row["per_class"], row["strategy"], row["aggregation"])

    # Find best
    best = max(results, key=lambda r: r["balanced_acc"])
    print(
        f"\nBest: strategy='{best['strategy']}' agg='{best['aggregation']}' "
        f"balanced_acc={best['balanced_acc']:.4f}"
    )

    # Save results (drop non-serialisable numpy arrays)
    import json
    _SKIP = {"preds", "labels"}
    serialisable = [{k: v for k, v in r.items() if k not in _SKIP} for r in results]
    out_path = f"tta_results_{args.dataset}_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()