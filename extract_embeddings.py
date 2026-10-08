"""
Extract and cache backbone embeddings for all datasets.

Saves to $HISTO_EMB_DIR/{model}/{dataset}/{split}/
  embeddings.npy  — float32, shape (N, D) for train, (N, 8, D) for test
  labels.npy      — int64,   shape (N,)

Train split: single (original) view — enough for linear probing / fast head training.
Test split:  all 8 D4 views — for TTA evaluation from cached embeddings.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime

from models import get_model
from data.data import HistoDataset, MHISTDataset
from data.transforms import get_tta_transforms

OUT_BASE = Path(os.environ.get("HISTO_EMB_DIR", "embeddings"))

BATCH_FOR = {
    "gigapath": 128, "hoptimus": 128,  # ViT-g, 80GB A100: mega-batch=1024
    "virchow": 256,  "virchow2": 256,  # ViT-L: mega-batch=2048
    "uni": 256,      "uni2": 256,
    "phikon": 256,   "phikon2": 256,
    "ctranspath": 512,
    "dinov2_b": 512, "dinov2_s": 512,
    "convnextv2_tiny": 128, "convnextv2_base": 128,
    "resnet18": 1024, "resnet50": 1024,
}


def _get_dataset(dataset: str, split: str, transform, cache_dir: str | None):
    if dataset == "mhist":
        return MHISTDataset(split=split, transform=transform)
    return HistoDataset(dataset, split=split, transform=transform, cache_dir=cache_dir)


def _get_sample_ids(ds, dataset: str) -> dict:
    """
    Return a dict of ID arrays (all length N, in dataset order):
      - sample_id  : unique patch/sample identifier
      - slide_id   : slide-level ID (where available, else None)
      - patient_id : patient-level ID (where available, else None)
    """
    N = len(ds)

    if dataset == "mhist":
        sample_ids = np.array(ds.filenames)
        return {"sample_id": sample_ids, "slide_id": None, "patient_id": None}

    if dataset == "tcga-ut":
        keys = ds._hf["__key__"]                                   # e.g. "TCGA-XX-XXXX-01A/.../patch"
        slide_ids   = np.array([k.split("/")[0] for k in keys])
        patient_ids = np.array(["-".join(s.split("-")[:3]) for s in slide_ids])
        return {"sample_id": np.array(keys), "slide_id": slide_ids, "patient_id": patient_ids}

    # nct-crc datasets: no patient column, use sequential index
    return {"sample_id": np.arange(N), "slide_id": None, "patient_id": None}


def _collate_pil(batch):
    imgs, labels = zip(*batch)
    return list(imgs), torch.tensor(labels)


@torch.no_grad()
def extract_single_view(model, dataset, batch_size, device, num_workers, amp):
    """Single-pass train extraction. Returns (N, D), (N,)."""
    tf = get_tta_transforms("none")[0]
    dataset.transform = None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=_collate_pil)
    all_embs, all_labels = [], []
    for images, labels in tqdm(loader, desc="  train", leave=False):
        batch = torch.stack([tf(img) for img in images]).to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            embs = model.extract_features(batch).float().cpu()
        all_embs.append(embs)
        all_labels.append(labels)
    return torch.cat(all_embs).numpy(), torch.cat(all_labels).numpy()


@torch.no_grad()
def extract_d4_views(model, dataset, batch_size, device, num_workers, amp):
    """Single-pass 8-view test extraction. Returns (N, 8, D), (N,)."""
    d4_transforms = get_tta_transforms("d4")  # 8 transforms
    n_views = len(d4_transforms)
    dataset.transform = None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=_collate_pil)
    all_embs, all_labels = [], []
    for images, labels in tqdm(loader, desc=f"  test ({n_views} views)", leave=False):
        B = len(images)
        # Stack all views into one mega-batch: (n_views * B, C, H, W)
        mega = torch.cat([
            torch.stack([tf(img) for img in images])
            for tf in d4_transforms
        ], dim=0).to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            embs = model.extract_features(mega).float().cpu()  # (n_views*B, D)
        all_embs.append(embs.reshape(n_views, B, -1).permute(1, 0, 2))  # (B, 8, D)
        all_labels.append(labels)
    view_names = [tf.name for tf in d4_transforms]
    return torch.cat(all_embs).numpy(), torch.cat(all_labels).numpy(), view_names


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",   required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--amp",     action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cache_dir", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = BATCH_FOR.get(args.model, 32)
    print(f"Model={args.model}  Dataset={args.dataset}  "
          f"Device={device}  Batch={batch_size}")

    # Load backbone once for this job
    model = get_model(args.model, num_classes=2, freeze_backbone=True).to(device)
    model.eval()
    D = model.feature_dim
    print(f"Backbone loaded — embed dim={D}")

    out_dir = OUT_BASE / args.model / args.dataset

    # Save class names once (same for both splits)
    # Load dataset briefly just to get class list, then reuse below
    _ds_tmp = _get_dataset(args.dataset, "test", transform=None, cache_dir=args.cache_dir)
    (out_dir / "classes.txt").write_text("\n".join(_ds_tmp.classes))
    print(f"Classes ({len(_ds_tmp.classes)}): {_ds_tmp.classes}")
    del _ds_tmp

    for split in ("train", "test"):
        out_path = out_dir / split
        emb_file = out_path / "embeddings.npy"
        lbl_file = out_path / "labels.npy"

        if emb_file.exists() and lbl_file.exists():
            print(f"  [{split}] already exists, skipping.")
            continue

        print(f"\n[{split}] extracting...")
        ds = _get_dataset(args.dataset, split, transform=None, cache_dir=args.cache_dir)

        # Save sample/slide/patient IDs (order matches embeddings)
        ids = _get_sample_ids(ds, args.dataset)
        np.save(out_path / "sample_ids.npy", ids["sample_id"])
        if ids["slide_id"] is not None:
            np.save(out_path / "slide_ids.npy",   ids["slide_id"])
            np.save(out_path / "patient_ids.npy", ids["patient_id"])

        if split == "train":
            # No 8x view multiplier — use 8x batch to match GPU load of test
            embs, labels = extract_single_view(model, ds, batch_size * 8, device,
                                               args.num_workers, args.amp)
        else:
            embs, labels, view_names = extract_d4_views(model, ds, batch_size, device,
                                                        args.num_workers, args.amp)
            (out_path / "view_names.txt").write_text("\n".join(view_names))
            print(f"  [test] view order: {view_names}")

        np.save(emb_file, embs)
        np.save(lbl_file, labels)
        print(f"  [{split}] saved: embeddings={embs.shape}  labels={labels.shape}")

        meta = {
            "model": args.model, "dataset": args.dataset, "split": split,
            "embed_dim": D, "n_samples": int(embs.shape[0]),
            "n_views": int(embs.shape[1]) if split == "test" else 1,
            "extracted": datetime.datetime.now().isoformat(),
        }
        (out_path / "meta.json").write_text(json.dumps(meta, indent=2))

    print("\nDone.")


if __name__ == "__main__":
    main()
