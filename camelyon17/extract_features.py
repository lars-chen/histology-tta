"""
Extract patch-level features from Camelyon17 WSIs using a frozen backbone.

Reads coords from CLAM-generated .h5 files (level-0 coords), loads the
corresponding region from the .tif slide at level 1 (20×, 256×256 px),
encodes with a frozen backbone, and saves per-slide tensors.

Output layout:
  <out_dir>/<model>/<tta_mode>/<slide_id>.pt

TTA modes:
  none   — one forward pass per patch → (N, d)
  d4_all — all 8 D4 views saved → (N, 8, d)

Efficiency:
  - SLURM array: pass --task_id $SLURM_ARRAY_TASK_ID --n_tasks $SLURM_ARRAY_TASK_COUNT
    to shard 441 slides evenly across array workers (one GPU each)
  - Prefetch: OpenSlide reads for batch i+1 happen in a background thread
    while the GPU encodes batch i
  - AMP: fp16 inference via torch.cuda.amp.autocast

Usage (single GPU):
  python -m camelyon17.extract_features --model uni --tta_modes none d4_all

Usage (SLURM array, 8 GPUs):
  #SBATCH --array=0-7
  python -m camelyon17.extract_features \
      --model uni --tta_modes none d4_all \
      --task_id $SLURM_ARRAY_TASK_ID --n_tasks $SLURM_ARRAY_TASK_COUNT
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# D4 geometric ops (PIL-space, no resize/normalize)
# ──────────────────────────────────────────────────────────────────────────────

_D4_OPS = [
    lambda img: img,
    lambda img: img.rotate(90),
    lambda img: img.rotate(180),
    lambda img: img.rotate(270),
    lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
    lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90),
    lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).rotate(180),
    lambda img: img.transpose(Image.FLIP_LEFT_RIGHT).rotate(270),
]


# ──────────────────────────────────────────────────────────────────────────────
# Backbone loader
# ──────────────────────────────────────────────────────────────────────────────

_MODEL_REGISTRY = {
    # name → (loader, repo, feature_dim, timm_kwargs)
    # Histology foundation models
    "phikon":    ("transformers", "owkin/phikon",                              768,  {}),
    "phikon2":   ("transformers", "owkin/phikon-v2",                           1024, {}),
    "uni":       ("timm",         "MahmoodLab/UNI",                            1024, {"init_values": 1e-5, "dynamic_img_size": True}),
    "virchow2":  ("timm",         "paige-ai/Virchow2",                         1280, {}),
    "gigapath":  ("timm",         "prov-gigapath/prov-gigapath",               1536, {}),
    # General-purpose models
    "dinov2-s":  ("transformers", "facebook/dinov2-small",                     384,  {}),
    "dinov2-b":  ("transformers", "facebook/dinov2-base",                      768,  {}),
    "dinov2-l":  ("transformers", "facebook/dinov2-large",                     1024, {}),
    "resnet50":  ("timm_local",    "resnet50.a1_in1k",                          2048, {"global_pool": "avg"}),
    "convnextv2":("timm_local",   "convnextv2_base.fcmae_ft_in22k_in1k",       1024, {}),
}


def _swiglu_kwargs() -> dict:
    from timm.layers import SwiGLUPacked
    import torch.nn as nn
    return {"mlp_layer": SwiGLUPacked, "act_layer": nn.SiLU}


def load_backbone(model_name: str, device: torch.device):
    """
    Load a frozen backbone and preprocessing/forward callables.
    Returns (backbone, preprocess_fn, forward_fn, feature_dim).

    preprocess_fn(pil_list) → (B, C, H, W) float32 CPU tensor
    forward_fn(pixel_values_gpu) → (B, d) float32 GPU tensor
    """
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Choices: {list(_MODEL_REGISTRY)}")

    loader, repo, feat_dim, timm_kwargs = _MODEL_REGISTRY[model_name]

    if loader == "transformers":
        from transformers import AutoImageProcessor, AutoModel
        processor = AutoImageProcessor.from_pretrained(repo, trust_remote_code=True)
        backbone = AutoModel.from_pretrained(repo, trust_remote_code=True)

        def preprocess_fn(pil_list):
            return processor(images=pil_list, return_tensors="pt")["pixel_values"]

        def forward_fn(x):
            return backbone(pixel_values=x).last_hidden_state[:, 0, :]

    elif loader in ("timm", "timm_local"):
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        if model_name in ("virchow", "virchow2"):
            timm_kwargs = {**_swiglu_kwargs(), "global_pool": "token"}

        model_id = repo if loader == "timm_local" else f"hf_hub:{repo}"
        backbone = timm.create_model(
            model_id, pretrained=True, num_classes=0, **timm_kwargs,
        )
        transform = create_transform(**resolve_data_config(backbone.pretrained_cfg))

        def preprocess_fn(pil_list):
            return torch.stack([transform(img) for img in pil_list])

        def forward_fn(x):
            return backbone(x)

    else:
        raise ValueError(f"Unknown loader: {loader}")

    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)

    return backbone, preprocess_fn, forward_fn, feat_dim


# ──────────────────────────────────────────────────────────────────────────────
# Patch reading
# ──────────────────────────────────────────────────────────────────────────────

def read_patches(slide, coords: np.ndarray, patch_size: int, patch_level: int) -> list[Image.Image]:
    """Read a batch of patches from an OpenSlide object (CPU, called from thread)."""
    return [
        slide.read_region((int(x), int(y)), patch_level, (patch_size, patch_size)).convert("RGB")
        for x, y in coords
    ]


def _batch_starts(n: int, batch_size: int) -> list[int]:
    return list(range(0, n, batch_size))


def prefetch_reader(slide, coords, patch_size, patch_level, batch_size):
    """
    Generator that yields batches of PIL patches.

    Submits the read for batch i+1 to a background thread while the caller
    processes (GPU-encodes) batch i, hiding OpenSlide I/O latency.
    """
    starts = _batch_starts(len(coords), batch_size)
    with ThreadPoolExecutor(max_workers=1) as pool:
        # Prime the first future
        fut = pool.submit(
            read_patches, slide, coords[starts[0]:starts[0] + batch_size],
            patch_size, patch_level,
        )
        for i, start in enumerate(starts):
            # Submit next batch read before waiting for current
            if i + 1 < len(starts):
                next_start = starts[i + 1]
                next_fut = pool.submit(
                    read_patches, slide, coords[next_start:next_start + batch_size],
                    patch_size, patch_level,
                )
            patches = fut.result()          # blocks only if GPU finished early
            fut = next_fut if i + 1 < len(starts) else None
            yield patches


# ──────────────────────────────────────────────────────────────────────────────
# Encoding
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_batch(
    pil_patches: list[Image.Image],
    preprocess_fn,
    forward_fn,
    device: torch.device,
) -> torch.Tensor:
    """Encode a list of PIL patches → (B, d) float32 on CPU. Uses AMP if CUDA."""
    pixel_values = preprocess_fn(pil_patches).to(device, non_blocking=True)
    with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        feats = forward_fn(pixel_values)
    return feats.cpu().float()


# ──────────────────────────────────────────────────────────────────────────────
# Per-slide extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_slide(
    h5_path: Path,
    slides_dir: Path,
    preprocess_fn,
    forward_fn,
    device: torch.device,
    tta_modes: list[str],
    patch_size: int,
    patch_level: int,
    batch_size: int,
    out_dirs: dict[str, Path],
) -> None:
    import openslide

    slide_id = h5_path.stem
    slide_path = slides_dir / f"{slide_id}.tif"
    if not slide_path.exists():
        slide_path = slides_dir / f"{slide_id}.tiff"
    if not slide_path.exists():
        print(f"  [SKIP] slide not found: {slide_path}")
        return

    if all((out_dirs[m] / f"{slide_id}.pt").exists() for m in tta_modes):
        print(f"  [SKIP] {slide_id} — all outputs exist")
        return

    with h5py.File(h5_path, "r") as f:
        coords = f["coords"][:]   # (N, 2)

    n_patches = len(coords)
    slide = openslide.OpenSlide(str(slide_path))
    buffers: dict[str, list[torch.Tensor]] = {m: [] for m in tta_modes}

    for raw_patches in prefetch_reader(slide, coords, patch_size, patch_level, batch_size):
        if "none" in tta_modes:
            buffers["none"].append(encode_batch(raw_patches, preprocess_fn, forward_fn, device))

        if "d4_all" in tta_modes:
            view_feats = [
                encode_batch([op(p) for p in raw_patches], preprocess_fn, forward_fn, device)
                for op in _D4_OPS
            ]
            buffers["d4_all"].append(torch.stack(view_feats, dim=1))  # (B, 8, d)

    slide.close()

    for mode in tta_modes:
        torch.save(
            torch.cat(buffers[mode], dim=0),
            out_dirs[mode] / f"{slide_id}.pt",
        )

    print(f"  {slide_id}: {n_patches} patches → {tta_modes}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--patches_dir",  type=Path,
                   default=Path("/gpfs/data/mankowskilab/chen/camelyon17_patched/patches"))
    p.add_argument("--slides_dir",   type=Path,
                   default=Path("/gpfs/data/mankowskilab/chen/camelyon17_patched/slides_symlinks"))
    p.add_argument("--out_dir",      type=Path,
                   default=Path("/gpfs/data/mankowskilab/chen/histology-tta/camelyon17_features"))
    p.add_argument("--model",        type=str, default="uni",
                   choices=list(_MODEL_REGISTRY))
    p.add_argument("--tta_modes",    nargs="+", default=["none", "d4_all"],
                   choices=["none", "d4_all"])
    p.add_argument("--batch_size",   type=int, default=512,
                   help="Patches per GPU forward pass (increase if VRAM allows)")
    p.add_argument("--patch_level",  type=int, default=1)
    p.add_argument("--patch_size",   type=int, default=256)
    p.add_argument("--device",       type=str, default="cuda")
    p.add_argument("--task_id",      type=int, default=0,
                   help="SLURM_ARRAY_TASK_ID — which shard of slides this worker handles")
    p.add_argument("--n_tasks",      type=int, default=1,
                   help="SLURM_ARRAY_TASK_COUNT — total number of array workers")
    p.add_argument("--slide_list",   type=Path, default=None,
                   help="Optional .txt with one slide_id per line (applied before sharding)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Task {args.task_id}/{args.n_tasks}")

    out_dirs = {}
    for mode in args.tta_modes:
        d = args.out_dir / args.model / mode
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[mode] = d

    print(f"Loading backbone: {args.model}")
    backbone, preprocess_fn, forward_fn, feat_dim = load_backbone(args.model, device)
    print(f"Feature dim: {feat_dim}")

    # Write metadata once per output dir (task 0 only to avoid races)
    if args.task_id == 0:
        metadata = {
            "model": args.model,
            "feature_dim": feat_dim,
            "patch_level": args.patch_level,
            "patch_size": args.patch_size,
            "patches_dir": str(args.patches_dir),
            "slides_dir": str(args.slides_dir),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        for mode, d in out_dirs.items():
            meta_path = d / "metadata.json"
            if not meta_path.exists():
                with open(meta_path, "w") as f:
                    json.dump({**metadata, "tta_mode": mode}, f, indent=2)

    # Collect and shard slides
    h5_files = sorted(args.patches_dir.glob("*.h5"))
    if args.slide_list is not None:
        allowed = set(args.slide_list.read_text().splitlines())
        h5_files = [f for f in h5_files if f.stem in allowed]

    # Each task handles its own contiguous slice
    h5_files = h5_files[args.task_id :: args.n_tasks]
    print(f"Processing {len(h5_files)} slides (shard {args.task_id}/{args.n_tasks}), "
          f"TTA modes: {args.tta_modes}")

    for i, h5_path in enumerate(h5_files):
        print(f"[{i+1}/{len(h5_files)}] {h5_path.stem}")
        try:
            extract_slide(
                h5_path=h5_path,
                slides_dir=args.slides_dir,
                preprocess_fn=preprocess_fn,
                forward_fn=forward_fn,
                device=device,
                tta_modes=args.tta_modes,
                patch_size=args.patch_size,
                patch_level=args.patch_level,
                batch_size=args.batch_size,
                out_dirs=out_dirs,
            )
        except Exception as e:
            print(f"  [ERROR] {h5_path.stem}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
