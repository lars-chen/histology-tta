"""
Patch-level label utilities for PANDA (Radboud subset).

Radboud mask encoding:
  0 = background   1 = stroma   2 = benign epithelium
  3 = GG3          4 = GG4      5 = GG5

patch_labels() returns binary labels: 1 = cancer (GG3/4/5), 0 = non-cancer.
The mask is read at the lowest pyramid level for speed, then coords are
mapped from level-0 space to that resolution.
"""

from __future__ import annotations
import os

from pathlib import Path

import numpy as np
import openslide

_CANCER_VALUES = {3, 4, 5}
_MASK_DIR = Path(os.path.join(os.environ.get("PANDA_DIR", "data/panda"), "train_label_masks"))


def patch_labels(
    slide_id: str,
    coords: np.ndarray,      # (N, 2) level-0 x,y
    patch_size_lv0: int,
    mask_dir: Path = _MASK_DIR,
) -> np.ndarray:             # (N,) int  0=non-cancer  1=cancer
    """
    Assign binary cancer label to each patch by majority vote in the mask.

    Reads the mask at its lowest pyramid level and maps coords accordingly.
    Returns zeros for all patches if mask file is missing.
    """
    mask_path = mask_dir / f"{slide_id}_mask.tiff"
    if not mask_path.exists():
        return np.zeros(len(coords), dtype=np.int64)

    slide = openslide.OpenSlide(str(mask_path))
    # Use lowest resolution level for speed
    read_level = slide.level_count - 1
    lv0_w, lv0_h = slide.level_dimensions[0]
    lv_w,  lv_h  = slide.level_dimensions[read_level]
    downsample = slide.level_downsamples[read_level]

    mask = np.array(slide.read_region((0, 0), read_level, (lv_w, lv_h)))[:, :, 0]
    slide.close()

    sx = lv_w / lv0_w
    sy = lv_h / lv0_h
    pw = max(1, round(patch_size_lv0 * sx))
    ph = max(1, round(patch_size_lv0 * sy))

    labels = np.zeros(len(coords), dtype=np.int64)
    for i, (cx, cy) in enumerate(coords):
        x0 = int(cx * sx)
        y0 = int(cy * sy)
        region = mask[y0 : y0 + ph, x0 : x0 + pw]
        if region.size == 0:
            continue
        # cancer if majority of non-background pixels are cancerous
        cancer_px = np.isin(region, list(_CANCER_VALUES)).sum()
        tissue_px = (region > 0).sum()
        if tissue_px > 0 and cancer_px / tissue_px > 0.5:
            labels[i] = 1

    return labels
