"""
Histology dataset loader.

Usage — downloaded (caches to disk, recommended for training):
    ds = HistoDataset("nct-crc-100k", split="train", transform=tf)
    train_loader, val_loader, test_loader, num_classes = get_dataloaders(
        "nct-crc-100k", train_tf, val_tf
    )

Usage — streaming (no disk, good for quick exploration):
    ds = HistoDataset("tcga-ut", split="train", transform=tf, streaming=True)

Supported datasets:
    "tcga-ut"         dakomura/tcga-ut          31-class pan-cancer (256×256)
    "nct-crc-100k"    1aurent/NCT-CRC-HE         9-class colorectal, stain-norm (224×224)
    "nct-crc-7k"      1aurent/NCT-CRC-HE         same 9 classes, held-out val set
    "nct-crc-nonorm"  1aurent/NCT-CRC-HE         same 9 classes, no stain norm

To add a dataset, add one entry to DATASETS below.
"""

import io
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from datasets import load_dataset


# Registry: name → (hf_repo, hf_subset, hf_train_split, hf_test_split,
#                    image_col, label_col, classes)
#
# image_col  : HF column name for the image
# label_col  : HF column name for the label; use "outer.inner" for nested dicts
# classes    : list of class names when labels are strings, or None to detect
#              from a ClassLabel feature automatically
_TCGA_UT_CLASSES = [
    "Adrenocortical_carcinoma",
    "Bladder_Urothelial_Carcinoma",
    "Brain_Lower_Grade_Glioma",
    "Breast_invasive_carcinoma",
    "Cervical_squamous_cell_carcinoma_and_endocervical_adenocarcinoma",
    "Cholangiocarcinoma",
    "Colon_Rectum_adenocarcinoma",
    "Esophageal_carcinoma",
    "Glioblastoma_multiforme",
    "Head_and_Neck_squamous_cell_carcinoma",
    "Kidney_Chromophobe",
    "Kidney_renal_clear_cell_carcinoma",
    "Kidney_renal_papillary_cell_carcinoma",
    "Liver_hepatocellular_carcinoma",
    "Lung_adenocarcinoma",
    "Lung_squamous_cell_carcinoma",
    "Lymphoid_Neoplasm_Diffuse_Large_B-cell_Lymphoma",
    "Mesothelioma",
    "Ovarian_serous_cystadenocarcinoma",
    "Pancreatic_adenocarcinoma",
    "Pheochromocytoma_and_Paraganglioma",
    "Prostate_adenocarcinoma",
    "Sarcoma",
    "Skin_Cutaneous_Melanoma",
    "Stomach_adenocarcinoma",
    "Testicular_Germ_Cell_Tumors",
    "Thymoma",
    "Thyroid_carcinoma",
    "Uterine_Carcinosarcoma",
    "Uterine_Corpus_Endometrial_Carcinoma",
    "Uveal_Melanoma",
]

DATASETS = {
    #                  repo                   subset                     train   test    img_col  lbl_col       classes
    "tcga-ut":        ("dakomura/tcga-ut",    None,                      "train","test", "jpg",   "json.label", _TCGA_UT_CLASSES),
    "nct-crc-100k":   ("1aurent/NCT-CRC-HE", "NCT_CRC_HE_100K",         "train", None, "image", "label",      None),
    "nct-crc-7k":     ("1aurent/NCT-CRC-HE", "CRC_VAL_HE_7K",           "train", None, "image", "label",      None),
    "nct-crc-nonorm": ("1aurent/NCT-CRC-HE", "NCT_CRC_HE_100K_NONORM",  "train", None, "image", "label",      None),
}


def _get_field(sample: dict, col: str):
    """Access sample[col], supporting 'outer.inner' dot notation."""
    if "." in col:
        outer, inner = col.split(".", 1)
        return sample[outer][inner]
    return sample[col]


def _to_pil(raw) -> Image.Image:
    """Convert whatever HF returns for an image column into a PIL RGB image."""
    if isinstance(raw, Image.Image):
        return raw.convert("RGB")
    if isinstance(raw, dict):
        return Image.open(io.BytesIO(raw["bytes"])).convert("RGB")
    if isinstance(raw, (bytes, bytearray)):
        return Image.open(io.BytesIO(raw)).convert("RGB")
    import numpy as np
    return Image.fromarray(np.asarray(raw)).convert("RGB")


class HistoDataset(Dataset):
    """
    Wraps a HuggingFace histology dataset as a standard PyTorch Dataset.

    Args:
        name      : one of the keys in DATASETS above
        split     : "train" or "test"
        transform : torchvision transform applied to each PIL image
        streaming : if True, stream from HF Hub without saving to disk
                    (disables len() and random access)
        cache_dir : where to store the downloaded dataset (None = HF default)
    """

    def __init__(
        self,
        name: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        streaming: bool = False,
        cache_dir: Optional[str] = None,
    ):
        assert name in DATASETS, f"Unknown dataset '{name}'. Choose from: {list(DATASETS)}"
        repo, subset, train_split, test_split, image_col, label_col, predefined_classes = DATASETS[name]

        hf_split = train_split if split == "train" else test_split
        assert hf_split is not None, f"'{name}' has no '{split}' split."

        self.transform = transform
        self.streaming = streaming
        self._image_col = image_col
        self._label_col = label_col
        self._hf = load_dataset(repo, name=subset, split=hf_split,
                                streaming=streaming, cache_dir=cache_dir)

        # Build class list
        if predefined_classes is not None:
            self.classes = predefined_classes
        else:
            # Try to read from a ClassLabel feature
            top_col = label_col.split(".")[0] if "." in label_col else label_col
            label_feat = self._hf.features.get(top_col)
            if hasattr(label_feat, "names"):
                self.classes = label_feat.names
            else:
                # Fallback: scan all labels (slow, but correct)
                self.classes = sorted({str(_get_field(s, label_col)) for s in self._hf})
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        if not streaming:
            print(f"Loaded '{name}' ({split}): {len(self):,} samples, {len(self.classes)} classes")

    def __len__(self):
        if self.streaming:
            raise TypeError("len() not available in streaming mode")
        return len(self._hf)

    def __getitem__(self, idx):
        sample = self._hf[idx]
        img = _to_pil(_get_field(sample, self._image_col))
        label = _get_field(sample, self._label_col)
        if isinstance(label, str):
            label = self.class_to_idx[label]
        if self.transform:
            img = self.transform(img)
        return img, int(label)

    def __iter__(self):
        """Enables use as an IterableDataset in streaming mode."""
        for sample in self._hf:
            img = _to_pil(_get_field(sample, self._image_col))
            label = _get_field(sample, self._label_col)
            if isinstance(label, str):
                label = self.class_to_idx[label]
            if self.transform:
                img = self.transform(img)
            yield img, int(label)

    @property
    def num_classes(self):
        return len(self.classes)


def get_dataloaders(
    name: str,
    train_transform: Callable,
    val_transform: Callable,
    val_fraction: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
):
    """
    Returns (train_loader, val_loader, test_loader, num_classes).

    Val set is carved from the training split via random_split.
    test_loader is None if the dataset has no dedicated test split.
    """
    train_ds = HistoDataset(name, split="train", transform=train_transform, cache_dir=cache_dir)
    num_classes = train_ds.num_classes

    n_val = int(len(train_ds) * val_fraction)
    train_set, val_set = random_split(
        train_ds, [len(train_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    # Give val its own transform (val_set.dataset is the same object,
    # so we wrap it to swap out the transform cleanly)
    val_set = _TransformSubset(train_ds, val_set.indices, val_transform)

    _, _, train_split, test_split, *_ = DATASETS[name]
    test_loader = None
    if test_split is not None:
        test_ds = HistoDataset(name, split="test", transform=val_transform, cache_dir=cache_dir)
        test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    loader_kw = dict(batch_size=batch_size, num_workers=num_workers)
    return (
        DataLoader(train_set, shuffle=True,  **loader_kw),
        DataLoader(val_set,   shuffle=False, **loader_kw),
        test_loader,
        num_classes,
    )


class _TransformSubset(Dataset):
    """Subset of an HFHistoDataset with an independent transform."""
    def __init__(self, base: HistoDataset, indices, transform):
        self._base = base
        self._indices = indices
        self._transform = transform

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        sample = self._base._hf[self._indices[idx]]
        img = _to_pil(_get_field(sample, self._base._image_col))
        label = _get_field(sample, self._base._label_col)
        if isinstance(label, str):
            label = self._base.class_to_idx[label]
        if self._transform:
            img = self._transform(img)
        return img, int(label)