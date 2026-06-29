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
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from datasets import load_dataset


# Registry: name → (hf_repo, hf_subset, hf_train_split, hf_test_split,
#                    image_col, label_col, classes, patient_col)
#
# image_col   : HF column name for the image
# label_col   : HF column name for the label; use "outer.inner" for nested dicts
# classes     : list of class names when labels are strings, or None to detect
#               from a ClassLabel feature automatically
# patient_col : HF column whose value encodes a patient/slide ID, used for
#               patient-level val splitting to avoid slide-leakage.
#               None → fall back to random patch-level splitting.
#               "__key__" → TCGA webdataset key (parsed as TCGA-XX-XXXX/...)
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
    #                  repo                   subset  train_split               test             img_col  lbl_col       classes               patient_col
    "tcga-ut":        ("dakomura/tcga-ut",    None,   "train",                 "test",          "jpg",   "json.label", _TCGA_UT_CLASSES,     "__key__"),
    "nct-crc-100k":   ("1aurent/NCT-CRC-HE", None,   "NCT_CRC_HE_100K",       "CRC_VAL_HE_7K", "image", "label",      None,                 None),
    "nct-crc-nonorm": ("1aurent/NCT-CRC-HE", None,   "NCT_CRC_HE_100K_NONORM","CRC_VAL_HE_7K", "image", "label",      None,                 None),
}

# ---------------------------------------------------------------------------
# MHIST — local dataset (not on HuggingFace)
# ---------------------------------------------------------------------------
_MHIST_CLASSES = ["HP", "SSA"]
_MHIST_ROOT = Path(__file__).resolve().parent / "mhist"


class MHISTDataset(Dataset):
    """Local MHIST dataset (images/ dir + annotations.csv)."""

    def __init__(self, split: str = "train", transform: Optional[Callable] = None,
                 root: Optional[Path] = None):
        import pandas as pd
        self.root = Path(root) if root else _MHIST_ROOT
        ann = pd.read_csv(self.root / "annotations.csv")
        partition = "train" if split == "train" else "test"
        ann = ann[ann["Partition"] == partition].reset_index(drop=True)
        self.filenames = ann["Image Name"].tolist()
        self.labels = ann["Majority Vote Label"].tolist()
        self.agreement = ann["Number of Annotators who Selected SSA (Out of 7)"].tolist()
        self.classes = _MHIST_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform
        print(f"Loaded 'mhist' ({split}): {len(self):,} samples, {len(self.classes)} classes")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = Image.open(self.root / "images" / self.filenames[idx]).convert("RGB")
        label = self.class_to_idx[self.labels[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label

    @property
    def num_classes(self):
        return len(self.classes)


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
        repo, subset, train_split, test_split, image_col, label_col, predefined_classes, *_ = DATASETS[name]

        hf_split = train_split if split == "train" else test_split
        assert hf_split is not None, f"'{name}' has no '{split}' split."

        self.transform = transform
        self.streaming = streaming
        self._image_col = image_col
        self._label_col = label_col
        load_kw = {} if subset is None else {"name": subset}
        self._hf = load_dataset(repo, split=hf_split,
                                streaming=streaming, cache_dir=cache_dir,
                                num_proc=1, **load_kw)

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


def _patient_id_from_tcga_key(key: str) -> str:
    """
    Extract a TCGA patient barcode from a webdataset __key__ field.

    Example key:  "TCGA-FS-A1Z7-06Z-00-DX7/0_8_521"
    Slide ID:     "TCGA-FS-A1Z7-06Z-00-DX7"   (part before '/')
    Patient ID:   "TCGA-FS-A1Z7"               (first 3 hyphen-fields)
    """
    slide_id = key.split("/")[0]
    return "-".join(slide_id.split("-")[:3])


def _patient_level_split(hf_dataset, patient_col: str, val_fraction: float, seed: int = 42):
    """
    Split dataset indices by patient rather than by patch.

    Reads only the patient_col (no image decoding), groups patch indices by
    patient ID, then assigns whole patients to train or val.  This prevents
    patches from the same patient appearing in both splits (slide-leakage).

    Args:
        hf_dataset   : HuggingFace Dataset (non-streaming, arrow-backed)
        patient_col  : column name containing patient/slide info
        val_fraction : fraction of *patients* (not patches) to hold out
        seed         : shuffle seed for reproducibility

    Returns:
        (train_indices, val_indices)  — lists of integer patch indices
    """
    print(f"  Building patient-level val split from '{patient_col}' column...", flush=True)

    # Arrow-backed column access — fast, no image decoding
    keys = hf_dataset[patient_col]

    patient_to_indices: dict = defaultdict(list)
    for idx, key in enumerate(keys):
        pid = _patient_id_from_tcga_key(key)
        patient_to_indices[pid].append(idx)

    patients = sorted(patient_to_indices.keys())
    rng = random.Random(seed)
    rng.shuffle(patients)

    n_val_patients = max(1, int(len(patients) * val_fraction))
    val_patients = set(patients[:n_val_patients])

    train_indices, val_indices = [], []
    for pid, indices in patient_to_indices.items():
        if pid in val_patients:
            val_indices.extend(indices)
        else:
            train_indices.extend(indices)

    print(
        f"  Patient-level split: "
        f"{len(patients) - n_val_patients} train patients ({len(train_indices):,} patches) / "
        f"{n_val_patients} val patients ({len(val_indices):,} patches)"
    )
    return train_indices, val_indices


def _compute_class_weights(labels, num_classes: int) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for balanced training.

    Returns a tensor of shape (num_classes,) where each weight is
    n_total / (num_classes * n_class), clamped to [0.1, 10.0] for stability.
    """
    counts = torch.zeros(num_classes)
    for label in labels:
        counts[int(label)] += 1
    # Inverse frequency: n_total / (num_classes * count_per_class)
    weights = counts.sum() / (num_classes * counts.clamp(min=1))
    weights = weights.clamp(min=0.1, max=10.0)
    print(f"  Class weights: min={weights.min():.2f}, max={weights.max():.2f}, "
          f"ratio={weights.max()/weights.min():.1f}x")
    return weights


def _get_mhist_dataloaders(
    train_transform, val_transform, val_fraction, batch_size, num_workers, seed,
):
    """Build train/val/test loaders for the local MHIST dataset."""
    train_ds = MHISTDataset(split="train", transform=train_transform)
    num_classes = train_ds.num_classes

    n_val = int(len(train_ds) * val_fraction)
    _train, _val = random_split(
        train_ds, [len(train_ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    # Wrap with independent transforms
    train_set = _MHISTSubset(train_ds, list(_train.indices), train_transform)
    val_set = _MHISTSubset(train_ds, list(_val.indices), val_transform)

    # Class weights from training split
    train_labels = [train_ds.class_to_idx[train_ds.labels[i]] for i in _train.indices]
    class_weights = _compute_class_weights(train_labels, num_classes)

    test_ds = MHISTDataset(split="test", transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    g = torch.Generator().manual_seed(seed)
    loader_kw = dict(batch_size=batch_size, num_workers=num_workers)
    return (
        DataLoader(train_set, shuffle=True, generator=g, **loader_kw),
        DataLoader(val_set, shuffle=False, **loader_kw),
        test_loader,
        num_classes,
        class_weights,
    )


class _MHISTSubset(Dataset):
    """Subset of MHISTDataset with an independent transform."""
    def __init__(self, base: MHISTDataset, indices, transform):
        self._base = base
        self._indices = indices
        self._transform = transform

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        img = Image.open(self._base.root / "images" / self._base.filenames[self._indices[idx]]).convert("RGB")
        label = self._base.class_to_idx[self._base.labels[self._indices[idx]]]
        if self._transform:
            img = self._transform(img)
        return img, label


def get_dataloaders(
    name: str,
    train_transform: Callable,
    val_transform: Callable,
    val_fraction: float = 0.1,
    batch_size: int = 32,
    num_workers: int = 4,
    cache_dir: Optional[str] = None,
    seed: int = 42,
    train_subset: Optional[int] = None,
):
    """
    Returns (train_loader, val_loader, test_loader, num_classes, class_weights).

    Val set is carved from the training split.  When the dataset registry
    specifies a patient_col, splitting is done at the patient level to avoid
    slide-leakage (patches from the same patient in both train and val).
    Otherwise falls back to random patch-level splitting.

    class_weights is a tensor of shape (num_classes,) with inverse-frequency
    weights computed from the training split, suitable for CrossEntropyLoss.
    """
    # Local datasets not in HF registry
    if name == "mhist":
        return _get_mhist_dataloaders(
            train_transform, val_transform, val_fraction, batch_size,
            num_workers, seed,
        )
    train_ds = HistoDataset(name, split="train", transform=train_transform, cache_dir=cache_dir)
    num_classes = train_ds.num_classes

    *_, patient_col = DATASETS[name]

    if patient_col is not None:
        train_indices, val_indices = _patient_level_split(
            train_ds._hf, patient_col, val_fraction, seed=seed
        )
        train_set = _TransformSubset(train_ds, train_indices, train_transform)
        val_set   = _TransformSubset(train_ds, val_indices,   val_transform)
    else:
        n_val = int(len(train_ds) * val_fraction)
        _train, _val = random_split(
            train_ds, [len(train_ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(seed),
        )
        train_indices = list(_train.indices)
        train_set = _TransformSubset(train_ds, train_indices, train_transform)
        val_set   = _TransformSubset(train_ds, list(_val.indices),   val_transform)

    # Optionally subsample training set (for ablation studies)
    if train_subset is not None and train_subset < len(train_indices):
        rng = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(train_indices), generator=rng)[:train_subset]
        train_indices = [train_indices[i] for i in perm.tolist()]
        train_set = _TransformSubset(train_ds, train_indices, train_transform)
        print(f"  Subsampled training set: {train_subset} / {len(train_ds)} samples")

    # Compute inverse-frequency class weights from training labels
    label_col = train_ds._label_col
    train_labels = [_get_field(train_ds._hf[i], label_col) for i in train_indices]
    train_labels = [train_ds.class_to_idx[l] if isinstance(l, str) else int(l)
                    for l in train_labels]
    class_weights = _compute_class_weights(train_labels, num_classes)

    _, _, _, test_split, *_ = DATASETS[name]
    test_loader = None
    if test_split is not None:
        test_ds = HistoDataset(name, split="test", transform=val_transform, cache_dir=cache_dir)
        test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    g = torch.Generator().manual_seed(seed)
    loader_kw = dict(batch_size=batch_size, num_workers=num_workers)
    return (
        DataLoader(train_set, shuffle=True,  generator=g, **loader_kw),
        DataLoader(val_set,   shuffle=False, **loader_kw),
        test_loader,
        num_classes,
        class_weights,
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