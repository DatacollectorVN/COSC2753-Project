"""Dataset and data loaders for Task 3 — gender + occasion prediction with metadata.

Each sample returns:
    Training:  (image, meta_article, meta_master, meta_colour, gender_label, occasion_label)
    Test:      (image, meta_article, meta_master, meta_colour, image_id)

Metadata columns (articleType, masterCategory, baseColour) are encoded as integer
indices via learned vocabularies. Index 0 is reserved for unknown/missing values.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset

from .transforms import get_train_transforms, get_val_transforms

# Occasion labels with fewer than this many samples are dropped.
# From analysis: Home(1), Party(13), Travel(25), Smart Casual(55) are rare.
RARE_OCCASION_LABELS = {"Home", "Party", "Travel", "Smart Casual"}


def _build_vocab(series: pd.Series) -> dict[str, int]:
    """Build a string → int vocabulary from a pandas Series.

    Index 0 is reserved for unknown/missing values.
    """
    unique_vals = sorted(series.dropna().unique())
    return {"<unk>": 0, **{v: i + 1 for i, v in enumerate(unique_vals)}}


def _encode_meta_col(series: pd.Series, vocab: dict[str, int]) -> np.ndarray:
    """Map a string Series to integer indices using a vocabulary. Unknown → 0."""
    return series.map(lambda x: vocab.get(x, 0) if pd.notna(x) else 0).values.astype(np.int64)


class FashionDataset(Dataset):
    """Dataset for Task 3: two-label (gender + occasion) with metadata features.

    Args:
        data_dir: Root directory containing CSV and image folders.
        split: "train" or "test".
        transform: Image transform pipeline.
        gender_label_map: Pre-built label map for gender (pass from train to val).
        occasion_label_map: Pre-built label map for occasion.
        meta_vocabs: Dict of {col_name: {str: int}} vocabularies for metadata.
        min_occasion_count: Occasions with fewer samples are dropped (default 50).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        gender_label_map: dict[str, int] | None = None,
        occasion_label_map: dict[str, int] | None = None,
        meta_vocabs: dict[str, dict[str, int]] | None = None,
        min_occasion_count: int = 50,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.has_labels = split == "train"

        # Load CSV
        if split == "train":
            csv_path = self.data_dir / "styles_train.csv"
        else:
            csv_path = self.data_dir / "styles_prediction.csv"
        self.df = pd.read_csv(csv_path)

        # Resolve image paths and filter missing
        self.image_dir = self.data_dir / f"images_{split}"
        self.df["image_path"] = self.df["id"].apply(lambda x: self.image_dir / f"{x}.jpg")
        self.df = self.df[self.df["image_path"].apply(lambda p: p.exists())].reset_index(drop=True)

        if self.has_labels:
            # Drop rows missing either target
            self.df = self.df.dropna(subset=["gender", "usage"]).reset_index(drop=True)

            # Drop rare occasion labels
            if min_occasion_count > 0:
                occasion_counts = self.df["usage"].value_counts()
                rare = occasion_counts[occasion_counts < min_occasion_count].index
                self.df = self.df[~self.df["usage"].isin(rare)].reset_index(drop=True)

            # Build or reuse label maps
            if gender_label_map is not None:
                self.gender_label_map = gender_label_map
            else:
                unique_gender = sorted(self.df["gender"].unique())
                self.gender_label_map = {g: i for i, g in enumerate(unique_gender)}

            if occasion_label_map is not None:
                self.occasion_label_map = occasion_label_map
            else:
                unique_occasion = sorted(self.df["usage"].unique())
                self.occasion_label_map = {o: i for i, o in enumerate(unique_occasion)}

            # Encode labels
            self.df["gender_label"] = self.df["gender"].map(self.gender_label_map)
            self.df["occasion_label"] = self.df["usage"].map(self.occasion_label_map)

            # Drop rows that don't map (e.g. val set has a class not in train label map)
            self.df = self.df.dropna(subset=["gender_label", "occasion_label"]).reset_index(drop=True)
            self.df["gender_label"] = self.df["gender_label"].astype(int)
            self.df["occasion_label"] = self.df["occasion_label"].astype(int)
        else:
            self.gender_label_map = gender_label_map or {}
            self.occasion_label_map = occasion_label_map or {}

        # Build or reuse metadata vocabularies
        meta_cols = ["articleType", "masterCategory", "baseColour"]
        if meta_vocabs is not None:
            self.meta_vocabs = meta_vocabs
        else:
            self.meta_vocabs = {col: _build_vocab(self.df[col]) for col in meta_cols}

        # Encode metadata columns as integer indices
        for col in meta_cols:
            self.df[f"meta_{col}"] = _encode_meta_col(self.df[col], self.meta_vocabs[col])

    @property
    def num_gender_classes(self) -> int:
        return len(self.gender_label_map)

    @property
    def num_occasion_classes(self) -> int:
        return len(self.occasion_label_map)

    @property
    def gender_class_names(self) -> list[str]:
        return [n for n, _ in sorted(self.gender_label_map.items(), key=lambda x: x[1])]

    @property
    def occasion_class_names(self) -> list[str]:
        return [n for n, _ in sorted(self.occasion_label_map.items(), key=lambda x: x[1])]

    @property
    def labels(self) -> np.ndarray:
        """Combined label for stratified splitting (gender * N_occasion + occasion)."""
        n_occ = self.num_occasion_classes
        return self.df["gender_label"].values * n_occ + self.df["occasion_label"].values

    def get_class_weights(self, target: str = "gender") -> torch.Tensor:
        """Inverse-frequency weights for CrossEntropyLoss.

        Args:
            target: "gender" or "occasion".
        """
        col = "gender_label" if target == "gender" else "occasion_label"
        num_classes = self.num_gender_classes if target == "gender" else self.num_occasion_classes
        counts = np.bincount(self.df[col].values, minlength=num_classes)
        counts = np.maximum(counts, 1)
        weights = len(self.df) / (num_classes * counts)
        return torch.FloatTensor(weights)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Image
        image = cv2.imread(str(row["image_path"]))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {row['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)

        # Metadata indices
        meta_article = row["meta_articleType"]
        meta_master = row["meta_masterCategory"]
        meta_colour = row["meta_baseColour"]

        if self.has_labels:
            return image, meta_article, meta_master, meta_colour, row["gender_label"], row["occasion_label"]
        return image, meta_article, meta_master, meta_colour, row["id"]


class FashionDatasetTest(Dataset):
    """Test dataset — loads images from a directory with optional CSV metadata."""

    def __init__(
        self,
        image_dir: str,
        transform=None,
        csv_path: str | None = None,
        meta_vocabs: dict[str, dict[str, int]] | None = None,
    ):
        self.image_dir = Path(image_dir)
        self.image_files = sorted(self.image_dir.glob("*.jpg"))
        self.transform = transform
        self.meta_vocabs = meta_vocabs or {}

        # Try to load metadata from CSV if provided
        self.meta_df = None
        if csv_path is not None and Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            df["id"] = df["id"].astype(str)
            self.meta_df = df.set_index("id")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        file_path = self.image_files[idx]
        image_id = file_path.stem

        image = cv2.imread(str(file_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {file_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)

        # Metadata — use CSV if available, else zeros (unknown)
        meta_article, meta_master, meta_colour = 0, 0, 0
        if self.meta_df is not None and image_id in self.meta_df.index:
            row = self.meta_df.loc[image_id]
            for col, attr in [("articleType", "meta_article"), ("masterCategory", "meta_master"), ("baseColour", "meta_colour")]:
                vocab = self.meta_vocabs.get(col, {})
                val = row.get(col, None)
                idx_val = vocab.get(val, 0) if pd.notna(val) else 0
                if attr == "meta_article":
                    meta_article = idx_val
                elif attr == "meta_master":
                    meta_master = idx_val
                else:
                    meta_colour = idx_val

        return image, meta_article, meta_master, meta_colour, image_id


def create_train_val_loaders(
    data_dir: str,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
    min_occasion_count: int = 50,
    transform_params: dict | None = None,
) -> tuple:
    """Create train/val DataLoaders with stratified splitting on combined labels.

    Returns:
        (train_loader, val_loader, gender_label_map, occasion_label_map,
         gender_weights, occasion_weights, meta_vocabs)
    """
    train_transform = get_train_transforms(transform_params)
    val_transform = get_val_transforms(transform_params)

    # Build full dataset to get label maps and vocabs
    train_dataset = FashionDataset(
        data_dir=data_dir, split="train", transform=train_transform,
        min_occasion_count=min_occasion_count,
    )
    gender_label_map = train_dataset.gender_label_map
    occasion_label_map = train_dataset.occasion_label_map
    meta_vocabs = train_dataset.meta_vocabs
    gender_weights = train_dataset.get_class_weights("gender")
    occasion_weights = train_dataset.get_class_weights("occasion")

    # Val dataset (same label maps and vocabs)
    val_dataset = FashionDataset(
        data_dir=data_dir, split="train", transform=val_transform,
        gender_label_map=gender_label_map, occasion_label_map=occasion_label_map,
        meta_vocabs=meta_vocabs, min_occasion_count=min_occasion_count,
    )

    # Stratified split on combined gender×occasion label.
    # Some gender×occasion pairs may have very few samples, so we merge
    # any class with < 2 members into a shared bucket for splitting.
    all_labels = train_dataset.labels
    stratify_labels = all_labels.copy()
    max_label = int(all_labels.max())
    label_counts = np.bincount(all_labels, minlength=max_label + 1)
    rare_mask = label_counts[all_labels] < 2
    if rare_mask.any():
        fake_label = max_label + 1
        stratify_labels[rare_mask] = fake_label
        # If the merged bucket itself is still < 2, merge it with the
        # most common class so StratifiedShuffleSplit can proceed.
        if rare_mask.sum() < 2:
            most_common = int(np.argmax(label_counts))
            stratify_labels[rare_mask] = most_common

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(stratify_labels)), stratify_labels))

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader, val_loader,
        gender_label_map, occasion_label_map,
        gender_weights, occasion_weights,
        meta_vocabs,
    )


def create_test_loader(
    image_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    transform_params: dict | None = None,
    csv_path: str | None = None,
    meta_vocabs: dict[str, dict[str, int]] | None = None,
) -> DataLoader:
    """Create test DataLoader for inference."""
    transform = get_val_transforms(transform_params)
    dataset = FashionDatasetTest(
        image_dir=image_dir, transform=transform,
        csv_path=csv_path, meta_vocabs=meta_vocabs,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
