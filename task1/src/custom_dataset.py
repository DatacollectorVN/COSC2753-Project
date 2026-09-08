from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, Subset

from .transforms import get_train_transforms, get_val_transforms


def _apply_rare_grouping(df: pd.DataFrame, label_col: str, min_class_count: int) -> pd.DataFrame:
    """Map classes with fewer than min_class_count samples to 'Other'.

    From the EDA analysis (task1_analysis.md):
        MIN_CLASS_COUNT=50 preserves 59 named classes covering 97.63% of images,
        mapping 65 rare classes into a single 'Other' bucket.
    """
    if min_class_count <= 0:
        df["model_target"] = df[label_col]
        return df

    counts = df[label_col].value_counts()
    rare_labels = counts[counts < min_class_count].index
    df["model_target"] = df[label_col].where(
        ~df[label_col].isin(rare_labels),
        "Other",
    )
    return df


class FashionDataset(Dataset):
    """Custom Dataset for FashionDataset with OpenCV image loading.

    Expects:
        data_dir/images_{split}/  — folder of .jpg images named by id
        data_dir/styles_{split}.csv or styles_prediction.csv

    Args:
        min_class_count: Classes with fewer samples are grouped into 'Other'.
            Set to 0 to keep all original classes (Experiment A).
            Recommended: 50 (Experiment C from analysis).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        label_col: str = "articleType",
        label_map: dict[str, int] | None = None,
        min_class_count: int = 0,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.label_col = label_col

        if split == "train":
            csv_path = self.data_dir / "styles_train.csv"
        else:
            csv_path = self.data_dir / "styles_prediction.csv"

        self.df = pd.read_csv(csv_path)
        self.image_dir = self.data_dir / f"images_{split}"

        self.df["image_path"] = self.df["id"].apply(
            lambda x: self.image_dir / f"{x}.jpg"
        )
        self.df = self.df[self.df["image_path"].apply(lambda p: p.exists())].reset_index(drop=True)

        self.has_labels = split == "train"
        if self.has_labels:
            self.df = self.df.dropna(subset=[label_col]).reset_index(drop=True)

            # Apply rare-class grouping
            self.df = _apply_rare_grouping(self.df, label_col, min_class_count)

            if label_map is not None:
                self.label_map = label_map
            else:
                unique_labels = sorted(self.df["model_target"].unique())
                self.label_map = {label: idx for idx, label in enumerate(unique_labels)}

            self.df["label"] = self.df["model_target"].map(self.label_map)
            self.df = self.df.dropna(subset=["label"]).reset_index(drop=True)
            self.df["label"] = self.df["label"].astype(int)
        else:
            self.label_map = label_map or {}

    @property
    def num_classes(self) -> int:
        return len(self.label_map)

    @property
    def class_names(self) -> list[str]:
        return [name for name, _ in sorted(self.label_map.items(), key=lambda x: x[1])]

    @property
    def labels(self) -> np.ndarray:
        """Return all labels as numpy array (for stratified splitting)."""
        return self.df["label"].values

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for CrossEntropyLoss.

        weight_i = total_samples / (num_classes * count_i)
        """
        counts = np.bincount(self.df["label"].values, minlength=self.num_classes)
        counts = np.maximum(counts, 1)  # avoid division by zero
        weights = len(self.df) / (self.num_classes * counts)
        return torch.FloatTensor(weights)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        image = cv2.imread(str(row["image_path"]))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {row['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        if self.has_labels:
            return image, row["label"]
        return image, row["id"]


class FashionDatasetTest(Dataset):
    """Test dataset that loads all images from a directory (no labels)."""

    def __init__(self, image_dir: str, transform=None):
        self.image_dir = Path(image_dir)
        self.image_files = sorted(self.image_dir.glob("*.jpg"))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        file_path = self.image_files[idx]
        image = cv2.imread(str(file_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {file_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, file_path.stem


def create_train_val_loaders(
    data_dir: str,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
    label_col: str = "articleType",
    min_class_count: int = 0,
    transform_params: dict | None = None,
) -> tuple[DataLoader, DataLoader, dict[str, int], torch.Tensor]:
    """Create train/val DataLoaders with stratified splitting.

    Uses StratifiedShuffleSplit (sklearn) to guarantee every class appears
    in both train and val sets, as recommended by the EDA analysis.

    Returns:
        (train_loader, val_loader, label_map, class_weights)
    """
    train_transform = get_train_transforms(transform_params)
    val_transform = get_val_transforms(transform_params)

    # Build dataset to get label_map and class distribution
    train_dataset = FashionDataset(
        data_dir=data_dir, split="train", transform=train_transform,
        label_col=label_col, min_class_count=min_class_count,
    )
    label_map = train_dataset.label_map
    class_weights = train_dataset.get_class_weights()

    # Val dataset with val transforms (same label_map)
    val_dataset = FashionDataset(
        data_dir=data_dir, split="train", transform=val_transform,
        label_col=label_col, label_map=label_map, min_class_count=min_class_count,
    )

    # Stratified split — ensures every class is in both train and val.
    # Classes with only 1 sample cannot be stratified, so we merge them into
    # a temporary group for splitting, then restore original labels.
    all_labels = train_dataset.labels
    label_counts = np.bincount(all_labels, minlength=len(label_map))
    stratify_labels = all_labels.copy()
    singleton_mask = label_counts[all_labels] < 2
    if singleton_mask.any():
        # Assign singletons a shared fake label so StratifiedShuffleSplit can handle them
        fake_label = all_labels.max() + 1
        stratify_labels[singleton_mask] = fake_label

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
    return train_loader, val_loader, label_map, class_weights


def create_test_loader(
    image_dir: str,
    batch_size: int = 64,
    num_workers: int = 4,
    transform_params: dict | None = None,
) -> DataLoader:
    """Create test DataLoader for inference."""
    transform = get_val_transforms(transform_params)
    dataset = FashionDatasetTest(image_dir=image_dir, transform=transform)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
