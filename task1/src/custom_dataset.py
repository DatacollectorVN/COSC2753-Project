from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

from .transforms import get_train_transforms, get_val_transforms


class FashionDataset(Dataset):
    """Custom Dataset for FashionDataset with OpenCV image loading.

    Expects:
        data_dir/images_{split}/  — folder of .jpg images named by id
        data_dir/styles_{split}.csv or styles_prediction.csv
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform=None,
        label_col: str = "articleType",
        label_map: dict[str, int] | None = None,
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
            if label_map is not None:
                self.label_map = label_map
            else:
                unique_labels = sorted(self.df[label_col].unique())
                self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
            self.df["label"] = self.df[label_col].map(self.label_map)
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
    image_size: int = 224,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    num_workers: int = 4,
    seed: int = 42,
    label_col: str = "articleType",
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    """Create train/val DataLoaders using SubsetRandomSampler.

    Returns:
        (train_loader, val_loader, label_map)
    """
    train_transform = get_train_transforms(image_size)
    val_transform = get_val_transforms(image_size)

    train_dataset = FashionDataset(
        data_dir=data_dir, split="train", transform=train_transform, label_col=label_col,
    )
    label_map = train_dataset.label_map

    val_dataset = FashionDataset(
        data_dir=data_dir, split="train", transform=val_transform,
        label_col=label_col, label_map=label_map,
    )

    num_samples = len(train_dataset)
    indices = list(range(num_samples))
    np.random.seed(seed)
    np.random.shuffle(indices)

    split = int(np.floor(val_ratio * num_samples))
    val_idx, train_idx = indices[:split], indices[split:]

    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, label_map


def create_test_loader(
    image_dir: str,
    image_size: int = 224,
    batch_size: int = 64,
    num_workers: int = 4,
) -> DataLoader:
    """Create test DataLoader for inference."""
    transform = get_val_transforms(image_size)
    dataset = FashionDatasetTest(image_dir=image_dir, transform=transform)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
