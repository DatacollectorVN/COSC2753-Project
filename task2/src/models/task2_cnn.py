"""Scikit-learn-compatible CNN for Task 2 season classification.

The estimator accepts image paths instead of a precomputed feature matrix. This
keeps GridSearchCV memory usage manageable and lets every convolution operate on
RGB pixels. All weights are learned from scratch on the supplied FashionDataset.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from ..transforms import PadAndResize


class _ImagePathDataset(Dataset):
    """Load and normalize images lazily from paths supplied by GridSearchCV."""

    def __init__(
        self,
        image_paths,
        labels=None,
        *,
        width: int,
        height: int,
        pad_value: int,
        augment: bool,
    ):
        self.image_paths = np.asarray(image_paths, dtype=object).reshape(-1)
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        self.transform = PadAndResize(width=width, height=height, pad_value=pad_value)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image = self.transform(Path(self.image_paths[index]))
        if self.augment:
            if random.random() < 0.5:
                image = np.flip(image, axis=1).copy()
            contrast = random.uniform(0.9, 1.1)
            brightness = random.uniform(-0.05, 0.05)
            image = np.clip((image - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)

        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        tensor = (tensor.float() - 0.5) / 0.5
        if self.labels is None:
            return tensor
        return tensor, int(self.labels[index])


class _ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class Task2CNN(nn.Module):
    """Three-block CNN with a compact spatial classification head."""

    def __init__(
        self,
        num_classes: int,
        input_width: int = 60,
        input_height: int = 80,
        channels=(32, 64, 128),
        dropout_rate: float = 0.3,
        hidden_units: int = 128,
    ):
        super().__init__()
        if len(channels) != 3 or any(int(channel) <= 0 for channel in channels):
            raise ValueError("channels must contain three positive integers")
        if int(input_width) < 16 or int(input_height) < 16:
            raise ValueError("input_width and input_height must be at least 16")

        blocks = []
        previous_channels = 3
        for output_channels in channels:
            blocks.append(_ConvBlock(previous_channels, int(output_channels)))
            previous_channels = int(output_channels)
        self.features = nn.Sequential(*blocks)
        pooled_height = int(input_height) // 16
        pooled_width = int(input_width) // 16
        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Dropout(float(dropout_rate)),
            nn.Linear(
                previous_channels * pooled_height * pooled_width,
                int(hidden_units),
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(float(dropout_rate) / 2.0),
            nn.Linear(int(hidden_units), int(num_classes)),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, images):
        return self.classifier(self.features(images))


class Task2CNNClassifier(ClassifierMixin, BaseEstimator):
    """Expose the PyTorch Task2CNN through the scikit-learn estimator protocol."""

    def __init__(
        self,
        width: int = 60,
        height: int = 80,
        pad_value: int = 255,
        channels=(32, 64, 128),
        dropout_rate: float = 0.3,
        hidden_units: int = 128,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        batch_size: int = 128,
        max_epochs: int = 25,
        patience: int = 5,
        min_delta: float = 0.0001,
        validation_fraction: float = 0.1,
        class_weight_power: float = 0.5,
        augment: bool = True,
        num_workers: int = 0,
        device: str = "auto",
        random_state: int = 42,
        verbose: int = 0,
    ):
        self.width = width
        self.height = height
        self.pad_value = pad_value
        self.channels = channels
        self.dropout_rate = dropout_rate
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.validation_fraction = validation_fraction
        self.class_weight_power = class_weight_power
        self.augment = augment
        self.num_workers = num_workers
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _validate_parameters(self) -> None:
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("width and height must be positive")
        if not 0.0 <= float(self.dropout_rate) < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")
        if int(self.max_epochs) <= 0 or int(self.patience) <= 0:
            raise ValueError("max_epochs and patience must be positive")
        if not 0.0 < float(self.validation_fraction) < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1")
        if not 0.0 <= float(self.class_weight_power) <= 1.0:
            raise ValueError("class_weight_power must be in [0, 1]")

    def _resolve_device(self) -> torch.device:
        requested = str(self.device).lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        resolved = torch.device(requested)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("device='cuda' was requested but CUDA is unavailable")
        if resolved.type == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise ValueError("device='mps' was requested but MPS is unavailable")
        return resolved

    def _set_seed(self) -> None:
        random.seed(int(self.random_state))
        np.random.seed(int(self.random_state))
        torch.manual_seed(int(self.random_state))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.random_state))

    def _loader(self, paths, labels=None, *, augment: bool, shuffle: bool) -> DataLoader:
        dataset = _ImagePathDataset(
            paths,
            labels,
            width=int(self.width),
            height=int(self.height),
            pad_value=int(self.pad_value),
            augment=augment,
        )
        generator = torch.Generator()
        generator.manual_seed(int(self.random_state))
        return DataLoader(
            dataset,
            batch_size=int(self.batch_size),
            shuffle=shuffle,
            num_workers=int(self.num_workers),
            pin_memory=self._training_device_.type == "cuda",
            generator=generator,
        )

    @staticmethod
    def _epoch(model, loader, criterion, device, optimizer=None):
        is_training = optimizer is not None
        model.train(is_training)
        total_loss = 0.0
        total_correct = 0
        total_rows = 0
        predictions = []
        targets = []

        context = torch.enable_grad() if is_training else torch.no_grad()
        with context:
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if is_training:
                    optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                if is_training:
                    loss.backward()
                    optimizer.step()

                batch_predictions = logits.argmax(dim=1)
                batch_rows = labels.size(0)
                total_loss += float(loss.item()) * batch_rows
                total_correct += int((batch_predictions == labels).sum().item())
                total_rows += batch_rows
                predictions.append(batch_predictions.detach().cpu().numpy())
                targets.append(labels.detach().cpu().numpy())

        y_pred = np.concatenate(predictions)
        y_true = np.concatenate(targets)
        return {
            "loss": total_loss / total_rows,
            "accuracy": total_correct / total_rows,
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }

    def fit(self, X, y):
        self._validate_parameters()
        image_paths = np.asarray(X, dtype=object).reshape(-1)
        labels = np.asarray(y).reshape(-1)
        if len(image_paths) != len(labels):
            raise ValueError("X and y must contain the same number of rows")
        if len(labels) < 2:
            raise ValueError("At least two training rows are required")
        missing_paths = [str(path) for path in image_paths if not Path(path).is_file()]
        if missing_paths:
            raise FileNotFoundError(
                f"Missing Task2CNN input images; first paths: {missing_paths[:5]}"
            )

        self._set_seed()
        self.classes_, encoded_labels = np.unique(labels, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("Task2CNN training requires at least two classes")

        indices = np.arange(len(labels))
        train_indices, validation_indices = train_test_split(
            indices,
            test_size=float(self.validation_fraction),
            random_state=int(self.random_state),
            stratify=encoded_labels,
        )
        self._training_device_ = self._resolve_device()
        train_loader = self._loader(
            image_paths[train_indices],
            encoded_labels[train_indices],
            augment=bool(self.augment),
            shuffle=True,
        )
        validation_loader = self._loader(
            image_paths[validation_indices],
            encoded_labels[validation_indices],
            augment=False,
            shuffle=False,
        )

        model = Task2CNN(
            num_classes=len(self.classes_),
            input_width=int(self.width),
            input_height=int(self.height),
            channels=self.channels,
            dropout_rate=float(self.dropout_rate),
            hidden_units=int(self.hidden_units),
        ).to(self._training_device_)
        if int(self.verbose) > 0:
            parameter_count = sum(parameter.numel() for parameter in model.parameters())
            print(
                f"Task2CNN fit | rows={len(train_indices)} train + "
                f"{len(validation_indices)} validation | "
                f"device={self._training_device_.type} | parameters={parameter_count:,}",
                flush=True,
            )

        counts = np.bincount(encoded_labels[train_indices], minlength=len(self.classes_))
        class_weights = (len(train_indices) / (len(self.classes_) * counts)) ** float(
            self.class_weight_power
        )
        criterion = nn.CrossEntropyLoss(
            weight=torch.as_tensor(
                class_weights, dtype=torch.float32, device=self._training_device_
            )
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=max(1, int(self.patience) // 2)
        )

        self.history_ = []
        best_state = None
        best_validation_f1 = -np.inf
        epochs_without_improvement = 0
        for epoch in range(1, int(self.max_epochs) + 1):
            training = self._epoch(
                model, train_loader, criterion, self._training_device_, optimizer=optimizer
            )
            validation = self._epoch(
                model, validation_loader, criterion, self._training_device_
            )
            scheduler.step(validation["macro_f1"])
            row = {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train_loss": float(training["loss"]),
                "validation_loss": float(validation["loss"]),
                "train_accuracy": float(training["accuracy"]),
                "validation_accuracy": float(validation["accuracy"]),
                "train_error": float(1.0 - training["accuracy"]),
                "validation_error": float(1.0 - validation["accuracy"]),
                "train_macro_f1": float(training["macro_f1"]),
                "validation_macro_f1": float(validation["macro_f1"]),
            }
            self.history_.append(row)
            if int(self.verbose) > 0:
                print(
                    f"Task2CNN epoch {epoch:02d}/{int(self.max_epochs)} | "
                    f"train loss={training['loss']:.4f} acc={training['accuracy']:.4f} | "
                    f"validation loss={validation['loss']:.4f} "
                    f"acc={validation['accuracy']:.4f} macro-F1={validation['macro_f1']:.4f}",
                    flush=True,
                )

            if validation["macro_f1"] > best_validation_f1 + float(self.min_delta):
                best_validation_f1 = validation["macro_f1"]
                best_state = copy.deepcopy(model.state_dict())
                self.best_epoch_ = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= int(self.patience):
                    break

        if best_state is None:
            raise RuntimeError("Task2CNN training did not produce a checkpoint")
        model.load_state_dict(best_state)
        self.model_ = model.cpu().eval()
        self.n_iter_ = len(self.history_)
        self.best_validation_macro_f1_ = float(best_validation_f1)
        self.model_parameter_count_ = int(
            sum(parameter.numel() for parameter in self.model_.parameters())
        )
        self._training_device_ = torch.device("cpu")
        return self

    def _predict_logits(self, X) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("Task2CNNClassifier must be fitted before prediction")
        image_paths = np.asarray(X, dtype=object).reshape(-1)
        prediction_device = self._resolve_device()
        self._training_device_ = prediction_device
        loader = self._loader(image_paths, augment=False, shuffle=False)
        self.model_.to(prediction_device).eval()
        logits = []
        with torch.no_grad():
            for images in loader:
                outputs = self.model_(images.to(prediction_device, non_blocking=True))
                logits.append(outputs.cpu().numpy())
        self.model_.cpu()
        self._training_device_ = torch.device("cpu")
        return np.concatenate(logits, axis=0)

    def predict_proba(self, X) -> np.ndarray:
        logits = self._predict_logits(X)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        indices = self.predict_proba(X).argmax(axis=1)
        return self.classes_[indices]


def build_task2_cnn(random_state: int, **model_params):
    """Build Task2CNN inside a pipeline with stable GridSearchCV parameter keys."""
    from sklearn.pipeline import Pipeline

    return Pipeline(
        steps=[
            (
                "classifier",
                Task2CNNClassifier(random_state=random_state, **model_params),
            )
        ]
    )
