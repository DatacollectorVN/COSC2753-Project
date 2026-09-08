import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .custom_dataset import create_train_val_loaders
from .metric_evaluation import compute_accuracy
from .models import build_model
from .utils import EarlyStopping, SettingConfig, create_run_dir, plot_training_curves, set_seed, setup_logger


class FashionTrainer(SettingConfig):
    """Training pipeline — instantiated from JSON config via **kwargs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._raw_params = kwargs

    def _build_model(self, num_classes: int) -> nn.Module:
        model_params = getattr(self, "MODEL_PARAMS", {})
        return build_model(
            model_name=self.MODEL_NAME,
            in_channels=self.IN_CHANNELS,
            num_classes=num_classes,
            **model_params,
        ).to(self.device)

    def _feedforward(self, model, criterion, images, labels):
        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = compute_accuracy(outputs, labels)
        return outputs, loss, acc

    def train(self):
        # Create run directory: SAVE_MODEL_DIR/train/yyyymmdd_hhmmss/
        run_dir = create_run_dir(self.SAVE_MODEL_DIR, "train")
        logger = setup_logger("train", run_dir / "logs.txt")

        set_seed(self.SEED)
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model: {self.MODEL_NAME} | Params: {getattr(self, 'MODEL_PARAMS', {})}")

        # Save parameters.json
        with open(run_dir / "parameters.json", "w") as f:
            json.dump(self._raw_params, f, indent=2)

        # Data
        min_class_count = getattr(self, "MIN_CLASS_COUNT", 0)
        transform_params = getattr(self, "TRANSFORM_PARAMS", {})
        train_loader, val_loader, label_map, class_weights = create_train_val_loaders(
            data_dir=self.DATA_DIR_TRAIN,
            batch_size=self.BATCH_SIZE,
            val_ratio=self.VAL_RATIO,
            num_workers=self.NUM_WORKERS,
            seed=self.SEED,
            label_col=self.LABEL_COL,
            min_class_count=min_class_count,
            transform_params=transform_params,
        )
        num_classes = len(label_map)
        if min_class_count > 0:
            logger.info(f"Rare-class grouping: classes with < {min_class_count} samples → 'Other'")
        logger.info(f"Classes: {num_classes} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

        # Save label_map.json
        with open(run_dir / "label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)

        # Model, optimizer, scheduler
        model = self._build_model(num_classes)

        use_class_weights = getattr(self, "USE_CLASS_WEIGHTS", False)
        if use_class_weights:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            logger.info("Using class-weighted CrossEntropyLoss")
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.LEARNING_RATE,
            weight_decay=self.WEIGHT_DECAY,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.LR_SCHEDULE_FACTOR,
            patience=self.LR_PATIENCE,
        )

        # Determine which metric to use for best-model saving
        save_best = getattr(self, "SAVE_BEST", "val_acc")
        if save_best in ("val_acc", "train_acc"):
            early_stopping = EarlyStopping(patience=self.EARLY_STOPPING_PATIENCE, mode="max")
        else:
            early_stopping = EarlyStopping(patience=self.EARLY_STOPPING_PATIENCE, mode="min")
        logger.info(f"Save best model by: {save_best}")

        # Training loop — track scores per epoch
        scores = {"epochs": []}

        for epoch in range(self.MAX_EPOCHS):
            # --- Train ---
            model.train()
            running_loss, running_acc, total = 0.0, 0.0, 0
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                _, loss, acc = self._feedforward(model, criterion, images, labels)
                loss.backward()
                optimizer.step()

                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                running_acc += acc * batch_size
                total += batch_size

            train_loss = running_loss / total
            train_acc = running_acc / total

            # --- Validate ---
            model.eval()
            val_running_loss, val_running_acc, val_total = 0.0, 0.0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    _, loss, acc = self._feedforward(model, criterion, images, labels)

                    batch_size = images.size(0)
                    val_running_loss += loss.item() * batch_size
                    val_running_acc += acc * batch_size
                    val_total += batch_size

            val_loss = val_running_loss / val_total
            val_acc = val_running_acc / val_total

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # Record epoch scores
            epoch_scores = {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
            }
            scores["epochs"].append(epoch_scores)

            logger.info(
                f"Epoch [{epoch+1}/{self.MAX_EPOCHS}] "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

            # Determine current metric for best-model check
            metric_map = {
                "val_acc": val_acc,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "train_loss": train_loss,
            }
            current_metric = metric_map[save_best]

            # Save best checkpoint
            if early_stopping.best is None or (
                current_metric > early_stopping.best if early_stopping.mode == "max"
                else current_metric < early_stopping.best
            ):
                checkpoint = {
                    "state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "num_classes": num_classes,
                    "label_map": label_map,
                    "model_name": self.MODEL_NAME,
                    "model_params": getattr(self, "MODEL_PARAMS", {}),
                    "min_class_count": min_class_count,
                    "transform_params": transform_params,
                }
                torch.save(checkpoint, run_dir / "best_model.pth")
                logger.info(f"  -> Saved best model ({save_best}: {current_metric:.4f})")

            if early_stopping.step(current_metric):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Save scores.json (all epoch metrics for later visualization)
        scores["best_epoch"] = (early_stopping.best is not None) and {
            "metric": save_best,
            "value": round(early_stopping.best, 6),
        }
        with open(run_dir / "scores.json", "w") as f:
            json.dump(scores, f, indent=2)

        # Save training curves plot
        train_losses = [e["train_loss"] for e in scores["epochs"]]
        val_losses = [e["val_loss"] for e in scores["epochs"]]
        train_accs = [e["train_acc"] for e in scores["epochs"]]
        val_accs = [e["val_acc"] for e in scores["epochs"]]
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, run_dir)

        logger.info(f"Training complete. Artifacts saved to: {run_dir}")
