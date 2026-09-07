import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .custom_dataset import create_train_val_loaders
from .metric_evaluation import compute_accuracy
from .model import CNN
from .utils import EarlyStopping, SettingConfig, plot_training_curves, set_seed, setup_logger


class FashionTrainer(SettingConfig):
    """Training pipeline — instantiated from JSON config via **kwargs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = setup_logger("train", self.LOG_DIR)

    def _build_model(self, num_classes: int) -> CNN:
        return CNN(
            in_channels=self.IN_CHANNELS,
            num_classes=num_classes,
            dropout_rate=self.DROPOUT_RATE,
        ).to(self.device)

    def _feedforward(self, model, criterion, images, labels):
        outputs = model(images)
        loss = criterion(outputs, labels)
        acc = compute_accuracy(outputs, labels)
        return outputs, loss, acc

    def train(self):
        set_seed(self.SEED)
        self.logger.info(f"Using device: {self.device}")

        # Data
        train_loader, val_loader, label_map = create_train_val_loaders(
            data_dir=self.DATA_DIR_TRAIN,
            image_size=self.IMAGE_SIZE,
            batch_size=self.BATCH_SIZE,
            val_ratio=self.VAL_RATIO,
            num_workers=self.NUM_WORKERS,
            seed=self.SEED,
            label_col=self.LABEL_COL,
        )
        num_classes = len(label_map)
        self.logger.info(f"Classes: {num_classes} | Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

        # Save label_map
        save_path = Path(self.SAVE_MODEL_DIR)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)

        # Model, optimizer, scheduler
        model = self._build_model(num_classes)
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
        early_stopping = EarlyStopping(patience=self.EARLY_STOPPING_PATIENCE, mode="max")

        # Training loop
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

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

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            self.logger.info(
                f"Epoch [{epoch+1}/{self.MAX_EPOCHS}] "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

            # Save best checkpoint
            if early_stopping.best is None or val_acc > early_stopping.best:
                checkpoint = {
                    "state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "num_classes": num_classes,
                    "label_map": label_map,
                }
                torch.save(checkpoint, save_path / "best_model.pth")
                self.logger.info(f"  -> Saved best model (Val Acc: {val_acc:.4f})")

            if early_stopping.step(val_acc):
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Save final model & plots
        torch.save(model.state_dict(), save_path / "final_model.pth")
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, self.SAVE_MODEL_DIR)
        self.logger.info("Training complete.")
