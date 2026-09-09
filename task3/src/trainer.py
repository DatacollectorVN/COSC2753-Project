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
    """Training pipeline for the two-head hybrid model (gender + occasion)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._raw_params = kwargs

    def _build_model(
        self,
        num_gender_classes: int,
        num_occasion_classes: int,
        meta_vocabs: dict,
    ) -> nn.Module:
        model_params = getattr(self, "MODEL_PARAMS", {})
        return build_model(
            model_name=self.MODEL_NAME,
            in_channels=self.IN_CHANNELS,
            num_gender_classes=num_gender_classes,
            num_occasion_classes=num_occasion_classes,
            num_article_types=len(meta_vocabs["articleType"]),
            num_master_categories=len(meta_vocabs["masterCategory"]),
            num_colours=len(meta_vocabs["baseColour"]),
            **model_params,
        ).to(self.device)

    def train(self):
        run_dir = create_run_dir(self.SAVE_MODEL_DIR, "train")
        logger = setup_logger("train", run_dir / "logs.txt")

        set_seed(self.SEED)
        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model: {self.MODEL_NAME} | Params: {getattr(self, 'MODEL_PARAMS', {})}")

        with open(run_dir / "parameters.json", "w") as f:
            json.dump(self._raw_params, f, indent=2)

        # Data
        min_occasion_count = getattr(self, "MIN_OCCASION_COUNT", 50)
        transform_params = getattr(self, "TRANSFORM_PARAMS", {})

        (
            train_loader, val_loader,
            gender_label_map, occasion_label_map,
            gender_weights, occasion_weights,
            meta_vocabs,
        ) = create_train_val_loaders(
            data_dir=self.DATA_DIR_TRAIN,
            batch_size=self.BATCH_SIZE,
            val_ratio=self.VAL_RATIO,
            num_workers=self.NUM_WORKERS,
            seed=self.SEED,
            min_occasion_count=min_occasion_count,
            transform_params=transform_params,
        )

        num_gender = len(gender_label_map)
        num_occasion = len(occasion_label_map)
        logger.info(f"Gender classes: {num_gender} | Occasion classes: {num_occasion}")
        logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

        # Save label maps and meta vocabs
        with open(run_dir / "gender_label_map.json", "w") as f:
            json.dump(gender_label_map, f, indent=2)
        with open(run_dir / "occasion_label_map.json", "w") as f:
            json.dump(occasion_label_map, f, indent=2)
        with open(run_dir / "meta_vocabs.json", "w") as f:
            json.dump(meta_vocabs, f, indent=2)

        # Model
        model = self._build_model(num_gender, num_occasion, meta_vocabs)

        # Loss — separate criterion per head
        use_class_weights = getattr(self, "USE_CLASS_WEIGHTS", False)
        if use_class_weights:
            criterion_gender = nn.CrossEntropyLoss(weight=gender_weights.to(self.device))
            criterion_occasion = nn.CrossEntropyLoss(weight=occasion_weights.to(self.device))
            logger.info("Using class-weighted CrossEntropyLoss for both heads")
        else:
            criterion_gender = nn.CrossEntropyLoss()
            criterion_occasion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.LEARNING_RATE,
            weight_decay=self.WEIGHT_DECAY,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min",
            factor=self.LR_SCHEDULE_FACTOR,
            patience=self.LR_PATIENCE,
        )

        save_best = getattr(self, "SAVE_BEST", "val_loss")
        if save_best in ("val_gender_acc", "val_occasion_acc"):
            early_stopping = EarlyStopping(patience=self.EARLY_STOPPING_PATIENCE, mode="max")
        else:
            early_stopping = EarlyStopping(patience=self.EARLY_STOPPING_PATIENCE, mode="min")
        logger.info(f"Save best model by: {save_best}")

        scores = {"epochs": []}

        for epoch in range(self.MAX_EPOCHS):
            # ── Train ──
            model.train()
            r_loss, r_g_acc, r_o_acc, total = 0.0, 0.0, 0.0, 0
            for images, m_art, m_mst, m_col, g_labels, o_labels in train_loader:
                images = images.to(self.device)
                m_art = m_art.to(self.device, dtype=torch.long)
                m_mst = m_mst.to(self.device, dtype=torch.long)
                m_col = m_col.to(self.device, dtype=torch.long)
                g_labels = g_labels.to(self.device, dtype=torch.long)
                o_labels = o_labels.to(self.device, dtype=torch.long)

                optimizer.zero_grad()
                g_logits, o_logits = model(images, m_art, m_mst, m_col)
                loss_g = criterion_gender(g_logits, g_labels)
                loss_o = criterion_occasion(o_logits, o_labels)
                loss = loss_g + loss_o
                loss.backward()
                optimizer.step()

                bs = images.size(0)
                r_loss += loss.item() * bs
                r_g_acc += compute_accuracy(g_logits, g_labels) * bs
                r_o_acc += compute_accuracy(o_logits, o_labels) * bs
                total += bs

            train_loss = r_loss / total
            train_g_acc = r_g_acc / total
            train_o_acc = r_o_acc / total

            # ── Validate ──
            model.eval()
            v_loss, v_g_acc, v_o_acc, v_total = 0.0, 0.0, 0.0, 0
            with torch.no_grad():
                for images, m_art, m_mst, m_col, g_labels, o_labels in val_loader:
                    images = images.to(self.device)
                    m_art = m_art.to(self.device, dtype=torch.long)
                    m_mst = m_mst.to(self.device, dtype=torch.long)
                    m_col = m_col.to(self.device, dtype=torch.long)
                    g_labels = g_labels.to(self.device, dtype=torch.long)
                    o_labels = o_labels.to(self.device, dtype=torch.long)

                    g_logits, o_logits = model(images, m_art, m_mst, m_col)
                    loss_g = criterion_gender(g_logits, g_labels)
                    loss_o = criterion_occasion(o_logits, o_labels)
                    loss = loss_g + loss_o

                    bs = images.size(0)
                    v_loss += loss.item() * bs
                    v_g_acc += compute_accuracy(g_logits, g_labels) * bs
                    v_o_acc += compute_accuracy(o_logits, o_labels) * bs
                    v_total += bs

            val_loss = v_loss / v_total
            val_g_acc = v_g_acc / v_total
            val_o_acc = v_o_acc / v_total

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            epoch_scores = {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train_loss": round(train_loss, 6),
                "train_gender_acc": round(train_g_acc, 6),
                "train_occasion_acc": round(train_o_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_gender_acc": round(val_g_acc, 6),
                "val_occasion_acc": round(val_o_acc, 6),
            }
            scores["epochs"].append(epoch_scores)

            logger.info(
                f"Epoch [{epoch+1}/{self.MAX_EPOCHS}] "
                f"LR: {current_lr:.2e} | "
                f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                f"Gender: {train_g_acc:.4f}/{val_g_acc:.4f} | "
                f"Occasion: {train_o_acc:.4f}/{val_o_acc:.4f}"
            )

            metric_map = {
                "val_loss": val_loss,
                "val_gender_acc": val_g_acc,
                "val_occasion_acc": val_o_acc,
                "train_loss": train_loss,
            }
            current_metric = metric_map[save_best]

            if early_stopping.best is None or (
                current_metric > early_stopping.best if early_stopping.mode == "max"
                else current_metric < early_stopping.best
            ):
                checkpoint = {
                    "state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_gender_acc": val_g_acc,
                    "val_occasion_acc": val_o_acc,
                    "num_gender_classes": num_gender,
                    "num_occasion_classes": num_occasion,
                    "gender_label_map": gender_label_map,
                    "occasion_label_map": occasion_label_map,
                    "meta_vocabs": meta_vocabs,
                    "model_name": self.MODEL_NAME,
                    "model_params": getattr(self, "MODEL_PARAMS", {}),
                    "min_occasion_count": min_occasion_count,
                    "transform_params": transform_params,
                }
                torch.save(checkpoint, run_dir / "best_model.pth")
                logger.info(f"  -> Saved best model ({save_best}: {current_metric:.4f})")

            if early_stopping.step(current_metric):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        scores["best_epoch"] = (early_stopping.best is not None) and {
            "metric": save_best,
            "value": round(early_stopping.best, 6),
        }
        with open(run_dir / "scores.json", "w") as f:
            json.dump(scores, f, indent=2)

        # Plot training curves
        train_losses = [e["train_loss"] for e in scores["epochs"]]
        val_losses = [e["val_loss"] for e in scores["epochs"]]
        train_accs = [e["train_gender_acc"] for e in scores["epochs"]]
        val_accs = [e["val_gender_acc"] for e in scores["epochs"]]
        plot_training_curves(train_losses, val_losses, train_accs, val_accs, run_dir)

        logger.info(f"Training complete. Artifacts saved to: {run_dir}")
