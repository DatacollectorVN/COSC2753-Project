import csv
import json
from pathlib import Path

import torch

from .custom_dataset import create_test_loader, create_train_val_loaders
from .metric_evaluation import compute_metrics
from .model import CNN
from .utils import SettingConfig, setup_logger


class FashionEvaluator(SettingConfig):
    """Evaluation & prediction pipeline — instantiated from JSON config via **kwargs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = setup_logger("evaluate", self.LOG_DIR)

    def _load_model(self) -> tuple[CNN, dict[str, int]]:
        checkpoint = torch.load(self.CHECKPOINT_PATH, map_location=self.device)
        label_map = checkpoint["label_map"]
        num_classes = checkpoint["num_classes"]

        model = CNN(
            in_channels=self.IN_CHANNELS,
            num_classes=num_classes,
            dropout_rate=self.DROPOUT_RATE,
        ).to(self.device)
        model.load_state_dict(checkpoint["state"])
        model.eval()
        return model, label_map

    def evaluate(self) -> dict:
        """Evaluate on validation split and print classification report."""
        model, label_map = self._load_model()
        class_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]

        _, val_loader, _ = create_train_val_loaders(
            data_dir=self.DATA_DIR_TRAIN,
            image_size=self.IMAGE_SIZE,
            batch_size=self.BATCH_SIZE,
            val_ratio=self.VAL_RATIO,
            num_workers=self.NUM_WORKERS,
            seed=self.SEED,
            label_col=self.LABEL_COL,
        )

        all_preds, all_targets = [], []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                outputs = model(images)
                _, predicted = torch.max(outputs, dim=1)
                all_preds.extend(predicted.cpu().tolist())
                all_targets.extend(labels.tolist())

        num_classes = len(label_map)
        metrics = compute_metrics(all_preds, all_targets, class_names, num_classes)

        self.logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        self.logger.info(
            f"Macro    — P: {metrics['macro_precision']:.4f} | "
            f"R: {metrics['macro_recall']:.4f} | F1: {metrics['macro_f1']:.4f}"
        )
        self.logger.info(
            f"Weighted — P: {metrics['weighted_precision']:.4f} | "
            f"R: {metrics['weighted_recall']:.4f} | F1: {metrics['weighted_f1']:.4f}"
        )
        self.logger.info("Per-class breakdown:")
        for name, m in metrics["per_class"].items():
            if m["support"] > 0:
                self.logger.info(
                    f"  {name:30s} | Acc: {m['accuracy']:.4f} | "
                    f"P: {m['precision']:.4f} | R: {m['recall']:.4f} | "
                    f"F1: {m['f1']:.4f} | Support: {m['support']}"
                )

        return metrics

    def predict(self) -> None:
        """Run inference on test images and save predictions to CSV."""
        model, label_map = self._load_model()
        idx_to_class = {v: k for k, v in label_map.items()}

        test_loader = create_test_loader(
            image_dir=self.TEST_IMAGE_DIR,
            image_size=self.IMAGE_SIZE,
            batch_size=self.BATCH_SIZE,
            num_workers=self.NUM_WORKERS,
        )

        results = []
        with torch.no_grad():
            for images, image_ids in test_loader:
                images = images.to(self.device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probs, dim=1)

                for img_id, pred_idx, conf in zip(
                    image_ids, predicted.cpu().tolist(), confidences.cpu().tolist()
                ):
                    results.append({
                        "id": img_id,
                        "predicted_class": idx_to_class[pred_idx],
                        "confidence": round(conf, 4),
                    })

        # Save predictions
        save_path = Path(self.SAVE_PREDICTIONS_CSV)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "predicted_class", "confidence"])
            writer.writeheader()
            writer.writerows(results)

        self.logger.info(f"Saved {len(results)} predictions to {save_path}")
