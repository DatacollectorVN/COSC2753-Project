import csv
import json
from pathlib import Path

import cv2
import torch

from .custom_dataset import create_test_loader, create_train_val_loaders
from .metric_evaluation import compute_metrics
from .models import build_model
from .transforms import get_val_transforms
from .utils import SettingConfig, create_run_dir, setup_logger


class FashionEvaluator(SettingConfig):
    """Evaluation & prediction pipeline — instantiated from JSON config via **kwargs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._raw_params = kwargs

    def _load_model(self) -> tuple[torch.nn.Module, dict[str, int], int, dict]:
        checkpoint = torch.load(self.CHECKPOINT_PATH, map_location=self.device)
        label_map = checkpoint["label_map"]
        num_classes = checkpoint["num_classes"]
        model_name = checkpoint["model_name"]
        model_params = checkpoint.get("model_params", {})
        min_class_count = checkpoint.get("min_class_count", 0)
        transform_params = checkpoint.get("transform_params", {})

        model = build_model(
            model_name=model_name,
            in_channels=self.IN_CHANNELS,
            num_classes=num_classes,
            **model_params,
        ).to(self.device)
        model.load_state_dict(checkpoint["state"])
        model.eval()
        return model, label_map, min_class_count, transform_params

    def evaluate(self) -> dict:
        """Evaluate on validation split. Saves all artifacts to a timestamped run dir."""
        run_dir = create_run_dir(self.SAVE_DIR, "eval")
        logger = setup_logger("evaluate", run_dir / "logs.txt")

        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Checkpoint: {self.CHECKPOINT_PATH}")

        model, label_map, min_class_count, transform_params = self._load_model()
        class_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]

        logger.info(f"Model loaded | Classes: {len(label_map)}")
        if min_class_count > 0:
            logger.info(f"Trained with MIN_CLASS_COUNT={min_class_count} (rare → 'Other')")
        if transform_params:
            logger.info(f"Transform: {transform_params}")

        _, val_loader, _, _ = create_train_val_loaders(
            data_dir=self.DATA_DIR_TRAIN,
            batch_size=self.BATCH_SIZE,
            val_ratio=self.VAL_RATIO,
            num_workers=self.NUM_WORKERS,
            seed=self.SEED,
            label_col=self.LABEL_COL,
            min_class_count=min_class_count,
            transform_params=transform_params,
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

        logger.info(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
        logger.info(
            f"Macro    — P: {metrics['macro_precision']:.4f} | "
            f"R: {metrics['macro_recall']:.4f} | F1: {metrics['macro_f1']:.4f}"
        )
        logger.info(
            f"Weighted — P: {metrics['weighted_precision']:.4f} | "
            f"R: {metrics['weighted_recall']:.4f} | F1: {metrics['weighted_f1']:.4f}"
        )
        logger.info("Per-class breakdown:")
        for name, m in metrics["per_class"].items():
            if m["support"] > 0:
                logger.info(
                    f"  {name:30s} | Acc: {m['accuracy']:.4f} | "
                    f"P: {m['precision']:.4f} | R: {m['recall']:.4f} | "
                    f"F1: {m['f1']:.4f} | Support: {m['support']}"
                )

        # Save parameters.json
        with open(run_dir / "parameters.json", "w") as f:
            json.dump(self._raw_params, f, indent=2)

        # Save scores.json (confusion_matrix as list for JSON serialization)
        scores_out = {
            k: v for k, v in metrics.items() if k != "confusion_matrix"
        }
        scores_out["confusion_matrix"] = metrics["confusion_matrix"].tolist()
        with open(run_dir / "scores.json", "w") as f:
            json.dump(scores_out, f, indent=2)

        logger.info(f"Evaluation complete. Artifacts saved to: {run_dir}")
        return metrics

    def predict(self) -> None:
        """Run inference on test images. Saves predictions to a timestamped run dir."""
        run_dir = create_run_dir(self.SAVE_DIR, "predict")
        logger = setup_logger("predict", run_dir / "logs.txt")

        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Checkpoint: {self.CHECKPOINT_PATH}")

        model, label_map, _, transform_params = self._load_model()
        idx_to_class = {v: k for k, v in label_map.items()}

        logger.info(f"Model loaded | Classes: {len(label_map)}")

        test_loader = create_test_loader(
            image_dir=self.TEST_IMAGE_DIR,
            batch_size=self.BATCH_SIZE,
            num_workers=self.NUM_WORKERS,
            transform_params=transform_params,
        )
        logger.info(f"Test loader created | {len(test_loader)} images")
        
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

        # Save predictions.csv
        predictions_path = run_dir / "predictions.csv"
        with open(predictions_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "predicted_class", "confidence"])
            writer.writeheader()
            writer.writerows(results)

        # Save parameters.json
        with open(run_dir / "parameters.json", "w") as f:
            json.dump(self._raw_params, f, indent=2)

        logger.info(f"Saved {len(results)} predictions to {predictions_path}")
        logger.info(f"Prediction complete. Artifacts saved to: {run_dir}")

    def predict_single(self, image_path: str) -> dict:
        """Run inference on a single image. Returns dict with predicted_class and confidence."""
        model, label_map, _, transform_params = self._load_model()
        idx_to_class = {v: k for k, v in label_map.items()}

        transform = get_val_transforms(transform_params)

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, dim=1)

        return {
            "image_id": Path(image_path).stem,
            "predicted_class": idx_to_class[predicted.item()],
            "confidence": confidence.item(),
        }
