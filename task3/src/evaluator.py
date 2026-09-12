import csv
import json
from pathlib import Path

import cv2
import pandas as pd
import torch

from torch.utils.data import DataLoader

from .custom_dataset import FashionDataset, create_test_loader, create_train_val_loaders
from .metric_evaluation import compute_metrics
from .models import build_model
from .transforms import get_val_transforms
from .utils import SettingConfig, create_run_dir, setup_logger


class FashionEvaluator(SettingConfig):
    """Evaluation & prediction pipeline for the two-head hybrid model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._raw_params = kwargs

    def _load_model(self) -> tuple:
        """Load model from checkpoint.

        Returns:
            (model, gender_label_map, occasion_label_map, meta_vocabs,
             min_occasion_count, transform_params)
        """
        checkpoint = torch.load(self.CHECKPOINT_PATH, map_location=self.device)
        gender_label_map = checkpoint["gender_label_map"]
        occasion_label_map = checkpoint["occasion_label_map"]
        meta_vocabs = checkpoint["meta_vocabs"]
        model_name = checkpoint["model_name"]
        model_params = checkpoint.get("model_params", {})
        min_occasion_count = checkpoint.get("min_occasion_count", 50)
        transform_params = checkpoint.get("transform_params", {})

        model = build_model(
            model_name=model_name,
            in_channels=self.IN_CHANNELS,
            num_gender_classes=checkpoint["num_gender_classes"],
            num_occasion_classes=checkpoint["num_occasion_classes"],
            num_article_types=len(meta_vocabs["articleType"]),
            num_master_categories=len(meta_vocabs["masterCategory"]),
            num_colours=len(meta_vocabs["baseColour"]),
            **model_params,
        ).to(self.device)
        model.load_state_dict(checkpoint["state"])
        model.eval()
        return model, gender_label_map, occasion_label_map, meta_vocabs, min_occasion_count, transform_params

    def evaluate(self, evaluate_all: bool = False) -> dict:
        """Evaluate on validation split or entire dataset.

        Args:
            evaluate_all: If True, evaluate on the entire training dataset.
                          If False (default), evaluate only on the validation split.
        """
        run_dir = create_run_dir(self.SAVE_DIR, "eval")
        logger = setup_logger("evaluate", run_dir / "logs.txt")

        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Checkpoint: {self.CHECKPOINT_PATH}")
        logger.info(f"Mode: {'entire dataset' if evaluate_all else 'validation split'}")

        model, gender_label_map, occasion_label_map, meta_vocabs, min_occasion_count, transform_params = self._load_model()
        gender_names = [n for n, _ in sorted(gender_label_map.items(), key=lambda x: x[1])]
        occasion_names = [n for n, _ in sorted(occasion_label_map.items(), key=lambda x: x[1])]

        logger.info(f"Gender classes: {len(gender_label_map)} | Occasion classes: {len(occasion_label_map)}")

        if evaluate_all:
            dataset = FashionDataset(
                data_dir=self.DATA_DIR_TRAIN,
                split="train",
                transform=get_val_transforms(transform_params),
                gender_label_map=gender_label_map,
                occasion_label_map=occasion_label_map,
                meta_vocabs=meta_vocabs,
                min_occasion_count=min_occasion_count,
            )
            val_loader = DataLoader(
                dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=False,
                num_workers=self.NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
            )
        else:
            (
                _, val_loader,
                _, _, _, _, _,
            ) = create_train_val_loaders(
                data_dir=self.DATA_DIR_TRAIN,
                batch_size=self.BATCH_SIZE,
                val_ratio=self.VAL_RATIO,
                num_workers=self.NUM_WORKERS,
                seed=self.SEED,
                min_occasion_count=min_occasion_count,
                transform_params=transform_params,
            )

        g_preds, g_targets = [], []
        o_preds, o_targets = [], []
        with torch.no_grad():
            for images, m_art, m_mst, m_col, g_labels, o_labels in val_loader:
                images = images.to(self.device)
                m_art = m_art.to(self.device, dtype=torch.long)
                m_mst = m_mst.to(self.device, dtype=torch.long)
                m_col = m_col.to(self.device, dtype=torch.long)

                g_logits, o_logits = model(images, m_art, m_mst, m_col)

                _, g_pred = torch.max(g_logits, dim=1)
                _, o_pred = torch.max(o_logits, dim=1)

                g_preds.extend(g_pred.cpu().tolist())
                g_targets.extend(g_labels.tolist())
                o_preds.extend(o_pred.cpu().tolist())
                o_targets.extend(o_labels.tolist())

        # Compute metrics per head
        gender_metrics = compute_metrics(g_preds, g_targets, gender_names, len(gender_label_map))
        occasion_metrics = compute_metrics(o_preds, o_targets, occasion_names, len(occasion_label_map))

        # Log gender results
        logger.info("=== GENDER HEAD ===")
        logger.info(f"Accuracy: {gender_metrics['overall_accuracy']:.4f}")
        logger.info(f"Macro F1: {gender_metrics['macro_f1']:.4f}")
        for name, m in gender_metrics["per_class"].items():
            if m["support"] > 0:
                logger.info(f"  {name:20s} | F1: {m['f1']:.4f} | P: {m['precision']:.4f} | R: {m['recall']:.4f} | N: {m['support']}")

        # Log occasion results
        logger.info("=== OCCASION HEAD ===")
        logger.info(f"Accuracy: {occasion_metrics['overall_accuracy']:.4f}")
        logger.info(f"Macro F1: {occasion_metrics['macro_f1']:.4f}")
        for name, m in occasion_metrics["per_class"].items():
            if m["support"] > 0:
                logger.info(f"  {name:20s} | F1: {m['f1']:.4f} | P: {m['precision']:.4f} | R: {m['recall']:.4f} | N: {m['support']}")

        # Save
        with open(run_dir / "parameters.json", "w") as f:
            json.dump(self._raw_params, f, indent=2)

        scores_out = {
            "gender": {k: v for k, v in gender_metrics.items() if k != "confusion_matrix"},
            "occasion": {k: v for k, v in occasion_metrics.items() if k != "confusion_matrix"},
        }
        scores_out["gender"]["confusion_matrix"] = gender_metrics["confusion_matrix"].tolist()
        scores_out["occasion"]["confusion_matrix"] = occasion_metrics["confusion_matrix"].tolist()
        with open(run_dir / "scores.json", "w") as f:
            json.dump(scores_out, f, indent=2)

        logger.info(f"Evaluation complete. Artifacts saved to: {run_dir}")
        return {"gender": gender_metrics, "occasion": occasion_metrics}

    def predict(self) -> None:
        """Run inference on test images. Outputs both gender and occasion predictions."""
        run_dir = create_run_dir(self.SAVE_DIR, "predict")
        logger = setup_logger("predict", run_dir / "logs.txt")

        logger.info(f"Run directory: {run_dir}")
        logger.info(f"Checkpoint: {self.CHECKPOINT_PATH}")

        model, gender_label_map, occasion_label_map, meta_vocabs, _, transform_params = self._load_model()
        g_idx_to_class = {v: k for k, v in gender_label_map.items()}
        o_idx_to_class = {v: k for k, v in occasion_label_map.items()}

        logger.info(f"Gender classes: {len(gender_label_map)} | Occasion classes: {len(occasion_label_map)}")

        test_csv = getattr(self, "TEST_CSV_PATH", None)
        test_loader = create_test_loader(
            image_dir=self.TEST_IMAGE_DIR,
            batch_size=self.BATCH_SIZE,
            num_workers=self.NUM_WORKERS,
            transform_params=transform_params,
            csv_path=test_csv,
            meta_vocabs=meta_vocabs,
        )

        results = []
        with torch.no_grad():
            for images, m_art, m_mst, m_col, image_ids in test_loader:
                images = images.to(self.device)
                m_art = m_art.to(self.device, dtype=torch.long)
                m_mst = m_mst.to(self.device, dtype=torch.long)
                m_col = m_col.to(self.device, dtype=torch.long)

                g_logits, o_logits = model(images, m_art, m_mst, m_col)
                g_probs = torch.softmax(g_logits, dim=1)
                o_probs = torch.softmax(o_logits, dim=1)
                g_conf, g_pred = torch.max(g_probs, dim=1)
                o_conf, o_pred = torch.max(o_probs, dim=1)

                for img_id, gp, gc, op, oc in zip(
                    image_ids,
                    g_pred.cpu().tolist(), g_conf.cpu().tolist(),
                    o_pred.cpu().tolist(), o_conf.cpu().tolist(),
                ):
                    results.append({
                        "id": img_id,
                        "gender": g_idx_to_class[gp],
                        "gender_confidence": round(gc, 4),
                        "usage": o_idx_to_class[op],
                        "usage_confidence": round(oc, 4),
                    })

        predictions_path = run_dir / "predictions.csv"
        with open(predictions_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "gender", "gender_confidence", "usage", "usage_confidence"])
            writer.writeheader()
            writer.writerows(results)

        with open(run_dir / "parameters.json", "w") as f:
            json.dump(self._raw_params, f, indent=2)

        logger.info(f"Saved {len(results)} predictions to {predictions_path}")
        logger.info(f"Prediction complete. Artifacts saved to: {run_dir}")

    def predict_single(self, image_path: str, csv_path: str | None = None) -> dict:
        """Run inference on a single image. Returns dict with gender, usage and confidences."""
        model, gender_label_map, occasion_label_map, meta_vocabs, _, transform_params = self._load_model()
        g_idx_to_class = {v: k for k, v in gender_label_map.items()}
        o_idx_to_class = {v: k for k, v in occasion_label_map.items()}

        transform = get_val_transforms(transform_params)

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = transform(image).unsqueeze(0).to(self.device)

        # Encode metadata from CSV if available
        meta_article, meta_master, meta_colour = 0, 0, 0
        image_id = Path(image_path).stem
        if csv_path is not None and Path(csv_path).exists():
            df = pd.read_csv(csv_path)
            df["id"] = df["id"].astype(str)
            row = df[df["id"] == image_id]
            if not row.empty:
                row = row.iloc[0]
                for col, vocab_key in [("articleType", "articleType"), ("masterCategory", "masterCategory"), ("baseColour", "baseColour")]:
                    vocab = meta_vocabs.get(vocab_key, {})
                    val = row.get(col, None)
                    idx_val = vocab.get(val, 0) if pd.notna(val) else 0
                    if col == "articleType":
                        meta_article = idx_val
                    elif col == "masterCategory":
                        meta_master = idx_val
                    else:
                        meta_colour = idx_val

        m_art = torch.tensor([[meta_article]], dtype=torch.long).to(self.device)
        m_mst = torch.tensor([[meta_master]], dtype=torch.long).to(self.device)
        m_col = torch.tensor([[meta_colour]], dtype=torch.long).to(self.device)

        with torch.no_grad():
            g_logits, o_logits = model(image, m_art, m_mst, m_col)
            g_probs = torch.softmax(g_logits, dim=1)
            o_probs = torch.softmax(o_logits, dim=1)
            g_conf, g_pred = torch.max(g_probs, dim=1)
            o_conf, o_pred = torch.max(o_probs, dim=1)

        return {
            "image_id": image_id,
            "gender": g_idx_to_class[g_pred.item()],
            "gender_confidence": g_conf.item(),
            "usage": o_idx_to_class[o_pred.item()],
            "usage_confidence": o_conf.item(),
        }
