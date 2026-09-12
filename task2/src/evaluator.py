"""Saved-model evaluation and test prediction workflows."""

import shutil
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd

from .custom_dataset import PREDICTION_COLUMNS, load_prediction_metadata, load_training_metadata
from .data_fingerprint import dataset_fingerprint
from .metric_evaluation import compute_metrics, save_evaluation_artifacts
from .utils import (
    SettingConfig,
    create_run_dir,
    prediction_confidence,
    save_json,
    setup_logger,
)


class FashionEvaluator(SettingConfig):
    def _load_bundle(self) -> dict:
        model_path = Path(self.MODEL_PATH)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Saved model not found: {model_path}. Run `uv run python train.py` first."
            )
        bundle = joblib.load(model_path)
        required = {
            "model",
            "target_column",
            "expected_labels",
            "training_scope",
            "input_kind",
            "input_params",
            "data_fingerprint",
        }
        missing = required - set(bundle)
        if missing:
            raise ValueError(f"Model bundle is missing fields: {sorted(missing)}")
        if bundle["input_kind"] != "image_paths":
            raise ValueError("Task 2 supports only Task2CNN image-path model bundles.")
        return bundle

    @staticmethod
    def _prepare_inputs(metadata, bundle):
        inputs = metadata["image_path"].astype(str).to_numpy(dtype=object)
        fingerprint = dataset_fingerprint(
            metadata,
            {"input_kind": "image_paths", **bundle["input_params"]},
        )
        return inputs, fingerprint

    def evaluate(self) -> dict:
        run_dir = create_run_dir(self.SAVE_DIR, "eval")
        logger = setup_logger("task2.evaluate", run_dir / "logs.txt")
        save_json(self._raw_params, run_dir / "parameters.json")
        bundle = self._load_bundle()
        if bundle["training_scope"] != "development_split_only":
            raise ValueError("Evaluation requires the development-only holdout model bundle.")

        metadata, dataset_audit = load_training_metadata(
            self.DATA_DIR_TRAIN,
            label_col=bundle["target_column"],
            expected_labels=bundle["expected_labels"],
        )
        inputs, fingerprint = self._prepare_inputs(metadata, bundle)
        if fingerprint != bundle["data_fingerprint"]:
            raise ValueError("The current training data does not match the evaluated model bundle.")

        id_to_index = {image_id: index for index, image_id in enumerate(metadata["id"].astype(str))}
        missing_ids = [image_id for image_id in bundle["holdout_ids"] if image_id not in id_to_index]
        if missing_ids:
            raise ValueError(f"Holdout IDs are absent from the current data: {missing_ids[:10]}")
        holdout_indices = np.asarray([id_to_index[image_id] for image_id in bundle["holdout_ids"]])
        holdout_inputs = inputs[holdout_indices]
        holdout_labels = metadata.iloc[holdout_indices][bundle["target_column"]].astype(str).to_numpy()

        evaluation_start = perf_counter()
        predictions = bundle["model"].predict(holdout_inputs)
        prediction_scores = prediction_confidence(bundle["model"], holdout_inputs)
        evaluation_seconds = perf_counter() - evaluation_start
        metrics = compute_metrics(holdout_labels, predictions, bundle["expected_labels"])
        metrics["evaluation_seconds"] = evaluation_seconds
        metrics["evaluated_rows"] = int(len(holdout_indices))
        metrics["dataset_audit"] = dataset_audit
        save_evaluation_artifacts(
            metrics,
            run_dir,
            image_ids=bundle["holdout_ids"],
            y_true=holdout_labels,
            y_pred=predictions,
            prediction_scores=prediction_scores,
        )
        logger.info(
            "Evaluated %d rows | accuracy %.4f | macro-F1 %.4f",
            len(holdout_indices),
            metrics["accuracy"],
            metrics["macro_f1"],
        )
        return metrics

    def predict(self) -> dict:
        run_dir = create_run_dir(self.SAVE_DIR, "predict")
        logger = setup_logger("task2.predict", run_dir / "logs.txt")
        save_json(self._raw_params, run_dir / "parameters.json")
        bundle = self._load_bundle()
        if bundle["training_scope"] != "all_usable_labelled_rows":
            raise ValueError("Prediction requires the final model fitted on all usable rows.")

        metadata = load_prediction_metadata(self.TEST_CSV_PATH, self.TEST_IMAGE_DIR)
        inputs, _ = self._prepare_inputs(metadata, bundle)

        prediction_start = perf_counter()
        predictions = bundle["model"].predict(inputs)
        prediction_scores = prediction_confidence(bundle["model"], inputs)
        prediction_seconds = perf_counter() - prediction_start
        unexpected = sorted(set(predictions) - set(bundle["expected_labels"]))
        if unexpected:
            raise ValueError(f"Model generated unexpected season labels: {unexpected}")

        submission = metadata[PREDICTION_COLUMNS].copy()
        original_ids = submission["id"].copy()
        submission[bundle["target_column"]] = predictions
        if list(submission.columns) != PREDICTION_COLUMNS:
            raise ValueError("Prediction output no longer matches the supplied CSV schema.")
        if not submission["id"].equals(original_ids):
            raise ValueError("Prediction changed the supplied test ID order.")
        if submission[bundle["target_column"]].isna().any():
            raise ValueError("Prediction output contains missing season values.")

        output_path = run_dir / self.OUTPUT_FILENAME
        submission.to_csv(output_path, index=False)
        diagnostics = pd.DataFrame(
            {
                "id": metadata["id"].astype(str),
                "predicted_season": predictions,
            }
        )
        if prediction_scores is not None:
            diagnostics["confidence_score"] = prediction_scores
        diagnostics.to_csv(run_dir / "season_predictions.csv", index=False)

        stable_output = Path(self.STABLE_OUTPUT_PATH)
        stable_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, stable_output)
        summary = {
            "run_directory": str(run_dir),
            "model_path": self.MODEL_PATH,
            "prediction_rows": int(len(submission)),
            "prediction_seconds": prediction_seconds,
            "prediction_path": str(output_path),
            "stable_output_path": str(stable_output),
            "columns": list(submission.columns),
            "class_counts": {
                str(label): int(count)
                for label, count in pd.Series(predictions).value_counts().items()
            },
        }
        save_json(summary, run_dir / "scores.json")
        logger.info("Saved %d predictions to %s", len(submission), output_path)
        return summary
