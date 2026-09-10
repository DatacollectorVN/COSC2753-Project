"""End-to-end GridSearchCV training workflow for Task 2."""

import shutil
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from .custom_dataset import load_training_metadata
from .features import load_or_extract_features
from .metric_evaluation import compute_metrics, save_evaluation_artifacts
from .models import build_model
from .utils import SettingConfig, create_run_dir, save_json, set_seed, setup_logger


SCORING = {
    "macro_f1": "f1_macro",
    "accuracy": "accuracy",
    "weighted_f1": "f1_weighted",
}


class FashionTrainer(SettingConfig):
    """Train, tune, evaluate, refit, and save the season classifier."""

    def train(self) -> dict:
        run_dir = create_run_dir(self.SAVE_MODEL_DIR, "train")
        logger = setup_logger("task2.train", run_dir / "logs.txt")
        save_json(self._raw_params, run_dir / "parameters.json")
        set_seed(self.SEED)

        logger.info("Run directory: %s", run_dir)
        logger.info("Loading Task 2 metadata from %s", self.DATA_DIR_TRAIN)
        metadata, dataset_audit = load_training_metadata(
            self.DATA_DIR_TRAIN,
            label_col=self.LABEL_COL,
            expected_labels=self.EXPECTED_LABELS,
        )
        save_json(dataset_audit, run_dir / "dataset_audit.json")
        logger.info(
            "Using %d rows; removed %d missing-label and %d missing-image rows",
            dataset_audit["usable_rows"],
            dataset_audit["missing_label_rows"],
            dataset_audit["missing_image_rows"],
        )

        feature_start = perf_counter()
        features, feature_fingerprint = load_or_extract_features(
            metadata,
            feature_params=self.FEATURE_PARAMS,
            cache_path=self.TRAIN_FEATURE_CACHE,
            force_rebuild=self.FORCE_REBUILD_FEATURES,
            logger=logger,
        )
        feature_seconds = perf_counter() - feature_start
        labels = metadata[self.LABEL_COL].astype(str).to_numpy()
        all_indices = np.arange(len(metadata))
        development_indices, holdout_indices = train_test_split(
            all_indices,
            test_size=self.VAL_RATIO,
            random_state=self.SEED,
            stratify=labels,
        )
        development_features = features[development_indices]
        holdout_features = features[holdout_indices]
        development_labels = labels[development_indices]
        holdout_labels = labels[holdout_indices]

        split_manifest = {
            "seed": self.SEED,
            "holdout_ratio": self.VAL_RATIO,
            "development_rows": int(len(development_indices)),
            "holdout_rows": int(len(holdout_indices)),
            "development_ids": metadata.iloc[development_indices]["id"].astype(str).tolist(),
            "holdout_ids": metadata.iloc[holdout_indices]["id"].astype(str).tolist(),
        }
        save_json(split_manifest, run_dir / "split_manifest.json")

        dummy_model = DummyClassifier(strategy="most_frequent")
        dummy_model.fit(np.zeros((len(development_labels), 1)), development_labels)
        baseline_predictions = dummy_model.predict(np.zeros((len(holdout_labels), 1)))
        baseline_metrics = compute_metrics(
            holdout_labels, baseline_predictions, self.EXPECTED_LABELS
        )
        save_json(baseline_metrics, run_dir / "baseline_metrics.json")
        logger.info(
            "Majority baseline | accuracy %.4f | macro-F1 %.4f",
            baseline_metrics["accuracy"],
            baseline_metrics["macro_f1"],
        )

        estimator = build_model(
            self.MODEL_NAME,
            random_state=self.SEED,
            **self.MODEL_PARAMS,
        )
        cross_validation = StratifiedKFold(
            n_splits=self.CV_FOLDS,
            shuffle=True,
            random_state=self.SEED,
        )
        search = GridSearchCV(
            estimator=estimator,
            param_grid=self.PARAM_GRID,
            scoring=SCORING,
            refit=self.REFIT_METRIC,
            cv=cross_validation,
            n_jobs=self.N_JOBS,
            verbose=self.VERBOSE,
            return_train_score=True,
            error_score="raise",
            pre_dispatch=self.N_JOBS,
        )
        candidate_count = int(np.prod([len(values) for values in self.PARAM_GRID.values()]))
        logger.info(
            "Starting GridSearchCV: %d candidates x %d folds; refit=%s",
            candidate_count,
            self.CV_FOLDS,
            self.REFIT_METRIC,
        )
        search_start = perf_counter()
        with parallel_backend(self.PARALLEL_BACKEND, n_jobs=self.N_JOBS):
            search.fit(development_features, development_labels)
        search_seconds = perf_counter() - search_start

        cv_results = pd.DataFrame(search.cv_results_).sort_values("rank_test_macro_f1")
        cv_results.to_csv(run_dir / "grid_search_results.csv", index=False)
        stable_cv_path = Path(self.CV_RESULTS_PATH)
        stable_cv_path.parent.mkdir(parents=True, exist_ok=True)
        cv_results.to_csv(stable_cv_path, index=False)
        logger.info("Best parameters: %s", search.best_params_)
        logger.info("Best cross-validation macro-F1: %.4f", search.best_score_)

        holdout_model = search.best_estimator_
        holdout_predictions = holdout_model.predict(holdout_features)
        holdout_decisions = holdout_model.decision_function(holdout_features)
        holdout_scores = np.max(holdout_decisions, axis=1)
        holdout_metrics = compute_metrics(
            holdout_labels, holdout_predictions, self.EXPECTED_LABELS
        )
        save_evaluation_artifacts(
            holdout_metrics,
            run_dir,
            image_ids=metadata.iloc[holdout_indices]["id"].astype(str),
            y_true=holdout_labels,
            y_pred=holdout_predictions,
            prediction_scores=holdout_scores,
        )

        bundle_metadata = {
            "task": "Fashion Season Classification",
            "target_column": self.LABEL_COL,
            "expected_labels": self.EXPECTED_LABELS,
            "feature_params": self.FEATURE_PARAMS,
            "feature_fingerprint": feature_fingerprint,
            "model_name": self.MODEL_NAME,
            "best_params": search.best_params_,
            "best_cv_macro_f1": float(search.best_score_),
            "seed": self.SEED,
            "holdout_ratio": self.VAL_RATIO,
            "dataset_audit": dataset_audit,
        }
        holdout_bundle = {
            **bundle_metadata,
            "model": holdout_model,
            "training_scope": "development_split_only",
            "holdout_ids": split_manifest["holdout_ids"],
        }
        run_holdout_path = run_dir / "holdout_model.joblib"
        joblib.dump(holdout_bundle, run_holdout_path, compress=3)
        stable_holdout_path = Path(self.HOLDOUT_MODEL_PATH)
        stable_holdout_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_holdout_path, stable_holdout_path)

        logger.info("Refitting the selected pipeline on all %d labelled rows", len(metadata))
        final_model = clone(holdout_model)
        final_fit_start = perf_counter()
        final_model.fit(features, labels)
        final_fit_seconds = perf_counter() - final_fit_start
        final_bundle = {
            **bundle_metadata,
            "model": final_model,
            "training_scope": "all_usable_labelled_rows",
            "training_rows": int(len(metadata)),
        }
        run_final_path = run_dir / "season_model.joblib"
        joblib.dump(final_bundle, run_final_path, compress=3)
        stable_final_path = Path(self.FINAL_MODEL_PATH)
        stable_final_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_final_path, stable_final_path)

        label_map = {label: index for index, label in enumerate(final_model.classes_)}
        save_json(label_map, run_dir / "label_map.json")
        summary = {
            "run_directory": str(run_dir),
            "usable_rows": int(len(metadata)),
            "feature_dimensions": int(features.shape[1]),
            "feature_preparation_seconds": feature_seconds,
            "grid_search_seconds": search_seconds,
            "final_refit_seconds": final_fit_seconds,
            "grid_candidates": int(len(cv_results)),
            "total_cross_validation_fits": int(len(cv_results) * self.CV_FOLDS),
            "best_params": search.best_params_,
            "best_cv_macro_f1": float(search.best_score_),
            "baseline_accuracy": baseline_metrics["accuracy"],
            "baseline_macro_f1": baseline_metrics["macro_f1"],
            "holdout_accuracy": holdout_metrics["accuracy"],
            "holdout_macro_f1": holdout_metrics["macro_f1"],
            "holdout_weighted_f1": holdout_metrics["weighted_f1"],
            "holdout_balanced_accuracy": holdout_metrics["balanced_accuracy"],
            "final_model_path": str(stable_final_path),
            "final_model_size_bytes": int(stable_final_path.stat().st_size),
        }
        save_json(summary, run_dir / "scores.json")
        save_json(
            {"run_directory": str(run_dir), "summary": summary},
            Path(self.ARTIFACT_DIR) / "latest_train_run.json",
        )
        logger.info(
            "Holdout | accuracy %.4f | macro-F1 %.4f | weighted-F1 %.4f",
            holdout_metrics["accuracy"],
            holdout_metrics["macro_f1"],
            holdout_metrics["weighted_f1"],
        )
        logger.info("Final model saved to %s", stable_final_path)
        return summary
