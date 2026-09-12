"""Classification metrics and evidence artifacts for Task 2."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .utils import save_json


def compute_metrics(y_true, y_pred, class_names: list[str]) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=class_names,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=class_names)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=class_names, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=class_names, average="weighted", zero_division=0)
        ),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_order": class_names,
    }


def save_evaluation_artifacts(
    metrics: dict,
    output_dir: str | Path,
    image_ids,
    y_true,
    y_pred,
    prediction_scores=None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, output_dir / "metrics.json")

    pd.DataFrame(metrics["classification_report"]).transpose().to_csv(
        output_dir / "classification_report.csv"
    )
    matrix = pd.DataFrame(
        metrics["confusion_matrix"],
        index=metrics["class_order"],
        columns=metrics["class_order"],
    )
    matrix.to_csv(output_dir / "confusion_matrix.csv", index_label="actual")

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set(title="Season classification confusion matrix", xlabel="Predicted", ylabel="Actual")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    predictions = pd.DataFrame(
        {
            "id": list(image_ids),
            "actual_season": list(y_true),
            "predicted_season": list(y_pred),
        }
    )
    predictions["correct"] = predictions["actual_season"] == predictions["predicted_season"]
    if prediction_scores is not None:
        predictions["confidence_score"] = prediction_scores
    predictions.to_csv(output_dir / "holdout_predictions.csv", index=False)
