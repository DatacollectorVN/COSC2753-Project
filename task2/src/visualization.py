"""Task2CNN epoch diagnostic plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_task2_cnn_epoch_curves(
    history: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Plot Task2CNN accuracy and error with legends below each graph."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    accuracy_path = output_dir / "task2_cnn_epoch_accuracy.png"
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(
        history["epoch"],
        history["train_accuracy"],
        marker="o",
        label="Training accuracy",
    )
    axis.plot(
        history["epoch"],
        history["validation_accuracy"],
        marker="s",
        label="Validation accuracy",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Accuracy")
    axis.set_title("Task2CNN accuracy by epoch")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    axis.xaxis.get_major_locator().set_params(integer=True)
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    figure.savefig(accuracy_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    error_path = output_dir / "task2_cnn_epoch_error.png"
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(
        history["epoch"],
        history["train_error"],
        marker="o",
        label="Training error",
    )
    axis.plot(
        history["epoch"],
        history["validation_error"],
        marker="s",
        label="Validation error",
    )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Classification error (1 - accuracy)")
    axis.set_title("Task2CNN error by epoch")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    axis.xaxis.get_major_locator().set_params(integer=True)
    figure.tight_layout(rect=(0, 0.1, 1, 1))
    figure.savefig(error_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    return {"accuracy": accuracy_path, "error": error_path}
