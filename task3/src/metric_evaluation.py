import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute overall accuracy from model outputs and targets."""
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == targets).sum().item()
    return correct / targets.size(0)


def compute_per_class_accuracy(
    all_preds: list[int],
    all_targets: list[int],
    num_classes: int,
    class_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute accuracy for each class individually.

    Per-class accuracy = (correct predictions for class i) / (total samples of class i).
    """
    preds = np.array(all_preds)
    targets = np.array(all_targets)
    per_class = {}

    for cls_idx in range(num_classes):
        mask = targets == cls_idx
        if mask.sum() == 0:
            acc = 0.0
        else:
            acc = (preds[mask] == cls_idx).sum() / mask.sum()

        name = class_names[cls_idx] if class_names else str(cls_idx)
        per_class[name] = float(acc)

    return per_class


def compute_per_class_metrics(
    all_preds: list[int],
    all_targets: list[int],
    num_classes: int,
    class_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute precision, recall, F1-score, and accuracy per class.

    Returns:
        Dict mapping class name -> {precision, recall, f1, accuracy, support}.
    """
    preds = np.array(all_preds)
    targets = np.array(all_targets)

    labels = list(range(num_classes))
    precisions = precision_score(targets, preds, labels=labels, average=None, zero_division=0)
    recalls = recall_score(targets, preds, labels=labels, average=None, zero_division=0)
    f1s = f1_score(targets, preds, labels=labels, average=None, zero_division=0)

    per_class = {}
    for cls_idx in range(num_classes):
        mask = targets == cls_idx
        support = int(mask.sum())
        if support == 0:
            acc = 0.0
        else:
            acc = float((preds[mask] == cls_idx).sum() / support)

        name = class_names[cls_idx] if class_names else str(cls_idx)
        per_class[name] = {
            "precision": float(precisions[cls_idx]),
            "recall": float(recalls[cls_idx]),
            "f1": float(f1s[cls_idx]),
            "accuracy": acc,
            "support": support,
        }

    return per_class


def compute_metrics(
    all_preds: list[int],
    all_targets: list[int],
    class_names: list[str] | None = None,
    num_classes: int | None = None,
) -> dict:
    """Compute full evaluation metrics.

    Returns dict with:
        - overall_accuracy: float
        - macro_precision, macro_recall, macro_f1: macro-averaged scores
        - weighted_precision, weighted_recall, weighted_f1: weighted-averaged scores
        - per_class: dict of per-class {precision, recall, f1, accuracy, support}
        - report: sklearn classification_report dict
        - confusion_matrix: np.ndarray
    """
    preds = np.array(all_preds)
    targets = np.array(all_targets)

    if num_classes is None:
        num_classes = max(int(targets.max()), int(preds.max())) + 1
    labels = list(range(num_classes))

    # Overall
    overall_acc = accuracy_score(targets, preds)

    # Macro averages
    macro_p = precision_score(targets, preds, labels=labels, average="macro", zero_division=0)
    macro_r = recall_score(targets, preds, labels=labels, average="macro", zero_division=0)
    macro_f1 = f1_score(targets, preds, labels=labels, average="macro", zero_division=0)

    # Weighted averages
    weighted_p = precision_score(targets, preds, labels=labels, average="weighted", zero_division=0)
    weighted_r = recall_score(targets, preds, labels=labels, average="weighted", zero_division=0)
    weighted_f1 = f1_score(targets, preds, labels=labels, average="weighted", zero_division=0)

    # Per-class breakdown
    per_class = compute_per_class_metrics(all_preds, all_targets, num_classes, class_names)

    # sklearn report (for backward compatibility)
    report = classification_report(
        targets, preds, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(targets, preds, labels=labels)

    return {
        "overall_accuracy": float(overall_acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_p),
        "weighted_recall": float(weighted_r),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "report": report,
        "confusion_matrix": cm,
    }
