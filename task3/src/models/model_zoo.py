"""Model Zoo — central registry for Task 3 hybrid architecture.

Usage:
    from src.models import build_model

    model = build_model(
        "hybrid_cnn",
        in_channels=3,
        num_gender_classes=5,
        num_occasion_classes=4,
        num_article_types=126,
        num_master_categories=8,
        num_colours=47,
    )
"""

import torch.nn as nn

from .hybrid_cnn import HybridCNN

REGISTRY: dict[str, type[nn.Module]] = {
    "hybrid_cnn": HybridCNN,
}


def list_models() -> list[str]:
    """Return names of all registered models."""
    return sorted(REGISTRY.keys())


def build_model(
    model_name: str,
    in_channels: int,
    num_gender_classes: int,
    num_occasion_classes: int,
    **model_params,
) -> nn.Module:
    """Instantiate a model from the registry.

    Args:
        model_name: Key in REGISTRY (e.g. "hybrid_cnn").
        in_channels: Number of input image channels (3 for RGB).
        num_gender_classes: Number of gender output classes.
        num_occasion_classes: Number of occasion output classes.
        **model_params: Extra keyword args forwarded to the model constructor
                        (e.g. num_article_types, channels, dropout_rate).

    Raises:
        ValueError: If model_name is not found in REGISTRY.
    """
    if model_name not in REGISTRY:
        available = ", ".join(list_models())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    model_cls = REGISTRY[model_name]
    return model_cls(
        in_channels=in_channels,
        num_gender_classes=num_gender_classes,
        num_occasion_classes=num_occasion_classes,
        **model_params,
    )
