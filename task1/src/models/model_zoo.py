"""Model Zoo — central registry for all CNN architectures.

Usage:
    from src.models import build_model, list_models

    # List available models
    list_models()

    # Build a model by name
    model = build_model("simple_cnn", in_channels=3, num_classes=124, dropout_rate=0.5)

    # Build from config dict (as loaded from JSON)
    model = build_model("resnet_cnn", in_channels=3, num_classes=124, **{"depth": 50})

Adding a new model:
    1. Create a new file in src/models/ (e.g., my_model.py)
    2. Define your nn.Module class — must accept (in_channels, num_classes, **kwargs)
    3. Register it in the REGISTRY dict below with a unique name
    4. Add docstring with CONFIG PARAMETERS section (see existing models for format)
"""

import torch.nn as nn

from .deep_cnn import DeepCNN
from .light_cnn import LightCNN
from .resnet_cnn import ResNetCNN
from .simple_cnn import SimpleCNN

# ──────────────────────────────────────────────────────────
# Registry: maps model name (str) → model class (nn.Module)
# ──────────────────────────────────────────────────────────
REGISTRY: dict[str, type[nn.Module]] = {
    "simple_cnn": SimpleCNN,
    "deep_cnn": DeepCNN,
    "light_cnn": LightCNN,
    "resnet_cnn": ResNetCNN,
}


def list_models() -> list[str]:
    """Return names of all registered models."""
    return sorted(REGISTRY.keys())


def build_model(
    model_name: str,
    in_channels: int,
    num_classes: int,
    **model_params,
) -> nn.Module:
    """Instantiate a model from the registry.

    Args:
        model_name: Key in REGISTRY (e.g. "simple_cnn").
        in_channels: Number of input image channels (3 for RGB).
        num_classes: Number of output classes.
        **model_params: Extra keyword args forwarded to the model constructor
                        (e.g. channels, dropout_rate, fc_hidden).

    Raises:
        ValueError: If model_name is not found in REGISTRY.
    """
    if model_name not in REGISTRY:
        available = ", ".join(list_models())
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {available}"
        )

    model_cls = REGISTRY[model_name]
    return model_cls(
        in_channels=in_channels,
        num_classes=num_classes,
        **model_params,
    )
