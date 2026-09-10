"""Registry for Task 2 estimators."""

from .linear_svc import build_linear_svc


REGISTRY = {
    "linear_svc": build_linear_svc,
}


def build_model(model_name: str, random_state: int, **model_params):
    try:
        builder = REGISTRY[model_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown model {model_name!r}. Available models: {sorted(REGISTRY)}"
        ) from error
    return builder(random_state=random_state, **model_params)
