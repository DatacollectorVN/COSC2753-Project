"""Registry for the Task 2 estimator."""

from .task2_cnn import build_task2_cnn


REGISTRY = {
    "task2_cnn": build_task2_cnn,
}


def build_model(model_name: str, random_state: int, **model_params):
    try:
        builder = REGISTRY[model_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown model {model_name!r}. Available models: {sorted(REGISTRY)}"
        ) from error
    return builder(random_state=random_state, **model_params)
