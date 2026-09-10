"""Linear support-vector classifier pipeline."""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def build_linear_svc(random_state: int, max_iter: int = 10000, tol: float = 1e-4) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LinearSVC(
                    dual="auto",
                    max_iter=max_iter,
                    tol=tol,
                    random_state=random_state,
                ),
            ),
        ]
    )
