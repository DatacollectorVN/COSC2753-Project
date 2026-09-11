"""Dataset identity checks for reproducible Task 2 model bundles."""

import hashlib
import json
from pathlib import Path

import pandas as pd


def dataset_fingerprint(metadata: pd.DataFrame, input_params: dict) -> str:
    """Fingerprint ordered IDs, image state, and input preparation settings."""
    digest = hashlib.sha256()
    digest.update(json.dumps(input_params, sort_keys=True).encode("utf-8"))
    for image_id, image_path in zip(metadata["id"], metadata["image_path"]):
        path = Path(image_path)
        stat = path.stat()
        digest.update(
            f"{image_id}|{path.name}|{stat.st_size}|{stat.st_mtime_ns}\n".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()
