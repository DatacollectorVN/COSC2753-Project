"""Metadata loading and validation for Task 2."""

from pathlib import Path

import pandas as pd


PREDICTION_COLUMNS = ["id", "gender", "articleType", "season", "usage"]


def load_training_metadata(
    data_dir: str | Path,
    label_col: str = "season",
    expected_labels: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Join labelled metadata rows to existing training images.

    The supplied files are never modified. Rows without a target or matching
    image are excluded because they cannot be used for supervised image
    classification.
    """
    data_dir = Path(data_dir)
    csv_path = data_dir / "styles_train.csv"
    image_dir = data_dir / "images_train"

    if not csv_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Training image directory not found: {image_dir}")

    raw = pd.read_csv(csv_path, dtype={"id": "string"})
    unnamed_columns = [column for column in raw.columns if column.startswith("Unnamed:")]
    metadata = raw.drop(columns=unnamed_columns).copy()

    required_columns = {"id", label_col}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(f"Training CSV is missing required columns: {sorted(missing_columns)}")

    metadata["id"] = metadata["id"].str.strip()
    metadata[label_col] = metadata[label_col].astype("string").str.strip()
    metadata.loc[metadata[label_col] == "", label_col] = pd.NA

    if metadata["id"].isna().any() or metadata["id"].duplicated().any():
        raise ValueError("Training IDs must be present and unique.")

    metadata["image_path"] = metadata["id"].map(lambda image_id: image_dir / f"{image_id}.jpg")
    metadata["image_exists"] = metadata["image_path"].map(Path.is_file)

    missing_label_count = int(metadata[label_col].isna().sum())
    missing_image_count = int((~metadata["image_exists"]).sum())
    usable = metadata.loc[metadata[label_col].notna() & metadata["image_exists"]].copy()
    usable.reset_index(drop=True, inplace=True)

    discovered_labels = sorted(usable[label_col].unique().tolist())
    if expected_labels is not None:
        unexpected = sorted(set(discovered_labels) - set(expected_labels))
        missing_expected = sorted(set(expected_labels) - set(discovered_labels))
        if unexpected or missing_expected:
            raise ValueError(
                f"Season-label mismatch. Unexpected={unexpected}; missing={missing_expected}"
            )

    audit = {
        "raw_rows": int(len(raw)),
        "empty_trailing_columns_removed": unnamed_columns,
        "missing_label_rows": missing_label_count,
        "missing_image_rows": missing_image_count,
        "usable_rows": int(len(usable)),
        "class_counts": {
            str(label): int(count)
            for label, count in usable[label_col].value_counts().items()
        },
    }
    return usable, audit


def load_prediction_metadata(
    test_csv_path: str | Path,
    test_image_dir: str | Path,
) -> pd.DataFrame:
    """Load the supplied prediction template in its original row order."""
    test_csv_path = Path(test_csv_path)
    test_image_dir = Path(test_image_dir)
    if not test_csv_path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {test_csv_path}")
    if not test_image_dir.is_dir():
        raise FileNotFoundError(f"Test image directory not found: {test_image_dir}")

    metadata = pd.read_csv(test_csv_path, dtype={"id": "string"})
    if list(metadata.columns) != PREDICTION_COLUMNS:
        raise ValueError(
            "Prediction CSV columns must remain exactly "
            f"{PREDICTION_COLUMNS}; found {list(metadata.columns)}"
        )

    metadata["id"] = metadata["id"].str.strip()
    if metadata["id"].isna().any() or metadata["id"].duplicated().any():
        raise ValueError("Prediction IDs must be present and unique.")

    metadata["image_path"] = metadata["id"].map(
        lambda image_id: test_image_dir / f"{image_id}.jpg"
    )
    missing = metadata.loc[~metadata["image_path"].map(Path.is_file), "id"].tolist()
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} test images; first IDs: {missing[:10]}")

    image_ids = {path.stem for path in test_image_dir.glob("*.jpg")}
    extra_image_ids = sorted(image_ids - set(metadata["id"]), key=int)
    if extra_image_ids:
        raise ValueError(
            f"Found {len(extra_image_ids)} test images with no prediction row; "
            f"first IDs: {extra_image_ids[:10]}"
        )
    return metadata
