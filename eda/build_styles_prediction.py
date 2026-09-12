"""Merge Task 1, Task 2, and Task 3 predictions into the submission schema.

The generated CSV contains: id, gender, articleType, season, usage.
Input paths are intentionally hardcoded for the completed prediction runs.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


TASK1_PREDICTIONS = Path(
    "/Users/nhan.ngo/rmit/COSC2753-Project/"
    "task1/results/predict/20260912_093903/predictions.csv"
)
TASK2_PREDICTIONS = Path(
    "/Users/nhan.ngo/rmit/COSC2753-Project/task2/task2_predictions.csv"
)
TASK3_PREDICTIONS = Path(
    "/Users/nhan.ngo/rmit/COSC2753-Project/"
    "task3/results/predict/20260912_100558/predictions.csv"
)

OUTPUT_COLUMNS = ["id", "gender", "articleType", "season", "usage"]


def read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    """Read a CSV and validate its required columns and unique IDs."""
    if not path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing = required_columns - columns
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)

    seen_ids: set[str] = set()
    duplicate_ids: set[str] = set()
    for row in rows:
        image_id = row["id"]
        if image_id in seen_ids:
            duplicate_ids.add(image_id)
        seen_ids.add(image_id)
    if duplicate_ids:
        raise ValueError(f"{path} contains duplicate IDs: {sorted(duplicate_ids)[:5]}")
    return rows


def index_by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["id"]: row for row in rows}


def build_predictions(output_path: Path) -> None:
    """Merge task outputs by ID and write the final prediction CSV."""
    task1_rows = read_csv(TASK1_PREDICTIONS, {"id", "predicted_class"})
    task2_rows = read_csv(TASK2_PREDICTIONS, {"id", "season"})
    task3_rows = read_csv(TASK3_PREDICTIONS, {"id", "gender", "usage"})

    task1_by_id = index_by_id(task1_rows)
    task3_by_id = index_by_id(task3_rows)
    expected_ids = {row["id"] for row in task2_rows}

    for name, actual_ids in (
        ("Task 1", set(task1_by_id)),
        ("Task 3", set(task3_by_id)),
    ):
        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids
        if missing or extra:
            raise ValueError(
                f"{name} IDs do not match Task 2: "
                f"{len(missing)} missing and {len(extra)} extra"
            )

    merged_rows = []
    for task2_row in task2_rows:
        image_id = task2_row["id"]
        merged_rows.append(
            {
                "id": image_id,
                "gender": task3_by_id[image_id]["gender"],
                "articleType": task1_by_id[image_id]["predicted_class"],
                "season": task2_row["season"],
                "usage": task3_by_id[image_id]["usage"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(merged_rows)

    print(f"Wrote {len(merged_rows)} predictions to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Task 1–3 predictions into styles_prediction format."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("styles_prediction.csv"),
        help="Output CSV path (default: eda/styles_prediction.csv)",
    )
    args = parser.parse_args()
    build_predictions(args.output)


if __name__ == "__main__":
    main()
