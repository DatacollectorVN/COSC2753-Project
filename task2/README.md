# Task 2 — Fashion Season Classification

Task 2 predicts the `season` label from a fashion-item image. The implementation is a complete, configuration-driven scikit-learn workflow: validated metadata loading, deterministic image preprocessing, HOG and HSV feature extraction, an untouched stratified holdout, a majority baseline, five-fold `GridSearchCV`, final refitting, saved models, repeatable evaluation, and prediction in the supplied CSV format.

The classifier is trained entirely on the supplied FashionDataset. It does not use pretrained weights or metadata predictors. `GridSearchCV` is the model-selection procedure; `LinearSVC` is the learning algorithm it tunes.

## Features

- Preserves the original 3:4 product-image ratio while padding unusual dimensions
- Uses 48 × 64 RGB images to limit memory and grid-search runtime
- Combines HOG shape descriptors with normalized HSV colour histograms
- Caches validated feature matrices so later runs do not reopen every image
- Removes rows with no `season` or no matching image without changing the source data
- Uses a fixed 80/20 stratified development/holdout split
- Compares against a majority-class `DummyClassifier`
- Tunes `LinearSVC` regularization and class weighting with five-fold stratified `GridSearchCV`
- Selects hyperparameters by macro-F1 and also records accuracy and weighted-F1
- Saves a development-only model for honest holdout evaluation
- Refits the selected pipeline on every usable labelled image for final prediction
- Preserves the five-column prediction template and original test-ID order
- Writes timestamped logs, parameters, metrics, confusion matrices, and predictions

## Quick start

Run all commands from `task2` because configuration paths are relative to this directory:

```bash
cd task2
uv sync
uv run jupyter execute --inplace notebooks/task2_season_eda.ipynb
uv run python train.py
uv run python evaluate.py
uv run python predict.py
```

Training creates stable model files in `artifacts/`, as well as a complete timestamped run under `results/train/`. Evaluation reads `artifacts/holdout_model.joblib`; prediction reads `artifacts/season_model.joblib`.

The final Task 2 prediction file is written to:

```text
task2_predictions.csv
```

It retains these columns exactly:

```text
id,gender,articleType,season,usage
```

Only `season` is filled by Task 2. Merge this column with the final Task 1 and Task 3 outputs when preparing the assignment-wide Canvas prediction submission.

## EDA evidence

The executed [season EDA notebook](notebooks/task2_season_eda.ipynb) found:

- 38,617 metadata rows and 38,612 training images
- 20 rows without a season label
- 5 metadata rows without an image
- 38,592 usable labelled examples
- no corrupt JPEG files
- 5,829 correctly aligned test rows and images
- Summer at 49.583% of usable data and Spring at only 4.058%
- a majority baseline of 0.4958 accuracy and 0.1657 macro-F1

These results justify stratification, macro-F1 model selection, class-weight tuning, and per-class reporting.

## Verified model results

The completed 40-fit search selected `classifier__C=0.01` and `classifier__class_weight=null`:

| Result | Value |
| --- | ---: |
| Mean five-fold CV macro-F1 | 0.6720 |
| Holdout accuracy | 0.6867 |
| Holdout balanced accuracy | 0.6299 |
| Holdout macro-F1 | 0.6684 |
| Holdout weighted-F1 | 0.6820 |
| Majority baseline macro-F1 | 0.1657 |

The saved holdout report contains per-class precision, recall, and F1. The final model was refitted on all 38,592 usable rows, and the generated 5,829-row prediction CSV passed schema, ID-order, missing-value, and class-label validation.

## Project structure

```text
task2/
├── config/
│   ├── eda_config.json
│   ├── train_config.json
│   ├── eval_config.json
│   └── predict_config.json
├── docs/
│   ├── architecture.md
│   └── installation.md
├── notebooks/
│   └── task2_season_eda.ipynb
├── src/
│   ├── models/
│   │   ├── linear_svc.py
│   │   └── model_zoo.py
│   ├── custom_dataset.py
│   ├── evaluator.py
│   ├── features.py
│   ├── metric_evaluation.py
│   ├── trainer.py
│   ├── transforms.py
│   └── utils.py
├── artifacts/                 # Stable models, feature caches, and grid results
├── results/                   # EDA and timestamped workflow outputs
├── train.py
├── evaluate.py
├── predict.py
├── pyproject.toml
└── uv.lock
```

## Default GridSearchCV investigation

The default grid contains eight candidates and five folds, producing 40 cross-validation fits:

| Parameter | Values | Purpose |
| --- | --- | --- |
| `classifier__C` | `0.01`, `0.1`, `1.0`, `10.0` | Compare regularization strengths. |
| `classifier__class_weight` | `null`, `balanced` | Measure the tradeoff between aggregate and minority-class performance. |

The scaler remains inside the scikit-learn `Pipeline`, so it is fitted separately in every cross-validation fold. The supplied test images are used only after model selection and final refitting.

## Documentation

- [Installation and usage](docs/installation.md)
- [Architecture and evaluation design](docs/architecture.md)
- [Task 2 report evidence and literature comparison](docs/report_notes.md)
- [Generated EDA findings](results/eda/eda_findings.md)
