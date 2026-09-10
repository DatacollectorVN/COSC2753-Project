# Installation and Usage

## Requirements

- Python 3.11 or newer
- `uv`
- the supplied FashionDataset at `../FashionDataset` relative to `task2`
- sufficient disk space for the local environment and cached feature matrices

The workflow uses CPU-based scikit-learn. A GPU is not required.

## Install

From the repository root:

```bash
cd task2
uv sync
```

`uv sync` creates `task2/.venv` and installs the versions recorded in `uv.lock`.

## Dataset layout

```text
COSC2753-Project/
├── FashionDataset/
│   ├── train/
│   │   ├── images_train/
│   │   │   └── <id>.jpg
│   │   └── styles_train.csv
│   └── test/
│       ├── images_test/
│       │   └── <id>.jpg
│       └── styles_prediction.csv
└── task2/
```

The training CSV must provide `id` and `season`. The prediction template must preserve this exact column order:

```text
id,gender,articleType,season,usage
```

## Run the EDA

```bash
uv run jupyter execute --inplace notebooks/task2_season_eda.ipynb
```

The notebook retains its displayed results and writes report-ready evidence under `results/eda/`.

## Train and tune

Review `config/train_config.json`, then run:

```bash
uv run python train.py
```

The first run extracts features for all usable training images and saves a validated cache. Later runs reuse it if the ordered image set and feature configuration are unchanged.

Training performs these stages:

1. Validate and filter the supplied metadata.
2. Extract or load HOG and HSV features.
3. Create a fixed stratified 80/20 split.
4. Evaluate the majority-class baseline on the holdout.
5. Run five-fold GridSearchCV only on the development split.
6. Evaluate the selected development model once on the holdout.
7. Save that model for reproducible evaluation.
8. Clone and refit the selected pipeline on all usable labelled rows.
9. Save the final all-data model for prediction.

Stable outputs are:

```text
artifacts/
├── features/train_features.npz
├── grid_search_results.csv
├── holdout_model.joblib
├── season_model.joblib
└── latest_train_run.json
```

The timestamped training folder also contains both models, the exact parameters, dataset audit, split IDs, baseline and holdout metrics, classification report, confusion matrix, predictions, grid results, and logs.

## Re-evaluate the holdout model

```bash
uv run python evaluate.py
```

Evaluation uses the holdout IDs embedded in `artifacts/holdout_model.joblib`. It refuses to evaluate the final all-data model because that model has already seen the holdout rows.

## Predict the test set

```bash
uv run python predict.py
```

Prediction writes a timestamped submission file and a stable copy:

```text
task2_predictions.csv
```

The diagnostic `season_predictions.csv` includes the decision score. This score is the maximum SVM decision value and is not a calibrated probability.

## Configuration guidance

- Keep `LABEL_COL` as `season` for Task 2.
- Keep `EXPECTED_LABELS` aligned with the four labels found in the supplied data.
- Increase `N_JOBS` only when sufficient RAM is available.
- Set `FORCE_REBUILD_FEATURES` to `true` after deliberately changing image files while retaining the same filenames and timestamps.
- Changing any `FEATURE_PARAMS` automatically invalidates the feature cache.
- Keep macro-F1 as `REFIT_METRIC` unless the report clearly justifies another model-selection objective.

## Submission notes

Include the source, lock file, required saved model, and README in the assignment code ZIP. The large feature caches can be regenerated and are not required to run prediction when `season_model.joblib` is supplied. The final Canvas prediction CSV must combine the Task 2 `season` values with the other task outputs without changing the supplied format.
