# Installation and Usage

## Requirements

- Python 3.11 or newer
- `uv`
- FashionDataset at `../FashionDataset` relative to `task2`
- enough time and memory for twelve CNN cross-validation fits

Task2CNN supports CUDA, Apple MPS, and CPU. A GPU is recommended.

## 1. Install

From the repository root:

```bash
cd task2
uv sync
```

## 2. Check the dataset

```text
COSC2753-Project/
├── FashionDataset/
│   ├── train/
│   │   ├── images_train/<id>.jpg
│   │   └── styles_train.csv
│   └── test/
│       ├── images_test/<id>.jpg
│       └── styles_prediction.csv
└── task2/
```

The training CSV must contain `id` and `season`. The prediction template must
retain `id,gender,articleType,season,usage` in that order.

## 3. Review configuration

The checked-in `config/train_config.json` uses:

- `MODEL_NAME`: `task2_cnn`
- `CV_FOLDS`: 3
- four parameter candidates, giving twelve CV fits
- at most 30 epochs per fit, with early stopping
- `device`: `auto`
- `N_JOBS`: 1

Keep `N_JOBS=1` when training on a GPU.

## 4. Train

```bash
uv run python train.py
```

Keep the terminal open. Each fit prints epoch progress. Training finishes after
GridSearchCV, holdout evaluation, and the final refit on all usable labelled
rows.

The two generated epoch figures are:

```text
docs/figures/task2_cnn_epoch_accuracy.png
docs/figures/task2_cnn_epoch_loss.png
```

Both legends appear below their graphs.

## 5. Evaluate

```bash
uv run python evaluate.py
```

This loads `artifacts/holdout_model.joblib` and evaluates only the saved holdout
IDs.

## 6. Predict

```bash
uv run python predict.py
```

This loads `artifacts/season_model.joblib` and writes
`task2_predictions.csv` plus timestamped diagnostic files under
`results/predict/`.

## Optional EDA

The notebook is stored in the repository-level `eda` folder:

```bash
uv run jupyter execute --inplace ../eda/task2_season_eda.ipynb
```

The completed Task2CNN reached 0.7257 holdout accuracy and 0.7249 macro-F1.
