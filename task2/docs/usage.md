# Usage

Task 2 predicts the four `season` classes with a PyTorch CNN exposed through a scikit-learn estimator. Training performs GridSearchCV, evaluates the selected development model on a fixed holdout, and then refits a final model on all usable labelled rows.

## 1. Prepare the environment and paths

Run commands from `task2`:

```bash
cd task2
uv sync --locked
```

The repository dataset is located at `../data/FashionDataset`. Before execution, update the checked-in `../FashionDataset/...` values as follows:

- `config/train_config.json`: `DATA_DIR_TRAIN` → `../data/FashionDataset/train`
- `config/eval_config.json`: `DATA_DIR_TRAIN` → `../data/FashionDataset/train`
- `config/predict_config.json`: `TEST_IMAGE_DIR` → `../data/FashionDataset/test/images_test`
- `config/predict_config.json`: `TEST_CSV_PATH` → `../data/FashionDataset/test/styles_prediction.csv`

The required structure is:

```text
../data/FashionDataset/
├── train/images_train/<id>.jpg
├── train/styles_train.csv
├── test/images_test/<id>.jpg
└── test/styles_prediction.csv
```

## 2. Configure training

Edit `config/train_config.json`:

| Key | Purpose |
| --- | --- |
| `DATA_DIR_TRAIN`, `LABEL_COL` | Training folder and target (`season`). |
| `EXPECTED_LABELS` | Allowed target classes; keep all four supplied seasons. |
| `MODEL_NAME` | Registry key; currently `task2_cnn`. |
| `MODEL_PARAMS` | Image shape, CNN channels, hidden units, dropout, optimizer, epochs, augmentation, workers, and device. |
| `PARAM_GRID` | GridSearchCV values. Keys use the pipeline prefix `classifier__`. |
| `VAL_RATIO`, `CV_FOLDS`, `REFIT_METRIC` | Holdout size and model-selection procedure. |
| `N_JOBS`, `PARALLEL_BACKEND` | Grid execution. Keep `N_JOBS=1` for GPU/MPS training. |
| `ARTIFACT_DIR`, model paths | Stable locations for the selected model bundles and search results. |
| `SAVE_MODEL_DIR`, `FIGURE_DIR` | Timestamped run outputs and stable diagnostic plots. |

`MODEL_PARAMS.device="auto"` chooses CUDA, MPS, or CPU. Use `"cpu"` to force CPU execution.

## 3. Train and select the model

```bash
uv run python train.py
```

The workflow:

1. validates labelled rows and image files;
2. creates a stratified development/holdout split;
3. searches `PARAM_GRID` with stratified cross-validation on development data;
4. evaluates the best development model once on the untouched holdout;
5. saves that model for reproducible evaluation;
6. refits the selected configuration on all usable data for prediction.

Stable outputs include:

```text
artifacts/holdout_model.joblib       # evaluation only
artifacts/season_model.joblib        # test prediction
artifacts/grid_search_results.csv
artifacts/latest_train_run.json
docs/figures/task2_cnn_epoch_accuracy.png
docs/figures/task2_cnn_epoch_loss.png
```

Detailed metrics, histories, plots, predictions, parameters, and model copies are also written to `results/train/<timestamp>/`.

## 4. Evaluate the holdout model

`config/eval_config.json` must reference `./artifacts/holdout_model.joblib`, not the final all-data model:

```bash
uv run python evaluate.py
```

The evaluator verifies the saved dataset fingerprint and holdout IDs. Changing the training data after fitting will intentionally fail validation. Results are written under `results/eval/<timestamp>/`, including metrics, per-row predictions, a confusion matrix, parameters, and logs.

## 5. Predict the test set

`config/predict_config.json` must reference `./artifacts/season_model.joblib`. Its main output settings are:

| Key | Purpose |
| --- | --- |
| `OUTPUT_FILENAME` | Filename inside the timestamped prediction run. |
| `STABLE_OUTPUT_PATH` | Copy of the latest submission, normally `./task2_predictions.csv`. |
| `SAVE_DIR` | Base folder for timestamped prediction artifacts. |

Run:

```bash
uv run python predict.py
```

The output preserves the supplied test CSV order and schema:

```text
id,gender,articleType,season,usage
```

Only `season` is filled by Task 2. The run directory also contains `season_predictions.csv` with confidence diagnostics, `scores.json`, `parameters.json`, and `logs.txt`.

## Troubleshooting

- Use the holdout bundle for evaluation and the final bundle for prediction; the code rejects the wrong training scope.
- Set `MODEL_PARAMS.num_workers` to `0` for multiprocessing issues.
- Reduce the grid, epochs, batch size, or channel widths when development runs take too long or run out of memory.
- Do not edit the dataset between training and holdout evaluation because fingerprint verification is deliberate.
