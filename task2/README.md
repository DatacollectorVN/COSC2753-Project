# Task 2 — Fashion Season Classification

Task 2 predicts `season` from each fashion-product image with `Task2CNN`, a
convolutional neural network trained from scratch on the supplied
FashionDataset. A scikit-learn-compatible wrapper lets `GridSearchCV` tune the
PyTorch network. No pretrained weights, external training data, or metadata
predictors are used.

## Task2CNN design

- Input: 60 × 80 RGB image, padded when necessary to preserve aspect ratio
- Feature learner: three Conv2d–BatchNorm–ReLU–MaxPool blocks
- Head: fixed average pooling, a 128-unit ReLU layer, dropout, and four logits
- Training: AdamW, class-aware cross-entropy, augmentation, and early stopping
- Selection metric: mean cross-validation macro-F1
- Classes: `Spring`, `Summer`, `Fall`, and `Winter`

The model loads images lazily from their paths. It does not use `articleType`,
`baseColour`, `productDisplayName`, or any other metadata predictor.

## Run the workflow

Run all commands from `task2`, because configuration paths are relative to this
folder:

```bash
cd task2
uv sync
uv run python train.py
uv run python evaluate.py
uv run python predict.py
```

`train.py` performs four parameter combinations across three folds, followed by
a development-model refit and an all-data refit. Keep the terminal running
until the final model path is printed. PyTorch automatically uses CUDA, Apple
MPS, or CPU. Set `MODEL_PARAMS.device` to `"cpu"` only when CPU execution is
required.

## Training and GridSearchCV

The workflow:

1. Validates the training CSV and matching images.
2. Creates a fixed stratified 80/20 development/holdout split.
3. Runs three-fold `GridSearchCV` on the development rows only.
4. Selects the parameters with the highest mean CV macro-F1.
5. Evaluates the selected development model once on the untouched holdout.
6. Refits the selected Task2CNN settings on all usable labelled rows.
7. Saves separate holdout and all-data model bundles.

The unlabelled assignment test set is used only for final prediction.

The configured grid searches:

| Parameter | Values |
| --- | --- |
| `classifier__channels` | `[24, 48, 96]`, `[32, 64, 128]` |
| `classifier__dropout_rate` | `0.25`, `0.40` |
| `classifier__learning_rate` | `0.001` |

Keep `N_JOBS=1` for GPU training so multiple CNN fits do not compete for GPU
memory.

## Epoch graphs

Training produces only these two learning-curve figures:

```text
docs/figures/task2_cnn_epoch_accuracy.png
docs/figures/task2_cnn_epoch_loss.png
```

Each graph contains training and internal-validation values over epochs. Its
legend is below the graph. The loss graph shows the class-weighted
cross-entropy optimized during training. The underlying values are
in `docs/task2_cnn_epoch_history.csv`. These validation curves do not use the
assignment test set, which has no season labels.

## Completed result

The completed run at `results/train/20260911_104332` selected channels
`[32, 64, 128]`, dropout `0.25`, and learning rate `0.001`. It achieved:

| Metric | Score |
| --- | ---: |
| Mean CV macro-F1 | 0.7155 |
| Holdout accuracy | 0.7257 |
| Holdout macro-F1 | 0.7249 |
| Holdout balanced accuracy | 0.7029 |

This result is below the requested 80% holdout-accuracy target and should be
reported as 72.57%, without rounding it to 80%.

## Outputs

Stable files are:

```text
artifacts/grid_search_results.csv
artifacts/holdout_model.joblib
artifacts/season_model.joblib
artifacts/latest_train_run.json
task2_predictions.csv
```

`evaluate.py` reproduces the holdout evaluation from the saved development
model. `predict.py` loads the all-data model and preserves the required output
columns and test-row order.

The Task 2 EDA notebook remains in the shared folder:

```bash
uv run jupyter execute --inplace ../eda/task2_season_eda.ipynb
```

EDA reads and describes the data; it does not alter the supplied dataset.

## Project structure

```text
task2/
├── config/
├── docs/
├── src/
│   ├── models/
│   │   ├── task2_cnn.py
│   │   └── model_zoo.py
│   ├── custom_dataset.py
│   ├── data_fingerprint.py
│   ├── evaluator.py
│   ├── metric_evaluation.py
│   ├── trainer.py
│   ├── transforms.py
│   ├── utils.py
│   └── visualization.py
├── artifacts/
├── results/
├── train.py
├── evaluate.py
├── predict.py
├── pyproject.toml
└── uv.lock
```

More detail is available in [installation.md](docs/installation.md),
[usage.md](docs/usage.md), [architecture.md](docs/architecture.md), and
[report_notes.md](docs/report_notes.md).
