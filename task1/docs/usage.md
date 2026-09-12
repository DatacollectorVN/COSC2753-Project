# Usage

Task 1 trains, evaluates, and runs inference with a single-label fashion image classifier. The default target is `articleType`.

## 1. Prepare the environment

Run every command from the `task1` directory because entry points and JSON paths are relative to it:

```bash
cd task1
uv sync --locked
```

The expected dataset layout is:

```text
../data/FashionDataset/
├── train/
│   ├── images_train/<id>.jpg
│   └── styles_train.csv
└── test/
    ├── images_test/<id>.jpg
    └── styles_prediction.csv
```

## 2. Configure and train

Edit `config/train_config.json`. The most important keys are:

| Key | Purpose |
| --- | --- |
| `DATA_DIR_TRAIN` | Folder containing `styles_train.csv` and `images_train/`. |
| `LABEL_COL` | CSV target column, currently `articleType`. |
| `MODEL_NAME` | `light_cnn`, `simple_cnn`, `deep_cnn`, or `resnet_cnn`. |
| `MODEL_PARAMS` | Parameters accepted by the selected model, such as channels, hidden width, dropout, or ResNet depth. |
| `TRANSFORM_PARAMS` | Output width/height, padding, normalization, and optional augmentation. |
| `MIN_CLASS_COUNT` | Maps classes below this count to `Other`; set `0` to disable. |
| `USE_CLASS_WEIGHTS` | Enables inverse-frequency cross-entropy weights. |
| `VAL_RATIO`, `SEED` | Control the reproducible stratified validation split. |
| `SAVE_BEST` | `val_acc`, `val_loss`, `train_acc`, or `train_loss`. |
| `BATCH_SIZE`, `NUM_WORKERS` | Data-loader settings. |
| `LEARNING_RATE`, `WEIGHT_DECAY` | Adam settings. |
| `MAX_EPOCHS`, `EARLY_STOPPING_PATIENCE` | Training duration and stopping. |

Start training:

```bash
uv run python train.py
```

Outputs are written to `results/train/<timestamp>/`:

```text
best_model.pth
label_map.json
parameters.json
scores.json
training_curves.png
logs.txt
```

The checkpoint contains the model architecture, label map, rare-class setting, and transforms. Keep it with the run artifacts.

## 3. Evaluate

Set `CHECKPOINT_PATH` in `config/eval_config.json` to the selected checkpoint. Keep `DATA_DIR_TRAIN`, `LABEL_COL`, `SEED`, and `VAL_RATIO` consistent with training to reproduce the validation split.

Evaluate the validation subset:

```bash
uv run python evaluate.py
```

Evaluate all usable labelled training rows:

```bash
uv run python evaluate.py --all
```

Each run writes `parameters.json`, `scores.json`, and `logs.txt` under `results/eval/<timestamp>/`. `scores.json` includes accuracy, macro/weighted precision, recall and F1, per-class metrics, and the confusion matrix.

## 4. Predict

Set `CHECKPOINT_PATH`, `SAVE_DIR`, `BATCH_SIZE`, and `NUM_WORKERS` in `config/predict_config.json`.

Predict every supported image in a folder:

```bash
uv run python predict.py \
  --data ../data/FashionDataset/test/images_test
```

This writes `predictions.csv`, `parameters.json`, and `logs.txt` to `results/predict/<timestamp>/`. The CSV columns are `id`, `predicted_class`, and `confidence`.

Predict one image without creating a prediction run directory:

```bash
uv run python predict.py \
  --image ../data/FashionDataset/test/images_test/52003.jpg
```

The class and confidence are printed to the terminal. `--data` and `--image` are mutually exclusive and one is required. The folder supplied with `--data` overrides `TEST_IMAGE_DIR` from JSON.

## Troubleshooting

- Run commands from `task1`; otherwise the fixed `config/*.json` paths will not resolve.
- Set `NUM_WORKERS` to `0` if worker processes fail.
- Reduce `BATCH_SIZE`, image dimensions, or model size for out-of-memory errors.
- Verify every CSV `id` has a matching `<id>.jpg` and that the checkpoint path exists.
