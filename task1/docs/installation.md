# Installation and Usage

## Requirements

- Python 3.11 or newer
- `uv` (recommended) or `pip`
- The FashionDataset arranged as described below
- Optional: an NVIDIA CUDA-capable GPU or Apple Silicon device with MPS support

The code also runs on CPU. Device selection is automatic: CUDA is preferred, followed by Apple MPS, then CPU.

## Install with uv

From the repository root:

```bash
cd task1
uv sync
```

`uv sync` creates a local `.venv` and installs the versions resolved in `uv.lock`. Confirm the environment with:

```bash
uv run python --version
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

To install `uv` when it is not already available, follow the official uv installation instructions for your operating system, then run `uv sync` again.

## Install with standard Python and pip

Using `uv` is preferred because this repository includes a lock file. A conventional virtual environment is also possible:

```bash
cd task1
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Commands in the remainder of this guide use `uv run`. When using the activated pip environment, remove the `uv run` prefix.

## Dataset layout

The checked-in configuration paths are relative to `task1`, so commands must be run from that directory unless the paths are changed.

```text
COSC2753-Project/
├── data/
│   └── FashionDataset/
│       ├── train/
│       │   ├── images_train/
│       │   │   ├── 10000.jpg
│       │   │   └── ...
│       │   └── styles_train.csv
│       └── test/
│           ├── images_test/
│           │   ├── 10001.jpg
│           │   └── ...
│           └── styles_prediction.csv
└── task1/
```

For training, `styles_train.csv` must contain:

- `id`: image identifier matching an `<id>.jpg` filename
- the configured label column, `articleType` by default

Rows with a missing label or without a matching image file are excluded. Prediction reads all `.jpg` files directly from `TEST_IMAGE_DIR`; `styles_prediction.csv` is part of the expected dataset layout but is not consumed by the current prediction loader.

## Configure and train

Edit `config/train_config.json` before starting a run. The default configuration trains a custom ResNet-50 using 96 × 128 RGB inputs, groups classes with fewer than 50 samples into `Other`, and applies class-weighted loss.

Common settings are:

| Key | Description |
| --- | --- |
| `DATA_DIR_TRAIN` | Directory containing `styles_train.csv` and `images_train/`. |
| `LABEL_COL` | CSV column to predict. |
| `SAVE_MODEL_DIR` | Base directory for timestamped training outputs. |
| `MODEL_NAME` | `light_cnn`, `simple_cnn`, `deep_cnn`, or `resnet_cnn`. |
| `MODEL_PARAMS` | Parameters passed to the selected model constructor. |
| `TRANSFORM_PARAMS` | Preprocessing and optional augmentation settings. |
| `MIN_CLASS_COUNT` | Minimum class frequency; lower-frequency classes become `Other`. Set to `0` to disable. |
| `USE_CLASS_WEIGHTS` | Whether to weight cross-entropy inversely by class frequency. |
| `VAL_RATIO` | Fraction used for the stratified validation split. |
| `SAVE_BEST` | Metric used to keep the best checkpoint. |
| `NUM_WORKERS` | Data-loader worker processes. Use `0` if multiprocessing causes problems. |

Architecture-specific `MODEL_PARAMS`:

| Model | Supported parameters |
| --- | --- |
| `light_cnn` | `channels`, `dropout_rate`, `fc_hidden` |
| `simple_cnn` | `channels`, `dropout_rate`, `fc_hidden` |
| `deep_cnn` | `channels`, `dropout_rate`, `fc_hidden` |
| `resnet_cnn` | `depth` (`50`, `101`, or `152`), `dropout_rate` |

Optional training augmentation can be enabled inside `TRANSFORM_PARAMS`:

```json
{
  "width": 96,
  "height": 128,
  "pad_value": 0,
  "flip_p": 0.5,
  "brightness": 0.2,
  "contrast": 0.2,
  "saturation": 0.2
}
```

Run training:

```bash
uv run python train.py
```

Each run creates:

```text
results/train/<timestamp>/
├── best_model.pth       # Model state and training metadata
├── label_map.json       # Class name to output-index mapping
├── logs.txt             # Console-equivalent training log
├── parameters.json      # Exact configuration used by the run
├── scores.json          # Per-epoch metrics and best metric
└── training_curves.png  # Loss and accuracy curves
```

The checkpoint is selected by `SAVE_BEST`. Training stops when that metric fails to improve for `EARLY_STOPPING_PATIENCE` epochs. `ReduceLROnPlateau` independently reduces the learning rate when validation loss plateaus.

## Evaluate a checkpoint

Update `CHECKPOINT_PATH` in `config/eval_config.json`, for example:

```json
"CHECKPOINT_PATH": "./results/train/20260907_234357/best_model.pth"
```

Keep `DATA_DIR_TRAIN`, `LABEL_COL`, `VAL_RATIO`, and `SEED` consistent with training. This recreates the same deterministic validation split. Model type, model parameters, class mapping, rare-class threshold, and transforms are restored from the checkpoint.

```bash
uv run python evaluate.py
```

Evaluation writes `parameters.json`, `scores.json`, and `logs.txt` under `results/eval/<timestamp>/`. The scores include overall, macro, weighted, and per-class metrics plus the confusion matrix.

## Predict test images

Set `CHECKPOINT_PATH` and `TEST_IMAGE_DIR` in `config/predict_config.json`, then run:

```bash
uv run python predict.py
```

Prediction writes these files under `results/predict/<timestamp>/`:

- `predictions.csv` with `id`, `predicted_class`, and `confidence`
- `parameters.json` containing the prediction configuration
- `logs.txt` containing run details

The transform settings and class mapping come from the checkpoint, which keeps inference consistent with training.

## Troubleshooting

### File or directory not found

Run the scripts from `task1`, or replace relative dataset and output paths in the JSON files with valid absolute paths. Verify that every CSV `id` has a corresponding `.jpg` file.

### Checkpoint placeholder error

The provided evaluation and prediction configurations contain `<run_timestamp>`. Replace it with an actual training run directory before executing those scripts.

### Data-loader multiprocessing issues

Set `NUM_WORKERS` to `0`, especially when debugging or when the operating system has trouble starting worker processes.

### Out-of-memory errors

Reduce `BATCH_SIZE`, choose `light_cnn` or `simple_cnn`, reduce model channel widths, or lower the input `width` and `height`. ResNet-101 and ResNet-152 require substantially more memory than the smaller CNNs.

### Slow CPU training

Start with `light_cnn`, a smaller batch or image size, and fewer `MAX_EPOCHS`. Checkpoint files are portable because evaluation loads them onto the automatically selected device.

