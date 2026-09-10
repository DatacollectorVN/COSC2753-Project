# Task 1 — Fashion Image Classification

Task 1 is a configurable PyTorch pipeline for classifying fashion product images by a metadata label such as `articleType`. It covers the complete workflow: loading and preprocessing images, training one of several CNN architectures, evaluating a saved checkpoint, and producing predictions for unseen images.

The implementation addresses class imbalance with optional rare-class grouping and inverse-frequency class weights. Every train, evaluation, and prediction run writes its outputs to a timestamped directory so experiments remain reproducible and easy to compare.

## Features

- OpenCV-based image loading and preprocessing
- Stratified train/validation split with a fixed random seed
- Aspect-ratio-preserving padding and configurable image resizing
- Optional horizontal flip and colour jitter augmentation
- Four selectable architectures: `light_cnn`, `simple_cnn`, `deep_cnn`, and `resnet_cnn`
- Optional rare-class grouping into an `Other` category
- Optional class-weighted cross-entropy loss
- Adam optimization, learning-rate reduction, and early stopping
- Accuracy, precision, recall, F1, per-class metrics, and confusion matrix output
- Timestamped checkpoints, configuration snapshots, logs, plots, and CSV predictions
- Automatic device selection in the order CUDA, Apple MPS, then CPU

## Documentation

- [Installation and usage](docs/installation.md)
- [Architecture and data flow](docs/architecture.md)

## Quick start

All commands below should be run from the `task1` directory because the JSON configurations use relative paths.

```bash
cd task1
uv sync
```

The default configuration expects the dataset at `../data/FashionDataset`:

```text
data/FashionDataset/
├── train/
│   ├── images_train/
│   │   └── <id>.jpg
│   └── styles_train.csv
└── test/
    ├── images_test/
    │   └── <id>.jpg
    └── styles_prediction.csv
```

Train a model after reviewing `config/train_config.json`:

```bash
uv run python train.py
```

Training creates `results/train/<timestamp>/best_model.pth`. Put that checkpoint path into both `config/eval_config.json` and `config/predict_config.json` as needed, then run:

```bash
uv run python evaluate.py
uv run python predict.py
```

See the [installation guide](docs/installation.md) for environment setup, configuration details, output files, and troubleshooting.

## Project structure

```text
task1/
├── config/
│   ├── train_config.json       # Training, model, and preprocessing settings
│   ├── eval_config.json        # Validation evaluation settings
│   └── predict_config.json     # Test inference settings
├── docs/
│   ├── installation.md
│   └── architecture.md
├── src/
│   ├── models/                 # CNN implementations and model registry
│   ├── custom_dataset.py       # Datasets, class handling, and data loaders
│   ├── evaluator.py            # Evaluation and prediction workflows
│   ├── metric_evaluation.py    # Classification metrics
│   ├── trainer.py              # Training workflow
│   ├── transforms.py           # Image preprocessing and augmentation
│   └── utils.py                # Device, seeds, logging, and run artifacts
├── evaluate.py                 # Evaluation entry point
├── predict.py                  # Prediction entry point
├── train.py                    # Training entry point
├── pyproject.toml              # Package metadata and dependencies
└── uv.lock                     # Reproducible dependency lock file
```

## Configuration overview

The entry-point scripts load their corresponding JSON file and pass its keys directly to `FashionTrainer` or `FashionEvaluator`. Important training choices include:

| Setting | Purpose |
| --- | --- |
| `MODEL_NAME` | Selects a model registered in `src/models/model_zoo.py`. |
| `MODEL_PARAMS` | Supplies architecture-specific channels, dropout, hidden width, or ResNet depth. |
| `TRANSFORM_PARAMS` | Controls image size, padding, normalization, and optional augmentation. |
| `MIN_CLASS_COUNT` | Groups labels with fewer samples than this threshold into `Other`; use `0` to disable. |
| `USE_CLASS_WEIGHTS` | Enables inverse-frequency weights in cross-entropy loss. |
| `SAVE_BEST` | Chooses `val_acc`, `val_loss`, `train_acc`, or `train_loss` for checkpoint selection. |

For the full component and runtime design, see [architecture.md](docs/architecture.md).

