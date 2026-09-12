# Task 3 — Hybrid Fashion Classification

Task 3 is a configuration-driven PyTorch pipeline that predicts two attributes for each fashion product:

- `gender`
- `usage` (the product occasion)

The model combines an image representation with categorical product metadata (`articleType`, `masterCategory`, and `baseColour`). The fused representation feeds two independent classification heads, allowing both targets to be learned in one training run.

## Features

- Joint gender and occasion classification
- Image branch with either a lightweight CNN or a custom ResNet-50/101/152 backbone
- Learned embeddings and an MLP for three metadata fields
- Metadata dropout so inference can still run when metadata is missing
- Aspect-ratio-preserving OpenCV preprocessing and configurable augmentation
- Combined gender-by-occasion stratified train/validation split
- Optional inverse-frequency class weighting
- Adam optimization, learning-rate reduction, checkpointing, and early stopping
- Per-head aggregate, per-class, and confusion-matrix metrics
- Timestamped logs, configuration snapshots, checkpoints, plots, and CSV predictions
- Automatic CUDA, Apple MPS, or CPU device selection

## Documentation

- [Installation](docs/installation.md)
- [Usage](docs/usage.md)
- [Architecture](docs/architecture.md)

## Quick start

Run all commands from `task3`, because the checked-in JSON configurations contain paths relative to this directory.

```bash
cd task3
uv sync
uv run python train.py
```

Training writes its best checkpoint to `results/train/<timestamp>/best_model.pth`. Replace `<run_timestamp>` in the evaluation and prediction configurations with that directory name, then run:

```bash
uv run python evaluate.py
uv run python predict.py
```

The default configuration expects this dataset layout:

```text
COSC2753-Project/
├── data/FashionDataset/
│   ├── train/
│   │   ├── images_train/<id>.jpg
│   │   └── styles_train.csv
│   └── test/
│       ├── images_test/<id>.jpg
│       └── styles_prediction.csv
└── task3/
```

See the [usage guide](docs/usage.md) before changing paths, model parameters, label filtering, or checkpoint selection.

## Project structure

```text
task3/
├── config/
│   ├── train_config.json       # Training, preprocessing, and model settings
│   ├── eval_config.json        # Validation evaluation settings
│   └── predict_config.json     # Test inference settings
├── docs/
│   ├── installation.md
│   ├── usage.md
│   └── architecture.md
├── src/
│   ├── models/
│   │   ├── hybrid_cnn.py       # Image/metadata model and both output heads
│   │   └── model_zoo.py        # Model registry and factory
│   ├── custom_dataset.py       # Datasets, vocabularies, split, and loaders
│   ├── evaluator.py            # Checkpoint evaluation and prediction
│   ├── metric_evaluation.py    # Classification metrics
│   ├── trainer.py              # Training and validation loop
│   ├── transforms.py           # OpenCV image transforms
│   └── utils.py                # Device, seed, logging, stopping, and plotting
├── train.py
├── evaluate.py
├── predict.py
├── pyproject.toml
└── uv.lock
```

## Model at a glance

```text
RGB image ──> CNN or ResNet ───────────────┐
                                            ├──> fusion ──> gender logits
3 metadata fields ──> embeddings ──> MLP ──┘           └──> occasion logits
```

Training minimizes the sum of the gender and occasion cross-entropy losses. Model construction settings, transforms, label maps, and metadata vocabularies are stored in the checkpoint and restored for evaluation and prediction.
