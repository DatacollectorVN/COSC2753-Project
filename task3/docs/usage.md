# Usage

## Before running a workflow

Run commands from the `task3` directory:

```bash
cd task3
```

Each entry point reads one fixed JSON file:

| Command | Configuration |
| --- | --- |
| `uv run python train.py` | `config/train_config.json` |
| `uv run python evaluate.py` | `config/eval_config.json` |
| `uv run python predict.py` | `config/predict_config.json` |

The scripts do not currently accept command-line overrides. Edit the corresponding JSON file before starting a run.

## Train

Review `config/train_config.json`, then run:

```bash
uv run python train.py
```

Important settings are:

| Key | Description |
| --- | --- |
| `DATA_DIR_TRAIN` | Directory containing `styles_train.csv` and `images_train/`. |
| `SAVE_MODEL_DIR` | Base directory for timestamped training artifacts. |
| `SAVE_BEST` | Checkpoint and early-stopping metric: `val_loss`, `val_gender_acc`, `val_occasion_acc`, or `train_loss`. |
| `SEED` | Python, NumPy, PyTorch, and split seed. |
| `MODEL_NAME` | Registry key; currently only `hybrid_cnn`. |
| `MODEL_PARAMS` | Backbone, embedding, fusion, head, and dropout settings. |
| `TRANSFORM_PARAMS` | Image dimensions, padding, normalization, and optional augmentation. |
| `MIN_OCCASION_COUNT` | Removes occasion classes occurring fewer times than this value; use `0` to disable. |
| `USE_CLASS_WEIGHTS` | Enables inverse-frequency weights for both cross-entropy losses. |
| `VAL_RATIO` | Validation fraction used by the combined-label stratified split. |
| `NUM_WORKERS` | Data-loader worker processes. |
| `LEARNING_RATE`, `WEIGHT_DECAY` | Adam optimizer settings. |
| `LR_SCHEDULE_FACTOR`, `LR_PATIENCE` | `ReduceLROnPlateau` settings. |
| `EARLY_STOPPING_PATIENCE` | Epochs without improvement before stopping. |
| `MAX_EPOCHS` | Maximum training epochs. |

### Model parameters

`MODEL_PARAMS.backbone` supports:

- `simple`: repeated Conv2d → BatchNorm → ReLU → MaxPool blocks. Set widths with `channels`.
- `resnet`: a custom bottleneck ResNet. Set `depth` to `50`, `101`, or `152`.

The remaining parameters control metadata embedding sizes (`article_embed_dim`, `master_embed_dim`, `colour_embed_dim`), metadata MLP output (`meta_dim`), head width (`fc_hidden`), head dropout (`dropout_rate`), and whole-vector metadata dropout (`meta_dropout_p`).

### Image transforms

The default training pipeline is:

```text
pad to 3:4 → resize to 96×128 → random horizontal flip → colour jitter
→ ImageNet normalization → CHW float tensor
```

Validation and prediction omit random augmentation. Remove `flip_p` to disable flipping. Remove all three of `brightness`, `contrast`, and `saturation` to disable colour jitter.

### Training outputs

Each run creates:

```text
results/train/<YYYYMMDD_HHMMSS>/
├── best_model.pth
├── gender_label_map.json
├── occasion_label_map.json
├── meta_vocabs.json
├── parameters.json
├── scores.json
├── training_curves.png
└── logs.txt
```

`best_model.pth` includes the weights plus everything required to reconstruct the model: output sizes, label maps, metadata vocabularies, model parameters, occasion threshold, and image transforms. Its optimizer state and saved-epoch metrics are retained as provenance, although training does not currently expose a resume operation.

The plotted accuracy curves show gender accuracy. Occasion accuracy remains available for every epoch in `scores.json` and `logs.txt`.

## Evaluate a checkpoint

Replace the placeholder in `config/eval_config.json`:

```json
"CHECKPOINT_PATH": "./results/train/20260908_232652/best_model.pth"
```

Keep `DATA_DIR_TRAIN`, `SEED`, and `VAL_RATIO` consistent with training so evaluation recreates the same validation split. The checkpoint supplies model parameters, transforms, mappings, vocabularies, and `MIN_OCCASION_COUNT`.

```bash
uv run python evaluate.py
```

Evaluation writes `logs.txt`, `parameters.json`, and `scores.json` to `results/eval/<timestamp>/`. For each head, `scores.json` contains:

- overall accuracy
- macro and weighted precision, recall, and F1
- precision, recall, F1, accuracy, and support per class
- a scikit-learn classification report
- a confusion matrix

### Current evaluation summary issue

The current `evaluate.py` entry point tries to print `metrics['report']['accuracy']`, but `FashionEvaluator.evaluate()` returns top-level `gender` and `occasion` dictionaries. Consequently, evaluation artifacts are saved successfully and then the entry point raises `KeyError: 'report'`. The per-head results are still available in the new run directory. A correct programmatic summary would read `metrics['gender']['overall_accuracy']` and `metrics['occasion']['overall_accuracy']`.

## Predict test images

Set an actual checkpoint path in `config/predict_config.json`, verify `TEST_IMAGE_DIR`, and optionally set `TEST_CSV_PATH`:

```bash
uv run python predict.py
```

The loader scans only lowercase `*.jpg` files immediately inside `TEST_IMAGE_DIR`. It sorts them by path and uses each filename stem as the output `id`.

If the prediction CSV exists, metadata is matched after converting CSV IDs to strings. Values found in the checkpoint vocabularies are embedded normally; missing or unseen values use unknown index `0`. If the CSV does not exist, every metadata field uses `0`, and the image branch still produces predictions.

Each prediction run creates:

```text
results/predict/<YYYYMMDD_HHMMSS>/
├── predictions.csv
├── parameters.json
└── logs.txt
```

`predictions.csv` has this schema:

| Column | Meaning |
| --- | --- |
| `id` | Test image filename without `.jpg`. |
| `gender` | Predicted gender label. |
| `gender_confidence` | Maximum softmax probability for the gender head. |
| `usage` | Predicted occasion label. |
| `usage_confidence` | Maximum softmax probability for the occasion head. |

The confidence values are rounded to four decimal places and should be interpreted as model scores, not guaranteed calibrated probabilities.

## Common problems

- **Checkpoint placeholder:** replace `<run_timestamp>` in evaluation and prediction configuration before running.
- **Empty or incomplete results:** verify CSV IDs match image filename stems and images use lowercase `.jpg` extensions.
- **Different evaluation split:** restore the same dataset contents, `SEED`, and `VAL_RATIO` used for training.
- **Stratification error:** very small datasets may not contain enough members of each combined gender/occasion group for the configured validation size.
- **Out-of-range configuration:** `SAVE_BEST` must be one of the metrics implemented in `src/trainer.py`, and ResNet depth must be `50`, `101`, or `152`.
