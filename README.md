# COSC2753 — Machine Learning Project

This repository contains four fashion-related machine learning tasks built on the shared **FashionDataset**. Each task lives in its own directory with independent dependencies, configuration, and documentation.

## Dataset layout

All tasks expect the dataset at `data/FashionDataset/`:

```text
data/FashionDataset/
├── train/
│   ├── images_train/<id>.jpg
│   └── styles_train.csv
└── test/
    ├── images_test/<id>.jpg
    └── styles_prediction.csv
```

The test CSV contains only `id` — no metadata columns are populated.

## Tasks

### Task 1 — Fashion Article Type Classification

Classifies fashion product images by `articleType` using a custom CNN trained from scratch in PyTorch. Supports four selectable architectures (`light_cnn`, `simple_cnn`, `deep_cnn`, `resnet_cnn`), optional rare-class grouping, class-weighted loss, and configurable augmentation.

```bash
cd task1 && uv sync
uv run python train.py
uv run python predict.py --data ../data/FashionDataset/test/images_test
uv run python predict.py --image ../data/FashionDataset/test/images_test/52003.jpg
```

- [README](task1/README.md)
- [Installation](task1/docs/installation.md)
- [Architecture](task1/docs/architecture.md)

---

### Task 2 — Fashion Season Classification

Predicts `season` (Spring, Summer, Fall, Winter) from product images using `Task2CNN` — a CNN wrapped in a scikit-learn estimator for `GridSearchCV` hyperparameter tuning. No pretrained weights or metadata features are used.

```bash
cd task2 && uv sync
uv run python train.py
uv run python evaluate.py
uv run python predict.py
```

- [README](task2/README.md)
- [Installation](task2/docs/installation.md)
- [Architecture](task2/docs/architecture.md)
- [Report notes](task2/docs/report_notes.md)

---

### Task 3 — Hybrid Gender and Usage Classification

Jointly predicts `gender` and `usage` (occasion) using a hybrid model that fuses a CNN image encoder with learned embeddings for three metadata fields (`articleType`, `masterCategory`, `baseColour`). Metadata dropout during training allows the model to work even when metadata is unavailable at test time.

```bash
cd task3 && uv sync
uv run python train.py
uv run python predict.py --data ../data/FashionDataset/test/images_test
uv run python predict.py --image ../data/FashionDataset/test/images_test/52003.jpg
```

- [README](task3/README.md)
- [Installation](task3/docs/installation.md)
- [Usage guide](task3/docs/usage.md)
- [Architecture](task3/docs/architecture.md)

---

### Task 4 — Fashion Visual Search

Retrieves visually similar products for a query image. Encodes images as 512-dimensional CLIP vectors and searches with FAISS using exact KNN or approximate ANN (HNSW).

```bash
cd task4 && uv sync
uv run python build_index.py
uv run python search.py --query ../data/FashionDataset/train/images_train/1525.jpg --k 5
```

- [README](task4/README.md)
- [Installation](task4/docs/installation.md)
- [Architecture](task4/docs/architecture.md)
- [KNN vs ANN/HNSW](task4/docs/search-algorithms.md)

---

## Project structure

```text
COSC2753-Project/
├── data/FashionDataset/          # Shared dataset (not committed)
├── eda/                          # Exploratory data analysis notebooks
├── documents/                    # Project brief and specifications
├── task1/                        # Article type classification (CNN)
├── task2/                        # Season classification (CNN + GridSearchCV)
├── task3/                        # Gender & usage classification (Hybrid CNN)
├── task4/                        # Visual search (CLIP + FAISS)
└── README.md
```

Each task directory follows a consistent layout:

```text
task<N>/
├── config/         # JSON configuration files
├── docs/           # Installation, architecture, and usage guides
├── src/            # Source modules (models, datasets, transforms, etc.)
├── results/        # Timestamped run outputs (not committed)
├── train.py        # Training entry point
├── evaluate.py     # Evaluation entry point
├── predict.py      # Prediction entry point
├── pyproject.toml  # Dependencies
└── uv.lock         # Dependency lock file
```

## Requirements

Each task manages its own virtual environment via [uv](https://docs.astral.sh/uv/). Run `uv sync` inside any task directory to install dependencies.
