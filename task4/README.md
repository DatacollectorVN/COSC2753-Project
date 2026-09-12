# Task 4 — Fashion Visual Search

This project retrieves the most visually similar fashion products for a query
image. It encodes images as 512-dimensional, L2-normalized CLIP vectors and
searches those vectors with FAISS using either exact K-nearest neighbours (KNN)
or approximate nearest neighbours (ANN) through HNSW.

The repository currently contains an index for **38,612 training images**, built
with `openai/clip-vit-base-patch32`.

## Quick start

Run commands from this `task4` directory so that the relative paths in the JSON
configuration files resolve correctly.

```bash
cd task4
uv sync

# Use the existing index and run the configured default (ANN)
uv run python search.py \
  --query ../data/FashionDataset/train/images_train/1525.jpg \
  --k 5
```

To compare the two search modes:

```bash
uv run python search.py --query ../data/FashionDataset/train/images_train/1525.jpg --mode knn
uv run python search.py --query ../data/FashionDataset/train/images_train/1525.jpg --mode ann
```

To save a result montage, both `--save` and `--image-dir` are required:

```bash
uv run python search.py \
  --query ../data/FashionDataset/train/images_train/1525.jpg \
  --image-dir ../data/FashionDataset/train/images_train \
  --save results/search/example.png
```

Rebuild the embedding files after changing the dataset or CLIP model:

```bash
uv run python build_index.py --config config/build_config.json
```

The first model load may download CLIP weights from Hugging Face. See the
[installation guide](docs/installation.md) for prerequisites, dataset layout,
offline use, and troubleshooting.

## How it works

There are two distinct phases:

1. `build_index.py` scans the training images, generates normalized CLIP image
   embeddings in batches, and persists the embedding matrix and matching IDs.
2. `search.py` loads those files, builds the selected FAISS index in memory,
   encodes one query image with the same CLIP model, and returns the Top-K IDs
   ranked by cosine similarity.

```text
Offline: images -> CLIP -> normalized vectors -> embeddings.npy + image_ids.json
Online:  query  -> CLIP -> normalized vector  -> FAISS KNN/HNSW -> Top-K results
```

The numerical score is an inner product. Because both stored and query vectors
are L2-normalized, it is also cosine similarity: larger values indicate greater
similarity.

## Project structure

```text
task4/
├── README.md
├── build_index.py                 # Offline embedding pipeline
├── search.py                      # Command-line query pipeline
├── pyproject.toml                 # Python metadata and dependencies
├── uv.lock                        # Reproducible dependency lockfile
├── config/
│   ├── build_config.json
│   └── search_config.json
├── docs/
│   ├── installation.md
│   ├── architecture.md
│   └── search-algorithms.md
├── results/index/
│   ├── embeddings.npy             # N x 512 float32 vectors
│   ├── image_ids.json             # Row-to-image-ID mapping
│   ├── build_meta.json
│   └── build_log.txt
└── src/
    ├── feature_extractor.py        # CLIP preprocessing and inference
    ├── embedding_store.py          # Persistence and FAISS indexes
    └── utils.py                    # Config, seeding, logging, plots
```

## Configuration

- `config/build_config.json` controls the model, dataset location, batch size,
  and output directory used by `build_index.py`.
- `config/search_config.json` controls the model, index location, default Top-K,
  default mode, and HNSW parameters used by `search.py`.
- `--query`, `--k`, `--mode`, `--save`, and `--image-dir` are search-time CLI
  options. CLI values for `--k` and `--mode` override their configured defaults.

`MODEL_NAME` must be identical for index building and querying. Embeddings from
different model checkpoints do not share a meaningful vector space.

## Documentation

- [Installation and operation](docs/installation.md)
- [Developer usage guide](docs/usage.md)
- [System architecture](docs/architecture.md)
- [KNN vs ANN/HNSW in detail](docs/search-algorithms.md)
