# Usage

Task 4 performs image-to-image fashion retrieval in two stages: build reusable CLIP embeddings for the catalogue, then search them with exact KNN or HNSW approximate nearest neighbours.

## 1. Prepare the environment

Run commands from `task4` so relative configuration paths resolve correctly:

```bash
cd task4
uv sync --locked
```

The first use of `openai/clip-vit-base-patch32` may download model files from Hugging Face. The default catalogue is `../data/FashionDataset/train/images_train`.

## 2. Configure and build the embedding corpus

Edit `config/build_config.json`:

| Key | Purpose |
| --- | --- |
| `MODEL_NAME` | Hugging Face CLIP checkpoint used to encode catalogue images. |
| `DATA_DIR` | Dataset split containing the image subdirectory. |
| `IMAGE_SUBDIR` | Image folder inside `DATA_DIR`. |
| `BATCH_SIZE` | Number of images encoded in each CLIP batch. |
| `INDEX_DIR` | Destination for embeddings, IDs, metadata, and logs. |
| `EMBEDDING_DIM` | Informational in the current code; the model determines the actual dimension. |
| `NUM_WORKERS`, `CSV_FILE` | Present for configuration compatibility but not currently consumed. |

Build with the default config:

```bash
uv run python build_index.py
```

Or select another build config:

```bash
uv run python build_index.py --config config/build_config.json
```

The build scans `.jpg`, `.jpeg`, and `.png` files, skips unreadable images, extracts normalized embeddings, and writes:

```text
results/index/
├── embeddings.npy
├── image_ids.json
├── build_meta.json
└── build_log.txt
```

`embeddings.npy` row `i` must remain paired with `image_ids.json` entry `i`. Keep or replace these files together.

## 3. Configure search

Edit `config/search_config.json`:

| Key | Purpose |
| --- | --- |
| `MODEL_NAME` | Must exactly match the checkpoint used during the build. |
| `INDEX_DIR` | Folder containing `embeddings.npy` and `image_ids.json`. |
| `TOP_K` | Default number of neighbours. |
| `SEARCH_MODE` | `knn` for exact `IndexFlatIP` or `ann` for HNSW. |
| `HNSW_M` | HNSW graph connectivity; higher values generally improve recall at added build/memory cost. |
| `HNSW_EF_SEARCH` | HNSW query exploration budget; higher values generally improve recall at added latency. |

The command line overrides `TOP_K` and `SEARCH_MODE` but always reads the other settings from `config/search_config.json`.

## 4. Search

Exact Top-K search:

```bash
uv run python search.py \
  --query ../data/FashionDataset/train/images_train/1525.jpg \
  --k 5 \
  --mode knn
```

HNSW approximate search:

```bash
uv run python search.py \
  --query ../data/FashionDataset/train/images_train/1525.jpg \
  --k 5 \
  --mode ann
```

The terminal prints ranked image IDs, cosine-similarity scores, FAISS lookup time, and total command time. Because the CLI loads CLIP and rebuilds its in-memory FAISS index for every invocation, `Total time` is not the same as steady-state lookup latency.

Save a visual result grid by supplying both the catalogue image directory and output path:

```bash
uv run python search.py \
  --query ../data/FashionDataset/train/images_train/1525.jpg \
  --mode ann \
  --image-dir ../data/FashionDataset/train/images_train \
  --save results/search/1525_ann.png
```

If either `--save` or `--image-dir` is omitted, results are printed but no montage is created.

## Choosing a search mode

- Use `knn` for guaranteed exact neighbours, a correctness baseline, or a modest query rate.
- Use `ann` when a long-lived service needs lower lookup latency and a measured small recall trade-off is acceptable.
- Keep `HNSW_EF_SEARCH >= TOP_K`; benchmark recall against KNN after tuning `M` or `efSearch`.

## Troubleshooting

- If index files are missing, run `build_index.py` first.
- If the model cannot load offline, populate the Hugging Face cache before disconnecting.
- Lower build `BATCH_SIZE` if CLIP inference runs out of memory.
- Rebuild embeddings whenever `MODEL_NAME` changes; vectors from different CLIP checkpoints are not comparable.
