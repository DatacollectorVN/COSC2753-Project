# System architecture

## Purpose and boundaries

Task 4 is a command-line, image-to-image retrieval system. Given one fashion
image, it returns the IDs and cosine-similarity scores of the Top-K closest
training images. It does not train CLIP, use the fashion metadata CSV during
retrieval, expose a server API, or persist a ready-built FAISS index.

## High-level data flow

```text
                         OFFLINE BUILD

images_train/*.{jpg,jpeg,png}
           │
           ▼
      PIL RGB images
           │  batches of BATCH_SIZE
           ▼
 CLIPProcessor + CLIPModel.get_image_features
           │
           ▼
     L2 normalization
           │
           ├───────────────┐
           ▼               ▼
 embeddings.npy      image_ids.json
   [N, 512]          row i -> ID i


                         ONLINE QUERY

 embeddings.npy + image_ids.json       query image
              │                            │
              ▼                            ▼
     in-memory EmbeddingStore       same CLIP pipeline
              │                            │
              ▼                            ▼
 IndexFlatIP or IndexHNSWFlat ◄── normalized query [512]
              │
              ▼
       Top-K row indices + scores
              │  map rows through image_ids.json
              ▼
       [(image_id, cosine score), ...]
              │
              └── optional Matplotlib result image
```

The same model checkpoint and normalization procedure are required on both
sides. Otherwise, query and catalog vectors are not comparable.

## Components and responsibilities

### `build_index.py`: offline orchestration

The builder reads `config/build_config.json` unless `--config` specifies another
file. It creates `INDEX_DIR`, collects supported image paths from one directory,
and sorts them for deterministic ordering. Images are decoded one batch at a
time; invalid files are logged and omitted without aborting the build.

The valid images and matching filename stems are kept aligned within each batch.
`CLIPFeatureExtractor.extract_batch()` produces an `N x D` NumPy array, which is
appended to `EmbeddingStore`. At completion, the raw vectors and ordered IDs are
saved, followed by model/build metadata.

The builder does **not** build or serialize either FAISS index. In this project,
the word "index" in `results/index` primarily refers to the searchable embedding
corpus and its ID mapping.

### `src/feature_extractor.py`: representation layer

`CLIPFeatureExtractor` wraps Hugging Face's CLIP processor and model:

- device selection is CUDA, then Apple MPS, then CPU;
- the processor performs CLIP-specific resize/crop/tensor normalization;
- `get_image_features()` applies the vision encoder and CLIP projection;
- each output is divided by its L2 norm;
- the result is copied to CPU as a NumPy array.

For the configured ViT-B/32 checkpoint, the projection dimension is 512. The
implementation reads this dimension from the model rather than trusting the
unused `EMBEDDING_DIM` configuration key.

### `src/embedding_store.py`: storage and retrieval layer

The store owns three related data structures:

```text
embeddings: N x D float32 matrix
image_ids:  length-N ordered list
FAISS index: transient index whose row numbers address both structures
```

This positional invariant is essential: row `i` of the matrix must describe
`image_ids[i]`. `save()` writes the first two structures; `load()` restores them,
infers `D` from the matrix, and clears any old in-memory FAISS indexes.

`build_knn_index()` creates `IndexFlatIP`, optionally moving it to a FAISS GPU if
one is available. `build_ann_index()` creates CPU `IndexHNSWFlat` with inner
product and sets `efSearch`. `search()` dispatches to one of those indexes and
maps valid result indices back to IDs.

### `search.py`: online orchestration

The search command reads `config/search_config.json`, then lets `--k` and
`--mode` override the corresponding defaults. Its lifecycle is:

1. load the embedding matrix and ID list;
2. construct the selected FAISS index in memory;
3. instantiate the same CLIP feature extractor;
4. decode and embed the query;
5. ask FAISS for K results and print mapped IDs/scores;
6. optionally find the result image files and render a montage.

Because this is a one-query CLI, both the CLIP model and FAISS structure are
recreated on every invocation. A service handling many requests should load the
model and data once at startup and reuse them across queries.

### `src/utils.py`: cross-cutting utilities

`SettingConfig` exposes JSON keys as attributes and also records an available
PyTorch device. Other helpers seed random number generators, configure console
plus file logging, create run directories, and save result grids with a
non-interactive Matplotlib backend.

## Persistence contract

| Artifact | Producer | Consumer | Meaning |
|---|---|---|---|
| `embeddings.npy` | `EmbeddingStore.save` | `EmbeddingStore.load` | Contiguous `float32` matrix with one normalized vector per valid image |
| `image_ids.json` | `EmbeddingStore.save` | `EmbeddingStore.load` | IDs in exactly the same row order as the matrix |
| `build_meta.json` | `build_index.py` | Human/operator | Model name, dimension, image count, elapsed build time |
| `build_log.txt` | logger | Human/operator | Build progress, warnings, and throughput |

The persisted files have no schema version or checksum. Keep them together as
one atomic logical unit. Mixing an embedding file and ID list from different
builds can silently return incorrect product IDs.

## Similarity contract

For database vector `x` and query vector `q`, CLIP extraction enforces
`||x||₂ = ||q||₂ = 1` (up to floating-point precision). Therefore:

```text
FAISS inner product = x · q = (x · q) / (||x||₂ ||q||₂) = cosine(x, q)
```

Both FAISS modes consequently use `METRIC_INNER_PRODUCT`, and larger scores are
better. A score reflects proximity in CLIP's learned semantic space; it is not a
calibrated probability or a guarantee that two items share a category.

## Configuration and precedence

Build configuration affects persisted data; search configuration affects query
execution. The most important invariant is matching `MODEL_NAME` values.

| Setting | Used by current code | Architectural effect |
|---|---:|---|
| `MODEL_NAME` | Yes | Defines the embedding space |
| `BATCH_SIZE` | Yes | Build throughput and peak memory |
| `INDEX_DIR` | Yes | Persistence location |
| `DATA_DIR`, `IMAGE_SUBDIR` | Yes | Source image directory |
| `EMBEDDING_DIM` | No | Dimension is obtained from CLIP |
| `NUM_WORKERS`, `CSV_FILE` | No | Reserved/legacy configuration |
| `TOP_K`, `SEARCH_MODE` | Yes | CLI defaults |
| `HNSW_M`, `HNSW_EF_SEARCH` | Yes in ANN | HNSW graph quality/cost trade-off |

## Scaling and production considerations

- **Process lifetime:** reuse loaded model and indexes for repeated queries.
- **Index persistence:** serialize the FAISS structure if HNSW construction time
  becomes material; the current program persists only raw vectors.
- **Updates:** the current workflow rebuilds the vector files as a batch. There
  is no deletion, transactional update, or concurrent writer support.
- **Validation:** verify equal vector/ID counts, expected dimension, finite
  values, and model identity when loading artifacts.
- **Input safety:** validate query decode failures and constrain image sizes in a
  public-facing service.
- **Quality evaluation:** measure Recall@K against exact KNN and, for product
  relevance, evaluate labelled retrieval metrics such as Precision@K or mAP.
- **Metadata filtering:** category, availability, or price filtering is not part
  of the present retrieval path and would require an additional mapping/filter
  layer.

See [KNN vs ANN/HNSW](search-algorithms.md) for the retrieval algorithms and
their tuning trade-offs.

