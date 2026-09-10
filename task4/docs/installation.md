# Installation and operation

## Requirements

- Python 3.11 or newer (the checked-in environment currently uses Python 3.12)
- Enough disk space for the Python environment, CLIP model cache, dataset, and
  index artifacts
- The FashionDataset in the layout shown below
- Internet access for the first Hugging Face model download, unless the model is
  already present in the local cache

A GPU is optional. PyTorch automatically selects CUDA, Apple Metal (MPS), or
CPU for CLIP inference, in that order. The declared dependency is `faiss-cpu`;
the project therefore does not require a FAISS GPU installation.

## Recommended installation with uv

From the repository root:

```bash
cd task4
uv sync
```

`uv sync` uses `pyproject.toml` and `uv.lock` to create/update `.venv` with the
locked dependency set. Run the application through uv:

```bash
uv run python search.py --help
```

If uv is not installed, follow the installation instructions at
<https://docs.astral.sh/uv/getting-started/installation/> and repeat the command.

## Alternative installation with pip

The project does not include a `requirements.txt`. Pip can install the package
dependencies declared in `pyproject.toml`:

```bash
cd task4
python3 -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install .
```

This resolves versions from the ranges in `pyproject.toml`; unlike `uv sync`, it
does not necessarily reproduce the exact versions in `uv.lock`.

## Dataset layout

The default build configuration is resolved relative to `task4`, and expects:

```text
COSC2753-Project/
├── data/FashionDataset/
│   └── train/
│       ├── images_train/
│       │   ├── 1525.jpg
│       │   └── ...
│       └── styles_train.csv
└── task4/
```

Only `.jpg`, `.jpeg`, and `.png` files immediately inside `images_train/` are
indexed. The builder sorts paths for a stable row order and uses each filename
stem (for example, `1525`) as its image ID. `CSV_FILE` and `NUM_WORKERS` exist in
the current build configuration but are not read by `build_index.py`.

For another dataset, edit `DATA_DIR` and `IMAGE_SUBDIR` in
`config/build_config.json`. Keep paths relative to `task4`, or use absolute
paths.

## Build or rebuild the embeddings

```bash
cd task4
uv run python build_index.py
```

Or provide another configuration:

```bash
uv run python build_index.py --config config/build_config.json
```

The builder:

1. loads supported images and converts them to RGB;
2. skips unreadable images with a warning;
3. embeds valid images in batches with CLIP;
4. writes `embeddings.npy`, `image_ids.json`, `build_meta.json`, and
   `build_log.txt` to `INDEX_DIR`.

With the current data, `build_meta.json` records 38,612 vectors of dimension 512
and a build time of 206.7 seconds on the machine used to create the index. That
is an observed result, not a performance guarantee.

Rebuild whenever the source images, `MODEL_NAME`, or embedding implementation
changes. Changing only KNN/ANN search settings does not require regenerating the
embeddings.

## Search

Basic search using defaults from `config/search_config.json`:

```bash
uv run python search.py --query path/to/query.jpg
```

Useful options:

```text
--query PATH       required query image
--k INTEGER        result count; default comes from TOP_K
--mode knn|ann     exact KNN or HNSW ANN
--image-dir PATH   dataset image folder used to render matched IDs
--save PATH        write a result montage; requires --image-dir to have effect
```

Example:

```bash
uv run python search.py \
  --query ../data/FashionDataset/test/images_test/52112.jpg \
  --mode ann \
  --k 10 \
  --image-dir ../data/FashionDataset/train/images_train \
  --save results/search/52112.png
```

`Search time` measures only the FAISS lookup. `Total time` also includes loading
the persisted embeddings, constructing the in-memory FAISS index, loading CLIP,
and computing the query embedding. It does not include result plotting, which
happens after the timing is printed.

## Model caching and offline use

`CLIPProcessor.from_pretrained()` and `CLIPModel.from_pretrained()` use the
Hugging Face cache. On the first run they download
`openai/clip-vit-base-patch32`; later runs can reuse it. For an offline machine,
populate that cache beforehand or change `MODEL_NAME` to a local model directory
in both configuration files.

## Troubleshooting

**`FileNotFoundError` for config or index files**

Run the commands from `task4`. `search.py` always opens
`config/search_config.json` relative to the current working directory.

**Dataset directory does not exist**

Check `DATA_DIR` plus `IMAGE_SUBDIR` in `config/build_config.json`. With the
defaults, their combined path is
`../data/FashionDataset/train/images_train` from `task4`.

**Model download or connection failure**

Verify network access or pre-populate the Hugging Face cache. Both build and
search load the CLIP checkpoint.

**Out of memory while building**

Reduce `BATCH_SIZE`. This lowers peak accelerator/RAM use at the cost of more
batches and usually lower throughput.

**No saved visualization**

Pass both `--save` and `--image-dir`. The search still prints IDs and scores when
only one is supplied, but the current code does not create a montage.

**OpenMP or duplicate-library errors on macOS**

The entry points set `KMP_DUPLICATE_LIB_OK=TRUE`, and HNSW construction is forced
to one FAISS OpenMP thread to avoid a known deadlock in this environment. Treat
the duplicate-library setting as a compatibility workaround; prefer compatible
PyTorch/FAISS builds in production.

