# Installation

## Requirements

- Python 3.11 or newer
- `uv` (recommended) or `pip`
- The FashionDataset in the expected directory layout
- Optional: an NVIDIA CUDA GPU or an Apple Silicon Mac with MPS

The pipeline also runs on CPU. At runtime it selects CUDA first, then Apple MPS, then CPU.

## Install with uv

From the repository root:

```bash
cd task3
uv sync --locked
```

This creates `task3/.venv` and installs the exact dependency resolution in `uv.lock`. Verify the environment:

```bash
uv run python --version
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If `uv` is unavailable, install it using the official instructions for your operating system and rerun the command.

## Install with Python and pip

`uv` is preferred because the repository includes its lock file. To use a conventional virtual environment instead:

```bash
cd task3
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

On Windows PowerShell, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

When using the activated pip environment, replace commands such as `uv run python train.py` with `python train.py`.

## Dataset setup

The default configuration paths are resolved from the current working directory. Run the entry points from `task3`, or update all paths in `config/*.json`.

```text
COSC2753-Project/
├── data/
│   └── FashionDataset/
│       ├── train/
│       │   ├── images_train/
│       │   │   ├── <id>.jpg
│       │   │   └── ...
│       │   └── styles_train.csv
│       └── test/
│           ├── images_test/
│           │   ├── <id>.jpg
│           │   └── ...
│           └── styles_prediction.csv
└── task3/
```

Training requires these CSV columns:

| Column | Role |
| --- | --- |
| `id` | Matches the image filename `<id>.jpg`. |
| `gender` | First prediction target. |
| `usage` | Second prediction target (occasion). |
| `articleType` | Metadata input. |
| `masterCategory` | Metadata input. |
| `baseColour` | Metadata input. |

Rows without a matching image, `gender`, or `usage` are removed. Occasion classes below `MIN_OCCASION_COUNT` are also removed. Metadata values may be missing; index `0` is reserved for missing or unseen values.

For prediction, only `.jpg` files in `TEST_IMAGE_DIR` are mandatory. `TEST_CSV_PATH` is optional at the code level. When it exists, matching rows contribute any available metadata columns; missing columns, values, rows, or the entire CSV fall back to unknown metadata.

## Installation troubleshooting

### Commands cannot find configuration or data

Run commands from `task3`. The entry points open paths such as `config/train_config.json` relative to the current working directory.

### Data-loader workers fail

Set `NUM_WORKERS` to `0` in the relevant configuration. This is useful on constrained environments and when debugging multiprocessing failures.

### CUDA or MPS runs out of memory

Lower `BATCH_SIZE`, reduce the input dimensions, or use the `simple` backbone. Custom ResNet-101 and ResNet-152 are substantially larger than the simple CNN.

### OpenCV display-library errors on a server

The project depends on `opencv-python-headless`, so it does not require desktop GUI libraries. Ensure a conflicting full `opencv-python` package has not been installed manually into the same environment.

Continue with the [usage guide](usage.md) after installation.
