import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class SettingConfig:
    """Base class that injects all JSON config keys as instance attributes."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.pin_memory = torch.cuda.is_available()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_run_dir(base_dir: str, phase: str) -> Path:
    """Create a timestamped run directory: base_dir/phase/yyyymmdd_hhmmss/"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / phase / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logger(name: str, log_file: Path) -> logging.Logger:
    """Setup logger that writes to console + a specific log file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    console = logging.StreamHandler()
    console.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    )

    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def plot_search_results(
    query_image: np.ndarray,
    result_images: list[np.ndarray],
    result_scores: list[float],
    result_ids: list[str],
    save_path: Path | None = None,
) -> None:
    """Plot query image alongside top-K search results with similarity scores."""
    n_results = len(result_images)
    fig, axes = plt.subplots(1, n_results + 1, figsize=(3 * (n_results + 1), 4))

    # Query image
    axes[0].imshow(query_image)
    axes[0].set_title("Query", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Result images
    for i, (img, score, img_id) in enumerate(zip(result_images, result_scores, result_ids)):
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(f"#{i+1} | {score:.4f}\n{img_id}", fontsize=9)
        axes[i + 1].axis("off")

    plt.suptitle("Visual Search Results (cosine similarity)", fontsize=13)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
