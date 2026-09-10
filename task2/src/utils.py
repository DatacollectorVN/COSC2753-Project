"""Shared configuration, logging, serialization, and run-directory helpers."""

import json
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np


class SettingConfig:
    def __init__(self, **parameters):
        self._raw_params = parameters
        for key, value in parameters.items():
            setattr(self, key, value)


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        return super().default(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def create_run_dir(base_dir: str | Path, phase: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / phase / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logger(name: str, log_file: str | Path) -> logging.Logger:
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def save_json(payload: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, cls=NumpyJSONEncoder)
