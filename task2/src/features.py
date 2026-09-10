"""HOG and HSV feature extraction with validated on-disk caching."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import hog

from .transforms import PadAndResize


class FashionFeatureExtractor:
    """Extract shape and colour information without pretrained weights."""

    def __init__(self, **feature_params):
        self.params = feature_params
        self.transform = PadAndResize(
            width=feature_params["width"],
            height=feature_params["height"],
            pad_value=feature_params.get("pad_value", 255),
        )
        self.hog_orientations = int(feature_params.get("hog_orientations", 9))
        self.hog_pixels_per_cell = tuple(feature_params.get("hog_pixels_per_cell", (8, 8)))
        self.hog_cells_per_block = tuple(feature_params.get("hog_cells_per_block", (2, 2)))
        self.hog_block_norm = feature_params.get("hog_block_norm", "L2-Hys")
        self.hsv_bins = int(feature_params.get("hsv_bins", 16))

    def transform_path(self, image_path: str | Path) -> np.ndarray:
        rgb_image = self.transform(image_path)
        grayscale = rgb2gray(rgb_image)
        hog_features = hog(
            grayscale,
            orientations=self.hog_orientations,
            pixels_per_cell=self.hog_pixels_per_cell,
            cells_per_block=self.hog_cells_per_block,
            block_norm=self.hog_block_norm,
            feature_vector=True,
        )

        hsv_image = rgb2hsv(rgb_image)
        pixels_per_channel = hsv_image.shape[0] * hsv_image.shape[1]
        colour_histograms = [
            np.histogram(hsv_image[..., channel], bins=self.hsv_bins, range=(0.0, 1.0))[0]
            .astype(np.float32)
            / pixels_per_channel
            for channel in range(3)
        ]
        return np.concatenate([hog_features, *colour_histograms]).astype(np.float32)


def _feature_fingerprint(metadata: pd.DataFrame, feature_params: dict) -> str:
    """Fingerprint ordered IDs, image state, and all feature settings."""
    digest = hashlib.sha256()
    digest.update(json.dumps(feature_params, sort_keys=True).encode("utf-8"))
    for image_id, image_path in zip(metadata["id"], metadata["image_path"]):
        path = Path(image_path)
        stat = path.stat()
        digest.update(f"{image_id}|{path.name}|{stat.st_size}|{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def load_or_extract_features(
    metadata: pd.DataFrame,
    feature_params: dict,
    cache_path: str | Path,
    force_rebuild: bool = False,
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, str]:
    """Load a matching cache or extract features in metadata row order."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    expected_ids = metadata["id"].astype(str).to_numpy(dtype=str)
    fingerprint = _feature_fingerprint(metadata, feature_params)

    if cache_path.is_file() and not force_rebuild:
        with np.load(cache_path, allow_pickle=False) as cache:
            cached_fingerprint = str(cache["fingerprint"].item())
            cached_ids = cache["ids"].astype(str)
            if cached_fingerprint == fingerprint and np.array_equal(cached_ids, expected_ids):
                features = cache["features"].astype(np.float32, copy=False)
                if logger:
                    logger.info("Loaded feature cache %s with shape %s", cache_path, features.shape)
                return features, fingerprint
        if logger:
            logger.info("Feature cache does not match the current data/configuration; rebuilding it")

    extractor = FashionFeatureExtractor(**feature_params)
    paths = metadata["image_path"].tolist()
    if not paths:
        raise ValueError("No images are available for feature extraction.")

    first_features = extractor.transform_path(paths[0])
    features = np.empty((len(paths), len(first_features)), dtype=np.float32)
    features[0] = first_features
    for index, image_path in enumerate(paths[1:], start=1):
        features[index] = extractor.transform_path(image_path)
        if logger and (index + 1) % 5000 == 0:
            logger.info("Extracted features for %d/%d images", index + 1, len(paths))

    temporary_path = cache_path.with_name(f"{cache_path.stem}.tmp.npz")
    np.savez_compressed(
        temporary_path,
        features=features,
        ids=expected_ids,
        fingerprint=np.asarray(fingerprint),
        feature_params=np.asarray(json.dumps(feature_params, sort_keys=True)),
    )
    temporary_path.replace(cache_path)
    if logger:
        logger.info("Saved feature cache %s with shape %s", cache_path, features.shape)
    return features, fingerprint
