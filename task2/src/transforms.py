"""Deterministic image geometry used by feature extraction."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


class PadAndResize:
    """Preserve aspect ratio, pad to the target ratio, and resize."""

    def __init__(self, width: int, height: int, pad_value: int = 255):
        if width <= 0 or height <= 0:
            raise ValueError("Image width and height must be positive.")
        if not 0 <= pad_value <= 255:
            raise ValueError("pad_value must be between 0 and 255.")
        self.size = (int(width), int(height))
        self.pad_colour = (int(pad_value),) * 3

    def __call__(self, image_path: str | Path) -> np.ndarray:
        image_path = Path(image_path)
        try:
            with Image.open(image_path) as image:
                rgb_image = image.convert("RGB")
                transformed = ImageOps.pad(
                    rgb_image,
                    self.size,
                    method=Image.Resampling.BILINEAR,
                    color=self.pad_colour,
                    centering=(0.5, 0.5),
                )
                return np.asarray(transformed, dtype=np.float32) / 255.0
        except Exception as error:
            raise ValueError(f"Could not preprocess image {image_path}: {error}") from error
