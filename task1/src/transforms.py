"""Image transforms using OpenCV.

All transforms are configurable via the TRANSFORM_PARAMS JSON config key.
The geometry pipeline (pad → resize) is deterministic and shared across
train/val/test. Augmentations run only during training, after geometry.

Default image size: 96×128 (width×height), preserving the original 3:4
aspect ratio of the FashionDataset images (60×80).
See eda/analysis/task1_analysis.md for the rationale.
"""

import cv2
import numpy as np
import torch


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: np.ndarray) -> torch.Tensor | np.ndarray:
        for t in self.transforms:
            image = t(image)
        return image


class PadToAspectRatio:
    """Pad image to a target aspect ratio (width:height) with a neutral fill.

    Places the original image centered on a canvas that matches the target
    ratio, filling the remaining area with pad_value. This ensures no
    stretching or cropping — the product stays geometrically intact.

    Config:
        target_w (int): Target width ratio component. Default: 3
        target_h (int): Target height ratio component. Default: 4
        pad_value (int): Fill colour (0–255, applied to all channels). Default: 0
    """

    def __init__(self, target_w: int = 3, target_h: int = 4, pad_value: int = 0):
        self.target_ratio = target_w / target_h
        self.pad_value = pad_value

    def __call__(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        current_ratio = w / h

        if abs(current_ratio - self.target_ratio) < 1e-3:
            return image

        if current_ratio > self.target_ratio:
            # Image is too wide → pad height
            new_h = int(w / self.target_ratio)
            pad_top = (new_h - h) // 2
            pad_bottom = new_h - h - pad_top
            image = cv2.copyMakeBorder(
                image, pad_top, pad_bottom, 0, 0,
                cv2.BORDER_CONSTANT, value=(self.pad_value,) * 3,
            )
        else:
            # Image is too tall → pad width
            new_w = int(h * self.target_ratio)
            pad_left = (new_w - w) // 2
            pad_right = new_w - w - pad_left
            image = cv2.copyMakeBorder(
                image, 0, 0, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(self.pad_value,) * 3,
            )
        return image


class Resize:
    """Resize image to (width, height).

    Config:
        width (int): Target width in pixels. Default: 96
        height (int): Target height in pixels. Default: 128
    """

    def __init__(self, width: int = 96, height: int = 128):
        self.size = (width, height)  # cv2.resize takes (w, h)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)


class RandomHorizontalFlip:
    """Randomly flip image horizontally.

    Config:
        p (float): Probability of flip. Default: 0.5
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return cv2.flip(image, 1)
        return image


class ColorJitter:
    """Randomly adjust brightness, contrast, and saturation.

    Config:
        brightness (float): Max brightness shift factor. Default: 0.2
        contrast (float): Max contrast shift factor. Default: 0.2
        saturation (float): Max saturation shift factor. Default: 0.2
    """

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, saturation: float = 0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32)
        factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
        image = np.clip(image * factor, 0, 255)
        factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
        mean = image.mean()
        image = np.clip((image - mean) * factor + mean, 0, 255)
        factor = 1.0 + np.random.uniform(-self.saturation, self.saturation)
        hsv = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
        return image.astype(np.uint8)


class Normalize:
    """Normalize to [0,1] then apply mean/std normalization.

    Config:
        mean (list[float]): Per-channel mean. Default: [0.485, 0.456, 0.406] (ImageNet)
        std (list[float]): Per-channel std. Default: [0.229, 0.224, 0.225] (ImageNet)
    """

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255.0
        return (image - self.mean) / self.std


class ToTensor:
    """Convert HWC numpy array to CHW torch tensor."""

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image.transpose(2, 0, 1)).float()


def get_train_transforms(transform_params: dict | None = None) -> Compose:
    """Build training transform pipeline from config.

    Pipeline: PadToAspectRatio → Resize → [optional augmentations] → Normalize → ToTensor

    Augmentations are only applied when their key is present in TRANSFORM_PARAMS:
        flip_p (float): Include RandomHorizontalFlip with this probability.
        brightness/contrast/saturation (float): Include ColorJitter when any of these keys exist.

    Always-applied config keys:
        width (int): Target width. Default: 96
        height (int): Target height. Default: 128
        pad_value (int): Padding fill value. Default: 0
        mean (list[float]): Normalize mean. Default: ImageNet
        std (list[float]): Normalize std. Default: ImageNet
    """
    p = transform_params or {}
    width = p.get("width", 96)
    height = p.get("height", 128)

    steps = [
        # Geometry (deterministic, shared with val/test)
        PadToAspectRatio(target_w=width, target_h=height, pad_value=p.get("pad_value", 0)),
        Resize(width=width, height=height),
    ]

    # Augmentation — only added when the key exists in config
    if "flip_p" in p:
        steps.append(RandomHorizontalFlip(p=p["flip_p"]))

    if any(k in p for k in ("brightness", "contrast", "saturation")):
        steps.append(ColorJitter(
            brightness=p.get("brightness", 0.0),
            contrast=p.get("contrast", 0.0),
            saturation=p.get("saturation", 0.0),
        ))

    # Normalization
    steps.append(Normalize(mean=p.get("mean", (0.485, 0.456, 0.406)), std=p.get("std", (0.229, 0.224, 0.225))))
    steps.append(ToTensor())

    return Compose(steps)


def get_val_transforms(transform_params: dict | None = None) -> Compose:
    """Build val/test transform pipeline from config.

    Pipeline: PadToAspectRatio → Resize → Normalize → ToTensor
    No augmentation — deterministic only.
    """
    p = transform_params or {}
    width = p.get("width", 96)
    height = p.get("height", 128)

    return Compose([
        PadToAspectRatio(target_w=width, target_h=height, pad_value=p.get("pad_value", 0)),
        Resize(width=width, height=height),
        Normalize(mean=p.get("mean", (0.485, 0.456, 0.406)), std=p.get("std", (0.229, 0.224, 0.225))),
        ToTensor(),
    ])
