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


class Resize:
    def __init__(self, size: int):
        self.size = (size, size)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return cv2.flip(image, 1)
        return image


class RandomRotation:
    def __init__(self, max_angle: float = 15.0):
        self.max_angle = max_angle

    def __call__(self, image: np.ndarray) -> np.ndarray:
        angle = np.random.uniform(-self.max_angle, self.max_angle)
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)


class ColorJitter:
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
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255.0
        return (image - self.mean) / self.std


class ToTensor:
    def __call__(self, image: np.ndarray) -> torch.Tensor:
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image.transpose(2, 0, 1)).float()


def get_train_transforms(image_size: int = 224) -> Compose:
    return Compose([
        Resize(image_size),
        RandomHorizontalFlip(),
        RandomRotation(15),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        Normalize(),
        ToTensor(),
    ])


def get_val_transforms(image_size: int = 224) -> Compose:
    return Compose([
        Resize(image_size),
        Normalize(),
        ToTensor(),
    ])
