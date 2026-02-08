"""
Validation preprocessors for different model architectures.

Each model type may have different preprocessing requirements:
- Image normalization (0-1 vs 0-255)
- Resize mode (simple resize vs letterbox)
- Padding values
"""

from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import numpy as np


class BaseValPreprocessor(ABC):
    """
    Abstract base class for validation preprocessors.

    Subclasses implement model-specific preprocessing logic.
    """

    def __init__(self, img_size: Tuple[int, int], max_labels: int = 120):
        """
        Initialize preprocessor.

        Args:
            img_size: Target image size (height, width).
            max_labels: Maximum number of labels per image for padding.
        """
        self.img_size = img_size
        self.max_labels = max_labels

    @abstractmethod
    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess image and targets.

        Args:
            img: Input image (H, W, C) in BGR format.
            targets: Target annotations (N, 5) with [x1, y1, x2, y2, class].
            input_size: Target input size (height, width).

        Returns:
            Tuple of (preprocessed_image, padded_targets).
        """
        pass

    @property
    @abstractmethod
    def normalize(self) -> bool:
        """Whether this preprocessor normalizes images to 0-1 range."""
        pass

    @property
    def custom_normalization(self) -> bool:
        """Whether this preprocessor applies its own normalization (e.g. ImageNet mean/std).
        When True, the validator should not rescale the images at all."""
        return False

    @property
    def uses_letterbox(self) -> bool:
        """Whether this preprocessor uses letterbox (aspect-preserving) resize."""
        return False

    def _pad_targets(self, targets: np.ndarray, n_valid: int) -> np.ndarray:
        """Pad targets to fixed size for batching."""
        padded = np.zeros((self.max_labels, 5), dtype=np.float32)
        if n_valid > 0:
            padded[:n_valid] = targets[:n_valid]
        return padded


class StandardValPreprocessor(BaseValPreprocessor):
    """
    Standard validation preprocessor for YOLOv9 models.

    Uses simple resize and normalizes to 0-1 range.
    """

    @property
    def normalize(self) -> bool:
        return True

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size

        # Simple resize
        resized_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Convert to CHW and float
        resized_img = resized_img.transpose(2, 0, 1)
        resized_img = np.ascontiguousarray(resized_img, dtype=np.float32)

        # Process targets
        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets) > 0:
            targets = np.array(targets).copy()
            n = min(len(targets), self.max_labels)

            # Undo letterbox scaling and apply simple resize scaling
            letterbox_r = min(target_h / orig_h, target_w / orig_w)
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h

            targets[:n, 0] = targets[:n, 0] / letterbox_r * scale_x
            targets[:n, 1] = targets[:n, 1] / letterbox_r * scale_y
            targets[:n, 2] = targets[:n, 2] / letterbox_r * scale_x
            targets[:n, 3] = targets[:n, 3] / letterbox_r * scale_y

            padded_targets[:n] = targets[:n]

        return resized_img, padded_targets


class YOLOXValPreprocessor(BaseValPreprocessor):
    """
    YOLOX validation preprocessor.

    Uses letterbox resize with gray padding (114, 114, 114).
    Does NOT normalize - keeps 0-255 range.
    """

    def __init__(self, img_size: Tuple[int, int], max_labels: int = 120, pad_value: int = 114):
        super().__init__(img_size, max_labels)
        self.pad_value = pad_value

    @property
    def normalize(self) -> bool:
        return False

    @property
    def uses_letterbox(self) -> bool:
        return True

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size

        # Letterbox resize maintaining aspect ratio
        ratio = min(target_h / orig_h, target_w / orig_w)
        new_h = int(orig_h * ratio)
        new_w = int(orig_w * ratio)

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image with gray background
        padded_img = np.full((target_h, target_w, 3), self.pad_value, dtype=np.uint8)
        padded_img[:new_h, :new_w] = resized_img

        # Keep BGR format (YOLOX is trained on BGR from cv2, do NOT convert to RGB)
        # The training uses BGR images directly from cv2.imread without conversion

        # Convert to CHW and float (keep 0-255 range)
        padded_img = padded_img.transpose(2, 0, 1)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

        # Process targets - they come scaled by letterbox r, which matches our preprocessing
        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets) > 0:
            targets = np.array(targets).copy()
            n = min(len(targets), self.max_labels)
            # Targets are already in letterbox coords, no conversion needed
            padded_targets[:n] = targets[:n]

        return padded_img, padded_targets


class RFDETRValPreprocessor(BaseValPreprocessor):
    """
    RF-DETR validation preprocessor.

    Uses simple resize (no letterbox) and ImageNet normalization.
    RF-DETR expects RGB input with ImageNet mean/std normalization.
    """

    # ImageNet normalization constants
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @property
    def normalize(self) -> bool:
        return False

    @property
    def custom_normalization(self) -> bool:
        # RF-DETR applies ImageNet mean/std normalization in the preprocessor.
        # The validator must not rescale these values.
        return True

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size

        # Simple resize (no letterbox)
        resized_img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB
        resized_img = resized_img[:, :, ::-1]

        # Convert to float and normalize to [0, 1]
        resized_img = resized_img.astype(np.float32) / 255.0

        # Apply ImageNet normalization
        resized_img = (resized_img - self.MEAN) / self.STD

        # Convert to CHW format
        resized_img = resized_img.transpose(2, 0, 1)
        resized_img = np.ascontiguousarray(resized_img, dtype=np.float32)

        # Process targets
        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets) > 0:
            targets = np.array(targets).copy()
            n = min(len(targets), self.max_labels)

            # Simple resize scaling (no letterbox)
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h

            # Scale bounding boxes
            targets[:n, 0] *= scale_x  # x1
            targets[:n, 1] *= scale_y  # y1
            targets[:n, 2] *= scale_x  # x2
            targets[:n, 3] *= scale_y  # y2

            padded_targets[:n] = targets[:n]

        return resized_img, padded_targets


class V9ValPreprocessor(BaseValPreprocessor):
    """
    YOLOv9 validation preprocessor.

    Uses letterbox resize with gray padding (114, 114, 114).
    Normalizes to 0-1 range.
    """

    def __init__(self, img_size: Tuple[int, int], max_labels: int = 120, pad_value: int = 114):
        super().__init__(img_size, max_labels)
        self.pad_value = pad_value

    @property
    def normalize(self) -> bool:
        return True

    @property
    def uses_letterbox(self) -> bool:
        return True

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size

        # Letterbox resize maintaining aspect ratio
        ratio = min(target_h / orig_h, target_w / orig_w)
        new_h = int(orig_h * ratio)
        new_w = int(orig_w * ratio)

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image with gray background
        padded_img = np.full((target_h, target_w, 3), self.pad_value, dtype=np.uint8)
        padded_img[:new_h, :new_w] = resized_img

        # Convert BGR to RGB (v9 expects RGB)
        padded_img = padded_img[:, :, ::-1]

        # Convert to CHW and float, normalize to 0-1
        padded_img = padded_img.transpose(2, 0, 1)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) / 255.0

        # Process targets - they come scaled by letterbox r
        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets) > 0:
            targets = np.array(targets).copy()
            n = min(len(targets), self.max_labels)
            padded_targets[:n] = targets[:n]

        return padded_img, padded_targets
