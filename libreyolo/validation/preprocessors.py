"""Validation preprocessors for different model architectures."""

from abc import ABC, abstractmethod
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


class BaseValPreprocessor(ABC):
    """Abstract base class for validation preprocessors."""

    def __init__(self, img_size: Tuple[int, int], max_labels: int = 120):
        self.img_size = img_size
        self.max_labels = max_labels

    @abstractmethod
    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess image (H, W, C BGR) and targets (N, 5) [x1,y1,x2,y2,class]."""
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

    @property
    def wants_unresized_image(self) -> bool:
        """If True, the dataset should hand over the original-resolution image
        and let the preprocessor own all resizing.

        ``COCODataset.load_resized_img`` letterbox-resizes by default to keep
        the YOLOX path on its happy path. Families that do plain stretch
        resize end up with a double-resize (letterbox → stretch) which costs
        ~1 mAP from the extra interpolation pass. Setting this True skips
        the dataset-level resize and lets the preprocessor go straight from
        the original image to the target size in a single ``cv2.resize``.
        """
        return False

    def _pad_targets(self, targets: np.ndarray, n_valid: int) -> np.ndarray:
        """Pad targets to fixed size for batching."""
        padded = np.zeros((self.max_labels, 5), dtype=np.float32)
        if n_valid > 0:
            padded[:n_valid] = targets[:n_valid]
        return padded


class StandardValPreprocessor(BaseValPreprocessor):
    """Default preprocessor: simple resize (no letterbox), normalizes to 0-1."""

    @property
    def normalize(self) -> bool:
        return True

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size

        resized_img = cv2.resize(
            img, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

        resized_img = resized_img.transpose(2, 0, 1)  # HWC → CHW
        resized_img = np.ascontiguousarray(resized_img, dtype=np.float32)

        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets) > 0:
            targets = np.array(targets).copy()
            n = min(len(targets), self.max_labels)

            # Undo letterbox scaling (applied by dataset) and apply simple resize scaling
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
    """YOLOX preprocessor: letterbox with gray padding, 0-255 range, BGR format."""

    def __init__(
        self, img_size: Tuple[int, int], max_labels: int = 120, pad_value: int = 114
    ):
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

        padded_img = np.full((target_h, target_w, 3), self.pad_value, dtype=np.uint8)
        padded_img[:new_h, :new_w] = resized_img

        # Keep BGR format — YOLOX is trained on BGR from cv2

        padded_img = padded_img.transpose(2, 0, 1)  # HWC → CHW, keep 0-255
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)

        # Targets are already in letterbox coords, no conversion needed
        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets) > 0:
            targets = np.array(targets).copy()
            n = min(len(targets), self.max_labels)
            padded_targets[:n] = targets[:n]

        return padded_img, padded_targets


class RFDETRValPreprocessor(BaseValPreprocessor):
    """RF-DETR preprocessor: simple resize, RGB, ImageNet mean/std normalization."""

    # ImageNet normalization constants (canonical source: models.rfdetr.utils)
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @property
    def normalize(self) -> bool:
        return False

    @property
    def custom_normalization(self) -> bool:
        return True  # ImageNet mean/std applied here; validator must not rescale

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size

        resized_img = cv2.resize(
            img, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

        resized_img = resized_img[:, :, ::-1]  # BGR → RGB
        resized_img = resized_img.astype(np.float32) / 255.0
        resized_img = (resized_img - self.MEAN) / self.STD  # ImageNet normalization

        resized_img = resized_img.transpose(2, 0, 1)  # HWC → CHW
        resized_img = np.ascontiguousarray(resized_img, dtype=np.float32)

        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets) > 0:
            targets = np.array(targets).copy()
            n = min(len(targets), self.max_labels)

            # Simple resize scaling (no letterbox)
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h

            targets[:n, 0] *= scale_x
            targets[:n, 1] *= scale_y
            targets[:n, 2] *= scale_x
            targets[:n, 3] *= scale_y

            padded_targets[:n] = targets[:n]

        return resized_img, padded_targets


class YOLO9ValPreprocessor(BaseValPreprocessor):
    """YOLOv9 preprocessor: letterbox with gray padding, 0-1 range, RGB format."""

    def __init__(
        self, img_size: Tuple[int, int], max_labels: int = 120, pad_value: int = 114
    ):
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

        padded_img = np.full((target_h, target_w, 3), self.pad_value, dtype=np.uint8)
        padded_img[:new_h, :new_w] = resized_img

        padded_img = padded_img[:, :, ::-1]  # BGR → RGB
        padded_img = padded_img.transpose(2, 0, 1)  # HWC → CHW
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) / 255.0

        # Targets are already in letterbox coords
        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets) > 0:
            targets = np.array(targets).copy()
            n = min(len(targets), self.max_labels)
            padded_targets[:n] = targets[:n]

        return padded_img, padded_targets


class YOLO9E2EValPreprocessor(YOLO9ValPreprocessor):
    """YOLOv9 E2E (NMS-free) preprocessor.

    Identical to YOLO9ValPreprocessor: letterbox with gray (114) padding,
    BGR→RGB, 0-1 normalization.  The one-to-one head does not change the
    preprocessing contract.
    """


class YOLONASValPreprocessor(YOLO9ValPreprocessor):
    """YOLO-NAS preprocessor.

    The current native port uses LibreYOLO's shared RGB 0-1 letterbox path for
    consistency across inference and validation. A later parity pass can tighten
    this toward the exact SG preprocessing contract if needed.
    """


class DFINEValPreprocessor(StandardValPreprocessor):
    """D-FINE preprocessor: plain resize + 0-1 + RGB, no letterbox, no ImageNet norm.

    Upstream D-FINE loads images via PIL (RGB) and feeds them through
    ``ConvertPILImage(scale=True)``; LibreYOLO's training transform mirrors
    this with an explicit BGR→RGB flip, and inference also runs on RGB. The
    validator's dataset, however, hands us BGR straight from ``cv2.imread``,
    so we flip channels here to keep validation aligned with train/inference.
    """

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        return super().__call__(img[:, :, ::-1].copy(), targets, input_size)


class DEIMValPreprocessor(DFINEValPreprocessor):
    """DEIM-D-FINE validation preprocessor: same RGB /255 plain resize as D-FINE."""


class DEIMv2ValPreprocessor(DEIMValPreprocessor):
    """DEIMv2 validation preprocessor matching upstream PIL/torchvision resize."""

    @property
    def wants_unresized_image(self) -> bool:
        # PIL BILINEAR on the original image is the whole point of this
        # preprocessor — matches upstream DEIMv2's torchvision val transform.
        # Without this opt-in, the dataset would letterbox first and we'd
        # be PIL-resizing a padded canvas instead of the source image.
        return True

    def _resize_image(
        self, img: np.ndarray, target_w: int, target_h: int
    ) -> np.ndarray:
        rgb = img[:, :, ::-1]
        return np.array(
            Image.fromarray(rgb).resize(
                (target_w, target_h), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size

        resized_img = self._resize_image(img, target_w, target_h)
        resized_img = resized_img.transpose(2, 0, 1)  # HWC -> CHW
        resized_img = np.ascontiguousarray(resized_img, dtype=np.float32)

        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets) > 0:
            targets = np.array(targets).copy()
            n = min(len(targets), self.max_labels)

            # COCO annotations are pre-scaled by the dataset's aspect-ratio r.
            # Undo that, then apply upstream's direct square resize scaling.
            letterbox_r = min(target_h / orig_h, target_w / orig_w)
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h

            targets[:n, 0] = targets[:n, 0] / letterbox_r * scale_x
            targets[:n, 1] = targets[:n, 1] / letterbox_r * scale_y
            targets[:n, 2] = targets[:n, 2] / letterbox_r * scale_x
            targets[:n, 3] = targets[:n, 3] / letterbox_r * scale_y

            padded_targets[:n] = targets[:n]

        return resized_img, padded_targets


class ECValPreprocessor(StandardValPreprocessor):
    """EC preprocessor: plain resize, RGB, /255, ImageNet normalize.

    Same skeleton as D-FINE's preprocessor but adds ImageNet (mean, std)
    normalization, matching upstream's val transforms:
        Resize -> ConvertPILImage(scale=True) -> Normalize(IMAGENET).
    Skipping ImageNet norm costs ~2 mAP on COCO val2017.
    """

    _IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    _IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    @property
    def custom_normalization(self) -> bool:
        # We apply /255 + ImageNet norm here; the validator must not rescale.
        return True

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        chw, padded_targets = super().__call__(
            img[:, :, ::-1].copy(), targets, input_size
        )
        chw = chw / 255.0
        chw = (chw - self._IMAGENET_MEAN) / self._IMAGENET_STD
        return chw.astype(np.float32), padded_targets


class DEIMv2DINOValPreprocessor(DEIMv2ValPreprocessor):
    """DEIMv2 DINOv3 validation preprocessor: PIL resize plus ImageNet norm."""

    _IMAGENET_MEAN = ECValPreprocessor._IMAGENET_MEAN
    _IMAGENET_STD = ECValPreprocessor._IMAGENET_STD

    @property
    def custom_normalization(self) -> bool:
        # We apply /255 + ImageNet norm here; the validator must not rescale.
        return True

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        chw, padded_targets = super().__call__(img, targets, input_size)
        chw = chw / 255.0
        chw = (chw - self._IMAGENET_MEAN) / self._IMAGENET_STD
        return chw.astype(np.float32), padded_targets


class PICODETValPreprocessor(StandardValPreprocessor):
    """PICODET preprocessor: simple resize, RGB, ImageNet mean/std in 0-255 space.

    Matches Bo's upstream val pipeline (``Resize(keep_ratio=False)`` then
    ``Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)``).
    Skipping the normalisation costs several mAP on COCO val2017.
    """

    _MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(3, 1, 1)
    _STD = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(3, 1, 1)

    @property
    def custom_normalization(self) -> bool:
        return True

    @property
    def wants_unresized_image(self) -> bool:
        return True  # avoid the dataset's letterbox-then-stretch double resize

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        # BGR -> RGB then standard simple-resize path; no /255 (mean/std are
        # already in 0-255 space).
        chw, padded_targets = super().__call__(
            img[:, :, ::-1].copy(), targets, input_size
        )
        chw = (chw - self._MEAN) / self._STD
        return chw.astype(np.float32), padded_targets


class RTDETRValPreprocessor(BaseValPreprocessor):
    """Preprocessor for RT-DETR validation: resize to fixed size, normalize to [0,1], no letterbox."""

    @property
    def normalize(self) -> bool:
        return True

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess image for RT-DETR validation."""
        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size

        # Simple resize (no letterbox)
        resized_img = cv2.resize(
            img, (target_w, target_h), interpolation=cv2.INTER_LINEAR
        )

        # BGR → RGB, normalize to [0, 1]
        resized_img = resized_img[:, :, ::-1]
        resized_img = resized_img.astype(np.float32) / 255.0

        resized_img = resized_img.transpose(2, 0, 1)  # HWC → CHW
        resized_img = np.ascontiguousarray(resized_img, dtype=np.float32)

        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets) > 0:
            targets = np.array(targets).copy()
            n = min(len(targets), self.max_labels)

            # Simple resize scaling (no letterbox)
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h

            targets[:n, 0] *= scale_x
            targets[:n, 1] *= scale_y
            targets[:n, 2] *= scale_x
            targets[:n, 3] *= scale_y

            padded_targets[:n] = targets[:n]

        return resized_img, padded_targets
