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

    def letterbox_scale(
        self, orig_h: int, orig_w: int, imgsz: int
    ) -> Tuple[float, float, float]:
        """Return (r, off_x, off_y) needed to invert the letterbox coordinate transform.

        Default: top-left padding — uniform scale r, zero offsets.
        Override for center-padded preprocessors (e.g. YOLO-NAS).
        """
        if isinstance(imgsz, (list, tuple)):
            input_h, input_w = int(imgsz[0]), int(imgsz[1])
        else:
            input_h = input_w = int(imgsz)
        r = min(input_h / orig_h, input_w / orig_w)
        return r, 0.0, 0.0

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

            scale_x = target_w / orig_w
            scale_y = target_h / orig_h

            if self.wants_unresized_image:
                # The dataset handed over original-coordinate labels (no
                # letterbox pre-scaling); only the simple resize scale applies.
                targets[:n, 0] *= scale_x
                targets[:n, 1] *= scale_y
                targets[:n, 2] *= scale_x
                targets[:n, 3] *= scale_y
            else:
                # Undo letterbox scaling (applied by dataset) then apply the
                # simple resize scaling.
                letterbox_r = min(target_h / orig_h, target_w / orig_w)
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


class EfficientDetValPreprocessor(BaseValPreprocessor):
    """EfficientDet eval resize-pad, RGB, and ImageNet normalization."""

    @property
    def normalize(self) -> bool:
        return False

    @property
    def custom_normalization(self) -> bool:
        return True

    @property
    def uses_letterbox(self) -> bool:
        return True

    @property
    def wants_unresized_image(self) -> bool:
        return True

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        from ..models.efficientdet.utils import preprocess_numpy

        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size
        if target_h != target_w:
            raise ValueError(
                "EfficientDet validation requires a square input, "
                f"got ({target_h}, {target_w})"
            )
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        chw, ratio = preprocess_numpy(rgb, input_size=target_h)

        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets):
            scaled = np.asarray(targets, dtype=np.float32).copy()
            count = min(len(scaled), self.max_labels)
            scaled[:count, :4] *= ratio
            padded_targets[:count] = scaled[:count]
        return np.ascontiguousarray(chw, dtype=np.float32), padded_targets


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

    @property
    def wants_unresized_image(self) -> bool:
        # RF-DETR's validation/inference pipeline is a single direct resize from
        # the source image to the square model canvas.
        return True

    @staticmethod
    def _family_preprocess_numpy():
        """Return the family's ``preprocess_numpy``; subclasses point elsewhere."""
        from ..models.rfdetr.utils import preprocess_numpy

        return preprocess_numpy

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size

        rgb_img = img[:, :, ::-1]  # BGR -> RGB
        if target_h == target_w:
            resized_img, _ = self._family_preprocess_numpy()(rgb_img, target_h)
        else:
            pil_img = Image.fromarray(rgb_img).resize(
                (target_w, target_h), Image.Resampling.BILINEAR
            )
            resized_img = np.array(pil_img, dtype=np.float32) / 255.0
            resized_img = (resized_img - self.MEAN) / self.STD
            resized_img = resized_img.transpose(2, 0, 1)
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


class LWDETRValPreprocessor(RFDETRValPreprocessor):
    """LW-DETR preprocessor: square resize, RGB, ImageNet mean/std.

    Upstream's val transform is ``SquareResize([640]) + ToTensor + Normalize``
    (PIL BILINEAR to a square canvas, no letterbox) — the same pipeline
    RF-DETR inherited from it, so only the family-local ``preprocess_numpy``
    differs.
    """

    @staticmethod
    def _family_preprocess_numpy():
        from ..models.lwdetr.utils import preprocess_numpy

        return preprocess_numpy


class DETRValPreprocessor(RFDETRValPreprocessor):
    """DETR preprocessor: fixed square RGB resize plus ImageNet normalization."""

    @staticmethod
    def _family_preprocess_numpy():
        from ..models.detr.utils import preprocess_numpy

        return preprocess_numpy


class FasterRCNNValPreprocessor(BaseValPreprocessor):
    """Keep source resolution for the model's in-graph detection transform."""

    @property
    def normalize(self) -> bool:
        # The validator performs /255; ImageNet normalization remains in-graph.
        return True

    @property
    def wants_unresized_image(self) -> bool:
        return True

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size
        rgb_chw = np.ascontiguousarray(
            img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32
        )

        # Images stay in original coordinates, but DetectionValidator's shared
        # GT parser expects non-letterboxed targets in validation-canvas space
        # and divides this scale back out. Scale only the metric copy.
        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets):
            scaled = np.asarray(targets, dtype=np.float32).copy()
            count = min(len(scaled), self.max_labels)
            scaled[:count, [0, 2]] *= target_w / orig_w
            scaled[:count, [1, 3]] *= target_h / orig_h
            padded_targets[:count] = scaled[:count]
        return rgb_chw, padded_targets


class RetinaNetValPreprocessor(BaseValPreprocessor):
    """Upstream min/max aspect resize, ImageNet normalization, and P3-P7 pad."""

    @property
    def normalize(self) -> bool:
        return False

    @property
    def custom_normalization(self) -> bool:
        return True

    @property
    def uses_letterbox(self) -> bool:
        return True

    @property
    def wants_unresized_image(self) -> bool:
        return True

    def letterbox_scale(
        self, orig_h: int, orig_w: int, imgsz: int
    ) -> Tuple[float, float, float]:
        from ..models.retinanet.utils import resize_scale

        return resize_scale((orig_w, orig_h), int(imgsz)), 0.0, 0.0

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        from ..models.retinanet.utils import preprocess_numpy

        target_h, target_w = input_size
        if target_h != target_w:
            raise ValueError("RetinaNet validation requires a scalar min-side size")
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        processed, ratio = preprocess_numpy(rgb, int(target_h))

        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets):
            scaled = np.asarray(targets, dtype=np.float32).copy()
            count = min(len(scaled), self.max_labels)
            scaled[:count, :4] *= ratio
            padded_targets[:count] = scaled[:count]
        return processed, padded_targets


class SSDValPreprocessor(BaseValPreprocessor):
    """SSD300 fixed-square RGB resize and source normalization."""

    @property
    def normalize(self) -> bool:
        return False

    @property
    def custom_normalization(self) -> bool:
        # The family helper applies torchvision SSD's 0-255-space mean
        # subtraction, so DetectionValidator must leave the tensor untouched.
        return True

    @property
    def uses_letterbox(self) -> bool:
        return False

    @property
    def wants_unresized_image(self) -> bool:
        # Avoid a dataset letterbox followed by SSD's direct square stretch.
        return True

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        from ..models.ssd.utils import preprocess_numpy

        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size
        if (target_h, target_w) != (300, 300):
            raise ValueError("SSD300 validation requires a 300 x 300 canvas")

        rgb = img[:, :, ::-1].copy()
        resized_img, _ = preprocess_numpy(rgb, input_size=300)
        resized_img = np.ascontiguousarray(resized_img, dtype=np.float32)

        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets):
            scaled = np.asarray(targets, dtype=np.float32).copy()
            count = min(len(scaled), self.max_labels)
            scaled[:count, [0, 2]] *= target_w / orig_w
            scaled[:count, [1, 3]] *= target_h / orig_h
            padded_targets[:count] = scaled[:count]
        return resized_img, padded_targets


class FCOSValPreprocessor(BaseValPreprocessor):
    """Torchvision FCOS aspect resize, ImageNet normalization, and stride pad."""

    @property
    def normalize(self) -> bool:
        return False

    @property
    def custom_normalization(self) -> bool:
        return True

    @property
    def uses_letterbox(self) -> bool:
        return True

    @property
    def wants_unresized_image(self) -> bool:
        return True

    def letterbox_scale(
        self, orig_h: int, orig_w: int, imgsz: int
    ) -> Tuple[float, float, float]:
        from ..models.fcos.utils import resize_dimensions

        if isinstance(imgsz, (tuple, list)):
            if len(imgsz) != 2 or int(imgsz[0]) != int(imgsz[1]):
                raise ValueError(f"FCOS requires a scalar/square imgsz, got {imgsz}")
            imgsz = int(imgsz[0])
        _, _, scale = resize_dimensions(orig_h, orig_w, int(imgsz))
        return scale, 0.0, 0.0

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        from ..models.fcos.utils import preprocess_numpy

        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size
        if target_h != target_w:
            raise ValueError(
                f"FCOS validation requires a scalar/square imgsz, got {input_size}"
            )

        rgb = np.ascontiguousarray(img[:, :, ::-1])
        image_chw, scale = preprocess_numpy(rgb, input_size=target_h)

        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets):
            scaled = np.asarray(targets, dtype=np.float32).copy()
            count = min(len(scaled), self.max_labels)
            scaled[:count, :4] *= scale
            padded_targets[:count] = scaled[:count]
        return image_chw, padded_targets


class DeformableDETRValPreprocessor(RFDETRValPreprocessor):
    """Deformable DETR fixed-square ImageNet-normalized preprocessor.

    Upstream evaluation preserves aspect ratio at a short side of 800 with a
    1333-pixel cap. LibreYOLO uses its fixed-shape deployment convention: PIL
    bilinear resize directly to 800 x 800, with independent x/y box scaling.
    Validation and interactive inference deliberately share that transform.
    """

    @staticmethod
    def _family_preprocess_numpy():
        from ..models.deformable_detr.utils import preprocess_numpy

        return preprocess_numpy


class CenterNetValPreprocessor(BaseValPreprocessor):
    """Official CenterNet BGR affine warp and dataset-metric target copy."""

    @property
    def normalize(self) -> bool:
        return False

    @property
    def custom_normalization(self) -> bool:
        return True

    @property
    def wants_unresized_image(self) -> bool:
        return True

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        from ..models.centernet.utils import preprocess_bgr

        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size
        if target_h != target_w:
            raise ValueError("CenterNet validation requires a square input canvas")
        processed, _ = preprocess_bgr(img, input_size=target_h)

        # CenterNet's pixels use a centered uniform affine transform. The
        # shared validation parser intentionally sees this independent
        # stretch-scaled metric copy so its non-letterbox inverse recovers the
        # original boxes exactly; the labels are not model inputs.
        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets):
            scaled = np.asarray(targets, dtype=np.float32).copy()
            count = min(len(scaled), self.max_labels)
            scaled[:count, [0, 2]] *= target_w / orig_w
            scaled[:count, [1, 3]] *= target_h / orig_h
            padded_targets[:count] = scaled[:count]
        return processed, padded_targets


class YOLO9ValPreprocessor(BaseValPreprocessor):
    """YOLOv9 preprocessor: letterbox with gray padding, 0-1 range, RGB format.

    Pad defaults to top-left (LibreYOLO <=1.5). Official MTL geometry is
    center-pad and is selected only when the checkpoint stamps
    ``letterbox_pad="center"``.
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        max_labels: int = 120,
        pad_value: int = 114,
        letterbox_pad: str | None = None,
    ):
        super().__init__(img_size, max_labels)
        self.pad_value = pad_value
        self.letterbox_pad = letterbox_pad

    @property
    def normalize(self) -> bool:
        return True

    @property
    def uses_letterbox(self) -> bool:
        return True

    def letterbox_scale(
        self, orig_h: int, orig_w: int, imgsz: int
    ) -> Tuple[float, float, float]:
        from libreyolo.preprocess.letterbox import letterbox_geometry

        if isinstance(imgsz, (list, tuple)):
            input_h, input_w = int(imgsz[0]), int(imgsz[1])
        else:
            input_h = input_w = int(imgsz)
        ratio, _nh, _nw, pad_left, pad_top = letterbox_geometry(
            orig_h, orig_w, input_h, input_w, self.letterbox_pad
        )
        return ratio, float(pad_left), float(pad_top)

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        from libreyolo.preprocess.letterbox import apply_letterbox_hwc

        target_h, target_w = input_size
        padded_img, _ratio, pad_left, pad_top = apply_letterbox_hwc(
            img, target_h, target_w, pad=self.letterbox_pad, fill=self.pad_value
        )

        padded_img = padded_img[:, :, ::-1]  # BGR → RGB
        padded_img = padded_img.transpose(2, 0, 1)  # HWC → CHW
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) / 255.0

        # Dataset already scaled boxes by the letterbox ratio onto the
        # unpadded resized frame. Only the pad offset is missing. Top-left
        # pad is 0 so this is a no-op for unmarked ≤1.5 checkpoints.
        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets) > 0:
            targets = np.array(targets).copy()
            n = min(len(targets), self.max_labels)
            targets[:n, 0] = targets[:n, 0] + pad_left
            targets[:n, 1] = targets[:n, 1] + pad_top
            targets[:n, 2] = targets[:n, 2] + pad_left
            targets[:n, 3] = targets[:n, 3] + pad_top
            padded_targets[:n] = targets[:n]

        return padded_img, padded_targets


class YOLO9E2EValPreprocessor(YOLO9ValPreprocessor):
    """YOLOv9 E2E (NMS-free) preprocessor.

    Identical to YOLO9ValPreprocessor: letterbox with gray (114) padding,
    BGR→RGB, 0-1 normalization.  The one-to-one head does not change the
    preprocessing contract.
    """


class DarknetValPreprocessor(YOLO9ValPreprocessor):
    """Darknet families (YOLOv2/v3/v4): letterbox, RGB, 0-1, ~0.5 gray pad.

    Identical to the YOLO9 preprocessor (letterbox top-left, BGR->RGB, /255)
    except the pad fill is 128 (~0.5), matching Darknet's ``letterbox_image``.
    """

    def __init__(
        self, img_size: Tuple[int, int], max_labels: int = 120, pad_value: int = 128
    ):
        super().__init__(img_size, max_labels, pad_value=pad_value)


class YOLONASValPreprocessor(YOLO9ValPreprocessor):
    """YOLO-NAS preprocessor: resize to 636 (longest side), center-pad to 640, RGB, 0-1.

    Matches ``preprocess_numpy`` in models/yolonas/utils.py exactly so that
    ``_postprocess(letterbox=True, resize_size=636)`` correctly undoes the
    coordinate transform.  Without this, the top-left-padded YOLO9 path is
    used here but the center-pad path is used for inference — an ~81-pixel
    offset that collapses baseline mAP to near zero on non-square images.
    """

    @property
    def wants_unresized_image(self) -> bool:
        # Need the original image so we can apply the 636-resize step in one
        # pass; the dataset's load_resized_img would give us a pre-letterboxed
        # frame and we'd double-resize with the wrong ratio.
        return True

    def letterbox_scale(
        self, orig_h: int, orig_w: int, imgsz: int
    ) -> Tuple[float, float, float]:
        # YOLO-NAS resizes to YOLO_NAS_RESIZE_SIZE first, then center-pads to imgsz.
        from ..models.yolonas.utils import YOLO_NAS_RESIZE_SIZE

        r = min(YOLO_NAS_RESIZE_SIZE / orig_h, YOLO_NAS_RESIZE_SIZE / orig_w)
        new_w = int(round(orig_w * r))
        new_h = int(round(orig_h * r))
        off_x = (imgsz - new_w) // 2
        off_y = (imgsz - new_h) // 2
        return r, float(off_x), float(off_y)

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        from ..models.yolonas.utils import YOLO_NAS_RESIZE_SIZE, preprocess_numpy

        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size  # e.g. (640, 640)
        if target_h != target_w:
            raise ValueError(
                f"YOLO-NAS validation does not support rectangular input sizes, "
                f"got ({target_h}, {target_w}). Use a square imgsz."
            )

        img_rgb = np.ascontiguousarray(img[:, :, ::-1])  # BGR → RGB
        img_chw, _ = preprocess_numpy(
            img_rgb,
            input_size=target_h,
            resize_size=YOLO_NAS_RESIZE_SIZE,
        )

        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets) > 0:
            targets = np.array(targets).copy()
            n = min(len(targets), self.max_labels)
            r = min(YOLO_NAS_RESIZE_SIZE / orig_h, YOLO_NAS_RESIZE_SIZE / orig_w)
            new_w = int(round(orig_w * r))
            new_h = int(round(orig_h * r))
            off_x = (target_w - new_w) // 2
            off_y = (target_h - new_h) // 2
            targets[:n, 0] = targets[:n, 0] * r + off_x
            targets[:n, 1] = targets[:n, 1] * r + off_y
            targets[:n, 2] = targets[:n, 2] * r + off_x
            targets[:n, 3] = targets[:n, 3] * r + off_y
            padded_targets[:n] = targets[:n]

        return img_chw, padded_targets


class YOLONASOBBValPreprocessor(YOLONASValPreprocessor):
    """YOLO-NAS-R preprocessor: longest side to 1024, bottom-right pad 114, BGR.

    Deliberately delegates the pixel work to ``preprocess_obb_numpy`` -- the
    exact function the OBB inference path uses -- so validation, training and
    prediction cannot drift apart (the centre-pad/bottom-right-pad mismatch
    this class's detect parent documents is precisely that failure mode).

    Targets are the OBB dataset's six columns
    ``[x1, y1, x2, y2, class, angle]`` where the ``xyxy`` block encodes
    ``cx, cy, w, h`` un-rotated. The rescale is uniform and the padding has no
    offset, so the angle passes through untouched.
    """

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        from ..preprocess.yolonas import preprocess_obb_numpy

        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size
        if target_h != target_w:
            raise ValueError(
                "YOLO-NAS OBB validation does not support rectangular input "
                f"sizes, got ({target_h}, {target_w}). Use a square imgsz."
            )

        # The dataset hands us BGR from cv2; preprocess_obb_numpy takes RGB and
        # reverses the channels itself, so flip once here rather than
        # duplicating its recipe.
        img_rgb = np.ascontiguousarray(img[:, :, ::-1])
        img_chw, ratio = preprocess_obb_numpy(img_rgb, input_size=target_h)

        ncol = targets.shape[1] if getattr(targets, "ndim", 0) == 2 else 6
        padded_targets = np.zeros((self.max_labels, ncol), dtype=np.float32)
        if len(targets) > 0:
            targets = np.array(targets, dtype=np.float32).copy()
            n = min(len(targets), self.max_labels)
            targets[:n, :4] *= ratio
            padded_targets[:n] = targets[:n]

        return img_chw, padded_targets

    def letterbox_scale(
        self, orig_h: int, orig_w: int, imgsz: int
    ) -> Tuple[float, float, float]:
        from ..postprocess.yolonas import YOLO_NAS_OBB_RESIZE_SIZE

        resize = min(YOLO_NAS_OBB_RESIZE_SIZE, imgsz)
        r = min(resize / orig_h, resize / orig_w)
        return r, 0.0, 0.0  # bottom-right padding: no offset


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

            # wants_unresized_image=True: pull_item already returns labels in
            # original coordinates, so only the direct square resize scaling
            # applies (no letterbox ratio to undo).
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h

            targets[:n, 0] *= scale_x
            targets[:n, 1] *= scale_y
            targets[:n, 2] *= scale_x
            targets[:n, 3] *= scale_y

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


class PPYOLOEValPreprocessor(StandardValPreprocessor):
    """PP-YOLOE preprocessor: stretch resize, RGB, ImageNet mean/std on 0-255.

    Reproduces the source validation path (``DetectionRescale`` to a fixed
    640x640 followed by ``Normalize``). The resize is a stretch, not a
    letterbox, so aspect ratio is not preserved and the postprocessor reverses
    the x and y scales independently.
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
        chw, padded_targets = super().__call__(
            img[:, :, ::-1].copy(), targets, input_size
        )
        chw = (chw - self._MEAN) / self._STD
        return chw.astype(np.float32), padded_targets


class RTDETRv2ValPreprocessor(BaseValPreprocessor):
    """RT-DETRv2 val preprocessor matching upstream's PIL/torchvision Resize.

    The only difference from ``RTDETRValPreprocessor`` is that this one uses
    PIL.Image.resize (BILINEAR) on the un-letterboxed source image, mirroring
    upstream's ``Resize -> ConvertPILImage(scale=True)`` chain. cv2.resize and
    PIL.Image.resize use different bilinear kernels; the pixel drift cascades
    to ~0.7 mAP on COCO val2017 for v2-r18 if cv2 is used instead.
    """

    @property
    def normalize(self) -> bool:
        return True

    @property
    def wants_unresized_image(self) -> bool:
        return True

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size

        # img is BGR uint8 from cv2.imread; flip to RGB then PIL-resize.
        rgb = img[:, :, ::-1]
        resized = np.array(
            Image.fromarray(rgb).resize(
                (target_w, target_h), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        resized = resized / 255.0
        resized = resized.transpose(2, 0, 1)
        resized = np.ascontiguousarray(resized, dtype=np.float32)

        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets) > 0:
            targets = np.array(targets).copy()
            n = min(len(targets), self.max_labels)
            # wants_unresized_image=True: labels are already in original
            # coordinates, so only the direct square resize scaling applies.
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
            targets[:n, 0] *= scale_x
            targets[:n, 1] *= scale_y
            targets[:n, 2] *= scale_x
            targets[:n, 3] *= scale_y
            padded_targets[:n] = targets[:n]

        return resized, padded_targets


class RTDETRv2OBBValPreprocessor(BaseValPreprocessor):
    """Aspect-preserving, bottom/right-padded RT-DETRv2 OBB preprocessing."""

    @property
    def normalize(self) -> bool:
        return True

    @property
    def uses_letterbox(self) -> bool:
        return True

    @property
    def wants_unresized_image(self) -> bool:
        return True

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(round(orig_w * scale)))
        new_h = max(1, int(round(orig_h * scale)))

        rgb = img[:, :, ::-1]
        resized = np.asarray(
            Image.fromarray(rgb).resize((new_w, new_h), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        chw = np.ascontiguousarray(padded.astype(np.float32).transpose(2, 0, 1) / 255.0)

        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets):
            scaled = np.asarray(targets, dtype=np.float32).copy()
            n = min(len(scaled), self.max_labels)
            scaled[:n, :4] *= scale
            padded_targets[:n] = scaled[:n]
        return chw, padded_targets


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


class RTMDetValPreprocessor(BaseValPreprocessor):
    """RTMDet preprocessor: BGR letterbox at pad 114, mmdet mean/std normalization.

    The mmdet config uses ``bgr_to_rgb=False`` with mean ``[103.53, 116.28, 123.675]``
    and std ``[57.375, 57.12, 58.395]`` applied to the BGR image (not RGB).
    """

    BGR_MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
    BGR_STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
    PAD_VALUE = 114

    @property
    def normalize(self) -> bool:
        return False

    @property
    def custom_normalization(self) -> bool:
        return True

    @property
    def uses_letterbox(self) -> bool:
        return True

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size

        ratio = min(target_h / orig_h, target_w / orig_w)
        new_h = int(orig_h * ratio)
        new_w = int(orig_w * ratio)

        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded_img = np.full((target_h, target_w, 3), self.PAD_VALUE, dtype=np.uint8)
        padded_img[:new_h, :new_w] = resized_img

        # img comes in as BGR (cv2). Apply mmdet mean/std in BGR space.
        normed = (padded_img.astype(np.float32) - self.BGR_MEAN) / self.BGR_STD
        normed = np.ascontiguousarray(normed.transpose(2, 0, 1), dtype=np.float32)

        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets) > 0:
            n = min(len(targets), self.max_labels)
            padded_targets[:n] = np.asarray(targets[:n], dtype=np.float32)

        return normed, padded_targets


class FOMOValPreprocessor(BaseValPreprocessor):
    """FOMO validation preprocessor: direct RGB stretch-resize + [-1, 1] normalisation."""

    MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    @property
    def normalize(self) -> bool:
        return False

    @property
    def custom_normalization(self) -> bool:
        return True

    @property
    def wants_unresized_image(self) -> bool:
        return True

    def __call__(
        self, img: np.ndarray, targets: np.ndarray, input_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        orig_h, orig_w = img.shape[:2]
        target_h, target_w = input_size

        rgb_img = img[:, :, ::-1]
        pil_img = Image.fromarray(rgb_img)
        resized_pil = pil_img.resize((target_w, target_h), Image.Resampling.BILINEAR)
        resized = np.array(resized_pil, dtype=np.float32) / 255.0
        resized = (resized - self.MEAN) / self.STD
        resized = np.ascontiguousarray(resized.transpose(2, 0, 1), dtype=np.float32)

        padded_targets = np.zeros((self.max_labels, 5), dtype=np.float32)
        if len(targets) > 0:
            targets = np.array(targets).copy()
            n = min(len(targets), self.max_labels)
            scale_x = target_w / orig_w
            scale_y = target_h / orig_h
            targets[:n, 0] *= scale_x
            targets[:n, 1] *= scale_y
            targets[:n, 2] *= scale_x
            targets[:n, 3] *= scale_y
            padded_targets[:n] = targets[:n]

        return resized, padded_targets


class DOMEDETRValPreprocessor(DFINEValPreprocessor):
    """Dome-DETR validation preprocessor: same RGB /255 plain resize as D-FINE.

    Only the input size differs (800 rather than 640), and that is carried by
    the model's ``INPUT_SIZES`` rather than by the preprocessor.
    """
