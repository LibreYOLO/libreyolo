"""Base class for LibreYOLO inference backends."""

# Annotations are deferred so that the ``-> torch.Tensor`` hints below do not
# resolve torch at class-definition time; see the lazy torch import further
# down.
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
from ..utils.lazy import lazy_module, module_available

from PIL import Image

# Constants and the yolo9 postprocess entry point live in the torch-free
# ``postprocess`` package, so importing them here does not drag in
# ``models/__init__.py`` (which eagerly builds every nn.Module to populate
# the can_load registry) and keeps this module importable without torch.
from ..postprocess.yolo9 import (
    _YOLO9_MAX_NMS_CANDIDATES,
    postprocess as yolo9_postprocess,
)
from ..postprocess.ppyoloe import PPYOLOE_PRE_NMS_TOP_K
from ..postprocess.yolonas import (
    YOLO_NAS_OBB_PRE_NMS_TOP_K,
    YOLO_NAS_OBB_RESIZE_SIZE,
    YOLO_NAS_PRE_NMS_TOP_K,
    YOLO_NAS_POSE_RESIZE_SIZE,
    YOLO_NAS_RESIZE_SIZE,
)
from ..preprocess import as_batched_input, as_input
from ..preprocess.yolo9 import preprocess_image
from ..preprocess.yolonas import (
    preprocess_image as yolonas_preprocess_image,
    preprocess_obb_image as yolonas_preprocess_obb_image,
    preprocess_pose_image as yolonas_preprocess_pose_image,
)
from ..preprocess.yolox import preprocess_image as yolox_preprocess_image
from ..tasks import normalize_supported_tasks, normalize_task, resolve_task
from ..utils.drawing import (
    draw_boxes,
    draw_keypoints,
    draw_masks,
    draw_obb,
    draw_points,
    draw_semantic_mask,
)
from ..utils.general import (
    COCO_CLASSES,
    get_safe_stem,
    log_saved_result,
    resolve_save_path,
)
from ..utils.image_loader import ImageLoader
from ..utils.model_info import build_model_info, format_model_info
from ..utils.predict_args import normalize_predict_kwargs
from ..utils.results import (
    Boxes,
    DepthMap,
    EdgeMap,
    Embeddings,
    Gaze,
    Keypoints,
    Matte,
    Masks,
    NormalMap,
    OBB,
    Points,
    Probs,
    Results,
    RestoredImage,
    SemanticMask,
)
from ..utils.screen import ScreenSource, grab_screen
from ..utils.source import SourceKind, build_stream_source, classify_source
from ..utils.video import (
    FrameSource,
    collect_video_results,
    run_video_inference,
)


@lru_cache(maxsize=1)
def _torch_installed() -> bool:
    """Whether torch is importable.

    Cached: the answer cannot change within a process, and this is consulted
    once per built result.
    """
    return module_available("torch")


def _f32(data):
    """float32 ``torch.Tensor`` when torch is installed, ``ndarray`` otherwise.

    Results containers hold ``TensorLike`` (tensor *or* ndarray) and never
    require a tensor, which is what lets a torch-free ONNX deployment build a
    Results object. With torch installed the tensor branch is taken and
    behaviour is byte-for-byte unchanged.
    """
    if _torch_installed():
        return torch.tensor(data, dtype=torch.float32)
    return np.asarray(data, dtype=np.float32)


def _zeros_f32(shape):
    """Empty-detection counterpart to :func:`_f32`."""
    if _torch_installed():
        return torch.zeros(shape, dtype=torch.float32)
    return np.zeros(shape, dtype=np.float32)


def _zeros_bool(length):
    """All-False mask, as a ``torch.Tensor`` if torch is installed."""
    if _torch_installed():
        return torch.zeros(length, dtype=torch.bool)
    return np.zeros(length, dtype=bool)


def _bool_array(data):
    """Boolean mask stack, as a ``torch.Tensor`` if torch is installed."""
    if _torch_installed():
        return torch.from_numpy(data).bool()
    return np.asarray(data, dtype=bool)


def _to_blob(input_tensor) -> np.ndarray:
    """Return the runtime input as a contiguous numpy array.

    Preprocessing yields a torch tensor when torch is installed and an ndarray
    otherwise (see ``libreyolo.preprocess.as_batched_input``). Every non-torch
    runtime (ONNX, OpenVINO, ncnn, ...) wants numpy either way, so normalise
    here instead of at each call site.
    """
    if isinstance(input_tensor, np.ndarray):
        return input_tensor
    return input_tensor.detach().cpu().numpy()



# torch is resolved on first use so this module stays importable in a
# torch-free ONNX deployment (discussions/711).
torch = lazy_module("torch")
F = lazy_module("torch.nn.functional")

logger = logging.getLogger(__name__)

ImageSize = Union[int, Tuple[int, int]]
_RECTANGULAR_BACKEND_FAMILIES = {
    "hrnet",
    "yolo9",
    "yolo9_e2e",
    "yolo9_p2",
    "nafnet",
    # Every PP-LiteSeg size is natively rectangular (512x1024 / 768x1536), so
    # its exported artifacts always carry a rectangular imgsz_h/imgsz_w pair.
    "ppliteseg",
    "realesrgan",
    "quicksrnet",
}

# Real-ESRGAN integer upscale factor per size, used by scale-aware restore decode.
_REALESRGAN_BACKEND_SCALE = {"x4": 4, "x2": 2, "x4t": 4}
_SWINIR_BACKEND_SCALE = {"s": 4, "m": 4, "l": 4}
_QUICKSRNET_BACKEND_SCALE = {"m2": 2}
_REALESRGAN_BACKEND_PAD_MULTIPLE = {"x4": 1, "x2": 2, "x4t": 1}

# Families removed from LibreYOLO. An exported artifact whose metadata still names
# one of these must fail loudly instead of being silently parsed as YOLO9.
_REMOVED_FAMILIES = {"damoyolo"}


class _BackendEvalProxy:
    def eval(self):
        return self


def _imgsz_hw(imgsz: ImageSize) -> Tuple[int, int]:
    if isinstance(imgsz, tuple):
        if len(imgsz) != 2:
            raise ValueError(f"imgsz must be int or (height, width), got {imgsz}")
        h, w = int(imgsz[0]), int(imgsz[1])
    else:
        h = w = int(imgsz)
    if h <= 0 or w <= 0:
        raise ValueError(f"imgsz values must be positive, got {(h, w)}")
    return h, w


def _normalize_imgsz(imgsz: ImageSize) -> ImageSize:
    h, w = _imgsz_hw(imgsz)
    return h if h == w else (h, w)


def _is_rectangular_imgsz(imgsz: ImageSize) -> bool:
    h, w = _imgsz_hw(imgsz)
    return h != w


class MetadataImageSizeError(ValueError):
    """Raised when exported input-size metadata is malformed."""


def _read_metadata_imgsz(
    meta: dict,
    model_family: Optional[str],
    *,
    artifact: str,
) -> ImageSize | None:
    """Read exported-runtime input size metadata.

    ``imgsz`` stays as the legacy square scalar. ``imgsz_h``/``imgsz_w`` are
    only allowed to describe rectangular runtime inputs for backend families
    that explicitly support them.
    """
    has_imgsz_h = "imgsz_h" in meta
    has_imgsz_w = "imgsz_w" in meta
    if has_imgsz_h != has_imgsz_w:
        raise MetadataImageSizeError(
            f"{artifact} must define both imgsz_h and imgsz_w, or neither."
        )

    if has_imgsz_h and has_imgsz_w:
        try:
            imgsz = _normalize_imgsz((int(meta["imgsz_h"]), int(meta["imgsz_w"])))
        except (TypeError, ValueError) as e:
            raise MetadataImageSizeError(
                f"{artifact} has invalid imgsz_h/imgsz_w metadata."
            ) from e
        if (
            _is_rectangular_imgsz(imgsz)
            and (model_family or "").lower() not in _RECTANGULAR_BACKEND_FAMILIES
        ):
            raise NotImplementedError(
                "Rectangular exported-backend inference is currently supported "
                "for YOLO9-family, HRNet, NAFNet, QuickSRNet, and "
                "Real-ESRGAN exports only. "
                f"{artifact} declares model_family={model_family or 'unknown'!r}."
            )
        return imgsz

    if "imgsz" in meta:
        try:
            return _normalize_imgsz(int(meta["imgsz"]))
        except (TypeError, ValueError) as e:
            raise MetadataImageSizeError(
                f"{artifact} has invalid imgsz metadata."
            ) from e

    return None


def _read_pose_metadata(meta: dict) -> dict[str, Any]:
    """Extract shared pose metadata from embedded or sidecar export metadata."""
    pose_meta: dict[str, Any] = {}
    if "num_keypoints" in meta:
        pose_meta["num_keypoints"] = int(meta["num_keypoints"])
    if "keypoint_dim" in meta:
        pose_meta["keypoint_dim"] = int(meta["keypoint_dim"])
    if "num_keypoints_per_class" in meta:
        raw_schema = meta["num_keypoints_per_class"]
        if isinstance(raw_schema, str):
            raw_schema = json.loads(raw_schema)
        if raw_schema is not None:
            pose_meta["num_keypoints_per_class"] = [int(count) for count in raw_schema]
    return pose_meta


def _read_runtime_metadata(meta: dict) -> dict[str, Any]:
    """Extract preprocessing and graph-contract metadata shared by backends."""
    runtime_meta: dict[str, Any] = {
        "embedded_nms": str(meta.get("nms", "")).lower() == "true",
    }
    if meta.get("crop_pct") is not None:
        runtime_meta["crop_pct"] = float(meta["crop_pct"])
    if meta.get("interpolation") is not None:
        runtime_meta["interpolation"] = str(meta["interpolation"])
    if meta.get("num_bins") is not None:
        runtime_meta["num_bins"] = int(meta["num_bins"])
    if meta.get("bin_width_deg") is not None:
        runtime_meta["bin_width_deg"] = float(meta["bin_width_deg"])
    if meta.get("offset_deg") is not None:
        runtime_meta["offset_deg"] = float(meta["offset_deg"])
    return runtime_meta


def _nms_numpy(
    boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45
) -> list:
    """Numpy-based Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]

    return keep


def _batched_nms_numpy(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float = 0.45,
) -> list:
    """Class-aware NMS matching torchvision.ops.batched_nms ordering."""
    keep = []
    for cls in np.unique(class_ids):
        cls_indices = np.where(class_ids == cls)[0]
        cls_keep = _nms_numpy(boxes[cls_indices], scores[cls_indices], iou_threshold)
        keep.extend(cls_indices[cls_keep].tolist())

    if not keep:
        return []
    keep = np.asarray(keep, dtype=np.int64)
    return keep[np.argsort(scores[keep])[::-1]].tolist()


def _is_pytorch_cuda_device(device_str: str) -> bool:
    """Return True only when device_str is a valid PyTorch CUDA device string.

    Non-PyTorch runtimes (OpenVINO "gpu", CoreML "coreml", ncnn "ncnn") store
    backend-specific device identifiers in self.device that are not parseable
    by torch.device(); calling torch.device() on them raises RuntimeError.
    """
    try:
        return torch.device(device_str).type == "cuda"
    except RuntimeError:
        return False


def _is_nms_free_family(model_family: Optional[str]) -> bool:
    """Whether backend outputs should bypass generic NMS.

    DETR-style families already emit a ranked set prediction after top-k
    selection. Applying YOLO-style IoU suppression on top of that can remove
    valid detections and make exported runtimes diverge from native PyTorch.
    """
    return model_family in {
        "centernet",
        "deformable_detr",
        "detr",
        "dinodetr",
        "dfine",
        "domedetr",
        "deim",
        "deimv2",
        "tinyformer",
        "ec",
        "faster_rcnn",
        "mask_rcnn",
        "lwdetr",
        "rfdetr",
        "rtdetr",
        "rtdetrv2",
        "rtdetrv4",
        "yolo9_e2e",
    }


def _lwdetr_num_select(model_size: Optional[str]) -> int:
    """Return LW-DETR's configured top-k selection for exported backends."""
    return 100 if model_size == "t" else 300


def _rfdetr_num_select(task: str, model_size: Optional[str]) -> int:
    """Return RF-DETR's configured top-k selection for exported backends."""
    if task == "segment":
        return {"n": 100, "s": 100, "m": 200, "l": 200}.get(model_size or "", 300)
    if task == "pose" and model_size == "x":
        return 100
    return 300


def _logsumexp_np(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(max_values, axis=axis) + np.log(
        np.sum(np.exp(values - max_values), axis=axis)
    )


def _rfdetr_keypoint_log_mean_trace_np(active_keypoints: np.ndarray) -> np.ndarray:
    log_l11 = active_keypoints[..., 4]
    l21 = active_keypoints[..., 5]
    log_l22 = active_keypoints[..., 6]
    w_find = 1.0 / (1.0 + np.exp(-active_keypoints[..., 2]))
    log_t1 = -2.0 * log_l11
    log_t2 = -2.0 * log_l22
    log_t3 = 2.0 * np.log(np.clip(np.abs(l21), 1e-12, None)) + log_t1 + log_t2
    log_trace_sigma = _logsumexp_np(
        np.stack([log_t1, log_t2, log_t3], axis=-1),
        axis=-1,
    )
    log_w_find = np.log(np.clip(w_find, 1e-12, None))
    return _logsumexp_np(log_trace_sigma + log_w_find, axis=-1) - _logsumexp_np(
        log_w_find,
        axis=-1,
    )


class BaseBackend(ABC):
    """Abstract base class for all inference backends.

    Subclasses must:
    1. Implement ``__init__`` to load the runtime-specific model, then call
       ``super().__init__(...)`` with the resolved common attributes.
    2. Implement ``_run_inference`` to execute the model and return raw outputs.
    """

    def __init__(
        self,
        *,
        model_path: str,
        nb_classes: int,
        device: str,
        imgsz: ImageSize,
        model_family: Optional[str],
        names: Dict[int, str],
        model_size: Optional[str] = None,
        task: str | None = None,
        supported_tasks=None,
        default_task: str | None = None,
        crop_pct: float | None = None,
        interpolation: str | None = None,
        num_keypoints: int | None = None,
        keypoint_dim: int | None = None,
        num_keypoints_per_class: list[int] | None = None,
        num_bins: int | None = None,
        bin_width_deg: float | None = None,
        offset_deg: float | None = None,
    ):
        self.model_path = model_path
        self.nb_classes = nb_classes
        self.device = device
        self.imgsz = _normalize_imgsz(imgsz)
        self.model_family = model_family
        self.family = model_family
        # DAMO-YOLO was removed; reject its exported artifacts loudly instead of
        # silently mis-parsing them as YOLO9 (DAMO used different pre/post-processing).
        if model_family in _REMOVED_FAMILIES:
            raise ValueError(
                f"model_family={model_family!r} is no longer supported: the "
                f"{model_family} family was removed from LibreYOLO. Re-export this "
                "model with a supported family, or pin an older LibreYOLO release "
                "to run an existing export."
            )
        self.model_size = model_size
        self.DEFAULT_TASK = normalize_task(default_task, default="detect")
        self.SUPPORTED_TASKS = normalize_supported_tasks(
            supported_tasks or (self.DEFAULT_TASK,)
        )
        self.task = resolve_task(
            explicit_task=task,
            default_task=self.DEFAULT_TASK,
            supported_tasks=self.SUPPORTED_TASKS,
        )
        if self.model_family == "yolo9" and self.task == "segment":
            raise NotImplementedError(
                "YOLO9 segmentation support was removed. Use a supported "
                "segmentation family instead of loading YOLO9 segment exports."
            )
        self.names = names
        self.FAMILY = model_family or "export"
        try:
            self.size = model_size or "export"
        except AttributeError:
            # Some concrete backends expose size as a computed read-only property.
            pass
        self.input_size = self.imgsz
        # Classification eval preprocessing (from export metadata); defaults keep
        # legacy behavior. Lets exported-backend classify inference match native.
        self.crop_pct = crop_pct if crop_pct is not None else 0.875
        self.interpolation = interpolation or "bilinear"
        # Set by backends that load a model with NMS baked into the graph; such
        # models emit final (1, max_det, 6) detections instead of raw tensors.
        if not hasattr(self, "embedded_nms"):
            self.embedded_nms = False
        if not hasattr(self, "embedded_nms_raw_output_index"):
            self.embedded_nms_raw_output_index = None
        if num_keypoints is not None:
            self.num_keypoints = int(num_keypoints)
        if keypoint_dim is not None:
            self.keypoint_dim = int(keypoint_dim)
        if num_keypoints_per_class is not None:
            self.num_keypoints_per_class = [
                int(count) for count in num_keypoints_per_class
            ]
        self.num_bins = int(num_bins if num_bins is not None else 90)
        self.bin_width_deg = float(bin_width_deg if bin_width_deg is not None else 4.0)
        self.offset_deg = float(offset_deg if offset_deg is not None else -180.0)
        if not hasattr(self, "model"):
            self.model = _BackendEvalProxy()

    # =========================================================================
    # Abstract interface
    # =========================================================================

    @abstractmethod
    def _run_inference(self, blob: np.ndarray) -> list:
        """Run backend-specific inference.

        Args:
            blob: Preprocessed input array of shape ``(1, C, H, W)``.

        Returns:
            List of numpy arrays, one per model output tensor.
        """

    # =========================================================================
    # Preprocessing
    # =========================================================================

    def _preprocess(self, image, effective_imgsz, color_format):
        """Dispatch to model-family-specific preprocessing.

        Returns:
            Tuple of (input_tensor, original_img, original_size, ratio).
        """
        if self.task == "restore" or self.model_family == "nafnet":
            if self.model_family in {"realesrgan", "quicksrnet"} and not getattr(
                self, "fixed_input_shape", False
            ):
                return self._preprocess_restore_native(image, color_format)
            return self._preprocess_restore(image, effective_imgsz, color_format)
        if self.task == "depth":
            return self._preprocess_depth(image, effective_imgsz, color_format)
        if self.task == "normal":
            return self._preprocess_normal(image, effective_imgsz, color_format)
        if self.task == "edge":
            return self._preprocess_edge(image, effective_imgsz, color_format)
        if self.task == "matte":
            return self._preprocess_matte(image, effective_imgsz, color_format)
        if self.task == "gaze":
            return self._preprocess_gaze(image, effective_imgsz, color_format)
        if self.model_family == "hrnet" and self.task == "pose":
            from ..models.hrnet.utils import preprocess_crop_image

            return preprocess_crop_image(
                image,
                input_size=effective_imgsz,
                color_format=color_format,
            )
        if self.model_family == "dekr" and self.task == "pose":
            # Bottom-up: the whole frame goes in, letterboxed to the square
            # canvas with pad value 127 and ImageNet normalization.
            from ..preprocess.dekr import preprocess_image as dekr_preprocess_image

            return dekr_preprocess_image(
                image,
                input_size=effective_imgsz,
                color_format=color_format,
            )
        if self.model_family in {"vjepa2", "levjepa"}:
            # These video-embedding graphs take a rank-5 clip (B, F, C, H, W). The
            # image preprocessing below produces a rank-4 batch, which the
            # runtime rejects with an opaque "Invalid rank" error. Fail here
            # with something actionable instead. Feeding these graphs a clip
            # through the exported-backend path is not wired up yet.
            raise NotImplementedError(
                f"Exported {self.model_family} graphs take a 5D video clip "
                "(B, F, C, H, W), and LibreYOLO's exported-backend "
                "preprocessing currently supplies a 4D image batch, so "
                f"{self.task!r} inference through this path is not supported "
                "yet. Use the PyTorch checkpoint "
                "with LibreYOLO for clip inference, "
                "or drive the exported graph directly with your own 5D input."
            )
        if self.task in {"classify", "embed"}:
            return self._preprocess_classify(image, effective_imgsz, color_format)
        if self.task == "point" and self.model_family == "fomo":
            from ..models.fomo.utils import preprocess_image as fomo_preprocess_image

            h, w = _imgsz_hw(effective_imgsz)
            if h != w:
                raise NotImplementedError(
                    "FOMO exported inference requires square imgsz."
                )
            return fomo_preprocess_image(image, h, color_format=color_format)
        if self.task == "semantic":
            return self._preprocess_semantic(image, effective_imgsz, color_format)
        if self.model_family == "yolox":
            return yolox_preprocess_image(
                image, input_size=effective_imgsz, color_format=color_format
            )
        elif self.model_family == "yolonas":
            if self.task == "pose":
                return yolonas_preprocess_pose_image(
                    image, input_size=effective_imgsz, color_format=color_format
                )
            if self.task == "obb":
                return yolonas_preprocess_obb_image(
                    image, input_size=effective_imgsz, color_format=color_format
                )
            return yolonas_preprocess_image(
                image, input_size=effective_imgsz, color_format=color_format
            )
        elif self.model_family == "rfdetr":
            tensor, img, size = self._preprocess_rfdetr(
                image,
                effective_imgsz,
                color_format,
                task=self.task,
            )
            return tensor, img, size, 1.0
        elif self.model_family == "lwdetr":
            tensor, img, size = self._preprocess_lwdetr(
                image, effective_imgsz, color_format
            )
            return tensor, img, size, 1.0
        elif self.model_family == "detr":
            tensor, img, size = self._preprocess_detr(
                image, effective_imgsz, color_format
            )
            return tensor, img, size, 1.0
        elif self.model_family in {"faster_rcnn", "mask_rcnn"}:
            return self._preprocess_faster_rcnn(image, effective_imgsz, color_format)
        elif self.model_family == "retinanet":
            from ..models.retinanet.utils import (
                preprocess_image as retinanet_preprocess_image,
            )

            input_h, input_w = _imgsz_hw(effective_imgsz)
            if input_h != input_w:
                raise NotImplementedError(
                    "RetinaNet exported inference requires a scalar or square imgsz."
                )
            return retinanet_preprocess_image(
                image,
                input_size=input_h,
                color_format=color_format,
            )
        elif self.model_family == "ssd":
            from ..models.ssd.utils import preprocess_image as ssd_preprocess_image

            return ssd_preprocess_image(
                image,
                input_size=effective_imgsz,
                color_format=color_format,
            )
        elif self.model_family == "fcos":
            return self._preprocess_fcos(image, effective_imgsz, color_format)
        elif self.model_family in {"deformable_detr", "dinodetr"}:
            tensor, img, size = self._preprocess_deformable_detr(
                image, effective_imgsz, color_format
            )
            return tensor, img, size, 1.0
        elif self.model_family == "centernet":
            return self._preprocess_centernet(image, effective_imgsz, color_format)
        elif self.model_family in ("dfine", "rtdetrv4"):
            tensor, img, size = self._preprocess_dfine(
                image, effective_imgsz, color_format
            )
            return tensor, img, size, 1.0
        elif self.model_family == "deim":
            tensor, img, size = self._preprocess_deim(
                image, effective_imgsz, color_format
            )
            return tensor, img, size, 1.0
        elif self.model_family == "deimv2":
            tensor, img, size = self._preprocess_deimv2(
                image, effective_imgsz, color_format, self.model_size
            )
            return tensor, img, size, 1.0
        elif self.model_family == "tinyformer":
            tensor, img, size = self._preprocess_tinyformer(
                image, effective_imgsz, color_format
            )
            return tensor, img, size, 1.0
        elif self.model_family == "ec":
            tensor, img, size = self._preprocess_ec(
                image, effective_imgsz, color_format
            )
            return tensor, img, size, 1.0
        elif self.model_family in ("rtdetr", "rtdetrv2"):
            if self.model_family == "rtdetrv2" and self.task == "obb":
                return self._preprocess_rtdetrv2_obb(
                    image, effective_imgsz, color_format
                )
            tensor, img, size = self._preprocess_rtdetr(
                image, effective_imgsz, color_format
            )
            return tensor, img, size, 1.0
        elif self.model_family == "ppyoloe":
            from ..models.ppyoloe.utils import (
                preprocess_image as ppyoloe_preprocess_image,
            )

            return ppyoloe_preprocess_image(
                image, input_size=effective_imgsz, color_format=color_format
            )
        elif self.model_family == "picodet":
            tensor, img, size = self._preprocess_picodet(
                image, effective_imgsz, color_format
            )
            return tensor, img, size, 1.0
        elif self.model_family == "efficientdet":
            return self._preprocess_efficientdet(image, effective_imgsz, color_format)
        elif self.model_family == "rtmdet":
            tensor, img, size, ratio = self._preprocess_rtmdet(
                image, effective_imgsz, color_format
            )
            return tensor, img, size, ratio
        elif self.model_family == "yolo1":
            from ..models.darknet.preprocess import (
                preprocess_image_stretch as _v1_pre,
            )

            sz = (
                effective_imgsz
                if isinstance(effective_imgsz, int)
                else max(effective_imgsz)
            )
            return _v1_pre(image, input_size=sz, color_format=color_format)
        elif self.model_family in ("yolo2", "yolo3", "yolo4"):
            from ..models.darknet.preprocess import preprocess_image as _dk_pre

            sz = (
                effective_imgsz
                if isinstance(effective_imgsz, int)
                else max(effective_imgsz)
            )
            return _dk_pre(image, input_size=sz, color_format=color_format)
        elif self.model_family == "yolo7":
            from ..models.yolo7.utils import preprocess_image as _y7_pre

            sz = (
                effective_imgsz
                if isinstance(effective_imgsz, int)
                else max(effective_imgsz)
            )
            return _y7_pre(image, input_size=sz, color_format=color_format)
        else:
            tensor, img, size = preprocess_image(
                image, input_size=effective_imgsz, color_format=color_format
            )
            return tensor, img, size, 1.0

    def _preprocess_classify(self, image, input_size, color_format):
        """Classification preprocessing: ImageNet-style resize/crop/normalize.

        Uses the per-family ``crop_pct``/``interpolation`` recorded in export
        metadata so exported-backend inference matches native predict()/val().
        """
        from ..data.classify_dataset import build_classify_transforms

        h, w = _imgsz_hw(input_size)
        if h != w:
            raise NotImplementedError(
                "Classification exported-backend inference supports square imgsz only."
            )

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        transform_kwargs = {
            "crop_pct": getattr(self, "crop_pct", 0.875),
            "interpolation": getattr(self, "interpolation", "bilinear"),
        }
        if self.model_family == "clip":
            from ..models.clip.model import CLIP_MEAN, CLIP_STD

            transform_kwargs.update(
                mean=CLIP_MEAN,
                std=CLIP_STD,
                crop_pct=1.0,
                interpolation="bicubic",
            )
        elif self.model_family == "siglip2":
            from ..models.siglip2.model import SIGLIP_MEAN, SIGLIP_STD

            transform_kwargs.update(
                mean=SIGLIP_MEAN,
                std=SIGLIP_STD,
                crop_pct=1.0,
                interpolation="bilinear",
                square_resize=True,
            )
        elif self.model_family == "pe":
            # Must mirror LibrePE._build_transform: symmetric [-1, 1]
            # normalization and a bilinear square resize, NOT ImageNet stats.
            from ..models.pe.nn import PE_MEAN, PE_STD

            transform_kwargs.update(
                mean=PE_MEAN,
                std=PE_STD,
                crop_pct=1.0,
                interpolation="bilinear",
                square_resize=True,
            )
        elif self.model_family == "vit":
            from ..models.vit.utils import VIT_MEAN, VIT_STD

            transform_kwargs.update(
                mean=VIT_MEAN,
                std=VIT_STD,
                crop_pct=0.9,
                interpolation="bicubic",
            )
        transform = build_classify_transforms(
            h,
            augment=False,
            **transform_kwargs,
        )
        img_tensor = transform(img).unsqueeze(0)
        return img_tensor, img, original_size, 1.0

    def _preprocess_semantic(self, image, input_size, color_format):
        """Dense semantic preprocessing for fixed-canvas exported graphs."""
        input_h, input_w = _imgsz_hw(input_size)
        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()
        arr = np.asarray(img.convert("RGB"))
        if self.model_family == "pidnet":
            from ..models.pidnet.model import preprocess_numpy

            chw, ratio = preprocess_numpy(arr, (input_h, input_w))
        elif self.model_family == "segformer":
            from ..models.segformer.model import preprocess_numpy

            chw, ratio = preprocess_numpy(arr, (input_h, input_w))
        elif self.model_family == "deeplabv3":
            from ..models.deeplabv3.utils import preprocess_numpy

            chw, ratio = preprocess_numpy(arr, (input_h, input_w))
        else:
            resized = cv2.resize(
                arr, (input_w, input_h), interpolation=cv2.INTER_LINEAR
            )
            chw = np.ascontiguousarray(
                resized.astype(np.float32).transpose(2, 0, 1) / 255.0
            )
            ratio = 1.0
        return (
            torch.from_numpy(chw).unsqueeze(0).float(),
            original_img,
            original_size,
            ratio,
        )

    def _preprocess_matte(self, image, input_size, color_format):
        """Family-specific fixed-canvas ImageNet-normalized matte preprocessing."""
        if self.model_family == "ben2":
            from ..models.ben2.utils import preprocess_numpy
        else:
            from ..models.birefnet.utils import preprocess_numpy

        input_h, input_w = _imgsz_hw(input_size)
        if input_h != input_w:
            raise NotImplementedError(
                "Matte exported-runtime inference requires square imgsz."
            )
        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        chw, ratio = preprocess_numpy(np.asarray(img.convert("RGB")), input_h)
        return (
            torch.from_numpy(chw).unsqueeze(0).float(),
            img.copy(),
            original_size,
            ratio,
        )

    @staticmethod
    def _preprocess_gaze(image, input_size, color_format):
        """Preprocess one already-cropped face for the L2CS gaze head."""
        from ..models.l2cs.utils import preprocess_face_crops

        input_h, input_w = _imgsz_hw(input_size)
        if (input_h, input_w) != (448, 448):
            raise ValueError(
                "L2CS exported inference requires the fixed 448x448 contract."
            )
        img = ImageLoader.load(image, color_format=color_format)
        return preprocess_face_crops([img]), img.copy(), img.size, 1.0

    @staticmethod
    def _preprocess_restore(image, input_size, color_format):
        """Restoration preprocessing for fixed-shape exported runtimes.

        Native NAFNet prediction runs at the input image's own resolution and
        reflect-pads only to the network stride. Exported runtimes use a fixed
        graph shape, so backend prediction accepts images that fit inside the
        exported canvas, pads bottom/right without resizing, and crops the
        restored output back to the original canvas.
        """
        input_h, input_w = _imgsz_hw(input_size)
        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()
        orig_w, orig_h = original_size
        if orig_h > input_h or orig_w > input_w:
            raise ValueError(
                "Restoration exported-runtime inference is fixed-resolution. "
                f"Input image is {orig_w}x{orig_h}, but the exported canvas is "
                f"{input_w}x{input_h}. Use a native .pt model for native-size "
                "large-image prediction, or export a matching fixed size."
            )

        arr = np.asarray(img, dtype=np.float32) / 255.0
        pad_h = input_h - orig_h
        pad_w = input_w - orig_w
        if pad_h or pad_w:
            mode = (
                "reflect"
                if orig_h > 1 and orig_w > 1 and pad_h < orig_h and pad_w < orig_w
                else "edge"
            )
            arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
        img_tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        return img_tensor.unsqueeze(0).float(), original_img, original_size, 1.0

    @staticmethod
    def _preprocess_depth(image, input_size, color_format):
        """Depth preprocessing for fixed-shape exported runtimes.

        Native depth prediction keeps the aspect ratio (short side to the
        model's native resolution). Exported runtimes use a fixed graph shape,
        so backend prediction stretch-resizes to the exported canvas and the
        depth map is resized back to the original canvas after inference
        (ADR 0006). Padding is deliberately avoided: padded pixels would leak
        fake depth context into real pixels through the receptive field.
        """
        input_h, input_w = _imgsz_hw(input_size)
        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()
        arr = np.asarray(img, dtype=np.uint8)
        resized = cv2.resize(arr, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        img_tensor = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)
        return img_tensor, original_img, original_size, 1.0

    @staticmethod
    def _preprocess_normal(image, input_size, color_format):
        """Surface-normal preprocessing for fixed-shape exported runtimes.

        The source aspect ratio must match the exported canvas. Stretching
        would change MoGe-2's image-plane geometry, while padding would change
        the coordinate field seen by the fixed graph. Native ``.pt`` inference
        remains available for arbitrary aspect ratios.
        """
        input_h, input_w = _imgsz_hw(input_size)
        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()
        orig_w, orig_h = original_size
        if orig_w * input_h != input_w * orig_h:
            raise ValueError(
                "Surface-normal exported-runtime inference requires the source "
                "aspect ratio to match the fixed export canvas. "
                f"Input image is {orig_w}x{orig_h}, but the exported canvas is "
                f"{input_w}x{input_h}. Stretching would change image-plane "
                "geometry and normal directions. Use a native .pt model for "
                "arbitrary-aspect-ratio prediction."
            )
        arr = np.asarray(img, dtype=np.uint8)
        resized = cv2.resize(arr, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        chw = resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        img_tensor = torch.from_numpy(np.ascontiguousarray(chw)).unsqueeze(0)
        return img_tensor, original_img, original_size, 1.0

    @staticmethod
    def _preprocess_edge(image, input_size, color_format):
        """Canonical fixed-square RGB preprocessing for edge specialists."""
        from ..models.edge_common import preprocess_numpy

        input_h, input_w = _imgsz_hw(input_size)
        if input_h != input_w:
            raise NotImplementedError(
                "Edge exported-runtime inference requires square imgsz."
            )
        img = ImageLoader.load(image, color_format=color_format).convert("RGB")
        original_size = img.size
        chw, ratio = preprocess_numpy(np.asarray(img), input_h)
        tensor = torch.from_numpy(chw).unsqueeze(0).float()
        return tensor, img.copy(), original_size, ratio

    @property
    def restore_scale(self) -> int:
        """Integer upscale factor for restore backends (1 unless super-resolution)."""

        if self.model_family == "realesrgan":
            return _REALESRGAN_BACKEND_SCALE.get(str(self.model_size), 1)
        if self.model_family == "swinir":
            return _SWINIR_BACKEND_SCALE.get(str(self.model_size), 1)
        if self.model_family == "quicksrnet":
            return _QUICKSRNET_BACKEND_SCALE.get(str(self.model_size), 1)
        return 1

    def _preprocess_restore_native(self, image, color_format):
        """Native-resolution preprocessing for dynamic super-resolution graphs.

        Loads RGB [0, 1], reflect-pads bottom/right to the network divisibility
        factor (2 for Real-ESRGAN x2, 1 otherwise). The dynamic
        ONNX graph accepts any spatial size, so no fixed canvas is imposed.
        """

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()
        arr = np.asarray(img, dtype=np.float32) / 255.0
        multiple = _REALESRGAN_BACKEND_PAD_MULTIPLE.get(str(self.model_size), 1)
        if multiple > 1:
            orig_h, orig_w = arr.shape[:2]
            pad_h = (multiple - orig_h % multiple) % multiple
            pad_w = (multiple - orig_w % multiple) % multiple
            if pad_h or pad_w:
                mode = (
                    "reflect"
                    if orig_h > 1 and orig_w > 1 and pad_h < orig_h and pad_w < orig_w
                    else "edge"
                )
                arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)
        img_tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))
        return img_tensor.unsqueeze(0).float(), original_img, original_size, 1.0

    @staticmethod
    def _preprocess_rfdetr(image, input_size, color_format, task=None):
        """RF-DETR preprocessing: direct resize + ImageNet normalization."""
        from ..preprocess.rfdetr import (
            IMAGENET_MEAN,
            IMAGENET_STD,
            preprocess_numpy as rfdetr_preprocess_numpy,
        )

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size  # (W, H)
        original_img = img.copy()

        if task == "pose":
            h, w = _imgsz_hw(input_size)
            arr = np.asarray(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            img_tensor = F.interpolate(
                img_tensor,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
            return (img_tensor - mean) / std, original_img, original_size

        img_chw, _ = rfdetr_preprocess_numpy(np.array(img), input_size)
        img_tensor = as_batched_input(img_chw)
        return img_tensor, original_img, original_size

    @staticmethod
    def _preprocess_lwdetr(image, input_size, color_format):
        """LW-DETR preprocessing: square resize + RGB + ImageNet mean/std."""
        from ..models.lwdetr.utils import preprocess_numpy as lwdetr_preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()

        img_chw, _ = lwdetr_preprocess_numpy(np.array(img), input_size)
        img_tensor = as_batched_input(img_chw)

        return img_tensor, original_img, original_size

    def _preprocess_faster_rcnn(self, image, input_size, color_format):
        """Feed raw RGB pixels to the in-graph GeneralizedRCNNTransform.

        Default Faster R-CNN ONNX exports have dynamic spatial axes, so the
        source image must not be resized, letterboxed, or ImageNet-normalized
        here.  The graph owns all three operations and returns boxes in source
        coordinates.  A fixed-shape artifact is retained as a compatibility
        fallback: it uses an explicit stretch resize whose independent x/y
        inverse is handled by ``_parse_faster_rcnn``.
        """
        if getattr(self, "_dynamic_spatial_axes", False):
            from ..models.faster_rcnn.utils import preprocess_image

            return preprocess_image(image, color_format=color_format)

        input_h, input_w = _imgsz_hw(input_size)
        img = ImageLoader.load(image, color_format=color_format).convert("RGB")
        original_size = img.size
        resized = cv2.resize(
            np.asarray(img), (input_w, input_h), interpolation=cv2.INTER_LINEAR
        )
        chw = np.ascontiguousarray(
            resized.astype(np.float32).transpose(2, 0, 1) / 255.0
        )
        return torch.from_numpy(chw).unsqueeze(0), img.copy(), original_size, 1.0

    @staticmethod
    def _preprocess_fcos(image, input_size, color_format):
        """Apply the out-of-graph torchvision FCOS transform."""
        from ..models.fcos.utils import preprocess_image

        input_h, input_w = _imgsz_hw(input_size)
        if input_h != input_w:
            raise ValueError(f"FCOS requires a scalar/square imgsz, got {input_size}")
        return preprocess_image(
            image,
            input_size=input_h,
            color_format=color_format,
        )

    @staticmethod
    def _preprocess_deformable_detr(image, input_size, color_format):
        """Deformable DETR preprocessing: square resize and ImageNet norm."""
        from ..models.deformable_detr.utils import preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()

        img_chw, _ = preprocess_numpy(np.array(img), input_size)
        img_tensor = as_batched_input(img_chw)
        return img_tensor, original_img, original_size

    @staticmethod
    def _preprocess_centernet(image, input_size, color_format):
        """CenterNet preprocessing: centered BGR affine warp and normalization."""
        from ..models.centernet.utils import preprocess_image

        input_h, input_w = _imgsz_hw(input_size)
        if input_h != input_w:
            raise NotImplementedError(
                "CenterNet exported inference requires a square input canvas."
            )
        return preprocess_image(image, input_size=input_h, color_format=color_format)

    @staticmethod
    def _preprocess_detr(image, input_size, color_format):
        """DETR preprocessing: square resize + RGB + ImageNet mean/std."""
        from ..models.detr.utils import preprocess_numpy as detr_preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()

        img_chw, _ = detr_preprocess_numpy(np.asarray(img), input_size)
        img_tensor = as_batched_input(img_chw)

        return img_tensor, original_img, original_size

    @staticmethod
    def _preprocess_dfine(image, input_size, color_format):
        """D-FINE preprocessing: plain resize + RGB + /255, no ImageNet norm."""
        from ..preprocess.dfine import preprocess_numpy as dfine_preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()

        img_chw, _ = dfine_preprocess_numpy(np.array(img), input_size)
        img_tensor = as_batched_input(img_chw)

        return img_tensor, original_img, original_size

    @staticmethod
    def _preprocess_deim(image, input_size, color_format):
        """DEIM-D-FINE preprocessing: plain resize + RGB + /255."""
        from ..preprocess.deim import preprocess_numpy as deim_preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()

        img_chw, _ = deim_preprocess_numpy(np.array(img), input_size)
        img_tensor = as_batched_input(img_chw)

        return img_tensor, original_img, original_size

    @staticmethod
    def _preprocess_deimv2(image, input_size, color_format, model_size=None):
        """DEIMv2 preprocessing; DINO-backed sizes use ImageNet normalization."""
        from ..preprocess.deimv2 import DINO_SIZES
        from ..preprocess.deimv2 import preprocess_numpy as deimv2_preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()

        img_chw, _ = deimv2_preprocess_numpy(
            np.array(img), input_size, imagenet_norm=model_size in DINO_SIZES
        )
        img_tensor = as_batched_input(img_chw)

        return img_tensor, original_img, original_size

    @staticmethod
    def _preprocess_tinyformer(image, input_size, color_format):
        """TinyFormer preprocessing; every size uses ImageNet normalization."""
        from ..preprocess.deimv2 import preprocess_numpy as deimv2_preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()

        img_chw, _ = deimv2_preprocess_numpy(
            np.array(img), input_size, imagenet_norm=True
        )
        img_tensor = as_batched_input(img_chw)

        return img_tensor, original_img, original_size

    @staticmethod
    def _preprocess_ec(image, input_size, color_format):
        """EC preprocessing: plain resize + RGB + /255 + ImageNet (mean, std)."""
        from ..preprocess.ec import preprocess_numpy as ec_preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()

        img_chw, _ = ec_preprocess_numpy(np.array(img), input_size)
        img_tensor = as_batched_input(img_chw)
        return img_tensor, original_img, original_size

    @staticmethod
    def _preprocess_picodet(image, input_size, color_format):
        """PICODET preprocessing: simple resize + RGB + ImageNet mean/std (0-255 space)."""
        from ..models.picodet.utils import preprocess_numpy as picodet_preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()

        img_chw, _ = picodet_preprocess_numpy(np.array(img), input_size)
        img_tensor = as_batched_input(img_chw)
        return img_tensor, original_img, original_size

    @staticmethod
    def _preprocess_efficientdet(image, input_size, color_format):
        """EfficientDet top-left resize-pad plus ImageNet normalization."""
        from ..models.efficientdet.utils import preprocess_image

        return preprocess_image(image, input_size=input_size, color_format=color_format)

    @staticmethod
    def _preprocess_rtmdet(image, input_size, color_format):
        """RTMDet preprocessing: BGR letterbox + mmdet mean/std normalization."""
        from ..models.rtmdet.utils import preprocess_numpy as rtmdet_preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size  # (W, H)
        original_img = img.copy()

        img_chw, ratio = rtmdet_preprocess_numpy(np.array(img), input_size)
        img_tensor = as_batched_input(img_chw)
        return img_tensor, original_img, original_size, ratio

    @staticmethod
    def _preprocess_rtdetr(image, input_size, color_format):
        """RT-DETR preprocessing: direct resize + normalize to [0,1]."""
        from ..preprocess.rtdetr import preprocess_numpy as rtdetr_preprocess_numpy

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size  # (W, H)
        original_img = img.copy()

        img_chw, _ = rtdetr_preprocess_numpy(np.array(img), input_size)
        img_tensor = as_input(img_chw)
        return img_tensor, original_img, original_size

    @staticmethod
    def _preprocess_rtdetrv2_obb(image, input_size, color_format):
        """RT-DETRv2 OBB uniform resize with bottom/right zero padding."""
        from PIL import Image

        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size
        original_img = img.copy()
        orig_w, orig_h = original_size
        target_h, target_w = _imgsz_hw(input_size)
        ratio = min(target_w / orig_w, target_h / orig_h)
        new_w = max(1, int(round(orig_w * ratio)))
        new_h = max(1, int(round(orig_h * ratio)))
        resized = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (target_w, target_h), color=0)
        canvas.paste(resized, (0, 0))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy()).unsqueeze(0)
        return tensor, original_img, original_size, ratio

    # =========================================================================
    # Output parsing
    # =========================================================================

    def _parse_outputs(
        self,
        all_outputs: list,
        effective_imgsz: ImageSize,
        original_size: tuple,
        conf: float,
        ratio: float | None = None,
        iou: float = 0.45,
        max_det: int = 300,
    ):
        """Parse raw outputs into boxes, scores, classes, masks, OBB, and keypoints."""
        orig_w, orig_h = original_size

        if getattr(self, "embedded_nms", False):
            raw_index = getattr(self, "embedded_nms_raw_output_index", None)
            if (
                self.model_family == "yolo9"
                and isinstance(raw_index, int)
                and raw_index < len(all_outputs)
            ):
                boxes, scores, cls = self._parse_yolo9(
                    [all_outputs[raw_index]],
                    effective_imgsz,
                    orig_w,
                    orig_h,
                    conf,
                    iou=iou,
                    max_det=max_det,
                )
                return boxes, scores, cls, None
            boxes, scores, cls = self._parse_embedded_nms(
                all_outputs, effective_imgsz, orig_w, orig_h, conf
            )
            return boxes, scores, cls, None

        if self.model_family == "hrnet" and self.task == "pose":
            return self._parse_hrnet_pose(
                all_outputs,
                effective_imgsz,
                original_size,
                max_det=max_det,
            )

        if self.model_family == "dekr" and self.task == "pose":
            return self._parse_dekr_pose(
                all_outputs,
                original_size,
                conf,
                ratio=ratio,
                max_det=max_det,
            )

        if self.model_family == "yolox":
            boxes, scores, cls = self._parse_yolox(
                all_outputs, effective_imgsz, orig_w, orig_h, conf, ratio
            )
            return boxes, scores, cls, None
        elif self.model_family == "yolonas":
            if self.task == "pose":
                return self._parse_yolonas_pose(
                    all_outputs,
                    effective_imgsz,
                    orig_w,
                    orig_h,
                    conf,
                    ratio=ratio,
                    max_det=max_det,
                )
            if self.task == "obb":
                return self._parse_yolonas_obb(
                    all_outputs,
                    effective_imgsz,
                    orig_w,
                    orig_h,
                    conf,
                    iou=iou,
                    ratio=ratio,
                    max_det=max_det,
                )
            boxes, scores, cls = self._parse_yolonas(
                all_outputs, effective_imgsz, orig_w, orig_h, conf, ratio=ratio
            )
            return boxes, scores, cls, None
        elif self.model_family == "rfdetr":
            return self._parse_rfdetr(
                all_outputs,
                orig_w,
                orig_h,
                conf,
                max_det=max_det,
            )
        elif self.model_family == "lwdetr":
            boxes, scores, cls = self._parse_lwdetr(
                all_outputs, orig_w, orig_h, conf, max_det=max_det
            )
            return boxes, scores, cls, None
        elif self.model_family == "detr":
            boxes, scores, cls = self._parse_detr(
                all_outputs, orig_w, orig_h, conf, max_det=max_det
            )
            return boxes, scores, cls, None
        elif self.model_family == "mask_rcnn":
            return self._parse_mask_rcnn(
                all_outputs, effective_imgsz, orig_w, orig_h, conf
            )
        elif self.model_family == "faster_rcnn":
            boxes, scores, cls = self._parse_faster_rcnn(
                all_outputs, effective_imgsz, orig_w, orig_h, conf
            )
            return boxes, scores, cls, None
        elif self.model_family == "retinanet":
            boxes, scores, cls = self._parse_retinanet(
                all_outputs,
                effective_imgsz,
                orig_w,
                orig_h,
                conf,
            )
            return boxes, scores, cls, None
        elif self.model_family == "fcos":
            boxes, scores, cls = self._parse_fcos(
                all_outputs,
                effective_imgsz,
                orig_w,
                orig_h,
                conf,
            )
            return boxes, scores, cls, None
        elif self.model_family == "ssd":
            boxes, scores, cls = self._parse_ssd(
                all_outputs,
                effective_imgsz,
                orig_w,
                orig_h,
                conf,
                iou=iou,
                max_det=max_det,
            )
            return boxes, scores, cls, None
        elif self.model_family in {"deformable_detr", "dinodetr"}:
            boxes, scores, cls = self._parse_deformable_detr(
                all_outputs, orig_w, orig_h, conf, max_det=max_det
            )
            return boxes, scores, cls, None
        elif self.model_family == "centernet":
            boxes, scores, cls = self._parse_centernet(
                all_outputs,
                effective_imgsz,
                orig_w,
                orig_h,
                conf,
                max_det=max_det,
            )
            return boxes, scores, cls, None
        elif self.model_family in ("dfine", "rtdetrv4"):
            if self.model_family == "dfine" and self.task == "segment":
                return self._parse_dfine_segment(
                    all_outputs, orig_w, orig_h, conf, max_det=max_det
                )
            boxes, scores, cls = self._parse_dfine(
                all_outputs, orig_w, orig_h, conf, max_det=max_det
            )
            return boxes, scores, cls, None
        elif self.model_family == "deim":
            boxes, scores, cls = self._parse_dfine(
                all_outputs, orig_w, orig_h, conf, max_det=max_det
            )
            return boxes, scores, cls, None
        elif self.model_family == "deimv2":
            boxes, scores, cls = self._parse_dfine(
                all_outputs, orig_w, orig_h, conf, max_det=max_det
            )
            return boxes, scores, cls, None
        elif self.model_family == "tinyformer":
            boxes, scores, cls = self._parse_dfine(
                all_outputs, orig_w, orig_h, conf, max_det=max_det
            )
            return boxes, scores, cls, None
        elif self.model_family == "ec":
            if self.task == "segment":
                return self._parse_ec_segment(
                    all_outputs, orig_w, orig_h, conf, max_det=max_det
                )
            if self.task == "pose":
                return self._parse_ec_pose(
                    all_outputs, orig_w, orig_h, conf, max_det=max_det
                )
            boxes, scores, cls = self._parse_dfine(
                all_outputs, orig_w, orig_h, conf, max_det=max_det
            )
            return boxes, scores, cls, None
        elif self.model_family in ("rtdetr", "rtdetrv2"):
            if self.model_family == "rtdetrv2" and self.task == "obb":
                return self._parse_rtdetr_obb(
                    all_outputs,
                    effective_imgsz,
                    orig_w,
                    orig_h,
                    conf,
                    max_det=max_det,
                )
            boxes, scores, cls = self._parse_rtdetr(
                all_outputs, orig_w, orig_h, conf, max_det=max_det
            )
            return boxes, scores, cls, None
        elif self.model_family == "ppyoloe":
            boxes, scores, cls = self._parse_ppyoloe(
                all_outputs, effective_imgsz, orig_w, orig_h, conf
            )
            return boxes, scores, cls, None
        elif self.model_family == "picodet":
            boxes, scores, cls = self._parse_picodet(
                all_outputs, effective_imgsz, orig_w, orig_h, conf
            )
            return boxes, scores, cls, None
        elif self.model_family == "efficientdet":
            boxes, scores, cls = self._parse_efficientdet(
                all_outputs,
                effective_imgsz,
                orig_w,
                orig_h,
                conf,
                ratio,
            )
            return boxes, scores, cls, None
        elif self.model_family == "rtmdet":
            boxes, scores, cls = self._parse_rtmdet(
                all_outputs, effective_imgsz, orig_w, orig_h, conf, ratio
            )
            return boxes, scores, cls, None
        else:
            parsed = self._parse_yolo9(
                all_outputs, effective_imgsz, orig_w, orig_h, conf, iou, max_det
            )
            if len(parsed) == 6:
                return parsed
            if len(parsed) == 5:
                return parsed
            if len(parsed) == 4:
                return parsed
            boxes, scores, cls = parsed
            return boxes, scores, cls, None

    @staticmethod
    def _parse_hrnet_pose(
        all_outputs: list,
        effective_imgsz: ImageSize,
        original_size: Tuple[int, int],
        *,
        max_det: int,
    ):
        """Decode one exported HRNet person-crop heatmap tensor."""
        if len(all_outputs) != 1:
            raise ValueError(
                "HRNet pose backend requires one heatmap output, "
                f"got {len(all_outputs)} outputs."
            )
        from ..models.hrnet.utils import box_to_center_scale
        from ..postprocess.hrnet import postprocess_hrnet

        original_width, original_height = original_size
        box = np.asarray(
            [[0.0, 0.0, float(original_width), float(original_height)]],
            dtype=np.float32,
        )
        center, scale = box_to_center_scale(box[0], effective_imgsz)
        decoded = postprocess_hrnet(
            np.asarray(all_outputs[0], dtype=np.float32),
            centers=center[None, :],
            scales=scale[None, :],
            boxes=box,
            box_scores=np.ones((1,), dtype=np.float32),
            keypoint_threshold=0.2,
            oks_threshold=0.9,
            max_det=min(int(max_det), 1),
        )
        return (
            decoded["boxes"],
            decoded["scores"],
            decoded["classes"],
            None,
            None,
            decoded["keypoints"],
        )

    @staticmethod
    def _parse_dekr_pose(
        all_outputs: list,
        original_size: Tuple[int, int],
        conf: float,
        *,
        ratio: float | None,
        max_det: int,
    ):
        """Decode exported DEKR heatmap logits and offsets into flat poses.

        The graph emits only the two dense maps; centre extraction, offset
        decode, pose NMS and the derived-box adapter all run here, so an
        exported artifact returns the same Results as the native model.
        """
        if len(all_outputs) != 2:
            raise ValueError(
                "DEKR pose backend requires two outputs (heatmap_logits, "
                f"offsets), got {len(all_outputs)}."
            )
        from ..postprocess.dekr import DEKR_MAX_NUM_PEOPLE, postprocess_dekr

        heatmap = torch.as_tensor(np.asarray(all_outputs[0], dtype=np.float32))
        offsets = torch.as_tensor(np.asarray(all_outputs[1], dtype=np.float32))
        decoded = postprocess_dekr(
            (heatmap, offsets),
            conf_thres=float(conf),
            original_size=original_size,
            ratio=float(ratio if ratio else 1.0),
            max_det=min(int(max_det), DEKR_MAX_NUM_PEOPLE),
        )
        return (
            decoded["boxes"],
            decoded["scores"],
            decoded["classes"],
            None,
            None,
            decoded["keypoints"],
        )

    def _parse_yolox(
        self, all_outputs, effective_imgsz, orig_w, orig_h, conf, ratio=1.0
    ):
        """Parse YOLOX output: (B, N, 5+nc) — cxcywh + objectness + class_scores."""
        outputs = all_outputs[0][0]  # (N, 5+nc)

        cx, cy, w, h = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]
        objectness = outputs[:, 4]
        class_scores = outputs[:, 5:]

        max_class_scores = np.max(class_scores, axis=1)
        max_scores = objectness * max_class_scores
        class_ids = np.argmax(class_scores, axis=1)

        mask = max_scores > conf
        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        max_scores, class_ids = max_scores[mask], class_ids[mask]

        if len(max_scores) == 0:
            return np.empty((0, 4)), max_scores, class_ids

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        if ratio is None or ratio == 1.0:
            input_h, input_w = _imgsz_hw(effective_imgsz)
            ratio = min(input_h / orig_h, input_w / orig_w)
        boxes /= ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_boxes]
        max_scores = max_scores[valid_boxes]
        class_ids = class_ids[valid_boxes]

        return boxes, max_scores, class_ids

    def _parse_rtmdet(
        self, all_outputs, effective_imgsz, orig_w, orig_h, conf, ratio=1.0
    ):
        """Parse RTMDet export-mode output: (B, N, 4 + nc) — xyxy (input-canvas pixels) + sigmoid scores.

        RTMDet exports use letterbox preprocessing, so the inverse scale is a
        single ``ratio`` (aspect-preserving), like YOLOX.
        """
        outputs = all_outputs[0][0]  # (N, 4 + nc)
        boxes_all = outputs[:, :4]
        scores = outputs[:, 4:]

        valid = scores > conf
        if not valid.any():
            return (
                np.empty((0, 4), dtype=boxes_all.dtype),
                np.empty((0,), dtype=scores.dtype),
                np.empty((0,), dtype=np.int64),
            )

        box_indices, class_ids = np.nonzero(valid)
        max_scores = scores[box_indices, class_ids]

        input_h, input_w = _imgsz_hw(effective_imgsz)
        strides = (8, 16, 32)
        level_sizes = [
            int(np.ceil(input_h / stride)) * int(np.ceil(input_w / stride))
            for stride in strides
        ]
        level_offsets = np.cumsum([0, *level_sizes])
        if level_offsets[-1] == boxes_all.shape[0]:
            nms_pre = 30000
            keep_parts = []
            for start, end in zip(level_offsets[:-1], level_offsets[1:]):
                level_mask = (box_indices >= start) & (box_indices < end)
                level_indices = np.nonzero(level_mask)[0]
                if level_indices.size > nms_pre:
                    level_scores = max_scores[level_indices]
                    keep = np.argpartition(-level_scores, nms_pre - 1)[:nms_pre]
                    keep = keep[np.argsort(-level_scores[keep])]
                    level_indices = level_indices[keep]
                keep_parts.append(level_indices)
            keep_indices = (
                np.concatenate(keep_parts)
                if keep_parts
                else np.empty((0,), dtype=np.int64)
            )
        else:
            nms_pre = min(30000, max_scores.size)
            keep_indices = np.argpartition(-max_scores, nms_pre - 1)[:nms_pre]
            keep_indices = keep_indices[np.argsort(-max_scores[keep_indices])]

        box_indices = box_indices[keep_indices]
        max_scores = max_scores[keep_indices]
        class_ids = class_ids[keep_indices]
        boxes = boxes_all[box_indices].astype(np.float32, copy=True)

        if len(boxes) == 0:
            return boxes, max_scores, class_ids

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, input_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, input_h)
        if ratio is None or ratio == 1.0:
            ratio = min(input_h / orig_h, input_w / orig_w)
        boxes = boxes / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_boxes]
        max_scores = max_scores[valid_boxes]
        class_ids = class_ids[valid_boxes]

        return boxes, max_scores, class_ids

    def _parse_picodet(self, all_outputs, effective_imgsz, orig_w, orig_h, conf):
        """Parse PICODET output: (B, N, 4+nc) — xyxy (input-canvas pixels) + sigmoid scores.

        PICODET exports use simple resize (not letterbox), so the inverse
        scale is independent x/y ratios from input canvas back to the
        original image.
        """
        outputs = all_outputs[0][0]  # (N, 4+nc)
        boxes_all = outputs[:, :4]
        scores = outputs[:, 4:]

        # Multi-label per anchor (every (anchor, class) pair above conf), matching the native
        # postprocess (postprocess/picodet.py). argmax kept only the best class per anchor and
        # dropped secondary-class detections, costing ~0.7 mAP vs native.
        valid = scores > conf
        if not valid.any():
            return (
                np.empty((0, 4), dtype=boxes_all.dtype),
                np.empty((0,), dtype=scores.dtype),
                np.empty((0,), dtype=np.int64),
            )

        box_indices, class_ids = np.nonzero(valid)
        max_scores = scores[box_indices, class_ids]

        # Per-level top-k (nms_pre), matching native postprocess/picodet.py: each FPN level is
        # capped separately so a busy level can't crowd out detections from other levels. The
        # exported output concatenates the 4 PicoDet levels (strides 8/16/32/64) in order, so we
        # map each candidate's anchor index to its level via the cumulative grid sizes. Falls back
        # to a single global cap if the layout doesn't match (unexpected stride/imgsz). The cap
        # also keeps numpy NMS fast (the uncapped multi-label flood at conf=0.001 was ~1.6-12 s/img).
        nms_pre = 1000
        # Ceil division: feature maps from stride-2 convs round up, so e.g. PicoDet-m (416) has a
        # 7x7 stride-64 P6 (416//64=6 would mismatch N and silently fall back to the global cap).
        level_sizes = [((effective_imgsz + s - 1) // s) ** 2 for s in (8, 16, 32, 64)]
        if sum(level_sizes) == scores.shape[0]:
            bounds = np.cumsum([0] + level_sizes)
            keep = []
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                idx = np.nonzero((box_indices >= lo) & (box_indices < hi))[0]
                if idx.size > nms_pre:
                    idx = idx[np.argpartition(max_scores[idx], -nms_pre)[-nms_pre:]]
                keep.append(idx)
            keep = np.concatenate(keep) if keep else np.empty(0, dtype=np.int64)
            box_indices, class_ids, max_scores = (
                box_indices[keep],
                class_ids[keep],
                max_scores[keep],
            )
        elif max_scores.shape[0] > nms_pre:
            top = np.argpartition(max_scores, -nms_pre)[-nms_pre:]
            box_indices, class_ids, max_scores = (
                box_indices[top],
                class_ids[top],
                max_scores[top],
            )

        boxes = boxes_all[box_indices].copy()

        scale_x = orig_w / effective_imgsz
        scale_y = orig_h / effective_imgsz
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        return boxes, max_scores, class_ids

    @staticmethod
    def _parse_ppyoloe(all_outputs, effective_imgsz, orig_w, orig_h, conf):
        """Parse PP-YOLOE output: boxes ``(B, A, 4)`` xyxy + scores ``(B, A, C)``.

        The graph emits both tensors already decoded into input-canvas pixels,
        so this only mirrors the native selection sequence: multi-label
        confidence filter, pre-NMS top-k, then the stretch-resize inverse with
        independent x and y scales. NMS runs downstream in the shared path.
        """
        first = np.asarray(all_outputs[0])[0]
        second = np.asarray(all_outputs[1])[0]
        if first.shape[-1] == 4:
            boxes_all, scores = first, second
        else:
            boxes_all, scores = second, first

        # Multi-label per anchor, matching postprocess/ppyoloe.py: an anchor
        # with two strong classes emits both candidates.
        valid = scores > conf
        if not valid.any():
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
        box_indices, class_ids = np.nonzero(valid)
        max_scores = scores[box_indices, class_ids]

        nms_top_k = PPYOLOE_PRE_NMS_TOP_K
        if max_scores.shape[0] > nms_top_k:
            top = np.argpartition(max_scores, -nms_top_k)[-nms_top_k:]
            box_indices, class_ids, max_scores = (
                box_indices[top],
                class_ids[top],
                max_scores[top],
            )

        boxes = boxes_all[box_indices].astype(np.float32, copy=True)
        input_h, input_w = _imgsz_hw(effective_imgsz)
        boxes[:, [0, 2]] *= orig_w / input_w
        boxes[:, [1, 3]] *= orig_h / input_h
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        return boxes, max_scores, class_ids.astype(np.int64)

    @staticmethod
    def _parse_efficientdet(
        all_outputs, effective_imgsz, orig_w, orig_h, conf, ratio=1.0
    ):
        """Parse ``(B, 5000, 6)`` decoded EfficientDet candidates."""
        candidates = np.asarray(all_outputs[0])[0]
        keep = (candidates[:, 4] > conf) & (candidates[:, 5] >= 0)
        candidates = candidates[keep]
        if candidates.shape[0] == 0:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

        input_h, input_w = _imgsz_hw(effective_imgsz)
        boxes = candidates[:, :4].astype(np.float32, copy=True)
        scores = candidates[:, 4].astype(np.float32, copy=False)
        class_ids = candidates[:, 5].astype(np.int64, copy=False)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, input_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, input_h)
        if ratio is None or ratio <= 0:
            ratio = min(input_h / orig_h, input_w / orig_w)
        boxes /= ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes[valid], scores[valid], class_ids[valid]

    def _parse_embedded_nms(self, all_outputs, effective_imgsz, orig_w, orig_h, conf):
        """Parse a graph-embedded-NMS detection output.

        Shape ``(1, max_det, 6)`` with rows ``[x1, y1, x2, y2, score, class]`` in
        input-canvas (letterbox) pixels. NMS already ran in the graph; here we
        drop zero-padding / sub-``conf`` rows and undo the letterbox scaling.
        """
        det = np.asarray(all_outputs[0], dtype=np.float32)
        if det.ndim == 3:
            det = det[0]  # (max_det, 6)
        keep = det[:, 4] > conf
        det = det[keep]
        if det.shape[0] == 0:
            empty = np.empty((0, 4), dtype=np.float32)
            return empty, np.empty((0,), np.float32), np.empty((0,), np.int64)

        boxes = det[:, :4].copy()
        scores = det[:, 4].astype(np.float32)
        class_ids = det[:, 5].astype(np.int64)

        input_h, input_w = _imgsz_hw(effective_imgsz)
        ratio = min(input_h / orig_h, input_w / orig_w)
        boxes /= ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]
        return boxes, scores, class_ids

    def _parse_faster_rcnn(
        self,
        all_outputs,
        effective_imgsz,
        orig_w,
        orig_h,
        conf,
    ):
        """Parse final, already-NMSed boxes emitted by the export wrapper."""
        boxes = np.asarray(all_outputs[0], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(all_outputs[1], dtype=np.float32).reshape(-1)
        class_ids = np.asarray(all_outputs[2], dtype=np.int64).reshape(-1)
        keep = scores > conf
        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
        if not len(boxes):
            return boxes, scores, class_ids

        boxes = boxes.copy()
        if not getattr(self, "_dynamic_spatial_axes", False):
            input_h, input_w = _imgsz_hw(effective_imgsz)
            boxes[:, [0, 2]] *= orig_w / input_w
            boxes[:, [1, 3]] *= orig_h / input_h
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes[valid], scores[valid], class_ids[valid]

    def _parse_retinanet(
        self,
        all_outputs,
        effective_imgsz,
        orig_w,
        orig_h,
        conf,
    ):
        """Select RetinaNet candidates per FPN level before backend NMS."""
        from ..postprocess.retinanet import level_anchor_counts, resize_geometry

        predictions = np.asarray(all_outputs[0], dtype=np.float32)
        if predictions.ndim == 3:
            if predictions.shape[0] != 1:
                raise ValueError(
                    "RetinaNet exported inference currently requires batch=1, "
                    f"got batch={predictions.shape[0]}"
                )
            predictions = predictions[0]
        if predictions.ndim != 2 or predictions.shape[1] < 5:
            raise ValueError(
                "RetinaNet output must have shape (1, anchors, 4 + classes), "
                f"got {np.asarray(all_outputs[0]).shape}"
            )

        input_h, input_w = _imgsz_hw(effective_imgsz)
        if input_h != input_w:
            raise NotImplementedError(
                "RetinaNet exported inference requires a scalar or square imgsz."
            )
        resized_h, resized_w, scale_x, scale_y = resize_geometry(
            (orig_w, orig_h), input_h
        )
        counts = level_anchor_counts(resized_h, resized_w)
        if sum(counts) != predictions.shape[0]:
            raise ValueError(
                "RetinaNet anchor count does not match preprocessing geometry: "
                f"output has {predictions.shape[0]}, expected {sum(counts)}."
            )

        selected_boxes = []
        selected_scores = []
        selected_classes = []
        offset = 0
        for count in counts:
            level = predictions[offset : offset + count]
            offset += count
            boxes, scores = level[:, :4], level[:, 4:]
            anchor_ids, class_ids = np.nonzero(scores > conf)
            if not len(anchor_ids):
                continue
            level_scores = scores[anchor_ids, class_ids]
            keep_count = min(1000, len(level_scores))
            if keep_count < len(level_scores):
                keep = np.argpartition(-level_scores, keep_count - 1)[:keep_count]
                level_scores = level_scores[keep]
                anchor_ids = anchor_ids[keep]
                class_ids = class_ids[keep]
            order = np.argsort(-level_scores, kind="stable")
            level_boxes = boxes[anchor_ids[order]].copy()
            level_boxes[:, [0, 2]] = np.clip(level_boxes[:, [0, 2]], 0, resized_w)
            level_boxes[:, [1, 3]] = np.clip(level_boxes[:, [1, 3]], 0, resized_h)
            selected_boxes.append(level_boxes)
            selected_scores.append(level_scores[order])
            selected_classes.append(class_ids[order].astype(np.int64, copy=False))

        if not selected_boxes:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )

        boxes = np.concatenate(selected_boxes)
        scores = np.concatenate(selected_scores).astype(np.float32, copy=False)
        class_ids = np.concatenate(selected_classes)
        finite = np.isfinite(boxes).all(axis=1) & np.isfinite(scores)
        boxes, scores, class_ids = boxes[finite], scores[finite], class_ids[finite]
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes, scores, class_ids = boxes[valid], scores[valid], class_ids[valid]
        boxes[:, [0, 2]] /= scale_x
        boxes[:, [1, 3]] /= scale_y
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        return boxes, scores, class_ids

    def _parse_ssd(
        self,
        all_outputs,
        effective_imgsz,
        orig_w,
        orig_h,
        conf,
        iou: float = 0.45,
        max_det: int = 300,
    ):
        """Parse SSD's decoded YOLO-grid tensor before shared class-wise NMS."""
        del iou, max_det
        input_h, input_w = _imgsz_hw(effective_imgsz)
        if (input_h, input_w) != (300, 300):
            raise ValueError(
                "SSD exported inference requires the fixed 300 x 300 canvas"
            )

        packed = np.asarray(all_outputs[0], dtype=np.float32)
        if packed.ndim == 3 and packed.shape[0] == 1:
            packed = packed[0]
        if packed.ndim != 2:
            raise ValueError("SSD backend expects one packed rank-3 output")
        if packed.shape[1] == 8732:
            packed = packed.T
        if packed.shape[0] != 8732 or packed.shape[1] != 4 + self.nb_classes:
            raise ValueError(
                f"SSD backend output must have shape (1, {4 + self.nb_classes}, 8732)"
            )

        boxes_all = packed[:, :4]
        scores_all = packed[:, 4:]
        image_boxes = []
        image_scores = []
        image_classes = []
        for class_id in range(scores_all.shape[1]):
            class_scores = scores_all[:, class_id]
            indices = np.flatnonzero(class_scores > conf)
            if indices.size > 400:
                top = np.argpartition(-class_scores[indices], 399)[:400]
                indices = indices[top]
            image_boxes.append(boxes_all[indices])
            image_scores.append(class_scores[indices])
            image_classes.append(np.full(indices.shape, class_id, dtype=np.int64))

        boxes = np.concatenate(image_boxes, axis=0).astype(np.float32, copy=True)
        scores = np.concatenate(image_scores, axis=0).astype(np.float32, copy=False)
        classes = np.concatenate(image_classes, axis=0)
        if not len(boxes):
            return boxes, scores, classes

        boxes[:, [0, 2]] *= orig_w / input_w
        boxes[:, [1, 3]] *= orig_h / input_h
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes[valid], scores[valid], classes[valid]

    def _parse_mask_rcnn(
        self,
        all_outputs,
        effective_imgsz,
        orig_w,
        orig_h,
        conf,
    ):
        """Parse aligned boxes and full-image masks from the export wrapper."""
        boxes = np.asarray(all_outputs[0], dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(all_outputs[1], dtype=np.float32).reshape(-1)
        class_ids = np.asarray(all_outputs[2], dtype=np.int64).reshape(-1)
        masks = None
        if self.task == "segment":
            if len(all_outputs) < 4:
                raise ValueError(
                    "Mask R-CNN segment export did not provide a masks output"
                )
            masks = np.asarray(all_outputs[3], dtype=np.float32)
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]

        keep = scores > conf
        boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
        if masks is not None:
            masks = masks[keep]
        if not len(boxes):
            if masks is not None:
                masks = np.zeros((0, orig_h, orig_w), dtype=np.bool_)
            return boxes, scores, class_ids, masks

        boxes = boxes.copy()
        if not getattr(self, "_dynamic_spatial_axes", False):
            input_h, input_w = _imgsz_hw(effective_imgsz)
            boxes[:, [0, 2]] *= orig_w / input_w
            boxes[:, [1, 3]] *= orig_h / input_h
            if masks is not None:
                masks = np.stack(
                    [
                        cv2.resize(
                            mask,
                            (orig_w, orig_h),
                            interpolation=cv2.INTER_LINEAR,
                        )
                        for mask in masks
                    ],
                    axis=0,
                )
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if masks is not None:
            masks = masks[valid] >= 0.5
        return boxes[valid], scores[valid], class_ids[valid], masks

    def _parse_fcos(
        self,
        all_outputs,
        effective_imgsz,
        orig_w,
        orig_h,
        conf,
    ):
        """Parse ``[xyxy, level_id, mapped class scores]`` FCOS rows."""
        output = np.asarray(all_outputs[0], dtype=np.float32)
        if output.ndim == 3:
            if output.shape[0] != 1:
                raise ValueError(
                    f"FCOS backend parsing expects batch 1, got {output.shape[0]}"
                )
            output = output[0]
        if output.ndim != 2 or output.shape[1] < 6:
            raise ValueError(
                "FCOS export output must have shape (1, anchors, 5 + classes)"
            )

        boxes_all = output[:, :4]
        level_ids = np.rint(output[:, 4]).astype(np.int64)
        class_scores = output[:, 5:]
        box_indices, class_ids = np.nonzero(class_scores > conf)
        if not len(box_indices):
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
        scores = class_scores[box_indices, class_ids]

        selected = []
        for level in range(5):
            level_candidates = np.nonzero(level_ids[box_indices] == level)[0]
            if level_candidates.size > 1000:
                level_scores = scores[level_candidates]
                top = np.argpartition(-level_scores, 999)[:1000]
                top = top[np.argsort(-level_scores[top])]
                level_candidates = level_candidates[top]
            selected.append(level_candidates)
        selected_indices = np.concatenate(selected)
        box_indices = box_indices[selected_indices]
        class_ids = class_ids[selected_indices]
        scores = scores[selected_indices]
        boxes = boxes_all[box_indices].copy()

        from ..models.fcos.utils import resize_dimensions

        input_h, input_w = _imgsz_hw(effective_imgsz)
        if input_h != input_w:
            raise ValueError(
                f"FCOS backend parsing requires a scalar/square imgsz, got {effective_imgsz}"
            )
        resized_h, resized_w, _ = resize_dimensions(orig_h, orig_w, input_h)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, resized_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, resized_h)
        boxes[:, [0, 2]] *= orig_w / resized_w
        boxes[:, [1, 3]] *= orig_h / resized_h
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes[valid], scores[valid], class_ids[valid]

    def _parse_yolo9(
        self,
        all_outputs,
        effective_imgsz,
        orig_w,
        orig_h,
        conf,
        iou: float = 0.45,
        max_det: int = 300,
    ):
        """Parse YOLO9 output: (B, 4+nc, N) — xyxy + class_scores."""
        if self.task == "obb":
            output = torch.from_numpy(np.asarray(all_outputs[0]))
            parsed = yolo9_postprocess(
                {"predictions": output, "obb": True},
                conf_thres=conf,
                iou_thres=iou,
                input_size=effective_imgsz,
                original_size=(orig_w, orig_h),
                max_det=max_det,
                letterbox=True,
            )
            boxes = np.asarray(parsed["boxes"], dtype=np.float32).reshape(-1, 4)
            max_scores = np.asarray(parsed["scores"], dtype=np.float32)
            class_ids = np.asarray(parsed["classes"], dtype=np.int64)
            obb = np.asarray(parsed["obb"], dtype=np.float32).reshape(-1, 7)
            return boxes, max_scores, class_ids, None, obb

        outputs = all_outputs[0][0].T  # (N, 4+nc)

        boxes_input_all = outputs[:, :4]
        scores = outputs[:, 4:]
        keypoints = None
        keypoints_all = None
        if self.task == "pose" and len(all_outputs) >= 2:
            keypoints_all = np.asarray(all_outputs[1][0], dtype=np.float32)

        if self.model_family == "yolo9_e2e" and self.task == "detect":
            topk_anchors = min(max_det, scores.shape[0])
            if topk_anchors == 0 or scores.shape[-1] == 0:
                return (
                    np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.float32),
                    np.empty((0,), dtype=np.int64),
                )

            anchor_scores = np.max(scores, axis=1)
            anchor_idx = np.argpartition(-anchor_scores, topk_anchors - 1)[
                :topk_anchors
            ]
            anchor_idx = anchor_idx[np.argsort(-anchor_scores[anchor_idx])]
            boxes_subset = boxes_input_all[anchor_idx]
            scores_subset = scores[anchor_idx]

            flat_scores = scores_subset.reshape(-1)
            topk_scores = min(max_det, flat_scores.size)
            flat_idx = np.argpartition(-flat_scores, topk_scores - 1)[:topk_scores]
            flat_idx = flat_idx[np.argsort(-flat_scores[flat_idx])]
            class_ids = flat_idx % scores_subset.shape[-1]
            box_indices = flat_idx // scores_subset.shape[-1]
            boxes_input = boxes_subset[box_indices]
            max_scores = flat_scores[flat_idx]
            keep = max_scores > conf
            boxes_input = boxes_input[keep]
            max_scores = max_scores[keep]
            class_ids = class_ids[keep]
        else:
            anchor_idx, class_ids = np.nonzero(scores > conf)
            boxes_input = boxes_input_all[anchor_idx]
            max_scores = scores[anchor_idx, class_ids]
            if keypoints_all is not None:
                keypoints = keypoints_all[anchor_idx].copy()
            max_nms = max(max_det, _YOLO9_MAX_NMS_CANDIDATES)
            if max_scores.size > max_nms:
                keep = np.argpartition(-max_scores, max_nms - 1)[:max_nms]
                keep = keep[np.argsort(-max_scores[keep])]
                boxes_input = boxes_input[keep]
                max_scores = max_scores[keep]
                class_ids = class_ids[keep]
                if keypoints is not None:
                    keypoints = keypoints[keep]

        boxes = boxes_input.copy()

        if len(boxes) == 0:
            if self.task == "pose" and keypoints_all is not None:
                return boxes, max_scores, class_ids, None, None, keypoints_all[:0]
            return boxes, max_scores, class_ids

        input_h, input_w = _imgsz_hw(effective_imgsz)
        if self.model_family == "yolo1":
            boxes[:, [0, 2]] *= orig_w / input_w
            boxes[:, [1, 3]] *= orig_h / input_h
        else:
            ratio = min(input_h / orig_h, input_w / orig_w)
            boxes[:, :4] /= ratio
            if keypoints is not None:
                keypoints[..., :2] /= ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        if keypoints is not None:
            keypoints[..., 0] = np.clip(keypoints[..., 0], 0, orig_w)
            keypoints[..., 1] = np.clip(keypoints[..., 1], 0, orig_h)
        valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if not valid_boxes.any():
            if self.task == "pose" and keypoints is not None:
                return (
                    boxes[:0],
                    max_scores[:0],
                    class_ids[:0],
                    None,
                    None,
                    keypoints[:0],
                )
            return boxes[:0], max_scores[:0], class_ids[:0]
        if not valid_boxes.all():
            boxes = boxes[valid_boxes]
            boxes_input = boxes_input[valid_boxes]
            max_scores = max_scores[valid_boxes]
            class_ids = class_ids[valid_boxes]
            if keypoints is not None:
                keypoints = keypoints[valid_boxes]

        if self.task == "pose" and keypoints is not None:
            return boxes, max_scores, class_ids, None, None, keypoints

        return boxes, max_scores, class_ids

    def _parse_yolonas(
        self,
        all_outputs,
        effective_imgsz,
        orig_w,
        orig_h,
        conf,
        ratio: Optional[float] = None,
    ):
        """Parse YOLO-NAS output: [boxes(B,N,4), scores(B,N,nc)] in input pixels."""
        first = all_outputs[0][0]
        second = all_outputs[1][0]
        if first.shape[-1] == 4 and second.shape[-1] != 4:
            boxes = first
            scores = second
        elif second.shape[-1] == 4 and first.shape[-1] != 4:
            boxes = second
            scores = first
        else:
            boxes = first
            scores = second

        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        mask = max_scores > conf
        boxes, max_scores, class_ids = boxes[mask], max_scores[mask], class_ids[mask]

        if len(boxes) == 0:
            return boxes, max_scores, class_ids

        boxes = boxes.astype(np.float32, copy=True)
        if YOLO_NAS_PRE_NMS_TOP_K and max_scores.size > YOLO_NAS_PRE_NMS_TOP_K:
            keep = np.argpartition(-max_scores, YOLO_NAS_PRE_NMS_TOP_K - 1)[
                :YOLO_NAS_PRE_NMS_TOP_K
            ]
            keep = keep[np.argsort(-max_scores[keep])]
            boxes = boxes[keep]
            max_scores = max_scores[keep]
            class_ids = class_ids[keep]

        input_h, input_w = _imgsz_hw(effective_imgsz)
        if ratio is None or ratio <= 0:
            resize_size = min(YOLO_NAS_RESIZE_SIZE, input_h, input_w)
            ratio = min(resize_size / orig_h, resize_size / orig_w)
        new_w = round(orig_w * ratio)
        new_h = round(orig_h * ratio)
        offset_x = (input_w - new_w) // 2
        offset_y = (input_h - new_h) // 2
        boxes[:, 0::2] = (boxes[:, 0::2] - offset_x) / ratio
        boxes[:, 1::2] = (boxes[:, 1::2] - offset_y) / ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid_boxes]
        max_scores = max_scores[valid_boxes]
        class_ids = class_ids[valid_boxes]
        return boxes, max_scores, class_ids

    @staticmethod
    def _parse_yolonas_obb(
        all_outputs,
        effective_imgsz,
        orig_w,
        orig_h,
        conf,
        iou: float = 0.25,
        ratio: Optional[float] = None,
        max_det: int = 300,
    ):
        """Parse YOLO-NAS-R OBB output: [boxes(B,N,5) cxcywhr, scores(B,N,nc)].

        Mirrors ``libreyolo.postprocess.yolonas.postprocess_obb`` step for step
        (threshold -> top-K -> undo the longest-side rescale and bottom-right
        pad -> canonicalise -> exact rotated NMS) but in numpy, so the ONNX
        backend stays importable without torch.
        """
        from ..data.obb import canonicalize_xywhr, xywhr_iou

        empty = (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            None,
            np.zeros((0, 7), dtype=np.float32),
        )

        first = np.asarray(all_outputs[0][0], dtype=np.float32)
        second = np.asarray(all_outputs[1][0], dtype=np.float32)
        if first.shape[-1] == 5 and second.shape[-1] != 5:
            boxes_raw, scores = first, second
        elif second.shape[-1] == 5 and first.shape[-1] != 5:
            boxes_raw, scores = second, first
        else:
            # A five-class checkpoint is indistinguishable by shape alone; the
            # LibreYOLO export wrapper always emits (boxes, scores).
            boxes_raw, scores = first, second

        max_scores = scores.max(axis=1)
        class_ids = scores.argmax(axis=1).astype(np.int64)
        mask = max_scores >= conf
        boxes_raw = boxes_raw[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]
        if boxes_raw.shape[0] == 0:
            return empty

        if YOLO_NAS_OBB_PRE_NMS_TOP_K and max_scores.size > YOLO_NAS_OBB_PRE_NMS_TOP_K:
            keep = np.argpartition(-max_scores, YOLO_NAS_OBB_PRE_NMS_TOP_K - 1)[
                :YOLO_NAS_OBB_PRE_NMS_TOP_K
            ]
            keep = keep[np.argsort(-max_scores[keep])]
            boxes_raw = boxes_raw[keep]
            max_scores = max_scores[keep]
            class_ids = class_ids[keep]

        input_h, input_w = _imgsz_hw(effective_imgsz)
        if ratio is None or ratio <= 0:
            resize_size = min(YOLO_NAS_OBB_RESIZE_SIZE, input_h, input_w)
            ratio = min(resize_size / orig_h, resize_size / orig_w)
        # Bottom-right padding leaves no offset to subtract, and the rescale is
        # uniform, so centres and sides divide by the ratio while the angle is
        # unchanged.
        xywhr = boxes_raw.astype(np.float32, copy=True)
        xywhr[:, :4] /= np.float32(ratio)
        xywhr[:, 0] = np.clip(xywhr[:, 0], 0, orig_w)
        xywhr[:, 1] = np.clip(xywhr[:, 1], 0, orig_h)

        # Non-finite rows are dropped here rather than in NMS: the native path
        # filters them inside rotated_nms_keep_indices, and canonicalization
        # would otherwise carry NaN centres and angles into public results.
        valid = (
            np.isfinite(xywhr).all(axis=1)
            & np.isfinite(max_scores)
            & (xywhr[:, 2] > 0)
            & (xywhr[:, 3] > 0)
        )
        xywhr = xywhr[valid]
        max_scores = max_scores[valid]
        class_ids = class_ids[valid]
        if xywhr.shape[0] == 0:
            return empty

        xywhr = np.stack([canonicalize_xywhr(row) for row in xywhr]).astype(np.float32)

        order = np.argsort(-max_scores).tolist()
        keep_idx: List[int] = []
        while order and len(keep_idx) < max_det:
            current = order.pop(0)
            keep_idx.append(current)
            order = [
                candidate
                for candidate in order
                if class_ids[candidate] != class_ids[current]
                or xywhr_iou(xywhr[current], xywhr[candidate]) <= iou
            ]
        if not keep_idx:
            return empty

        keep_arr = np.asarray(keep_idx, dtype=np.int64)
        xywhr = xywhr[keep_arr]
        max_scores = max_scores[keep_arr]
        class_ids = class_ids[keep_arr]

        half_w = xywhr[:, 2] / 2
        half_h = xywhr[:, 3] / 2
        cos = np.abs(np.cos(xywhr[:, 4]))
        sin = np.abs(np.sin(xywhr[:, 4]))
        extent_x = cos * half_w + sin * half_h
        extent_y = sin * half_w + cos * half_h
        boxes = np.stack(
            [
                np.clip(xywhr[:, 0] - extent_x, 0, orig_w),
                np.clip(xywhr[:, 1] - extent_y, 0, orig_h),
                np.clip(xywhr[:, 0] + extent_x, 0, orig_w),
                np.clip(xywhr[:, 1] + extent_y, 0, orig_h),
            ],
            axis=-1,
        ).astype(np.float32)
        obb = np.concatenate(
            [
                xywhr,
                max_scores[:, None].astype(np.float32),
                class_ids[:, None].astype(np.float32),
            ],
            axis=-1,
        ).astype(np.float32)
        return boxes, max_scores.astype(np.float32), class_ids, None, obb

    def _parse_yolonas_pose(
        self,
        all_outputs,
        effective_imgsz,
        orig_w,
        orig_h,
        conf,
        ratio: Optional[float] = None,
        max_det=300,
    ):
        """Parse YOLO-NAS pose: boxes, scores, keypoint xy, keypoint confidence."""
        boxes = all_outputs[0][0]
        scores = all_outputs[1][0]
        keypoints_xy = all_outputs[2][0]
        keypoints_conf = all_outputs[3][0]

        # scores: [A, nc]. Single-class pose keeps the historical squeeze;
        # multi-class pose takes the top-scoring class per anchor.
        if scores.ndim > 1 and scores.shape[-1] > 1:
            class_ids_full = scores.argmax(axis=-1).astype(np.int64)
            scores = scores.max(axis=-1)
        else:
            scores = scores.squeeze(-1)
            class_ids_full = None

        mask = scores >= conf
        boxes = boxes[mask].astype(np.float32, copy=True)
        max_scores = scores[mask].astype(np.float32, copy=False)
        keypoints_xy = keypoints_xy[mask].astype(np.float32, copy=True)
        keypoints_conf = keypoints_conf[mask].astype(np.float32, copy=False)
        if class_ids_full is not None:
            class_ids = class_ids_full[mask]
        else:
            class_ids = np.zeros((max_scores.shape[0],), dtype=np.int64)

        if len(boxes) == 0:
            keypoints = np.zeros((0, keypoints_xy.shape[-2], 3), dtype=np.float32)
            return boxes, max_scores, class_ids, None, None, keypoints

        pre_nms_top_k = max(1000, int(max_det))
        if max_scores.size > pre_nms_top_k:
            keep = np.argpartition(-max_scores, pre_nms_top_k - 1)[:pre_nms_top_k]
            keep = keep[np.argsort(-max_scores[keep])]
            boxes = boxes[keep]
            max_scores = max_scores[keep]
            keypoints_xy = keypoints_xy[keep]
            keypoints_conf = keypoints_conf[keep]
            class_ids = class_ids[keep]

        scale = ratio
        if scale is None or scale <= 0:
            scale = min(
                YOLO_NAS_POSE_RESIZE_SIZE / orig_h,
                YOLO_NAS_POSE_RESIZE_SIZE / orig_w,
            )
        boxes[:, 0::2] /= scale
        boxes[:, 1::2] /= scale
        keypoints_xy[..., 0] /= scale
        keypoints_xy[..., 1] /= scale

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if not valid.all():
            boxes = boxes[valid]
            max_scores = max_scores[valid]
            class_ids = class_ids[valid]
            keypoints_xy = keypoints_xy[valid]
            keypoints_conf = keypoints_conf[valid]

        keypoints = np.concatenate([keypoints_xy, keypoints_conf[..., None]], axis=-1)
        return boxes, max_scores, class_ids, None, None, keypoints

    def _parse_dfine(self, all_outputs, orig_w, orig_h, conf, max_det: int = 300):
        """Parse D-FINE outputs: pred_logits (B, Q, nc) + pred_boxes (B, Q, 4) cxcywh [0,1].

        Matches the upstream DFINEPostProcessor (use_focal_loss=True): sigmoid →
        topk over (queries × classes) flattened → labels = topk_idx % nc, query_idx
        = topk_idx // nc. No NMS (DETR set-prediction).
        """
        pred_logits = all_outputs[0][0]  # (Q, nc)
        pred_boxes = all_outputs[1][0]  # (Q, 4)

        Q, nc = pred_logits.shape
        prob = 1.0 / (1.0 + np.exp(-pred_logits.astype(np.float64)))
        prob = prob.astype(np.float32)

        flat = prob.reshape(-1)  # (Q * nc,)
        k = min(max_det, flat.size)
        # Top-k via argpartition (faster than full sort).
        idx = np.argpartition(-flat, k - 1)[:k]
        idx = idx[np.argsort(-flat[idx])]

        scores = flat[idx]
        query_idx = idx // nc
        class_ids = idx % nc

        # cxcywh -> xyxy in [0,1], then gather + scale.
        cx, cy, w, h = (
            pred_boxes[:, 0],
            pred_boxes[:, 1],
            pred_boxes[:, 2],
            pred_boxes[:, 3],
        )
        boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        boxes = boxes_xyxy[query_idx]

        boxes[:, [0, 2]] *= orig_w
        boxes[:, [1, 3]] *= orig_h

        mask = scores > conf
        return boxes[mask], scores[mask], class_ids[mask].astype(np.int64)

    def _parse_detr(self, all_outputs, orig_w, orig_h, conf, max_det: int = 100):
        """Parse vanilla DETR's softmax logits and one prediction per query."""
        pred_logits = np.asarray(all_outputs[0][0], dtype=np.float32)
        pred_boxes = np.asarray(all_outputs[1][0], dtype=np.float32)
        if pred_logits.ndim != 2 or pred_boxes.ndim != 2:
            raise ValueError(
                "DETR backend expects (Q, classes) logits and (Q, 4) boxes"
            )
        if pred_logits.shape[0] != pred_boxes.shape[0] or pred_boxes.shape[1] != 4:
            raise ValueError("DETR backend logits and boxes have incompatible shapes")

        shifted = pred_logits - pred_logits.max(axis=-1, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities = exponentials / exponentials.sum(axis=-1, keepdims=True)
        object_probabilities = probabilities[:, :-1]

        if pred_logits.shape[-1] == 92 and self.nb_classes == 80:
            from ..utils.coco import COCO91_CATEGORY_IDS

            object_probabilities = object_probabilities[
                :, np.asarray(COCO91_CATEGORY_IDS, dtype=np.int64)
            ]

        scores = object_probabilities.max(axis=-1)
        class_ids = object_probabilities.argmax(axis=-1).astype(np.int64)
        budget = min(max(int(max_det), 0), int(scores.size))
        if budget:
            query_indices = np.argpartition(-scores, budget - 1)[:budget]
            query_indices = query_indices[np.argsort(-scores[query_indices])]
            scores = scores[query_indices]
            class_ids = class_ids[query_indices]
        else:
            query_indices = np.zeros((0,), dtype=np.int64)
            scores = scores[:0]
            class_ids = class_ids[:0]

        center_x, center_y, width, height = pred_boxes.T
        boxes_xyxy = np.stack(
            (
                center_x - 0.5 * width,
                center_y - 0.5 * height,
                center_x + 0.5 * width,
                center_y + 0.5 * height,
            ),
            axis=-1,
        )
        boxes = boxes_xyxy[query_indices]
        boxes[:, [0, 2]] *= orig_w
        boxes[:, [1, 3]] *= orig_h

        keep = scores > conf
        return boxes[keep], scores[keep], class_ids[keep]

    def _parse_lwdetr(self, all_outputs, orig_w, orig_h, conf, max_det: int = 300):
        """Parse LW-DETR outputs — same top-K decode as D-FINE, plus COCO remap.

        Upstream never returns more than its configured ``num_select``, and the
        released COCO head has one column per COCO category id, so the ids are
        mapped down to the contiguous 80-class interface the native path exposes.

        The unmapped columns are sliced out *before* the top-K, matching
        ``postprocess.lwdetr``: filtering after selection would let one of the
        11 annotation-free COCO ids consume a slot of the max_det budget with no
        replacement, so exported graphs would drop a detection the native path
        keeps.
        """
        effective_max_det = min(max_det, _lwdetr_num_select(self.model_size))

        num_classes = all_outputs[0].shape[-1]
        if num_classes == 91 and self.nb_classes == 80:
            from ..utils.coco import COCO91_CATEGORY_IDS

            columns = np.asarray(COCO91_CATEGORY_IDS, dtype=np.int64)
            sliced = [all_outputs[0][:, :, columns], *all_outputs[1:]]
            boxes, scores, class_ids = self._parse_dfine(
                sliced, orig_w, orig_h, conf, max_det=effective_max_det
            )
            # _parse_dfine returns indices into the sliced 80-column head, which
            # is already the contiguous LibreYOLO ordering.
            return boxes, scores, class_ids

        return self._parse_dfine(
            all_outputs, orig_w, orig_h, conf, max_det=effective_max_det
        )

    def _parse_deformable_detr(
        self, all_outputs, orig_w, orig_h, conf, max_det: int = 300
    ):
        """Parse NMS-free outputs and remove unused COCO columns before top-K."""
        effective_max_det = min(max_det, 300)
        num_classes = all_outputs[0].shape[-1]
        if num_classes == 91 and self.nb_classes == 80:
            from ..utils.coco import COCO91_CATEGORY_IDS

            columns = np.asarray(COCO91_CATEGORY_IDS, dtype=np.int64)
            all_outputs = [all_outputs[0][:, :, columns], *all_outputs[1:]]
        return self._parse_dfine(
            all_outputs, orig_w, orig_h, conf, max_det=effective_max_det
        )

    @staticmethod
    def _parse_centernet(
        all_outputs,
        input_size,
        orig_w,
        orig_h,
        conf,
        max_det: int = 100,
    ):
        """Parse the export graph's baked top-100 CenterNet detections."""
        from ..postprocess.centernet import postprocess

        decoded = (
            all_outputs[0] if isinstance(all_outputs, (tuple, list)) else all_outputs
        )
        result = postprocess(
            decoded,
            conf_thres=conf,
            original_size=(orig_w, orig_h),
            input_size=input_size,
            max_det=min(max_det, 100),
        )
        return result["boxes"], result["scores"], result["classes"]

    def _parse_dfine_segment(
        self, all_outputs, orig_w, orig_h, conf, max_det: int = 300
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Parse D-FINE-seg raw exports into boxes, classes, and masks."""
        pred_logits = all_outputs[0][0]
        pred_boxes = all_outputs[1][0]
        pred_masks = all_outputs[2][0] if len(all_outputs) >= 3 else None

        _, nc = pred_logits.shape
        prob = 1.0 / (1.0 + np.exp(-pred_logits.astype(np.float64)))
        prob = prob.astype(np.float32)
        flat = prob.reshape(-1)
        k = min(max_det, flat.size)
        idx = np.argpartition(-flat, k - 1)[:k]
        idx = idx[np.argsort(-flat[idx])]

        scores = flat[idx]
        query_idx = idx // nc
        class_ids = idx % nc

        boxes = self._scale_cxcywh_boxes(
            pred_boxes[query_idx],
            orig_w,
            orig_h,
            clip=True,
        )
        keep = scores > conf
        boxes = boxes[keep]
        scores = scores[keep]
        query_idx = query_idx[keep]
        class_ids = class_ids[keep]

        masks_out = None
        if pred_masks is not None and query_idx.size > 0:
            masks_t = torch.from_numpy(pred_masks[query_idx]).unsqueeze(1).float()
            in_h, in_w = _imgsz_hw(self.input_size)
            masks_t = F.interpolate(
                masks_t,
                size=(int(in_h), int(in_w)),
                mode="bilinear",
                align_corners=False,
            )
            masks_t = F.interpolate(
                masks_t,
                size=(int(orig_h), int(orig_w)),
                mode="bilinear",
                align_corners=False,
            )[:, 0].clamp_(0, 1)
            boxes_t = torch.from_numpy(boxes).to(dtype=masks_t.dtype)
            if boxes_t.numel() > 0:
                ys = torch.arange(int(orig_h), dtype=masks_t.dtype)[None, :, None]
                xs = torch.arange(int(orig_w), dtype=masks_t.dtype)[None, None, :]
                x1, y1, x2, y2 = boxes_t.T
                inside = (
                    (xs >= x1[:, None, None])
                    & (xs < x2[:, None, None])
                    & (ys >= y1[:, None, None])
                    & (ys < y2[:, None, None])
                )
                masks_t = masks_t * inside.to(dtype=masks_t.dtype)
            masks_out = (masks_t >= 0.5).numpy()

        return boxes, scores, class_ids.astype(np.int64), masks_out

    def _parse_ec_segment(
        self, all_outputs, orig_w, orig_h, conf, max_det=300
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Parse EC segmentation outputs: logits, normalized cxcywh boxes, masks."""
        pred_logits = all_outputs[0][0]
        pred_boxes = all_outputs[1][0]
        pred_masks = all_outputs[2][0] if len(all_outputs) >= 3 else None

        query_idx, class_ids, scores = self._ec_topk(pred_logits, max_det=max_det)
        keep = scores > conf
        query_idx = query_idx[keep]
        class_ids = class_ids[keep]
        max_scores = scores[keep]

        boxes = self._scale_cxcywh_boxes(
            pred_boxes[query_idx],
            orig_w,
            orig_h,
            clip=False,
        )
        masks_out = None
        if pred_masks is not None and query_idx.size > 0:
            masks_t = torch.from_numpy(pred_masks[query_idx]).unsqueeze(1).float()
            masks_t = F.interpolate(
                masks_t,
                size=(int(orig_h), int(orig_w)),
                mode="bilinear",
                align_corners=False,
            )
            masks_out = (masks_t[:, 0] > 0.0).numpy()

        return boxes, max_scores, class_ids.astype(np.int64), masks_out

    def _parse_ec_pose(self, all_outputs, orig_w, orig_h, conf, max_det=300):
        """Parse EC pose outputs: logits and normalized flattened keypoints."""
        pred_logits = all_outputs[0][0]
        pred_boxes = None
        pred_keypoints = all_outputs[1][0]
        if len(all_outputs) >= 3:
            maybe_boxes = all_outputs[1][0]
            maybe_keypoints = all_outputs[2][0]
            if maybe_boxes.shape[-1] == 4:
                pred_boxes = maybe_boxes
                pred_keypoints = maybe_keypoints

        scores_per_class = 1.0 / (1.0 + np.exp(-pred_logits.astype(np.float64)))
        scores_per_class = scores_per_class.astype(np.float32)
        # Person class is the LAST logit (index 1 of ECPose's 2-class head); keep
        # this in lockstep with ``postprocess_pose`` so .pt and ONNX agree.
        query_scores = scores_per_class[..., -1]
        k = min(max_det, query_scores.size)
        query_idx = np.argpartition(-query_scores, k - 1)[:k]
        query_idx = query_idx[np.argsort(-query_scores[query_idx])]
        scores = query_scores[query_idx]
        keep = scores >= conf
        query_idx = query_idx[keep]
        max_scores = scores[keep]
        class_ids = np.zeros((max_scores.shape[0],), dtype=np.int64)

        if pred_keypoints.ndim >= 3 and pred_keypoints.shape[-1] == 2:
            num_keypoints = int(pred_keypoints.shape[-2])
        else:
            num_keypoints = int(pred_keypoints.shape[-1]) // 2
        if num_keypoints <= 0:
            num_keypoints = int(getattr(self, "num_keypoints", 17) or 17)
        if query_idx.size == 0:
            empty_boxes = np.zeros((0, 4), dtype=np.float32)
            empty_keypoints = np.zeros((0, num_keypoints, 3), dtype=np.float32)
            return empty_boxes, max_scores, class_ids, None, None, empty_keypoints

        keypoints_xy = pred_keypoints[query_idx].reshape(-1, num_keypoints, 2)
        keypoints_xy = keypoints_xy.astype(np.float32, copy=True)
        keypoints_xy[..., 0] *= float(orig_w)
        keypoints_xy[..., 1] *= float(orig_h)

        if pred_boxes is not None:
            boxes = self._scale_cxcywh_boxes(pred_boxes[query_idx], orig_w, orig_h)
        else:
            x_min = keypoints_xy[..., 0].min(axis=1)
            y_min = keypoints_xy[..., 1].min(axis=1)
            x_max = keypoints_xy[..., 0].max(axis=1)
            y_max = keypoints_xy[..., 1].max(axis=1)
            boxes = np.stack([x_min, y_min, x_max, y_max], axis=1)
        visibility = np.ones((*keypoints_xy.shape[:-1], 1), dtype=np.float32)
        keypoints = np.concatenate([keypoints_xy, visibility], axis=-1)
        return boxes, max_scores, class_ids, None, None, keypoints

    @staticmethod
    def _ec_topk(pred_logits, max_det: int):
        scores = 1.0 / (1.0 + np.exp(-pred_logits.astype(np.float64)))
        scores = scores.astype(np.float32)
        num_classes = scores.shape[-1]
        flat = scores.reshape(-1)
        k = min(max_det, flat.size)
        idx = np.argpartition(-flat, k - 1)[:k]
        idx = idx[np.argsort(-flat[idx])]
        query_idx = idx // num_classes
        class_ids = idx % num_classes
        return query_idx, class_ids, flat[idx]

    @staticmethod
    def _scale_cxcywh_boxes(boxes_cxcywh, orig_w, orig_h, *, clip: bool = True):
        cx, cy, w, h = (
            boxes_cxcywh[:, 0],
            boxes_cxcywh[:, 1],
            boxes_cxcywh[:, 2],
            boxes_cxcywh[:, 3],
        )
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        boxes = boxes.astype(np.float32, copy=False)
        boxes[:, [0, 2]] *= orig_w
        boxes[:, [1, 3]] *= orig_h
        if clip:
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        return boxes

    def _normalize_rfdetr_keypoint_output(
        self,
        raw_keypoint_output,
        *,
        query_count: int,
        num_classes: int,
    ) -> np.ndarray:
        raw = np.asarray(raw_keypoint_output)
        if raw.ndim >= 3 and raw.shape[0] == 1 and raw.shape[1] == query_count:
            raw = raw[0]
        elif raw.ndim == 4 and raw.shape[0] == 1:
            raw = raw[0]

        if raw.ndim == 2:
            schema = getattr(self, "num_keypoints_per_class", None)
            if schema:
                schema_counts = np.asarray(
                    [int(count) for count in schema], dtype=np.int64
                )
                if schema_counts.size != num_classes or schema_counts.max() <= 0:
                    raise ValueError(
                        "Invalid RF-DETR GroupPose num_keypoints_per_class metadata "
                        f"for {num_classes} classes: {list(schema_counts)}"
                    )
                slots = int(schema_counts.size * schema_counts.max())
                if slots <= 0 or raw.shape[-1] % slots != 0:
                    raise ValueError(
                        "RF-DETR GroupPose flattened keypoint output cannot be "
                        f"reshaped with schema {list(schema_counts)}: {raw.shape}"
                    )
                pred_dim = raw.shape[-1] // slots
                raw = raw.reshape(raw.shape[0], slots, pred_dim)
            else:
                keypoint_dim = int(getattr(self, "keypoint_dim", 3) or 3)
                if keypoint_dim not in (2, 3) or raw.shape[-1] % keypoint_dim != 0:
                    raise ValueError(
                        "RF-DETR flattened keypoint output cannot be reshaped "
                        f"with keypoint_dim={keypoint_dim}: {raw.shape}"
                    )
                raw = raw.reshape(
                    raw.shape[0], raw.shape[-1] // keypoint_dim, keypoint_dim
                )

        if raw.ndim != 3:
            raise ValueError(f"Unexpected RF-DETR keypoint output shape: {raw.shape}")
        return raw

    def _parse_rfdetr(self, all_outputs, orig_w, orig_h, conf, max_det=300):
        """Parse RF-DETR output: boxes (B,300,4) cxcywh [0,1] + logits (B,300,nc).

        For segmentation models a third output is present:
        masks (B,300,Hm,Wm) raw mask logits at model resolution.
        For pose models a third output is present:
        keypoints (B,300,K,3) with normalized xy and visibility logits.
        For OBB models a third output is present:
        angles (B,300,1) in radians.
        """
        first = all_outputs[0][0]
        second = all_outputs[1][0]
        if first.shape[-1] == 4:
            boxes_all = first
            logits = second
        else:
            logits = first
            boxes_all = second
        raw_masks = None
        raw_keypoints = None
        raw_keypoint_output = None
        raw_angles = None
        grouppose_active_keypoints = None
        if len(all_outputs) >= 3:
            if self.task == "obb":
                raw_angles = all_outputs[2][0]
            elif self.task == "pose":
                raw_keypoint_output = all_outputs[2]
            else:
                raw_masks = all_outputs[2][0]

        if raw_keypoint_output is not None and not getattr(
            self, "num_keypoints_per_class", None
        ):
            public_classes = int(self.nb_classes)
            if 0 < public_classes < logits.shape[-1]:
                logits = logits[:, :public_classes]
        scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64))).astype(np.float32)
        num_queries, num_classes = scores.shape
        if raw_keypoint_output is not None:
            raw_keypoints = self._normalize_rfdetr_keypoint_output(
                raw_keypoint_output,
                query_count=num_queries,
                num_classes=num_classes,
            )
        model_size = self.model_size or getattr(self, "size", None)
        num_select = (
            _rfdetr_num_select(self.task, model_size)
            if int(max_det) == 300
            else int(max_det)
        )
        k = min(
            num_select,
            num_queries * num_classes,
        )
        flat_indexes = np.argpartition(scores.reshape(-1), -k)[-k:]
        flat_indexes = flat_indexes[np.argsort(scores.reshape(-1)[flat_indexes])[::-1]]
        max_scores = scores.reshape(-1)[flat_indexes]
        query_idx = flat_indexes // num_classes
        class_ids = flat_indexes % num_classes
        boxes_raw = boxes_all[query_idx]
        angles_raw = raw_angles[query_idx] if raw_angles is not None else None
        keypoints_raw = (
            raw_keypoints[query_idx].copy() if raw_keypoints is not None else None
        )
        if raw_masks is not None:
            raw_masks = raw_masks[query_idx]

        if (
            self.task == "pose"
            and keypoints_raw is not None
            and keypoints_raw.ndim == 3
            and keypoints_raw.shape[-1] >= 7
            and num_classes > 1
            and keypoints_raw.shape[1] % num_classes == 0
        ):
            schema = getattr(self, "num_keypoints_per_class", None)
            keypoint_counts = None
            if schema:
                schema_counts = np.asarray(
                    [int(count) for count in schema], dtype=np.int64
                )
                if (
                    schema_counts.size == num_classes
                    and schema_counts.max() > 0
                    and keypoints_raw.shape[1]
                    == schema_counts.size * int(schema_counts.max())
                ):
                    keypoint_counts = schema_counts
                    max_num_keypoints = int(schema_counts.max())
                else:
                    raise ValueError(
                        "Invalid RF-DETR GroupPose num_keypoints_per_class metadata "
                        f"for keypoint output {keypoints_raw.shape}: {list(schema_counts)}"
                    )
            else:
                max_num_keypoints = keypoints_raw.shape[1] // num_classes
            grouped = keypoints_raw.reshape(
                keypoints_raw.shape[0],
                num_classes,
                max_num_keypoints,
                keypoints_raw.shape[-1],
            )
            selected = grouped[np.arange(len(class_ids)), class_ids]

            # GroupPose exports use internal class 0 for no-keypoint detections
            # and keypoint-bearing classes after it. Public pose labels are
            # contiguous over only the keypoint-bearing classes (person -> 0).
            if keypoint_counts is None:
                keypoint_counts = np.full(
                    num_classes, max_num_keypoints, dtype=np.int64
                )
                if self.nb_classes == num_classes - 1:
                    keypoint_counts[0] = 0
            active_counts = keypoint_counts[class_ids]
            valid_pose_class = active_counts > 0

            if np.any(valid_pose_class):
                trace_alpha = 0.2
                log_mean_traces = np.zeros(len(selected), dtype=np.float32)
                for class_idx, active_count in enumerate(keypoint_counts):
                    if active_count <= 0:
                        continue
                    class_mask = class_ids == class_idx
                    if not np.any(class_mask):
                        continue
                    log_mean_traces[class_mask] = _rfdetr_keypoint_log_mean_trace_np(
                        selected[class_mask, :active_count]
                    )
                max_scores = max_scores * np.exp(-trace_alpha * log_mean_traces)

            keypoints_selected = np.zeros(
                (len(selected), max_num_keypoints, 3),
                dtype=np.float32,
            )
            active_keypoint_mask = np.zeros(
                (len(selected), max_num_keypoints),
                dtype=bool,
            )
            for row_idx, active_count in enumerate(active_counts):
                if active_count <= 0:
                    continue
                keypoints_selected[row_idx, :active_count, :3] = selected[
                    row_idx,
                    :active_count,
                    :3,
                ]
                active_keypoint_mask[row_idx, :active_count] = True

            kp_classes = np.flatnonzero(keypoint_counts > 0)
            remap = np.full(num_classes, -1, dtype=class_ids.dtype)
            remap[kp_classes] = np.arange(len(kp_classes), dtype=class_ids.dtype)

            boxes_raw = boxes_raw[valid_pose_class]
            max_scores = max_scores[valid_pose_class]
            class_ids = remap[class_ids[valid_pose_class]]
            if angles_raw is not None:
                angles_raw = angles_raw[valid_pose_class]
            if keypoints_raw is not None:
                keypoints_raw = keypoints_selected[valid_pose_class]
                grouppose_active_keypoints = active_keypoint_mask[valid_pose_class]
            if raw_masks is not None:
                raw_masks = raw_masks[valid_pose_class]

        mask = max_scores > conf
        boxes_raw = boxes_raw[mask]
        max_scores, class_ids = max_scores[mask], class_ids[mask]
        if angles_raw is not None:
            angles_raw = angles_raw[mask]
        if keypoints_raw is not None:
            keypoints_raw = keypoints_raw[mask]
            if grouppose_active_keypoints is not None:
                grouppose_active_keypoints = grouppose_active_keypoints[mask]
        if raw_masks is not None:
            raw_masks = raw_masks[mask]

        if len(boxes_raw) == 0:
            if self.task == "obb":
                return (
                    boxes_raw,
                    max_scores,
                    class_ids,
                    None,
                    np.zeros((0, 7), dtype=np.float32),
                )
            if self.task == "pose" and keypoints_raw is not None:
                return boxes_raw, max_scores, class_ids, None, None, keypoints_raw
            return boxes_raw, max_scores, class_ids, None

        # COCO 91→80 class mapping
        if num_classes == 91 and self.nb_classes == 80:
            # Shared module, not models.rfdetr.model: that import pulls in the
            # optional transformers dependency, which LW-DETR exports (also a
            # 91-wide head) must not require.
            from ..utils.coco import COCO91_TO_COCO80

            mapped = np.array([COCO91_TO_COCO80.get(int(c), -1) for c in class_ids])
            valid = mapped >= 0
            boxes_raw = boxes_raw[valid]
            max_scores = max_scores[valid]
            class_ids = mapped[valid]
            if angles_raw is not None:
                angles_raw = angles_raw[valid]
            if keypoints_raw is not None:
                keypoints_raw = keypoints_raw[valid]
                if grouppose_active_keypoints is not None:
                    grouppose_active_keypoints = grouppose_active_keypoints[valid]
            if raw_masks is not None:
                raw_masks = raw_masks[valid]

        if len(boxes_raw) == 0:
            if self.task == "obb":
                return (
                    boxes_raw,
                    max_scores,
                    class_ids,
                    None,
                    np.zeros((0, 7), dtype=np.float32),
                )
            if self.task == "pose" and keypoints_raw is not None:
                return (
                    boxes_raw,
                    max_scores,
                    class_ids,
                    None,
                    None,
                    keypoints_raw,
                )
            return boxes_raw, max_scores, class_ids, None

        cx, cy, w, h = (
            boxes_raw[:, 0],
            boxes_raw[:, 1],
            boxes_raw[:, 2],
            boxes_raw[:, 3],
        )
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        obb_out = None
        if angles_raw is not None:
            angles = np.asarray(angles_raw, dtype=np.float32).reshape(-1)
            obb_out = np.stack(
                [
                    cx * orig_w,
                    cy * orig_h,
                    w * orig_w,
                    h * orig_h,
                    angles,
                    max_scores,
                    class_ids.astype(np.float32),
                ],
                axis=1,
            ).astype(np.float32, copy=False)

        # Resize and threshold masks to original image resolution
        masks_out = None
        if raw_masks is not None and len(raw_masks) > 0:
            masks_t = torch.from_numpy(raw_masks).unsqueeze(1).float()
            masks_t = F.interpolate(
                masks_t,
                size=(int(orig_h), int(orig_w)),
                mode="bilinear",
                align_corners=False,
            )
            masks_out = (masks_t[:, 0] > 0.0).numpy()  # (N, H, W)

        keypoints_out = None
        if keypoints_raw is not None:
            keypoints_out = np.asarray(keypoints_raw, dtype=np.float32).copy()
            keypoints_out[..., 0] *= float(orig_w)
            keypoints_out[..., 1] *= float(orig_h)
            if keypoints_out.shape[-1] == 2:
                visibility = np.ones((*keypoints_out.shape[:-1], 1), dtype=np.float32)
                keypoints_out = np.concatenate([keypoints_out, visibility], axis=-1)
            else:
                keypoints_out[..., 2] = 1.0 / (1.0 + np.exp(-keypoints_out[..., 2]))
                keypoints_out = keypoints_out[..., :3]
            if grouppose_active_keypoints is not None:
                keypoints_out[~grouppose_active_keypoints] = 0.0

        if self.task == "obb":
            return boxes, max_scores, class_ids, masks_out, obb_out
        if self.task == "pose":
            return boxes, max_scores, class_ids, masks_out, None, keypoints_out
        return boxes, max_scores, class_ids, masks_out

    def _parse_rtdetr(self, all_outputs, orig_w, orig_h, conf, max_det: int = 300):
        """Parse RT-DETR output: pred_boxes (B,Q,4) cxcywh [0,1] + pred_logits (B,Q,C).

        RTDETR outputs are already in the correct class indices (no COCO 91->80 mapping needed).
        """
        # all_outputs order depends on ONNX output naming; try both orderings
        first = all_outputs[0][0]  # (Q, 4) or (Q, C)
        second = all_outputs[1][0]  # (Q, C) or (Q, 4)

        # Detect which is boxes and which is logits by shape
        if first.shape[1] == 4 and len(second.shape) == 2 and second.shape[1] != 4:
            boxes_raw = first  # (Q, 4) normalized cxcywh
            logits = second  # (Q, C) raw logits
        elif second.shape[1] == 4 and len(first.shape) == 2 and first.shape[1] != 4:
            boxes_raw = second
            logits = first
        else:
            # Fallback: assume pred_logits has more columns (num_classes typically > 4)
            if first.shape[1] > second.shape[1]:
                logits = first
                boxes_raw = second
            else:
                logits = second
                boxes_raw = first

        # Match upstream RTDETRPostProcessor (and _parse_dfine): top-K across the
        # flattened (Q*nc) score matrix, allowing multiple classes per query.
        # Per-query argmax (the previous logic) silently dropped valid non-max
        # detections and cost ~0.7-0.9 mAP on COCO val2017.
        Q, nc = logits.shape
        prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        prob = prob.astype(np.float32)

        flat = prob.reshape(-1)
        k = min(max_det, flat.size)
        idx = np.argpartition(-flat, k - 1)[:k]
        idx = idx[np.argsort(-flat[idx])]

        scores = flat[idx]
        query_idx = idx // nc
        class_ids = idx % nc

        cx, cy, w, h = (
            boxes_raw[:, 0],
            boxes_raw[:, 1],
            boxes_raw[:, 2],
            boxes_raw[:, 3],
        )
        boxes_xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
        boxes = boxes_xyxy[query_idx]
        boxes[:, [0, 2]] *= orig_w
        boxes[:, [1, 3]] *= orig_h

        mask = scores > conf
        return boxes[mask], scores[mask], class_ids[mask]

    @staticmethod
    def _parse_rtdetr_obb(
        all_outputs,
        effective_imgsz,
        orig_w,
        orig_h,
        conf,
        max_det: int = 300,
    ):
        """Parse five-coordinate RT-DETRv2 OBB output without NMS."""
        first = np.asarray(all_outputs[0][0], dtype=np.float32)
        second = np.asarray(all_outputs[1][0], dtype=np.float32)
        if first.ndim != 2 or second.ndim != 2 or first.shape[0] != second.shape[0]:
            raise ValueError(
                "RT-DETRv2 OBB export must return matching [Q,C] logits and "
                f"[Q,5] boxes, got {first.shape} and {second.shape}"
            )

        if first.shape[-1] == second.shape[-1] == 5:
            # LibreYOLO's export wrapper defines the equal-width case as
            # (pred_logits, pred_boxes). Shape alone cannot distinguish a
            # five-class checkpoint from its five-coordinate OBB output.
            logits, boxes_raw = first, second
        elif first.shape[-1] == 5 and second.shape[-1] != 5:
            boxes_raw, logits = first, second
        elif second.shape[-1] == 5 and first.shape[-1] != 5:
            boxes_raw, logits = second, first
        else:
            raise ValueError(
                "RT-DETRv2 OBB export must return one [Q,5] box tensor and "
                f"one [Q,C] logit tensor, got {first.shape} and {second.shape}"
            )

        prob = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        flat = prob.astype(np.float32).reshape(-1)
        k = min(max_det, flat.size)
        if k == 0:
            empty = np.zeros((0,), dtype=np.float32)
            return (
                np.zeros((0, 4), dtype=np.float32),
                empty,
                empty.astype(np.int64),
                None,
                np.zeros((0, 7), dtype=np.float32),
            )
        indexes = np.argpartition(-flat, k - 1)[:k]
        indexes = indexes[np.argsort(-flat[indexes])]
        scores = flat[indexes]
        num_classes = logits.shape[-1]
        query_indexes = indexes // num_classes
        class_ids = indexes % num_classes

        target_h, target_w = _imgsz_hw(effective_imgsz)
        scale = min(target_w / orig_w, target_h / orig_h)
        selected = boxes_raw[query_indexes]
        xywh = (
            selected[:, :4]
            * np.asarray([target_w, target_h, target_w, target_h], dtype=np.float32)
            / np.float32(scale)
        )
        angles = selected[:, 4] * np.float32(np.pi)
        keep = scores > conf
        xywh = xywh[keep]
        angles = angles[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        half_w = xywh[:, 2] / 2
        half_h = xywh[:, 3] / 2
        cos = np.abs(np.cos(angles))
        sin = np.abs(np.sin(angles))
        extent_x = cos * half_w + sin * half_h
        extent_y = sin * half_w + cos * half_h
        boxes = np.stack(
            [
                xywh[:, 0] - extent_x,
                xywh[:, 1] - extent_y,
                xywh[:, 0] + extent_x,
                xywh[:, 1] + extent_y,
            ],
            axis=-1,
        )
        obb = np.concatenate(
            [
                xywh,
                angles[:, None],
                scores[:, None],
                class_ids.astype(np.float32)[:, None],
            ],
            axis=-1,
        )
        return boxes, scores, class_ids, None, obb

    # =========================================================================
    # Result building
    # =========================================================================

    @staticmethod
    def _parse_classify_probs(all_outputs) -> torch.Tensor:
        logits = np.asarray(all_outputs[0])
        if logits.ndim == 1:
            logits = logits[None, :]
        if logits.ndim != 2:
            raise ValueError(
                "Classification backend output must have shape (batch, classes), "
                f"got {tuple(logits.shape)}."
            )
        logits_t = torch.from_numpy(logits).float()
        return torch.softmax(logits_t, dim=1)[0]

    @staticmethod
    def _parse_embeddings(all_outputs) -> torch.Tensor:
        embeddings = np.asarray(all_outputs[0], dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]
        if embeddings.ndim != 2:
            raise ValueError(
                "Embedding backend output must have shape (batch, dimensions), "
                f"got {tuple(embeddings.shape)}."
            )
        embeddings_t = torch.from_numpy(np.ascontiguousarray(embeddings))
        return F.normalize(embeddings_t, dim=1)

    @staticmethod
    def _parse_restore_output(
        all_outputs, original_size: Tuple[int, int], scale: int = 1
    ) -> np.ndarray:
        """Decode backend restoration output to HWC uint8 RGB.

        For super-resolution the valid canvas is ``scale`` times the input, so
        the output is cropped to ``scale`` x the original size.
        """
        restored = np.asarray(all_outputs[0])
        if restored.ndim == 4:
            restored = restored[0]
        if restored.ndim == 3 and restored.shape[0] == 3:
            restored = np.transpose(restored, (1, 2, 0))
        if restored.ndim != 3 or restored.shape[-1] != 3:
            raise ValueError(
                "Restoration backend output must have shape [B, 3, H, W] "
                f"or [H, W, 3], got {tuple(restored.shape)}."
            )
        orig_w, orig_h = original_size
        restored = restored[: orig_h * int(scale), : orig_w * int(scale), :]
        return (np.clip(restored, 0.0, 1.0) * 255.0).round().astype(np.uint8)

    def _build_classify_result(
        self,
        all_outputs,
        *,
        orig_shape: Tuple[int, int],
        image_path,
    ) -> Results:
        return Results(
            boxes=None,
            probs=Probs(self._parse_classify_probs(all_outputs)),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
        )

    def _build_embedding_result(
        self,
        all_outputs,
        *,
        orig_shape: Tuple[int, int],
        image_path,
    ) -> Results:
        return Results(
            boxes=None,
            embeddings=Embeddings(self._parse_embeddings(all_outputs), orig_shape),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
        )

    @staticmethod
    def _parse_depth_output(
        all_outputs, original_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Decode backend depth output to an (H, W) float map on the original canvas."""
        depth = np.asarray(all_outputs[0], dtype=np.float32)
        if depth.ndim == 2:
            depth = depth[None, None]
        elif depth.ndim == 3:
            depth = depth[:, None] if depth.shape[0] == 1 else depth[None]
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError(
                "Depth backend output must have shape [B, 1, H, W], "
                f"got {tuple(np.asarray(all_outputs[0]).shape)}."
            )
        orig_w, orig_h = original_size
        depth_t = torch.from_numpy(np.ascontiguousarray(depth))
        # align_corners=True matches the native depth families' postprocess.
        depth_t = F.interpolate(
            depth_t, size=(orig_h, orig_w), mode="bilinear", align_corners=True
        )
        return depth_t[0, 0]

    @staticmethod
    def _parse_depth_anything3_output(
        all_outputs, original_size: Tuple[int, int]
    ) -> torch.Tensor:
        if len(all_outputs) != 2:
            raise ValueError(
                "Depth Anything 3 backend output must contain depth and sky maps."
            )
        depth = torch.from_numpy(
            np.ascontiguousarray(np.asarray(all_outputs[0], dtype=np.float32))
        )
        sky = torch.from_numpy(
            np.ascontiguousarray(np.asarray(all_outputs[1], dtype=np.float32))
        )
        if depth.ndim != 4 or sky.shape != depth.shape or depth.shape[1] != 1:
            raise ValueError(
                "Depth Anything 3 backend depth and sky outputs must both have "
                f"shape [B, 1, H, W], got {tuple(depth.shape)} and "
                f"{tuple(sky.shape)}."
            )

        corrected = depth
        for index in range(depth.shape[0]):
            non_sky = sky[index] < 0.3
            if non_sky.sum() <= 10 or (~non_sky).sum() <= 10:
                continue
            non_sky_depth = depth[index][non_sky]
            if non_sky_depth.numel() > 100_000:
                sample_indices = torch.randint(
                    0,
                    non_sky_depth.numel(),
                    (100_000,),
                )
                non_sky_depth = non_sky_depth[sample_indices]
            far_depth = torch.quantile(non_sky_depth, 0.99)
            if corrected is depth:
                corrected = depth.clone()
            corrected[index] = torch.where(non_sky, depth[index], far_depth)

        inverse_depth = torch.reciprocal(corrected.clamp_min(1e-6))
        return BaseBackend._parse_depth_output([inverse_depth.numpy()], original_size)

    def _parse_depth_outputs(
        self, all_outputs, original_size: Tuple[int, int]
    ) -> torch.Tensor:
        if self.model_family == "depth_anything3":
            return self._parse_depth_anything3_output(all_outputs, original_size)
        return self._parse_depth_output(all_outputs, original_size)

    def _build_depth_result(
        self,
        all_outputs,
        *,
        orig_shape: Tuple[int, int],
        original_size: Tuple[int, int],
        image_path,
    ) -> Results:
        depth = self._parse_depth_outputs(all_outputs, original_size)
        return Results(
            boxes=None,
            depth_map=DepthMap(depth, orig_shape),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
        )

    @staticmethod
    def _parse_normal_output(
        all_outputs, original_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Decode backend normals to an HWC unit field on the original canvas."""
        source = np.asarray(all_outputs[0])
        normal = np.asarray(source, dtype=np.float32)
        if normal.ndim == 3:
            if normal.shape[0] == 3:
                normal = normal[None]
            elif normal.shape[-1] == 3:
                normal = np.transpose(normal, (2, 0, 1))[None]
        elif normal.ndim == 4 and normal.shape[-1] == 3:
            normal = np.transpose(normal, (0, 3, 1, 2))
        if normal.ndim != 4 or normal.shape[0] != 1 or normal.shape[1] != 3:
            raise ValueError(
                "Normal backend output must have shape [1, 3, H, W] or "
                f"[1, H, W, 3], got {tuple(source.shape)}."
            )

        orig_w, orig_h = original_size
        normal_t = torch.from_numpy(np.ascontiguousarray(normal))
        normal_t = F.interpolate(
            normal_t,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        finite = torch.isfinite(normal_t).all(dim=1, keepdim=True)
        safe = torch.where(finite, normal_t, 0.0)
        norms = torch.linalg.vector_norm(safe, dim=1, keepdim=True)
        valid = finite & (norms > 1e-12)
        unit = safe / norms.clamp_min(1e-12)
        fallback = torch.zeros_like(unit)
        fallback[:, 2] = -1.0
        unit = torch.where(valid, unit, fallback)
        return unit[0].permute(1, 2, 0).contiguous()

    @staticmethod
    def _parse_edge_output(all_outputs, original_size: Tuple[int, int]) -> torch.Tensor:
        """Decode backend edge output to an (H, W) probability map."""
        edges = np.asarray(all_outputs[0], dtype=np.float32)
        if edges.ndim == 2:
            edges = edges[None, None]
        elif edges.ndim == 3:
            edges = edges[:, None] if edges.shape[0] == 1 else edges[None]
        if edges.ndim != 4 or edges.shape[1] != 1:
            raise ValueError(
                "Edge backend output must have shape [B, 1, H, W], "
                f"got {tuple(np.asarray(all_outputs[0]).shape)}."
            )
        orig_w, orig_h = original_size
        edge_tensor = torch.from_numpy(np.ascontiguousarray(edges))
        edge_tensor = F.interpolate(
            edge_tensor,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        return edge_tensor[0, 0].clamp(0.0, 1.0)

    def _build_normal_result(
        self,
        all_outputs,
        *,
        orig_shape: Tuple[int, int],
        original_size: Tuple[int, int],
        image_path,
    ) -> Results:
        normal = self._parse_normal_output(all_outputs, original_size)
        return Results(
            boxes=None,
            normal_map=NormalMap(normal, orig_shape),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
        )

    def _build_edge_result(
        self,
        all_outputs,
        *,
        orig_shape: Tuple[int, int],
        original_size: Tuple[int, int],
        image_path,
    ) -> Results:
        edges = self._parse_edge_output(all_outputs, original_size)
        return Results(
            boxes=None,
            edges=EdgeMap(edges, orig_shape),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
        )

    def _parse_semantic_output(
        self,
        all_outputs,
        original_size: Tuple[int, int],
        effective_imgsz: ImageSize,
        ratio: float,
    ) -> torch.Tensor:
        logits = np.asarray(all_outputs[0], dtype=np.float32)
        if logits.ndim == 3:
            logits = logits[None]
        if logits.ndim != 4:
            raise ValueError(
                "Semantic backend output must have shape [B, C, H, W], "
                f"got {tuple(np.asarray(all_outputs[0]).shape)}."
            )
        orig_w, orig_h = original_size
        logits_t = torch.from_numpy(np.ascontiguousarray(logits))
        align_corners = False
        if self.model_family in {"pidnet", "segformer"}:
            input_h, input_w = _imgsz_hw(effective_imgsz)
            scale_y = logits_t.shape[-2] / input_h
            scale_x = logits_t.shape[-1] / input_w
            valid_h = min(
                logits_t.shape[-2], max(int(round(orig_h * ratio * scale_y)), 1)
            )
            valid_w = min(
                logits_t.shape[-1], max(int(round(orig_w * ratio * scale_x)), 1)
            )
            logits_t = logits_t[..., :valid_h, :valid_w]
            align_corners = self.model_family == "pidnet"
        logits_t = F.interpolate(
            logits_t,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=align_corners,
        )
        return logits_t.argmax(dim=1)[0]

    def _build_semantic_result(
        self,
        all_outputs,
        *,
        orig_shape: Tuple[int, int],
        original_size: Tuple[int, int],
        effective_imgsz: ImageSize,
        ratio: float,
        image_path,
    ) -> Results:
        semantic = self._parse_semantic_output(
            all_outputs, original_size, effective_imgsz, ratio
        )
        return Results(
            boxes=None,
            semantic_mask=SemanticMask(semantic, orig_shape),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
        )

    @staticmethod
    def _parse_matte_output(
        all_outputs, original_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Decode matte logits to a soft alpha map on the original canvas."""
        logits = np.asarray(all_outputs[-1], dtype=np.float32)
        if logits.ndim == 2:
            logits = logits[None, None]
        elif logits.ndim == 3:
            logits = logits[:, None] if logits.shape[0] == 1 else logits[None]
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(
                "Matte backend output must have shape [B, 1, H, W], "
                f"got {tuple(np.asarray(all_outputs[-1]).shape)}."
            )
        orig_w, orig_h = original_size
        matte = torch.sigmoid(torch.from_numpy(np.ascontiguousarray(logits)))
        matte = F.interpolate(
            matte,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )
        return matte[0, 0].clamp(0.0, 1.0)

    def _build_matte_result(
        self,
        all_outputs,
        *,
        orig_shape: Tuple[int, int],
        original_size: Tuple[int, int],
        image_path,
    ) -> Results:
        matte = self._parse_matte_output(all_outputs, original_size)
        return Results(
            boxes=None,
            matte=Matte(matte, orig_shape),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
        )

    def _build_gaze_result(
        self,
        all_outputs,
        *,
        orig_shape: Tuple[int, int],
        image_path,
    ) -> Results:
        """Decode L2CS yaw/pitch logits for a single face-crop input."""
        if len(all_outputs) != 2:
            raise ValueError(
                f"Gaze backend requires yaw and pitch logits, got {len(all_outputs)} outputs."
            )
        from ..models.l2cs.utils import bin_logits_to_angles

        yaw = torch.from_numpy(
            np.ascontiguousarray(np.asarray(all_outputs[0], dtype=np.float32))
        )
        pitch = torch.from_numpy(
            np.ascontiguousarray(np.asarray(all_outputs[1], dtype=np.float32))
        )
        angles = bin_logits_to_angles(
            yaw,
            pitch,
            num_bins=self.num_bins,
            bin_width_deg=self.bin_width_deg,
            offset_deg=self.offset_deg,
        )
        orig_h, orig_w = orig_shape
        boxes = Boxes(
            torch.tensor([[0.0, 0.0, float(orig_w), float(orig_h)]]),
            torch.ones(1),
            torch.zeros(1),
            orig_shape=orig_shape,
        )
        return Results(
            boxes=boxes,
            gaze=Gaze(angles, orig_shape),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
        )

    def _build_point_result(
        self,
        all_outputs,
        *,
        orig_shape: Tuple[int, int],
        original_size: Tuple[int, int],
        effective_imgsz: ImageSize,
        conf: float,
        max_det: int,
        image_path,
    ) -> Results:
        if self.model_family != "fomo":
            raise NotImplementedError(
                f"Exported point parsing is not implemented for {self.model_family!r}."
            )
        from ..models.fomo.utils import postprocess as postprocess_fomo

        heatmap = torch.from_numpy(
            np.ascontiguousarray(np.asarray(all_outputs[0], dtype=np.float32))
        )
        input_h, input_w = _imgsz_hw(effective_imgsz)
        if input_h != input_w:
            raise NotImplementedError("FOMO exported inference requires square imgsz.")
        decoded = postprocess_fomo(
            heatmap,
            conf_thres=conf,
            input_size=input_h,
            original_size=original_size,
            max_det=max_det,
        )["points"]
        return Results(
            boxes=None,
            points=Points(decoded, orig_shape),
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
        )

    def _build_restore_result(
        self,
        all_outputs,
        *,
        orig_shape: Tuple[int, int],
        original_size: Tuple[int, int],
        image_path,
    ) -> Results:
        scale = self.restore_scale
        restored = self._parse_restore_output(all_outputs, original_size, scale)
        restored_hw = (int(restored.shape[0]), int(restored.shape[1]))
        return Results(
            boxes=None,
            restored=RestoredImage(torch.from_numpy(restored), restored_hw),
            orig_shape=orig_shape,
            restore_scale=scale,
            path=str(image_path) if image_path else None,
            names=self.names,
        )

    def _build_result(
        self,
        boxes: np.ndarray,
        max_scores: np.ndarray,
        class_ids: np.ndarray,
        *,
        masks: "np.ndarray | None" = None,
        obb: "np.ndarray | None" = None,
        keypoints: "np.ndarray | None" = None,
        orig_shape: Tuple[int, int],
        image_path,
        iou: float,
        classes: Optional[List[int]],
        max_det: int,
    ) -> Results:
        """Apply family-appropriate suppression/max_det/filtering and wrap."""
        if self.model_family == "ssd":
            max_det = min(max(0, int(max_det)), 200)

        if len(boxes) == 0:
            keypoints_obj = None
            if keypoints is not None:
                keypoints_obj = Keypoints(_f32(keypoints), orig_shape)
            return Results(
                boxes=Boxes(
                    _zeros_f32((0, 4)),
                    _zeros_f32((0,)),
                    _zeros_f32((0,)),
                ),
                obb=OBB(_zeros_f32((0, 7)), orig_shape)
                if self.task == "obb"
                else None,
                keypoints=keypoints_obj,
                orig_shape=orig_shape,
                path=str(image_path) if image_path else None,
                names=self.names,
            )

        if obb is None and not _is_nms_free_family(self.model_family):
            # YOLO9 needs class-aware NMS so multi-label detections
            # on a shared anchor (same box, different class) survive, matching
            # the native batched_nms path. Class-agnostic NMS would drop the
            # lower-scored class and make exported runtimes disagree with native.
            # ONNX models with graph-embedded NMS still pass through this after
            # backend clipping so letterboxed-image behavior stays aligned with
            # native YOLO9 postprocess.
            if self.model_family in (
                "efficientdet",
                "fcos",
                "picodet",
                "ppyoloe",
                "retinanet",
                "rtmdet",
                "ssd",
                "yolo9",
                "yolonas",
                "yolox",
            ):
                keep = _batched_nms_numpy(boxes, max_scores, class_ids, iou)
            else:
                keep = _nms_numpy(boxes, max_scores, iou)
            boxes, max_scores, class_ids = (
                boxes[keep],
                max_scores[keep],
                class_ids[keep],
            )
            if masks is not None:
                masks = masks[keep]
            if keypoints is not None:
                keypoints = keypoints[keep]

        if self.model_family == "fcos":
            max_det = min(int(max_det), 100)

        if len(boxes) > max_det:
            top_indices = np.argsort(max_scores)[::-1][:max_det]
            boxes = boxes[top_indices]
            max_scores = max_scores[top_indices]
            class_ids = class_ids[top_indices]
            if masks is not None:
                masks = masks[top_indices]
            if obb is not None:
                obb = obb[top_indices]
            if keypoints is not None:
                keypoints = keypoints[top_indices]

        boxes_t = _f32(boxes)
        conf_t = _f32(max_scores)
        cls_t = _f32(class_ids)
        obb_t = _f32(obb) if obb is not None else None

        if classes is not None and len(boxes_t) > 0:
            cls_mask = _zeros_bool(len(cls_t))
            for cid in classes:
                cls_mask |= cls_t == cid
            boxes_t = boxes_t[cls_mask]
            conf_t = conf_t[cls_mask]
            cls_t = cls_t[cls_mask]
            mask_np = _to_blob(cls_mask)
            if masks is not None:
                masks = masks[mask_np]
            if obb_t is not None:
                obb_t = obb_t[cls_mask]
            if keypoints is not None:
                keypoints = keypoints[mask_np]

        masks_obj = None
        if masks is not None and len(masks) > 0:
            masks_obj = Masks(_bool_array(masks), orig_shape=orig_shape)

        keypoints_obj = None
        if keypoints is not None:
            keypoints_obj = Keypoints(_f32(keypoints), orig_shape)

        obb_obj = None
        if obb_t is not None:
            obb_obj = OBB(obb_t, orig_shape)

        return Results(
            boxes=Boxes(boxes_t, conf_t, cls_t),
            masks=masks_obj,
            keypoints=keypoints_obj,
            obb=obb_obj,
            orig_shape=orig_shape,
            path=str(image_path) if image_path else None,
            names=self.names,
        )

    # =========================================================================
    # Save
    # =========================================================================

    def _save_annotated(self, result, original_img, image_path, output_path):
        """Save annotated image to disk."""
        annotated_img = original_img
        forced_ext = None
        if result.boxes is None and getattr(result, "probs", None) is not None:
            pass
        elif result.boxes is None and getattr(result, "restored", None) is not None:
            annotated_img = Image.fromarray(result.restored.array, mode="RGB")
        elif result.boxes is None and getattr(result, "depth_map", None) is not None:
            from ..utils.drawing import draw_depth_map

            depth_data = result.depth_map.data
            if isinstance(depth_data, torch.Tensor):
                depth_data = depth_data.cpu().numpy()
            annotated_img = draw_depth_map(original_img, depth_data)
        elif result.boxes is None and getattr(result, "normal_map", None) is not None:
            from ..utils.drawing import draw_normal_map

            normal_data = result.normal_map.data
            if isinstance(normal_data, torch.Tensor):
                normal_data = normal_data.cpu().numpy()
            annotated_img = draw_normal_map(original_img, normal_data)
        elif result.boxes is None and getattr(result, "edges", None) is not None:
            from ..utils.drawing import draw_edge_map

            edge_data = result.edges.data
            if isinstance(edge_data, torch.Tensor):
                edge_data = edge_data.cpu().numpy()
            annotated_img = draw_edge_map(original_img, edge_data)
        elif (
            result.boxes is None and getattr(result, "semantic_mask", None) is not None
        ):
            mask_data = result.semantic_mask.data
            if isinstance(mask_data, torch.Tensor):
                mask_data = mask_data.cpu().numpy()
            annotated_img = draw_semantic_mask(original_img, mask_data)
        elif result.boxes is None and getattr(result, "matte", None) is not None:
            annotated_img = Image.fromarray(result.cutout(original_img), mode="RGBA")
            forced_ext = "png"
        elif result.boxes is None and getattr(result, "points", None) is not None:
            if len(result.points) > 0:
                annotated_img = draw_points(
                    original_img,
                    result.points.xy.tolist(),
                    result.points.conf.tolist(),
                    result.points.cls.tolist(),
                    class_names=result.names,
                )
        elif len(result) > 0:
            if result.masks is not None:
                annotated_img = draw_masks(
                    annotated_img,
                    result.masks.data.numpy(),
                    result.boxes.cls.tolist(),
                )
            if result.obb is not None:
                annotated_img = draw_obb(
                    annotated_img,
                    result.obb.xywhr.tolist(),
                    result.obb.conf.tolist(),
                    result.obb.cls.tolist(),
                    class_names=self.names,
                )
            else:
                annotated_img = draw_boxes(
                    annotated_img,
                    result.boxes.xyxy.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.cls.tolist(),
                    class_names=self.names,
                )
            if result.keypoints is not None:
                kpts_np = result.keypoints.data
                if isinstance(kpts_np, torch.Tensor):
                    kpts_np = kpts_np.cpu().numpy()
                annotated_img = draw_keypoints(annotated_img, kpts_np)

        ext = forced_ext or (
            Path(image_path).suffix.lstrip(".") if image_path else "jpg"
        )
        if not ext:
            ext = "jpg"
        if output_path:
            final_path = resolve_save_path(output_path, image_path, ext=ext)
        else:
            stem = get_safe_stem(image_path) if image_path else "inference"
            model_tag = Path(self.model_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_dir = Path("runs/detections")
            save_dir.mkdir(parents=True, exist_ok=True)
            final_path = save_dir / f"{stem}_{model_tag}_{timestamp}.{ext}"
        if forced_ext is not None:
            final_path = Path(final_path).with_suffix(f".{forced_ext}")

        annotated_img.save(final_path)
        log_saved_result(result, final_path)

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def build_names(nb_classes: int) -> Dict[int, str]:
        """Build a class names dict — COCO for 80 classes, generic otherwise."""
        if nb_classes == 80:
            return {i: n for i, n in enumerate(COCO_CLASSES)}
        return {i: f"class_{i}" for i in range(nb_classes)}

    def eval(self):
        return self

    def _get_model_name(self) -> str:
        return self.model_family or "export"

    def _get_input_size(self) -> ImageSize:
        return self.imgsz

    def _get_val_preprocessor(self, img_size: ImageSize | None = None):
        if img_size is None:
            img_size = self._get_input_size()

        from ..validation.preprocessors import (
            DEIMValPreprocessor,
            DEIMv2DINOValPreprocessor,
            DEIMv2ValPreprocessor,
            DeformableDETRValPreprocessor,
            CenterNetValPreprocessor,
            DETRValPreprocessor,
            DFINEValPreprocessor,
            ECValPreprocessor,
            EfficientDetValPreprocessor,
            LWDETRValPreprocessor,
            PICODETValPreprocessor,
            PPYOLOEValPreprocessor,
            RFDETRValPreprocessor,
            RTDETRValPreprocessor,
            RTDETRv2OBBValPreprocessor,
            RTDETRv2ValPreprocessor,
            RTMDetValPreprocessor,
            StandardValPreprocessor,
            YOLO9E2EValPreprocessor,
            YOLO9ValPreprocessor,
            YOLONASValPreprocessor,
            YOLOXValPreprocessor,
        )

        if self.model_family == "deimv2":
            from ..models.deimv2.nn import DINO_SIZES

            model_size = self.model_size or getattr(self, "size", None)
            preprocessor_cls = (
                DEIMv2DINOValPreprocessor
                if model_size in DINO_SIZES
                else DEIMv2ValPreprocessor
            )
            return preprocessor_cls(img_size=_imgsz_hw(img_size))

        if self.model_family == "tinyformer":
            return DEIMv2DINOValPreprocessor(img_size=_imgsz_hw(img_size))

        preprocessor_cls = {
            "deim": DEIMValPreprocessor,
            "deformable_detr": DeformableDETRValPreprocessor,
            "centernet": CenterNetValPreprocessor,
            "detr": DETRValPreprocessor,
            "dinodetr": DeformableDETRValPreprocessor,
            "dfine": DFINEValPreprocessor,
            "ec": ECValPreprocessor,
            "efficientdet": EfficientDetValPreprocessor,
            "lwdetr": LWDETRValPreprocessor,
            "picodet": PICODETValPreprocessor,
            "ppyoloe": PPYOLOEValPreprocessor,
            "rfdetr": RFDETRValPreprocessor,
            "rtdetr": RTDETRValPreprocessor,
            "rtdetrv2": (
                RTDETRv2OBBValPreprocessor
                if getattr(self, "task", "detect") == "obb"
                else RTDETRv2ValPreprocessor
            ),
            "rtdetrv4": DFINEValPreprocessor,
            "rtmdet": RTMDetValPreprocessor,
            "yolo9": YOLO9ValPreprocessor,
            "yolo9_e2e": YOLO9E2EValPreprocessor,
            "yolo9_p2": YOLO9ValPreprocessor,
            "yolonas": YOLONASValPreprocessor,
            "yolox": YOLOXValPreprocessor,
        }.get(self.model_family, StandardValPreprocessor)
        return preprocessor_cls(img_size=_imgsz_hw(img_size))

    def _resolve_predict_imgsz(self, imgsz: ImageSize | None = None) -> ImageSize:
        effective = _normalize_imgsz(imgsz if imgsz is not None else self.imgsz)
        if (
            _is_rectangular_imgsz(effective)
            and (self.model_family or "").lower() not in _RECTANGULAR_BACKEND_FAMILIES
        ):
            raise NotImplementedError(
                "Rectangular imgsz backend inference is currently supported "
                "for YOLO9-family, HRNet, NAFNet, QuickSRNet, and "
                "Real-ESRGAN exports only."
            )
        return effective

    def _forward(self, input_tensor: torch.Tensor):
        blob = _to_blob(input_tensor)
        try:
            outputs = self._run_inference(blob)
        except Exception:
            if blob.shape[0] <= 1:
                raise
            per_image_outputs = [
                self._run_inference(blob[i : i + 1]) for i in range(blob.shape[0])
            ]
            outputs = [
                np.concatenate(
                    [np.asarray(item[j]) for item in per_image_outputs], axis=0
                )
                for j in range(len(per_image_outputs[0]))
            ]
        return [torch.from_numpy(np.asarray(output)) for output in outputs]

    @staticmethod
    def _as_numpy_outputs(output) -> list:
        if isinstance(output, torch.Tensor):
            return [output.detach().cpu().numpy()]
        if isinstance(output, np.ndarray):
            return [output]
        if isinstance(output, (list, tuple)):
            arrays = []
            for item in output:
                if isinstance(item, torch.Tensor):
                    arrays.append(item.detach().cpu().numpy())
                else:
                    arrays.append(np.asarray(item))
            return arrays
        return [np.asarray(output)]

    @staticmethod
    def _unpack_parsed_outputs(parsed):
        if len(parsed) == 6:
            return parsed
        if len(parsed) == 5:
            boxes, max_scores, class_ids, masks, obb = parsed
            return boxes, max_scores, class_ids, masks, obb, None
        boxes, max_scores, class_ids, masks = parsed
        return boxes, max_scores, class_ids, masks, None, None

    def _postprocess(
        self,
        output,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        input_size: ImageSize | None = None,
        letterbox: bool = False,
        max_det: int = 300,
        ratio: float | None = None,
        **kwargs,
    ) -> Dict:
        effective_imgsz = self._resolve_predict_imgsz(input_size)
        outputs = self._as_numpy_outputs(output)
        if self.task == "classify":
            return {"probs": self._parse_classify_probs(outputs)}
        if self.task == "embed":
            return {"embeddings": self._parse_embeddings(outputs)}
        if self.task == "restore":
            restored = np.asarray(outputs[0])
            if restored.ndim == 4:
                orig_w, orig_h = original_size
                restored = restored[:, :, :orig_h, :orig_w]
            return {"restored": torch.from_numpy(restored).float().clamp(0.0, 1.0)}
        if self.task == "depth":
            return {"depth": self._parse_depth_outputs(outputs, original_size)}
        if self.task == "normal":
            return {"normal": self._parse_normal_output(outputs, original_size)}
        if self.task == "edge":
            return {"edges": self._parse_edge_output(outputs, original_size)}
        if self.task == "matte":
            return {"matte": self._parse_matte_output(outputs, original_size)}
        if self.task == "gaze":
            result = self._build_gaze_result(
                outputs,
                orig_shape=(int(original_size[1]), int(original_size[0])),
                image_path=None,
            )
            return {"gaze": result.gaze.data}
        if self.task == "semantic":
            return {
                "semantic": self._parse_semantic_output(
                    outputs,
                    original_size,
                    effective_imgsz,
                    float(ratio or 1.0),
                )
            }
        if self.task == "point":
            result = self._build_point_result(
                outputs,
                orig_shape=(int(original_size[1]), int(original_size[0])),
                original_size=original_size,
                effective_imgsz=effective_imgsz,
                conf=conf_thres,
                max_det=max_det,
                image_path=None,
            )
            return {"points": result.points.data}
        parsed = self._parse_outputs(
            outputs,
            effective_imgsz,
            original_size,
            conf_thres,
            ratio=ratio,
            iou=iou_thres,
            max_det=max_det,
        )
        boxes, max_scores, class_ids, masks, obb, keypoints = (
            self._unpack_parsed_outputs(parsed)
        )
        result = self._build_result(
            boxes,
            max_scores,
            class_ids,
            masks=masks,
            obb=obb,
            keypoints=keypoints,
            orig_shape=(int(original_size[1]), int(original_size[0])),
            image_path=None,
            iou=iou_thres,
            classes=None,
            max_det=max_det,
        )

        det: Dict[str, object] = {
            "num_detections": len(result),
            "boxes": result.boxes.xyxy,
            "scores": result.boxes.conf,
            "classes": result.boxes.cls.to(torch.int64),
        }
        if result.masks is not None:
            det["masks"] = result.masks.data
        if result.keypoints is not None:
            det["keypoints"] = result.keypoints.data
        if result.obb is not None:
            det["obb"] = result.obb.data
        return det

    def val(
        self,
        data: str | None = None,
        batch: int = 16,
        imgsz: ImageSize | None = None,
        conf: float = 0.001,
        iou: float = 0.6,
        workers: int = 4,
        allow_download_scripts: bool = False,
        device: str | None = None,
        split: str = "val",
        augment: bool = False,
        save_json: bool = False,
        verbose: bool = True,
        *,
        plots: bool | None = None,
        **kwargs,
    ) -> Dict:
        from ..validation import (
            ClassifyValidator,
            DepthValidator,
            EdgeValidator,
            DetectionValidator,
            OBBValidator,
            PointValidator,
            PoseValidator,
            RestoreValidator,
            SemanticValidator,
            SegmentationValidator,
            ValidationConfig,
            MatteValidator,
            NormalValidator,
        )

        if augment:
            raise ValueError(
                "Augmented validation is not supported for exported backends"
            )
        if imgsz is None:
            imgsz = self._get_input_size()
        imgsz = self._resolve_predict_imgsz(imgsz)
        if self.model_family == "hrnet" and self.task == "pose":
            raise NotImplementedError(
                "Exported HRNet artifacts are person-crop pose heads. Validate the "
                "native composed HRNet model so a person detector can provide boxes."
            )
        if _is_rectangular_imgsz(imgsz):
            raise NotImplementedError(
                "Rectangular exported-backend validation is not supported yet."
            )
        if plots is not None and "save_plots" not in kwargs:
            kwargs["save_plots"] = plots

        validation_device = device or (
            self.device
            if _is_pytorch_cuda_device(self.device) and torch.cuda.is_available()
            else "cpu"
        )
        config = ValidationConfig(
            data=data,
            batch_size=batch,
            imgsz=imgsz,
            conf_thres=conf,
            iou_thres=iou,
            num_workers=workers,
            allow_download_scripts=allow_download_scripts,
            device=validation_device,
            split=split,
            augment=augment,
            save_json=save_json,
            verbose=verbose,
            **kwargs,
        )
        if self.task == "classify":
            validator_cls = ClassifyValidator
        elif self.task == "embed":
            raise NotImplementedError(
                "Exported embedding validation requires a retrieval dataset contract."
            )
        elif self.task == "point":
            validator_cls = PointValidator
        elif self.task == "segment":
            validator_cls = SegmentationValidator
        elif self.task == "pose":
            validator_cls = PoseValidator
        elif self.task == "obb":
            validator_cls = OBBValidator
        elif self.task == "restore":
            validator_cls = RestoreValidator
        elif self.task == "semantic":
            validator_cls = SemanticValidator
        elif self.task == "depth":
            validator_cls = DepthValidator
        elif self.task == "normal":
            validator_cls = NormalValidator
        elif self.task == "edge":
            validator_cls = EdgeValidator
        elif self.task == "matte":
            validator_cls = MatteValidator
        elif self.task == "gaze":
            raise NotImplementedError(
                "Exported gaze validation requires a gaze-labelled dataset contract."
            )
        else:
            validator_cls = DetectionValidator
        validator = validator_cls(model=self, config=config)
        return validator()

    # =========================================================================
    # Inference pipeline
    # =========================================================================

    def _predict_single(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        save: bool = False,
        output_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[ImageSize] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
        save_stem: Optional[str] = None,
    ) -> Results:
        """Run inference on a single image.

        ``save_stem`` overrides the saved filename stem for in-memory images
        (which have no path to derive one from).
        """
        image_path = image if isinstance(image, (str, Path)) else None
        effective_imgsz = self._resolve_predict_imgsz(imgsz)

        input_tensor, original_img, original_size, ratio = self._preprocess(
            image, effective_imgsz, color_format
        )

        blob = _to_blob(input_tensor)

        all_outputs = self._run_inference(blob)

        orig_w, orig_h = original_size
        orig_shape = (orig_h, orig_w)
        if self.task == "classify":
            result = self._build_classify_result(
                all_outputs,
                orig_shape=orig_shape,
                image_path=image_path,
            )
            if save:
                self._save_annotated(
                    result,
                    original_img,
                    image_path if image_path is not None else save_stem,
                    output_path,
                )
            return result
        if self.task == "embed":
            return self._build_embedding_result(
                all_outputs,
                orig_shape=orig_shape,
                image_path=image_path,
            )
        if self.task == "restore":
            result = self._build_restore_result(
                all_outputs,
                orig_shape=orig_shape,
                original_size=original_size,
                image_path=image_path,
            )
            if save:
                self._save_annotated(
                    result,
                    original_img,
                    image_path if image_path is not None else save_stem,
                    output_path,
                )
            return result
        if self.task == "depth":
            result = self._build_depth_result(
                all_outputs,
                orig_shape=orig_shape,
                original_size=original_size,
                image_path=image_path,
            )
            if save:
                self._save_annotated(
                    result,
                    original_img,
                    image_path if image_path is not None else save_stem,
                    output_path,
                )
            return result
        if self.task == "normal":
            result = self._build_normal_result(
                all_outputs,
                orig_shape=orig_shape,
                original_size=original_size,
                image_path=image_path,
            )
            if save:
                self._save_annotated(
                    result,
                    original_img,
                    image_path if image_path is not None else save_stem,
                    output_path,
                )
            return result
        if self.task == "edge":
            result = self._build_edge_result(
                all_outputs,
                orig_shape=orig_shape,
                original_size=original_size,
                image_path=image_path,
            )
            if save:
                self._save_annotated(
                    result,
                    original_img,
                    image_path if image_path is not None else save_stem,
                    output_path,
                )
            return result
        if self.task == "matte":
            result = self._build_matte_result(
                all_outputs,
                orig_shape=orig_shape,
                original_size=original_size,
                image_path=image_path,
            )
            if save:
                self._save_annotated(
                    result,
                    original_img,
                    image_path if image_path is not None else save_stem,
                    output_path,
                )
            return result
        if self.task == "gaze":
            result = self._build_gaze_result(
                all_outputs,
                orig_shape=orig_shape,
                image_path=image_path,
            )
            if save:
                self._save_annotated(
                    result,
                    original_img,
                    image_path if image_path is not None else save_stem,
                    output_path,
                )
            return result
        if self.task == "semantic":
            result = self._build_semantic_result(
                all_outputs,
                orig_shape=orig_shape,
                original_size=original_size,
                effective_imgsz=effective_imgsz,
                ratio=float(ratio or 1.0),
                image_path=image_path,
            )
            if save:
                self._save_annotated(
                    result,
                    original_img,
                    image_path if image_path is not None else save_stem,
                    output_path,
                )
            return result
        if self.task == "point":
            result = self._build_point_result(
                all_outputs,
                orig_shape=orig_shape,
                original_size=original_size,
                effective_imgsz=effective_imgsz,
                conf=conf,
                max_det=max_det,
                image_path=image_path,
            )
            if save:
                self._save_annotated(
                    result,
                    original_img,
                    image_path if image_path is not None else save_stem,
                    output_path,
                )
            return result

        parsed = self._parse_outputs(
            all_outputs,
            effective_imgsz,
            original_size,
            conf,
            ratio=ratio,
            iou=iou,
            max_det=max_det,
        )
        boxes, max_scores, class_ids, masks, obb, keypoints = (
            self._unpack_parsed_outputs(parsed)
        )

        result = self._build_result(
            boxes,
            max_scores,
            class_ids,
            masks=masks,
            obb=obb,
            keypoints=keypoints,
            orig_shape=orig_shape,
            image_path=image_path,
            iou=iou,
            classes=classes,
            max_det=max_det,
        )

        if save:
            self._save_annotated(
                result,
                original_img,
                image_path if image_path is not None else save_stem,
                output_path,
            )

        return result

    def _supports_batched_inference(self) -> bool:
        """Whether ``_run_inference`` accepts stacked (N, C, H, W) blobs.

        Default False: traced/compiled runtimes are typically baked to
        batch 1. Backends whose artifact declares a dynamic batch axis
        (ONNX, OpenVINO) override this; TensorRT manages batching itself
        in its own ``_process_in_batches``.
        """
        return False

    def _process_in_batches(
        self,
        images: List,
        batch: int = 1,
        save: bool = False,
        output_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[ImageSize] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
        start_idx: int = 0,
    ) -> List[Results]:
        """Process multiple images (file paths or in-memory).

        When ``batch > 1`` and the runtime accepts stacked blobs, each chunk
        of ``batch`` images runs as a single forward pass; otherwise images
        run sequentially.

        ``start_idx`` is the position of ``images[0]`` within the caller's full
        source. It only affects the indexed filename stems given to in-memory
        images, so streaming can hand over one chunk at a time without two
        chunks both saving an ``image0``.
        """
        use_batched = (
            batch > 1
            and self._supports_batched_inference()
            # Latched by _predict_batch after a runtime rejects a stacked
            # blob, so a long list does not retry (and warn) once per chunk.
            and not getattr(self, "_batched_inference_failed", False)
        )
        if use_batched:
            results = []
            for start in range(0, len(images), batch):
                results.extend(
                    self._predict_batch(
                        images[start : start + batch],
                        start_idx=start_idx + start,
                        save=save,
                        output_path=output_path,
                        conf=conf,
                        iou=iou,
                        imgsz=imgsz,
                        classes=classes,
                        max_det=max_det,
                        color_format=color_format,
                    )
                )
            return results

        results = []
        for idx, image in enumerate(images):
            results.append(
                self._predict_single(
                    image,
                    save=save,
                    output_path=output_path,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    classes=classes,
                    max_det=max_det,
                    color_format=color_format,
                    save_stem=(
                        None
                        if isinstance(image, (str, Path))
                        else f"image{start_idx + idx}"
                    ),
                )
            )
        return results

    def _stream_in_batches(
        self,
        images: List,
        batch: int = 1,
        **kwargs,
    ) -> Generator[Results, None, None]:
        """Yield Results for *images* one at a time, ``batch`` images per step.

        Peak memory holds one chunk of Results rather than the whole source.
        Each chunk still goes through ``_process_in_batches``, so runtime
        overrides (such as the TensorRT batching path) keep applying.
        """
        step = max(1, int(batch))
        for start in range(0, len(images), step):
            yield from self._process_in_batches(
                images[start : start + step],
                batch=batch,
                start_idx=start,
                **kwargs,
            )

    def _predict_batch(
        self,
        chunk: List,
        start_idx: int,
        *,
        save: bool = False,
        output_path: str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[ImageSize] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
    ) -> List[Results]:
        """Run one stacked forward pass over a chunk of images.

        Mirrors ``_predict_single`` step for step, except the preprocessed
        tensors are concatenated into a single blob for ``_run_inference``
        and the outputs are sliced back per image (``[i : i + 1]`` keeps the
        batch dim, which every output parser already expects). Falls back to
        the sequential path if the blob cannot be stacked or the runtime
        rejects the batched call.
        """
        effective_imgsz = self._resolve_predict_imgsz(imgsz)

        preprocessed = []
        for image in chunk:
            input_tensor, original_img, original_size, ratio = self._preprocess(
                image, effective_imgsz, color_format
            )
            image_path = image if isinstance(image, (str, Path)) else None
            preprocessed.append(
                (input_tensor, original_img, original_size, ratio, image_path)
            )

        tensors = [item[0] for item in preprocessed]
        all_outputs = None
        stackable = all(
            isinstance(t, torch.Tensor) and t.dim() == 4 and t.shape == tensors[0].shape
            for t in tensors
        )
        if stackable and not getattr(self, "_batched_inference_failed", False):
            blob = np.concatenate([_to_blob(t) for t in tensors], axis=0)
            try:
                all_outputs = self._run_inference(blob)
            except Exception as e:
                self._batched_inference_failed = True
                logger.warning(
                    "Batched inference failed for %s (%s); falling back to "
                    "sequential processing.",
                    Path(self.model_path).name,
                    e,
                )
        if all_outputs is None:
            return [
                self._predict_single(
                    image,
                    save=save,
                    output_path=output_path,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    classes=classes,
                    max_det=max_det,
                    color_format=color_format,
                    save_stem=(
                        None
                        if isinstance(image, (str, Path))
                        else f"image{start_idx + offset}"
                    ),
                )
                for offset, image in enumerate(chunk)
            ]

        results = []
        for offset, (_, original_img, original_size, ratio, image_path) in enumerate(
            preprocessed
        ):
            per_image = [
                np.asarray(output)[offset : offset + 1] for output in all_outputs
            ]
            save_name = (
                image_path if image_path is not None else f"image{start_idx + offset}"
            )
            orig_w, orig_h = original_size
            orig_shape = (orig_h, orig_w)

            if self.task == "classify":
                result = self._build_classify_result(
                    per_image,
                    orig_shape=orig_shape,
                    image_path=image_path,
                )
            elif self.task == "embed":
                result = self._build_embedding_result(
                    per_image,
                    orig_shape=orig_shape,
                    image_path=image_path,
                )
            elif self.task == "restore":
                result = self._build_restore_result(
                    per_image,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    image_path=image_path,
                )
            elif self.task == "depth":
                result = self._build_depth_result(
                    per_image,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    image_path=image_path,
                )
            elif self.task == "normal":
                result = self._build_normal_result(
                    per_image,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    image_path=image_path,
                )
            elif self.task == "edge":
                result = self._build_edge_result(
                    per_image,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    image_path=image_path,
                )
            elif self.task == "matte":
                result = self._build_matte_result(
                    per_image,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    image_path=image_path,
                )
            elif self.task == "gaze":
                result = self._build_gaze_result(
                    per_image,
                    orig_shape=orig_shape,
                    image_path=image_path,
                )
            elif self.task == "semantic":
                result = self._build_semantic_result(
                    per_image,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    effective_imgsz=effective_imgsz,
                    ratio=float(ratio or 1.0),
                    image_path=image_path,
                )
            elif self.task == "point":
                result = self._build_point_result(
                    per_image,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    effective_imgsz=effective_imgsz,
                    conf=conf,
                    max_det=max_det,
                    image_path=image_path,
                )
            else:
                parsed = self._parse_outputs(
                    per_image,
                    effective_imgsz,
                    original_size,
                    conf,
                    ratio=ratio,
                    iou=iou,
                    max_det=max_det,
                )
                boxes, max_scores, class_ids, masks, obb, keypoints = (
                    self._unpack_parsed_outputs(parsed)
                )
                result = self._build_result(
                    boxes,
                    max_scores,
                    class_ids,
                    masks=masks,
                    obb=obb,
                    keypoints=keypoints,
                    orig_shape=orig_shape,
                    image_path=image_path,
                    iou=iou,
                    classes=classes,
                    max_det=max_det,
                )

            if save:
                self._save_annotated(result, original_img, save_name, output_path)
            results.append(result)
        return results

    # =========================================================================
    # Public API
    # =========================================================================

    def info(self, detailed: bool = False, verbose: bool = True) -> Dict:
        """Return exported-runtime metadata and lightweight counts."""
        data = build_model_info(self, detailed=detailed)
        if verbose:
            logger.info(format_model_info(data))
        return data

    def __call__(
        self,
        source: Union[
            str, Path, int, Image.Image, np.ndarray, list, tuple, None
        ] = None,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[ImageSize] = None,
        device: str | None = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        save: bool = False,
        batch: int = 1,
        # video parameters
        stream: bool = False,
        stream_buffer: bool = False,
        vid_stride: int = 1,
        show: bool = False,
        output_path: str | None = None,
        color_format: str = "auto",
        **kwargs,
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        """Run inference on images, directories, videos, or screen captures."""
        normalize_predict_kwargs(kwargs)
        if device not in (None, "", "auto", self.device):
            logger.warning(
                "Backend was loaded on device=%s; predict(device=%s) is ignored. "
                "Load the backend with device=%s to change runtime device.",
                self.device,
                device,
                device,
            )

        source_spec = classify_source(source)

        # Handle finite video input.
        if source_spec.kind == SourceKind.VIDEO:
            gen = self._predict_video(
                source_spec.source,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                save=save,
                show=show,
                vid_stride=vid_stride,
                output_path=output_path,
            )
            if stream:
                return gen
            return collect_video_results(gen, source_spec.source, vid_stride)

        if source_spec.live:
            if not stream:
                raise ValueError(
                    "Live stream sources require stream=True so results are "
                    "consumed incrementally."
                )
            frame_source = build_stream_source(
                source_spec,
                vid_stride=vid_stride,
                stream_buffer=stream_buffer,
            )
            return self._predict_video(
                frame_source,
                source_label="stream",
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                save=save,
                show=show,
                output_path=output_path,
            )

        # Handle screen-capture input ("screen", "screen 1", "screen 1 x y w h")
        if source_spec.kind == SourceKind.SCREEN:
            return self._predict_screen(
                source_spec.source,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                save=save,
                show=show,
                stream=stream,
                vid_stride=vid_stride,
                output_path=output_path,
            )

        # Handle in-memory batch input (list/tuple of images) and directories.
        # Both collapse to a list of images run in batches.
        images = None
        if source_spec.kind == SourceKind.IMAGE_BATCH:
            images = list(source_spec.items)
        elif source_spec.kind == SourceKind.DIRECTORY:
            images = ImageLoader.collect_images(source_spec.source)
            if not images:
                return iter(()) if stream else []

        if images is not None or stream:
            if images is None:
                images = [source]
            batch_kwargs = dict(
                batch=batch,
                save=save,
                output_path=output_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                color_format=color_format,
            )
            if stream:
                return self._stream_in_batches(images, **batch_kwargs)
            return self._process_in_batches(images, **batch_kwargs)

        return self._predict_single(
            source,
            save=save,
            output_path=output_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=classes,
            max_det=max_det,
            color_format=color_format,
        )

    def predict(
        self, *args, **kwargs
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        """Alias for __call__ method."""
        return self(*args, **kwargs)

    def _predict_video(
        self,
        source: Union[str, Path, FrameSource],
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[ImageSize] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        save: bool = False,
        show: bool = False,
        vid_stride: int = 1,
        output_path: Optional[str] = None,
        source_label: Optional[str] = None,
    ) -> Generator[Results, None, None]:
        """Run inference on a video file, yielding per-frame Results."""
        source_label = str(source) if source_label is None else source_label
        effective_imgsz = self._resolve_predict_imgsz(imgsz)

        def predict_frame(pil_img):
            input_tensor, original_img, original_size, ratio = self._preprocess(
                pil_img, effective_imgsz, "rgb"
            )
            blob = _to_blob(input_tensor)
            all_outputs = self._run_inference(blob)
            orig_w, orig_h = original_size
            orig_shape = (orig_h, orig_w)
            if self.task == "classify":
                return self._build_classify_result(
                    all_outputs,
                    orig_shape=orig_shape,
                    image_path=source_label,
                )
            if self.task == "embed":
                return self._build_embedding_result(
                    all_outputs,
                    orig_shape=orig_shape,
                    image_path=source_label,
                )
            if self.task == "restore":
                return self._build_restore_result(
                    all_outputs,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    image_path=source_label,
                )
            if self.task == "depth":
                return self._build_depth_result(
                    all_outputs,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    image_path=source_label,
                )
            if self.task == "normal":
                return self._build_normal_result(
                    all_outputs,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    image_path=source_label,
                )
            if self.task == "edge":
                return self._build_edge_result(
                    all_outputs,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    image_path=source_label,
                )
            if self.task == "matte":
                return self._build_matte_result(
                    all_outputs,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    image_path=source_label,
                )
            if self.task == "gaze":
                return self._build_gaze_result(
                    all_outputs,
                    orig_shape=orig_shape,
                    image_path=source_label,
                )
            if self.task == "semantic":
                return self._build_semantic_result(
                    all_outputs,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    effective_imgsz=effective_imgsz,
                    ratio=float(ratio or 1.0),
                    image_path=source_label,
                )
            if self.task == "point":
                return self._build_point_result(
                    all_outputs,
                    orig_shape=orig_shape,
                    original_size=original_size,
                    effective_imgsz=effective_imgsz,
                    conf=conf,
                    max_det=max_det,
                    image_path=source_label,
                )
            parsed = self._parse_outputs(
                all_outputs,
                effective_imgsz,
                original_size,
                conf,
                ratio=ratio,
                iou=iou,
                max_det=max_det,
            )
            boxes, max_scores, class_ids, masks, obb, keypoints = (
                self._unpack_parsed_outputs(parsed)
            )
            return self._build_result(
                boxes,
                max_scores,
                class_ids,
                masks=masks,
                obb=obb,
                keypoints=keypoints,
                orig_shape=orig_shape,
                image_path=source_label,
                iou=iou,
                classes=classes,
                max_det=max_det,
            )

        yield from run_video_inference(
            source,
            predict_frame,
            vid_stride=vid_stride,
            save=save,
            show=show,
            output_path=output_path,
        )

    def _predict_screen(
        self,
        source: Union[str, Path],
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[ImageSize] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        save: bool = False,
        show: bool = False,
        stream: bool = False,
        vid_stride: int = 1,
        output_path: Optional[str] = None,
    ) -> Union[Results, Generator[Results, None, None]]:
        """Capture one screenshot, or continuously capture when streaming."""
        if stream:

            def stream_results():
                yield from self._predict_video(
                    ScreenSource(source, vid_stride=vid_stride),
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    classes=classes,
                    max_det=max_det,
                    save=save,
                    show=show,
                    output_path=output_path,
                    source_label=str(source),
                )

            return stream_results()

        result = self._predict_single(
            grab_screen(source),
            save=save,
            output_path=output_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=classes,
            max_det=max_det,
            color_format="rgb",
            save_stem="screen",
        )
        result.path = str(source)
        return result
