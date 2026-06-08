"""LibreRFDETR implementation for LibreYOLO."""

from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from libreyolo.training.ddp_spawn import ddp_aware
from PIL import Image

from ..base import BaseModel
from ...data import load_data_config
from ...tasks import normalize_task
from ...utils.image_loader import ImageInput, ImageLoader
from ...utils.serialization import load_trusted_torch_file
from .nn import LibreRFDETRModel, RFDETR_CONFIGS, RFDETR_SEG_CONFIGS
from .config import RFDETRConfig
from .utils import postprocess, preprocess_numpy
from .trainer import RFDETRTrainer
from ...validation.preprocessors import RFDETRValPreprocessor

# COCO 91-class to 80-class mapping.
# RF-DETR pretrained models output 91 COCO category IDs (1-90),
# but YOLO-format labels use a contiguous 80-class scheme (0-79).
_COCO91_TO_COCO80 = {
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 14,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 19,
    22: 20,
    23: 21,
    24: 22,
    25: 23,
    27: 24,
    28: 25,
    31: 26,
    32: 27,
    33: 28,
    34: 29,
    35: 30,
    36: 31,
    37: 32,
    38: 33,
    39: 34,
    40: 35,
    41: 36,
    42: 37,
    43: 38,
    44: 39,
    46: 40,
    47: 41,
    48: 42,
    49: 43,
    50: 44,
    51: 45,
    52: 46,
    53: 47,
    54: 48,
    55: 49,
    56: 50,
    57: 51,
    58: 52,
    59: 53,
    60: 54,
    61: 55,
    62: 56,
    63: 57,
    64: 58,
    65: 59,
    67: 60,
    70: 61,
    72: 62,
    73: 63,
    74: 64,
    75: 65,
    76: 66,
    77: 67,
    78: 68,
    79: 69,
    80: 70,
    81: 71,
    82: 72,
    84: 73,
    85: 74,
    86: 75,
    87: 76,
    88: 77,
    89: 78,
    90: 79,
}


_RFDETR_UPSTREAM_WEIGHT_URLS = {
    "rf-detr-nano.pth": "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
    "rf-detr-small.pth": "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
    "rf-detr-medium.pth": "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
    "rf-detr-large-2026.pth": "https://storage.googleapis.com/rfdetr/rf-detr-large-2026.pth",
    "rf-detr-seg-nano.pt": "https://storage.googleapis.com/rfdetr/rf-detr-seg-n-ft.pth",
    "rf-detr-seg-small.pt": "https://storage.googleapis.com/rfdetr/rf-detr-seg-s-ft.pth",
    "rf-detr-seg-medium.pt": "https://storage.googleapis.com/rfdetr/rf-detr-seg-m-ft.pth",
    "rf-detr-seg-large.pt": "https://storage.googleapis.com/rfdetr/rf-detr-seg-l-ft.pth",
    "rf-detr-seg-xlarge.pt": "https://storage.googleapis.com/rfdetr/rf-detr-seg-xl-ft.pth",
    "rf-detr-seg-xxlarge.pt": "https://storage.googleapis.com/rfdetr/rf-detr-seg-2xl-ft.pth",
}


def _checkpoint_model_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Extract a tensor state dict from RF-DETR/LibreYOLO checkpoint variants."""
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        checkpoint = checkpoint["model"]
    elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
        checkpoint = checkpoint["state_dict"]

    state = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = key.removeprefix("module.")
        key = key.removeprefix("model.")
        key = key.removeprefix("_orig_mod.")
        state[key] = value
    return state


class LibreRFDETR(BaseModel):
    """RF-DETR model for object detection and instance segmentation.

    RF-DETR is a Detection Transformer using DINOv2 backbone with
    multi-scale deformable attention. Segmentation variants add a
    lightweight mask head for instance segmentation.

    autobatch_fraction is lower than the default 0.60 because the probe's
    fake backward underestimates RF-DETR's real training memory (the loss
    backward runs through SetCriterion and 6 aux-loss decoder layers), and
    DDP adds gradient buckets on top.

    Args:
        model_path: Path to weights, pre-loaded state_dict, or None for pretrained.
        size: Model size variant ("n", "s", "m", "l").
        nb_classes: Number of classes (default: 80 for COCO).
        device: Device for inference.

    Example::

        >>> model = LibreRFDETR(size="s")
        >>> detections = model.predict("path/to/image.jpg")
    """

    autobatch_fraction: float = 0.45

    # Class-level metadata
    FAMILY = "rfdetr"
    FILENAME_PREFIX = "LibreRFDETR"
    INPUT_SIZES = {"n": 384, "s": 512, "m": 576, "l": 704}
    SEG_INPUT_SIZES = {"n": 312, "s": 384, "m": 432, "l": 504, "x": 624, "xx": 768}
    # Classification runs the DINOv2 backbone at 224 (divisible by patch_size 14).
    CLS_INPUT_SIZES = {"n": 224, "s": 224, "m": 224, "l": 224}
    SUPPORTED_TASKS = ("detect", "segment", "pose", "classify", "obb")
    TASK_INPUT_SIZES = {
        "detect": INPUT_SIZES,
        "segment": SEG_INPUT_SIZES,
        "pose": INPUT_SIZES,
        "classify": CLS_INPUT_SIZES,
        "obb": INPUT_SIZES,
    }
    EXPERIMENTAL_WEIGHT_FILENAMES = frozenset(
        {
            "librerfdetrn-pose.pt",
            "librerfdetrs-pose.pt",
            "librerfdetrm-pose.pt",
            "librerfdetrl-pose.pt",
        }
    )
    TRAIN_CONFIG = RFDETRConfig
    val_preprocessor_class = RFDETRValPreprocessor
    TTA_FIXED_SIZE = True  # resizes to a fixed square; multi-scale TTA is a no-op

    # CLI parameters intentionally ignored by native RF-DETR training.
    UNSUPPORTED_TRAIN_PARAMS: ClassVar[set[str]] = {
        "mosaic",
        "mixup",
        "degrees",
        "shear",
        "mosaic_scale",
        "mixup_scale",
        "optimizer",
        "momentum",
        "nesterov",
        "hsv_prob",
        "translate",
        "pretrained",
    }

    # =========================================================================
    # Registry classmethods
    # =========================================================================

    @classmethod
    def can_load(cls, weights_dict: dict) -> bool:
        keys_lower = [k.lower() for k in weights_dict]
        if any(
            "detr" in k
            or "dinov2" in k
            or "transformer" in k
            or ("encoder" in k and "decoder" in k)
            or "query_embed" in k
            or "class_embed" in k
            or "bbox_embed" in k
            for k in keys_lower
        ):
            return True
        # Classification checkpoints carry only the DINOv2 backbone + a linear
        # head, so they lack the detection/decoder markers above. Recognize the
        # backbone-plus-linear-head signature so the factory can route them.
        return "linear.weight" in weights_dict and any(
            k.startswith("backbone.") for k in weights_dict
        )

    @classmethod
    def detect_size(
        cls, weights_dict: dict, state_dict: dict | None = None
    ) -> Optional[str]:
        full_ckpt = state_dict if state_dict is not None else weights_dict
        if isinstance(full_ckpt, dict) and isinstance(full_ckpt.get("size"), str):
            return full_ckpt["size"]
        is_seg = any(k.startswith("segmentation_head") for k in weights_dict)

        RESOLUTION_TO_SIZE = {384: "n", 512: "s", 576: "m", 704: "l"}
        SEG_RESOLUTION_TO_SIZE = {
            312: "n",
            384: "s",
            432: "m",
            504: "l",
            624: "x",
            768: "xx",
        }
        res_map = SEG_RESOLUTION_TO_SIZE if is_seg else RESOLUTION_TO_SIZE

        args = full_ckpt.get("args")
        if args is not None:
            resolution = (
                getattr(args, "resolution", None)
                if hasattr(args, "resolution")
                else args.get("resolution")
                if isinstance(args, dict)
                else None
            )
            if resolution in res_map:
                return res_map[resolution]

        # Fallback: infer from backbone position_embeddings shape
        pos_key = "backbone.0.encoder.encoder.embeddings.position_embeddings"
        if pos_key in weights_dict:
            pos_tokens = weights_dict[pos_key].shape[1]
            token_map = (
                {
                    26 * 26 + 1: "n",
                    32 * 32 + 1: "s",
                    36 * 36 + 1: "m",
                    42 * 42 + 1: "l",
                    52 * 52 + 1: "x",
                    64 * 64 + 1: "xx",
                }
                if is_seg
                else {
                    24 * 24 + 1: "n",
                    32 * 32 + 1: "s",
                    36 * 36 + 1: "m",
                    44 * 44 + 1: "l",
                }
            )
            return token_map.get(pos_tokens)

        return None

    @classmethod
    def detect_nb_classes(cls, weights_dict: dict) -> Optional[int]:
        if "linear.weight" in weights_dict:
            return int(weights_dict["linear.weight"].shape[0])
        # RF-DETR class_embed has (num_classes + 1) outputs (includes background)
        if "class_embed.bias" in weights_dict:
            detected = int(weights_dict["class_embed.bias"].shape[0]) - 1
            if detected <= 0 and any(
                k.startswith("keypoint_head") for k in weights_dict
            ):
                return 1
            return detected
        return None

    @classmethod
    def detect_num_keypoints(cls, weights_dict: dict) -> Optional[int]:
        if "keypoint_head.layers.2.weight" in weights_dict:
            channels = int(weights_dict["keypoint_head.layers.2.weight"].shape[0])
            if channels % 3 == 0:
                return channels // 3
        return None

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        upstream_url = _RFDETR_UPSTREAM_WEIGHT_URLS.get(Path(filename).name.lower())
        if upstream_url is not None:
            return upstream_url
        return super().get_download_url(filename)

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        model_path: str | None = None,
        size: str | None = None,
        nb_classes: int = 80,
        device: str = "auto",
        segmentation: bool = False,
        task: str | None = None,
        num_keypoints: int = 17,
        keypoint_dim: int = 3,
        allow_detect_to_obb_transfer: bool = False,
        allow_detect_to_pose_transfer: bool = False,
        **kwargs,
    ):
        # Resolve task: explicit `task` > legacy `segmentation` flag > filename / checkpoint inference.
        if task is not None and segmentation and normalize_task(task) != "segment":
            raise ValueError(
                "Conflicting RF-DETR task options: segmentation=True requires task='segment'."
            )
        resolved_task = task
        if resolved_task is None and segmentation:
            resolved_task = "segment"
        if normalize_task(resolved_task) == "pose" and nb_classes == 80:
            nb_classes = 1
        self.num_keypoints = int(num_keypoints)
        self.keypoint_dim = int(keypoint_dim)
        if size is None and (
            model_path is None or (isinstance(model_path, dict) and not model_path)
        ):
            size = "s"

        if isinstance(model_path, dict) and not model_path:
            weight_source = None
        elif normalize_task(resolved_task) == "classify" and model_path is None:
            # Classification builds its own ImageNet-pretrained DINOv2 backbone;
            # the detection checkpoints do not apply.
            weight_source = None
        elif normalize_task(resolved_task) == "pose" and model_path is None:
            weight_source = None
        elif model_path is None:
            cfgs = (
                RFDETR_SEG_CONFIGS
                if normalize_task(resolved_task) == "segment"
                else RFDETR_CONFIGS
            )
            cfg = cfgs.get(size)
            default_weights = cfg.pretrain_weights if cfg is not None else None
            weight_source = (
                self._resolve_weights_path(default_weights)
                if default_weights is not None
                else None
            )
        elif isinstance(model_path, str):
            weight_source = self._resolve_weights_path(model_path)
        else:
            weight_source = model_path

        self._weight_source = weight_source
        self._allow_detect_to_obb_transfer = bool(allow_detect_to_obb_transfer)
        self._allow_detect_to_pose_transfer = bool(allow_detect_to_pose_transfer)
        if size is None:
            size = self._detect_size_from_source(weight_source)
            if size is None:
                raise ValueError(
                    "Could not automatically detect RF-DETR model size. "
                    "Pass size='n', 's', 'm', 'l', 'x', or 'xx'."
                )

        if weight_source is not None:
            checkpoint_task = self._detect_task_from_source(weight_source)
            if resolved_task is None:
                resolved_task = checkpoint_task
            elif checkpoint_task is not None:
                requested_task = normalize_task(resolved_task)
                allowed = requested_task == checkpoint_task or (
                    requested_task == "obb"
                    and checkpoint_task == "detect"
                    and self._allow_detect_to_obb_transfer
                ) or (
                    requested_task == "pose"
                    and checkpoint_task == "detect"
                    and self._allow_detect_to_pose_transfer
                )
                if not allowed:
                    raise ValueError(
                        f"RF-DETR checkpoint appears to be task={checkpoint_task!r}, "
                        f"but task={requested_task!r} was requested."
                    )

        self._model_num_classes = nb_classes
        if isinstance(weight_source, dict):
            weight_state = _checkpoint_model_state(weight_source)
            detected_classes = self.detect_nb_classes(weight_state)
            if detected_classes is not None:
                self._model_num_classes = (
                    max(1, detected_classes)
                    if normalize_task(resolved_task) == "pose"
                    else detected_classes
                )
            detected_k = self.detect_num_keypoints(weight_state)
            if detected_k is not None:
                self.num_keypoints = detected_k

        # RF-DETR COCO checkpoints have 90 arch-classes (91 outputs including
        # background), while LibreYOLO exposes the contiguous COCO-80 interface.
        user_nb_classes = 80 if nb_classes == 90 else nb_classes

        super().__init__(
            model_path=None,
            size=size,
            nb_classes=user_nb_classes,
            device=device,
            task=resolved_task,
            **kwargs,
        )

        if weight_source is not None:
            self._load_weights(weight_source)
            self.model.eval()
        elif self._is_pose and self.nb_classes == 1 and self.names.get(0) == "class_0":
            self.names = {0: "person"}

    @property
    def _is_segmentation(self) -> bool:
        """Adapter flag derived from the canonical task state."""
        return getattr(self, "task", "detect") == "segment"

    @property
    def _is_pose(self) -> bool:
        """Adapter flag derived from the canonical task state."""
        return getattr(self, "task", "detect") == "pose"

    @property
    def _is_classification(self) -> bool:
        return self.task == "classify"

    @property
    def _is_obb(self) -> bool:
        """Adapter flag derived from the canonical task state."""
        return self.task == "obb"

    @staticmethod
    def _detect_size_from_source(model_path: str | dict[str, Any] | None) -> str | None:
        if model_path is None:
            return None
        if isinstance(model_path, str):
            try:
                ckpt = load_trusted_torch_file(
                    model_path,
                    map_location="cpu",
                    context="RF-DETR size detection",
                )
            except Exception:
                return LibreRFDETR.detect_size_from_filename(model_path)
        else:
            ckpt = model_path

        if not isinstance(ckpt, dict):
            if isinstance(model_path, str):
                return LibreRFDETR.detect_size_from_filename(model_path)
            return None
        if isinstance(metadata_size := ckpt.get("size"), str):
            return metadata_size
        detected_size = LibreRFDETR.detect_size(_checkpoint_model_state(ckpt), ckpt)
        if detected_size is not None:
            return detected_size
        if isinstance(model_path, str):
            return LibreRFDETR.detect_size_from_filename(model_path)
        return None

    @staticmethod
    def _detect_task_from_source(model_path: str | dict[str, Any]) -> str | None:
        filename_task = (
            LibreRFDETR.detect_task_from_filename(str(model_path))
            if isinstance(model_path, str)
            else None
        )
        try:
            if isinstance(model_path, str):
                ckpt = load_trusted_torch_file(
                    model_path,
                    map_location="cpu",
                    context="RF-DETR task detection",
                )
            else:
                ckpt = model_path
        except Exception:
            return filename_task

        if isinstance(ckpt, dict) and isinstance(ckpt.get("task"), str):
            return normalize_task(ckpt["task"])
        if filename_task is not None:
            return filename_task

        state = _checkpoint_model_state(ckpt) if isinstance(ckpt, dict) else {}
        if "linear.weight" in state and any(k.startswith("backbone.") for k in state):
            return "classify"
        if any(k.startswith("segmentation_head") for k in state):
            return "segment"
        if any(k.startswith("keypoint_head") for k in state):
            return "pose"
        return None

    @staticmethod
    def _detect_segmentation(model_path: str | dict[str, Any]) -> bool:
        """Check if weights contain a segmentation head."""
        try:
            if isinstance(model_path, str):
                ckpt = load_trusted_torch_file(
                    model_path,
                    map_location="cpu",
                    context="RF-DETR segmentation detection",
                )
            else:
                ckpt = model_path
            if isinstance(ckpt, dict) and ckpt.get("task") is not None:
                return normalize_task(ckpt.get("task")) == "segment"
            state = _checkpoint_model_state(ckpt)
            return any(k.startswith("segmentation_head") for k in state)
        except Exception:
            return False

    @staticmethod
    def _detect_pose(model_path: str | dict[str, Any]) -> bool:
        """Check if weights contain a keypoint head."""
        try:
            if isinstance(model_path, str):
                ckpt = load_trusted_torch_file(
                    model_path,
                    map_location="cpu",
                    context="RF-DETR pose detection",
                )
            else:
                ckpt = model_path
            if isinstance(ckpt, dict) and ckpt.get("task") is not None:
                return normalize_task(ckpt.get("task")) == "pose"
            state = _checkpoint_model_state(ckpt)
            return any(k.startswith("keypoint_head") for k in state)
        except Exception:
            return False

    # =========================================================================
    # Model lifecycle
    # =========================================================================

    def _init_model(self) -> nn.Module:
        return LibreRFDETRModel(
            config=self.size,
            nb_classes=self._model_num_classes,
            device=str(self.device),
            segmentation=self._is_segmentation,
            pose=self._is_pose,
            classification=self._is_classification,
            obb=self._is_obb,
            num_keypoints=self.num_keypoints,
        )

    def _rebuild_for_new_classes(self, new_nc: int):
        """Swap the classifier head (classify) or rebuild the detector head."""
        if self._is_classification:
            self.nb_classes = new_nc
            self._model_num_classes = new_nc
            classifier = self.model.classifier
            in_features = classifier.linear.in_features
            classifier.linear = nn.Linear(in_features, new_nc)
            classifier.nb_classes = new_nc
            self.model.nb_classes = new_nc
            self.model.to(self.device)
            self.names = {i: f"class_{i}" for i in range(new_nc)}
            return
        super()._rebuild_for_new_classes(new_nc)

    def _get_available_layers(self) -> Dict[str, nn.Module]:
        layers = {}
        if hasattr(self.model, "model"):
            actual_model = self.model.model
            if hasattr(actual_model, "backbone"):
                layers["backbone"] = actual_model.backbone
            if hasattr(actual_model, "transformer"):
                layers["transformer"] = actual_model.transformer
                if hasattr(actual_model.transformer, "encoder"):
                    layers["encoder"] = actual_model.transformer.encoder
                if hasattr(actual_model.transformer, "decoder"):
                    layers["decoder"] = actual_model.transformer.decoder
            if hasattr(actual_model, "class_embed"):
                layers["class_embed"] = actual_model.class_embed
            if hasattr(actual_model, "bbox_embed"):
                layers["bbox_embed"] = actual_model.bbox_embed
            if getattr(actual_model, "angle_embed", None) is not None:
                layers["angle_embed"] = actual_model.angle_embed
            if getattr(actual_model, "segmentation_head", None) is not None:
                layers["segmentation_head"] = actual_model.segmentation_head
            if getattr(actual_model, "keypoint_head", None) is not None:
                layers["keypoint_head"] = actual_model.keypoint_head
        return layers

    def _strict_loading(self) -> bool:
        return False

    # =========================================================================
    # Inference pipeline
    # =========================================================================

    @staticmethod
    def _get_preprocess_numpy():
        from .utils import preprocess_numpy

        return preprocess_numpy

    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        """Preprocess: resize + ImageNet normalization (no letterbox)."""
        effective_res = input_size if input_size is not None else self.input_size

        img = ImageLoader.load(image, color_format=color_format)
        orig_w, orig_h = img.size
        orig_size = (orig_w, orig_h)

        if self._is_classification:
            from ...data.classify_dataset import build_classify_transforms

            transform = build_classify_transforms(effective_res, augment=False)
            img_tensor = transform(img).unsqueeze(0)
            return img_tensor, img, orig_size, 1.0

        img_chw, _ = preprocess_numpy(np.array(img), effective_res)
        img_tensor = torch.from_numpy(img_chw).unsqueeze(0)

        return img_tensor, img, orig_size, 1.0

    def _forward(self, input_tensor: torch.Tensor) -> Any:
        return self.model(input_tensor)

    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        **kwargs,
    ) -> Dict:
        if self._is_classification:
            logits = output
            if isinstance(logits, dict):
                logits = logits.get("logits", logits.get("predictions"))
            probs = torch.softmax(logits.float(), dim=1)[0]
            return {"probs": probs}
        if isinstance(output, tuple):
            tuple_output = output
            output = {"pred_boxes": tuple_output[0], "pred_logits": tuple_output[1]}
            if len(tuple_output) > 2:
                if self._is_pose:
                    output["pred_keypoints"] = tuple_output[2]
                elif self.task == "obb":
                    output["pred_angles"] = tuple_output[2]
                else:
                    output["pred_masks"] = tuple_output[2]

        logits = output["pred_logits"]
        if self._is_pose and logits.shape[-1] > self.nb_classes:
            output = dict(output)
            output["pred_logits"] = logits[..., : self.nb_classes]
            logits = output["pred_logits"]
        default_num_select = getattr(self.model, "num_select", max_det)
        requested_num_select = kwargs.get(
            "num_select",
            default_num_select if max_det == 300 else max_det,
        )
        num_select = min(requested_num_select, logits.shape[-2] * logits.shape[-1])

        # original_size is (width, height); rfdetr postprocess expects (height, width)
        orig_w, orig_h = original_size
        target_sizes = torch.tensor([(orig_h, orig_w)], device=self.device)

        results = postprocess(output, target_sizes, num_select=num_select)

        result = results[0]
        scores = result["scores"]
        labels = result["labels"]
        boxes = result["boxes"]
        masks = result.get("masks")  # (K, H, W) bool or None
        keypoints = result.get("keypoints")
        obb = result.get("obb")

        keep = scores > conf_thres
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]
        if masks is not None:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]
        if obb is not None:
            obb = obb[keep]

        # Map COCO 91-class IDs to YOLO 80-class indices if needed
        num_output_classes = output["pred_logits"].shape[-1]
        if num_output_classes == 91 and self.nb_classes == 80:
            mapped = torch.tensor(
                [_COCO91_TO_COCO80.get(int(c), -1) for c in labels.cpu()],
                dtype=labels.dtype,
                device=labels.device,
            )
            valid = mapped >= 0
            boxes = boxes[valid]
            scores = scores[valid]
            labels = mapped[valid]
            if masks is not None:
                masks = masks[valid]
            if keypoints is not None:
                keypoints = keypoints[valid]
            if obb is not None:
                obb = obb[valid]
                obb[:, 5] = scores
                obb[:, 6] = labels.float()

        det = {
            "boxes": boxes.cpu().tolist(),
            "scores": scores.cpu().tolist(),
            "classes": labels.cpu().tolist(),
            "num_detections": len(boxes),
        }
        if masks is not None:
            det["masks"] = masks.cpu()
        if keypoints is not None:
            det["keypoints"] = keypoints.cpu()
        if obb is not None:
            det["obb"] = obb.cpu().tolist()
        return det

    # =========================================================================
    # Weights
    # =========================================================================

    def _load_classify_weights(self, model_path: str | dict[str, Any]) -> None:
        """Load a LibreYOLO classification checkpoint into the classifier head."""
        if isinstance(model_path, str):
            loaded = load_trusted_torch_file(
                model_path,
                map_location="cpu",
                context="RF-DETR classify weights",
            )
        else:
            loaded = model_path
        if not isinstance(loaded, dict):
            raise TypeError("RF-DETR classification checkpoints must be dictionaries")

        # Guard against loading a detection/segmentation checkpoint into the
        # classifier: its keys would silently fail to match (strict=False),
        # leaving a randomly initialized head that "loads" successfully.
        ckpt_task = loaded.get("task")
        if isinstance(ckpt_task, str) and normalize_task(ckpt_task) != "classify":
            raise RuntimeError(
                f"Checkpoint was trained for task={normalize_task(ckpt_task)!r}, "
                "but is being loaded into an RF-DETR classification model. "
                "Load it with the matching task."
            )

        ckpt_nc = loaded.get("nc")
        if ckpt_nc is None:
            names = loaded.get("names")
            ckpt_nc = len(names) if names else None
        if ckpt_nc is None:
            ckpt_nc = self.detect_nb_classes(_checkpoint_model_state(loaded))
        if ckpt_nc is not None and ckpt_nc != self.nb_classes:
            self._rebuild_for_new_classes(int(ckpt_nc))

        # LibreRFDETRModel.load_state_dict (classification branch) unwraps the
        # checkpoint's "model" payload before loading into the classifier.
        result = self.model.load_state_dict(loaded, strict=False)
        missing = list(getattr(result, "missing_keys", []) or [])
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        # The linear head must have been populated; if it is missing (or the
        # archive is full of detection-only keys), this is not a classifier.
        if any(k.startswith("linear.") for k in missing) or any(
            ("class_embed" in k or "transformer" in k or "query" in k)
            for k in unexpected
        ):
            raise RuntimeError(
                "Checkpoint does not look like an RF-DETR classification model "
                "(its weights do not match the backbone + linear classifier). "
                "Load a classification checkpoint or the correct task."
            )

        ckpt_names = loaded.get("names")
        if ckpt_names is not None:
            self.names = self._sanitize_names(ckpt_names, self.nb_classes)
        self.model.to(self.device)

    def _load_weights(self, model_path: str | dict[str, Any]):
        if self._is_classification:
            return self._load_classify_weights(model_path)
        try:
            if isinstance(model_path, str):
                if not Path(model_path).exists():
                    from ...utils.download import download_weights

                    download_weights(model_path, self.size)
                loaded = load_trusted_torch_file(
                    model_path,
                    map_location="cpu",
                    context="RF-DETR weights",
                )
            else:
                loaded = model_path

            if not isinstance(loaded, dict):
                raise TypeError("RF-DETR checkpoints must be dictionaries")

            ckpt_family = loaded.get("model_family", "")
            if ckpt_family and ckpt_family != self.FAMILY:
                raise RuntimeError(
                    f"Checkpoint was trained with model_family='{ckpt_family}' "
                    f"but is being loaded into '{self.FAMILY}'."
                )

            ckpt_task = loaded.get("task")
            normalized_ckpt_task = None
            if ckpt_task is not None:
                normalized_ckpt_task = normalize_task(ckpt_task)
                allowed = normalized_ckpt_task == self.task or (
                    self.task == "obb"
                    and normalized_ckpt_task == "detect"
                    and self._allow_detect_to_obb_transfer
                ) or (
                    self._is_pose
                    and normalized_ckpt_task == "detect"
                    and self._allow_detect_to_pose_transfer
                )
                if not allowed:
                    raise RuntimeError(
                        f"Checkpoint was trained for task='{normalized_ckpt_task}' "
                        f"but this model was initialized for task='{self.task}'. "
                        "Pass the matching task or use explicit training transfer."
                    )

            # Replay LoRA injection for adapter checkpoints. A model trained with
            # lora=True saves its DINOv2 encoder under PeftModel keys; rebuild the
            # same wrapped graph here (the recipe is fixed, so re-running the
            # canonical injection reproduces matching modules) before loading, so
            # the adapter keys line up instead of being rejected as unexpected.
            # Merged/exported checkpoints carry no adapter keys and skip this.
            from ...training.lora import (
                apply_lora_to_rfdetr,
                module_has_lora,
                state_dict_has_lora,
            )

            loaded_state = _checkpoint_model_state(loaded)
            pose_checkpoint = any(k.startswith("keypoint_head.") for k in loaded_state)
            detect_pose_transfer = (
                self._is_pose
                and normalized_ckpt_task == "detect"
                and self._allow_detect_to_pose_transfer
            )
            if (
                self._is_pose
                and normalized_ckpt_task == "pose"
                and not pose_checkpoint
            ):
                raise RuntimeError(
                    "RF-DETR pose checkpoints must include keypoint_head.* weights. "
                    "Detect-to-pose initialization is only supported through "
                    "explicit training transfer."
                )
            already_lora = module_has_lora(self.model)
            if not already_lora and state_dict_has_lora(
                loaded_state
            ):
                apply_lora_to_rfdetr(self.model.model)

            missing, unexpected = self.model.load_state_dict(loaded, strict=False)
            if unexpected:
                raise RuntimeError(
                    f"Unexpected RF-DETR checkpoint keys: {sorted(unexpected)[:10]}"
                    + (
                        f" (+{len(unexpected) - 10} more)"
                        if len(unexpected) > 10
                        else ""
                    )
                )

            if self._is_pose and not pose_checkpoint and not detect_pose_transfer:
                raise RuntimeError(
                    "RF-DETR pose checkpoints must include keypoint_head.* weights. "
                    "Detect-to-pose initialization is only supported through "
                    "explicit training transfer."
                )

            if detect_pose_transfer:
                self.model.model.reinitialize_detection_head(self.nb_classes)
                self.model.nb_classes = 1
                self.model.args.num_classes = 0

            ckpt_nc = loaded.get("nc")
            if detect_pose_transfer:
                self.nb_classes = 1
            elif ckpt_nc is not None:
                self.nb_classes = int(ckpt_nc)
            else:
                self.nb_classes = (
                    80 if self.model.nb_classes == 90 else self.model.nb_classes
                )

            self._model_num_classes = self.model.nb_classes
            ckpt_k = loaded.get("num_keypoints")
            if ckpt_k is not None:
                self.num_keypoints = int(ckpt_k)
            else:
                detected_k = self.detect_num_keypoints(loaded_state)
                if detected_k is not None:
                    self.num_keypoints = detected_k
            ckpt_kd = loaded.get("keypoint_dim")
            if ckpt_kd is not None:
                self.keypoint_dim = int(ckpt_kd)
            if self.nb_classes == 80:
                from ...utils.general import COCO_CLASSES

                self.names = {i: n for i, n in enumerate(COCO_CLASSES)}
            else:
                self.names = {i: f"class_{i}" for i in range(self.nb_classes)}
            if self._is_pose and self.nb_classes == 1:
                self.names = {0: "person"}

            ckpt_names = loaded.get("names")
            if ckpt_names is not None:
                self.names = self._sanitize_names(ckpt_names, self.nb_classes)

            args = loaded.get("args") or loaded.get("hyper_parameters") or {}
            class_names = (
                args.get("class_names")
                if isinstance(args, dict)
                else getattr(args, "class_names", None)
            )
            if class_names:
                self.names = {
                    i: str(name)
                    for i, name in enumerate(class_names[: self.nb_classes])
                }
            if self._is_pose and self.nb_classes == 1:
                self.names = {0: "person"}

            if missing:
                # ``strict=False`` is expected for class/head adaptation and older
                # checkpoints, but missing non-head tensors should stay visible.
                ignored = ["class_embed.", "transformer.enc_out_class_embed."]
                if detect_pose_transfer:
                    ignored.append("keypoint_head.")
                missing_angle = [k for k in missing if k.startswith("angle_embed.")]
                if (
                    self.task == "obb"
                    and missing_angle
                    and not self._allow_detect_to_obb_transfer
                ):
                    raise RuntimeError(
                        "RF-DETR OBB checkpoints must include angle_embed.* weights. "
                        "Detect-to-OBB initialization is only supported through "
                        "explicit training transfer."
                    )

                ignored = [
                    "class_embed.",
                    "transformer.enc_out_class_embed.",
                ]
                if detect_pose_transfer:
                    ignored.append("keypoint_head.")
                if self._allow_detect_to_obb_transfer:
                    ignored.append("angle_embed.")
                important = [k for k in missing if not k.startswith(tuple(ignored))]
                if important:
                    raise RuntimeError(
                        f"Missing RF-DETR checkpoint keys: {sorted(important)[:10]}"
                        + (
                            f" (+{len(important) - 10} more)"
                            if len(important) > 10
                            else ""
                        )
                    )
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load RF-DETR weights: {e}") from e

    # =========================================================================
    # Public API
    # =========================================================================

    def export(self, format: str = "onnx", *, opset: int = 17, **kwargs) -> str:
        """Export model. RF-DETR requires opset >= 17 for LayerNormalization."""
        return super().export(format, opset=opset, **kwargs)

    def val(self, *args, workers: int = 0, **kwargs) -> Dict:
        """Run RF-DETR validation with a Windows-safe worker default."""
        return super().val(*args, workers=workers, **kwargs)

    def _restore_after_training(self, result: dict) -> None:
        """Reload the saved checkpoint and leave real torch models in eval mode."""
        checkpoint = None
        for key in ("best_checkpoint", "last_checkpoint"):
            path = result.get(key)
            if path and Path(path).exists():
                checkpoint = str(path)
                break

        if checkpoint is not None:
            self.model_path = checkpoint
            self._load_weights(checkpoint)

        model = getattr(self, "model", None)
        device = getattr(self, "device", None)
        if model is not None and device is not None and hasattr(model, "to"):
            model.to(device)
        if model is not None and hasattr(model, "eval"):
            model.eval()

    def _resume_checkpoint_uses_lora(self, resume_path: str | Path) -> bool:
        """Return True when a resume checkpoint needs a LoRA-wrapped graph."""
        path = Path(resume_path)
        if not path.exists():
            return False

        checkpoint = load_trusted_torch_file(
            path,
            map_location="cpu",
            context="RF-DETR resume checkpoint probe",
        )
        if not isinstance(checkpoint, dict):
            return False

        config = checkpoint.get("config")
        if isinstance(config, dict) and bool(config.get("lora", False)):
            return True

        from ...training.lora import state_dict_has_lora

        model_state = checkpoint.get("train_model", checkpoint.get("model", checkpoint))
        return isinstance(model_state, dict) and state_dict_has_lora(
            _checkpoint_model_state(model_state)
        )

    @ddp_aware(batch_key="batch_size")
    def train(
        self,
        data: str,
        epochs: int = 100,
        batch_size: int | None = None,
        lr: float | None = None,
        output_dir: str = "runs/train",
        resume: str | Path | bool | None = None,
        **kwargs,
    ) -> Dict:
        """Fine-tune RF-DETR through LibreYOLO's native trainer."""
        output_path = Path(output_dir)
        train_kwargs = dict(kwargs)
        project = train_kwargs.pop("project", None)
        name = train_kwargs.pop("name", None)
        exist_ok = train_kwargs.pop("exist_ok", True)
        batch = train_kwargs.pop("batch", None)
        lr0 = train_kwargs.pop("lr0", None)
        if project is None:
            project = output_path.parent
        if name is None:
            name = output_path.name
        run_dir = Path(project) / str(name)

        if batch is not None and batch_size is not None and batch != batch_size:
            raise ValueError(
                f"Conflicting RF-DETR batch values: batch={batch} and batch_size={batch_size}"
            )
        if lr0 is not None and lr is not None and lr0 != lr:
            raise ValueError(f"Conflicting RF-DETR LR values: lr0={lr0} and lr={lr}")

        resolved_batch = batch if batch is not None else batch_size
        resolved_lr0 = lr0 if lr0 is not None else lr
        if resolved_batch is None:
            resolved_batch = 4
        if resolved_lr0 is None:
            resolved_lr0 = 1e-4

        pose_train_metadata = {}
        if self._is_pose:
            data_cfg = load_data_config(
                data,
                allow_scripts=bool(train_kwargs.get("allow_download_scripts", False)),
            )
            kpt_shape = data_cfg.get("kpt_shape")
            if not kpt_shape or len(kpt_shape) < 1:
                raise ValueError("RF-DETR pose training requires kpt_shape in the dataset yaml")
            num_keypoints = int(kpt_shape[0])
            keypoint_dim = int(kpt_shape[1]) if len(kpt_shape) > 1 else 3
            if keypoint_dim not in (2, 3):
                raise ValueError(
                    f"RF-DETR pose training supports keypoint_dim 2 or 3, got {keypoint_dim}"
                )
            data_nc = int(data_cfg.get("nc", 1))
            if data_nc != 1:
                raise ValueError(
                    f"RF-DETR pose training expects a person-only dataset with nc=1, got nc={data_nc}"
                )
            if self.model.num_keypoints != num_keypoints:
                self.model.model.reinitialize_keypoint_head(num_keypoints)
                self.model.num_keypoints = num_keypoints
                self.model.args.num_keypoints = num_keypoints
            self.num_keypoints = num_keypoints
            self.keypoint_dim = keypoint_dim
            self.nb_classes = 1
            self.names = {0: "person"}
            oks_sigmas = train_kwargs.get(
                "oks_sigmas",
                data_cfg.get("oks_sigmas", data_cfg.get("sigmas")),
            )
            pose_train_metadata = {
                "num_keypoints": num_keypoints,
                "keypoint_dim": keypoint_dim,
                "num_classes": 1,
            }
            if oks_sigmas is not None:
                pose_train_metadata["oks_sigmas"] = oks_sigmas

        train_kwargs.update(
            {
                "data": data,
                "epochs": epochs,
                "batch": resolved_batch,
                "lr0": resolved_lr0,
                "project": str(project),
                "name": str(name),
                "exist_ok": exist_ok,
                "size": self.size,
                "num_classes": self.nb_classes,
            }
        )
        train_kwargs.update(pose_train_metadata)
        if train_kwargs.get("imgsz") is None:
            train_kwargs["imgsz"] = self.input_size

        aliases = {
            "num_workers": "workers",
            "use_ema": "ema",
            "checkpoint_interval": "save_period",
            "early_stopping_patience": "patience",
        }
        for src, dst in aliases.items():
            if src in train_kwargs:
                train_kwargs[dst] = train_kwargs.pop(src)
        train_kwargs.pop("early_stopping", None)

        resume_path = None
        if resume:
            resume_path = run_dir / "weights" / "last.pt" if resume is True else resume
            if not train_kwargs.get(
                "lora", False
            ) and self._resume_checkpoint_uses_lora(resume_path):
                train_kwargs["lora"] = True

        trainer = RFDETRTrainer(self.model, wrapper_model=self, **train_kwargs)
        if resume:
            trainer.setup()
            trainer.resume(str(resume_path))
        result = trainer.train()
        result["output_dir"] = result.get("save_dir", str(run_dir))

        self._restore_after_training(result)

        return result
