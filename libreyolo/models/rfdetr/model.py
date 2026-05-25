"""LibreRFDETR implementation for LibreYOLO."""

from pathlib import Path
from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from libreyolo.training.ddp_spawn import ddp_aware
from PIL import Image

from ..base import BaseModel
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

    Args:
        model_path: Path to weights, pre-loaded state_dict, or None for pretrained.
        size: Model size variant ("n", "s", "m", "l").
        nb_classes: Number of classes (default: 80 for COCO).
        device: Device for inference.

    Example::

        >>> model = LibreRFDETR(size="s")
        >>> detections = model.predict("path/to/image.jpg")
    """

    # Class-level metadata
    FAMILY = "rfdetr"
    FILENAME_PREFIX = "LibreRFDETR"
    INPUT_SIZES = {"n": 384, "s": 512, "m": 576, "l": 704}
    SEG_INPUT_SIZES = {"n": 312, "s": 384, "m": 432, "l": 504, "x": 624, "xx": 768}
    SUPPORTED_TASKS = ("detect", "segment")
    TASK_INPUT_SIZES = {
        "detect": INPUT_SIZES,
        "segment": SEG_INPUT_SIZES,
    }
    TRAIN_CONFIG = RFDETRConfig
    val_preprocessor_class = RFDETRValPreprocessor
    TTA_FIXED_SIZE = True  # resizes to a fixed square; multi-scale TTA is a no-op

    # CLI parameters intentionally ignored by native RF-DETR training.
    UNSUPPORTED_TRAIN_PARAMS: ClassVar[set[str]] = {
        "imgsz",
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
        return any(
            "detr" in k
            or "dinov2" in k
            or "transformer" in k
            or ("encoder" in k and "decoder" in k)
            or "query_embed" in k
            or "class_embed" in k
            or "bbox_embed" in k
            for k in keys_lower
        )

    @classmethod
    def detect_size(
        cls, weights_dict: dict, state_dict: dict | None = None
    ) -> Optional[str]:
        full_ckpt = state_dict if state_dict is not None else weights_dict
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
        # RF-DETR class_embed has (num_classes + 1) outputs (includes background)
        if "class_embed.bias" in weights_dict:
            return weights_dict["class_embed.bias"].shape[0] - 1
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
        size: str = "s",
        nb_classes: int = 80,
        device: str = "auto",
        segmentation: bool = False,
        task: str | None = None,
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

        if isinstance(model_path, dict) and not model_path:
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

        if weight_source is not None:
            filename_task = (
                self.detect_task_from_filename(str(weight_source))
                if isinstance(weight_source, str)
                else None
            )
            checkpoint_is_segment = False
            if filename_task != "segment":
                checkpoint_is_segment = self._detect_segmentation(weight_source)
            if resolved_task is None:
                resolved_task = filename_task or (
                    "segment" if checkpoint_is_segment else None
                )
            elif (
                normalize_task(resolved_task) == "detect"
                and (filename_task == "segment" or checkpoint_is_segment)
            ):
                raise ValueError(
                    "RF-DETR checkpoint appears to be a segmentation model, "
                    "but task='detect' was requested."
                )

        self._model_num_classes = nb_classes
        if isinstance(weight_source, dict):
            detected_classes = self.detect_nb_classes(_checkpoint_model_state(weight_source))
            if detected_classes is not None:
                self._model_num_classes = detected_classes

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

    @property
    def _is_segmentation(self) -> bool:
        """Adapter flag derived from the canonical task state."""
        return self.task == "segment"

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

    # =========================================================================
    # Model lifecycle
    # =========================================================================

    def _init_model(self) -> nn.Module:
        return LibreRFDETRModel(
            config=self.size,
            nb_classes=self._model_num_classes,
            device=str(self.device),
            segmentation=self._is_segmentation,
        )

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
            if getattr(actual_model, "segmentation_head", None) is not None:
                layers["segmentation_head"] = actual_model.segmentation_head
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
        if isinstance(output, tuple):
            output = {
                "pred_boxes": output[0],
                "pred_logits": output[1],
                **({"pred_masks": output[2]} if len(output) > 2 else {}),
            }

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

        keep = scores > conf_thres
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]
        if masks is not None:
            masks = masks[keep]

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

        det = {
            "boxes": boxes.cpu().tolist(),
            "scores": scores.cpu().tolist(),
            "classes": labels.cpu().tolist(),
            "num_detections": len(boxes),
        }
        if masks is not None:
            det["masks"] = masks.cpu()
        return det

    # =========================================================================
    # Weights
    # =========================================================================

    def _load_weights(self, model_path: str | dict[str, Any]):
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

            missing, unexpected = self.model.load_state_dict(loaded, strict=False)
            if unexpected:
                raise RuntimeError(
                    f"Unexpected RF-DETR checkpoint keys: {sorted(unexpected)[:10]}"
                    + (f" (+{len(unexpected) - 10} more)" if len(unexpected) > 10 else "")
                )

            ckpt_nc = loaded.get("nc")
            if ckpt_nc is not None:
                self.nb_classes = int(ckpt_nc)
            else:
                self.nb_classes = 80 if self.model.nb_classes == 90 else self.model.nb_classes

            self._model_num_classes = self.model.nb_classes
            if self.nb_classes == 80:
                from ...utils.general import COCO_CLASSES

                self.names = {i: n for i, n in enumerate(COCO_CLASSES)}
            else:
                self.names = {i: f"class_{i}" for i in range(self.nb_classes)}

            ckpt_names = loaded.get("names")
            if ckpt_names is not None:
                self.names = self._sanitize_names(ckpt_names, self.nb_classes)

            args = loaded.get("args") or loaded.get("hyper_parameters") or {}
            class_names = args.get("class_names") if isinstance(args, dict) else getattr(args, "class_names", None)
            if class_names:
                self.names = {i: str(name) for i, name in enumerate(class_names[: self.nb_classes])}

            if missing:
                # ``strict=False`` is expected for class/head adaptation and older
                # checkpoints, but missing non-head tensors should stay visible.
                ignored = ("class_embed.", "transformer.enc_out_class_embed.")
                important = [k for k in missing if not k.startswith(ignored)]
                if important:
                    raise RuntimeError(
                        f"Missing RF-DETR checkpoint keys: {sorted(important)[:10]}"
                        + (f" (+{len(important) - 10} more)" if len(important) > 10 else "")
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

    @ddp_aware(batch_key="batch_size")
    def train(
        self,
        data: str,
        epochs: int = 100,
        batch_size: int | None = None,
        lr: float | None = None,
        output_dir: str = "runs/train",
        resume: str | None = None,
        **kwargs,
    ) -> Dict:
        """Fine-tune RF-DETR through LibreYOLO's native trainer.

        Args:
            data: Path to data.yaml.
            epochs: Training epochs.
            batch_size: Global batch size (divided by world_size per GPU under DDP).
            lr: Initial learning rate.
            output_dir: Root output directory.
            resume: Path to checkpoint to resume from.
            **kwargs: Extra trainer kwargs, including ``device``. Pass
                ``device="0,1"`` (or a list) to enable multi-GPU training from
                a plain Python script — DDP workers are spawned automatically,
                no torchrun required.
        """
        output_path = Path(output_dir)
        train_kwargs = dict(kwargs)
        batch = train_kwargs.pop("batch", None)
        lr0 = train_kwargs.pop("lr0", None)

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

        train_kwargs.update(
            {
                "data": data,
                "epochs": epochs,
                "batch": resolved_batch,
                "lr0": resolved_lr0,
                "project": str(output_path.parent),
                "name": output_path.name,
                "exist_ok": True,
                "size": self.size,
                "num_classes": self.nb_classes,
                "imgsz": self.input_size,
            }
        )

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

        trainer = RFDETRTrainer(self.model, wrapper_model=self, **train_kwargs)
        if resume:
            trainer.setup()
            trainer.resume(str(resume))
        result = trainer.train()
        result["output_dir"] = result.get("save_dir", output_dir)

        best_ckpt = result.get("best_checkpoint")
        if best_ckpt and Path(best_ckpt).exists():
            self.model_path = best_ckpt
            self._load_weights(best_ckpt)
            self.model.to(self.device).eval()

        return result
