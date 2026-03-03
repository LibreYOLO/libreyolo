"""
Base model class for LibreYOLO model wrappers.

Provides shared functionality for all YOLO model variants.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from PIL import Image

from ...utils.general import COCO_CLASSES
from ...utils.image_loader import ImageInput
from ...utils.results import Results
from ...validation.preprocessors import StandardValPreprocessor


class BaseModel(ABC):
    """Abstract base class for LibreYOLO model wrappers.

    Subclasses must implement the abstract methods to provide model-specific
    behavior for initialization, forward pass, and postprocessing.

    Class constants subclasses should set:
        FAMILY: Model family identifier (e.g. "yolox").
        FILENAME_PREFIX: Prefix for weight filenames (e.g. "LibreYOLOX").
        INPUT_SIZES: Mapping of size code to input resolution.
        val_preprocessor_class: Preprocessor class for validation.
    """

    # Class-level model metadata — subclasses override these
    FAMILY: ClassVar[str] = ""
    FILENAME_PREFIX: ClassVar[str] = ""
    WEIGHT_EXT: ClassVar[str] = ".pt"
    INPUT_SIZES: ClassVar[dict[str, int]] = {}
    val_preprocessor_class = StandardValPreprocessor

    # Model registry — auto-populated by __init_subclass__
    _registry: ClassVar[List[Type["BaseModel"]]] = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if (
            hasattr(cls, "can_load")
            and not getattr(cls.can_load, "__isabstractmethod__", False)
            and cls not in BaseModel._registry
        ):
            BaseModel._registry.append(cls)

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        model_path: Union[str, dict, None],
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        valid_sizes = self._get_valid_sizes()
        if size not in valid_sizes:
            raise ValueError(
                f"Invalid size: {size}. Must be one of: {', '.join(valid_sizes)}"
            )

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.size = size
        self.nb_classes = nb_classes
        self.input_size = self.INPUT_SIZES[size]

        if nb_classes == 80:
            self.names: Dict[int, str] = {i: n for i, n in enumerate(COCO_CLASSES)}
        else:
            self.names: Dict[int, str] = {i: f"class_{i}" for i in range(nb_classes)}

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.model = self._init_model()

        if model_path is None:
            self.model_path = None
        elif isinstance(model_path, dict):
            self.model_path = None
            self.model.load_state_dict(
                model_path, strict=self._strict_loading()
            )
        else:
            self.model_path = model_path

        if model_path is None:
            self.model.train()
        else:
            self.model.eval()
        self.model.to(self.device)

    # =========================================================================
    # Abstract interface — subclasses must implement
    # =========================================================================

    @abstractmethod
    def _init_model(self) -> nn.Module:
        """Initialize and return the neural network model."""
        pass

    @abstractmethod
    def _get_available_layers(self) -> Dict[str, nn.Module]:
        """Return mapping of layer names to module objects."""
        pass

    @staticmethod
    @abstractmethod
    def _get_preprocess_numpy():
        """Return the ``preprocess_numpy(img_rgb_hwc, input_size)`` callable for this model family."""
        pass

    @abstractmethod
    def _preprocess(
        self,
        image: ImageInput,
        color_format: str = "auto",
        input_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        """Preprocess image for inference.

        Returns:
            Tuple of (input_tensor, original_image, original_size, ratio).
        """
        pass

    @abstractmethod
    def _forward(self, input_tensor: torch.Tensor) -> Any:
        """Run model forward pass."""
        pass

    @abstractmethod
    def _postprocess(
        self,
        output: Any,
        conf_thres: float,
        iou_thres: float,
        original_size: Tuple[int, int],
        max_det: int = 300,
        ratio: float = 1.0,
    ) -> Dict:
        """Postprocess model output to detections."""
        pass

    # =========================================================================
    # Concrete defaults — subclasses may override
    # =========================================================================

    def _get_valid_sizes(self) -> List[str]:
        return list(self.INPUT_SIZES.keys())

    def _get_model_name(self) -> str:
        return self.FAMILY

    def _get_input_size(self) -> int:
        return self.input_size

    def _strict_loading(self) -> bool:
        """Return whether to use strict mode when loading weights."""
        return True

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        """Transform state dict keys before loading.

        Override in subclasses that need to remap legacy key names.
        """
        return state_dict

    def _rebuild_for_new_classes(self, new_nb_classes: int):
        """Rebuild model with a new class count, preserving weights where shapes match."""
        old_state = self.model.state_dict()
        self.nb_classes = new_nb_classes
        self.model = self._init_model()

        new_state = self.model.state_dict()
        for key in old_state:
            if key in new_state and old_state[key].shape == new_state[key].shape:
                new_state[key] = old_state[key]

        self.model.load_state_dict(new_state)
        self.model.to(self.device)

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        """Extract model size from a weight filename."""
        if not cls.INPUT_SIZES or not cls.FILENAME_PREFIX:
            return None
        sizes_pattern = "".join(cls.INPUT_SIZES.keys())
        prefix = cls.FILENAME_PREFIX.lower()
        ext = re.escape(cls.WEIGHT_EXT)
        m = re.search(rf"{prefix}([{sizes_pattern}]){ext}", filename.lower())
        return m.group(1) if m else None

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        """Return the Hugging Face download URL for the given weight filename."""
        size = cls.detect_size_from_filename(filename)
        if size is None:
            return None
        repo = f"LibreYOLO/{cls.FILENAME_PREFIX}{size}"
        actual = f"{cls.FILENAME_PREFIX}{size}{cls.WEIGHT_EXT}"
        return f"https://huggingface.co/{repo}/resolve/main/{actual}"

    def _get_val_preprocessor(self, img_size: int | None = None):
        """Return the validation preprocessor for this model."""
        if img_size is None:
            img_size = self._get_input_size()
        return self.val_preprocessor_class(img_size=(img_size, img_size))

    # =========================================================================
    # Weight loading internals
    # =========================================================================

    @staticmethod
    def _strip_ddp_prefix(state_dict: dict) -> dict:
        """Strip 'module.' prefix from DDP-wrapped state_dict keys."""
        if any(k.startswith("module.") for k in state_dict):
            return {k.removeprefix("module."): v for k, v in state_dict.items()}
        return state_dict

    @staticmethod
    def _sanitize_names(names: dict, nc: int) -> Dict[int, str]:
        """Sanitize a class names dict: ensure int keys, fill gaps, trim to nc."""
        sanitized = {}
        for k, v in names.items():
            try:
                sanitized[int(k)] = str(v)
            except (ValueError, TypeError):
                continue

        result = {}
        for i in range(nc):
            result[i] = sanitized.get(i, f"class_{i}")
        return result

    def _load_weights(self, model_path: str):
        """Load model weights from file.

        Handles raw state_dicts and training checkpoint dicts.
        Auto-rebuilds model architecture if checkpoint has different nc.
        Also handles DDP prefix stripping and cross-family rejection.
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model weights file not found: {model_path}")

        try:
            loaded = torch.load(model_path, map_location="cpu", weights_only=False)

            if isinstance(loaded, dict):
                if "model" in loaded:
                    state_dict = loaded["model"]
                elif "state_dict" in loaded:
                    state_dict = loaded["state_dict"]
                else:
                    state_dict = loaded

                state_dict = self._strip_ddp_prefix(state_dict)

                # Reject cross-family loading
                own_family = self._get_model_name()
                ckpt_family = loaded.get("model_family", "")
                if ckpt_family and ckpt_family != own_family:
                    raise RuntimeError(
                        f"Checkpoint was trained with model_family='{ckpt_family}' "
                        f"but is being loaded into '{own_family}'. "
                        f"Use the correct model class for this checkpoint."
                    )

                ckpt_nc = loaded.get("nc")
                if ckpt_nc is not None and ckpt_nc != self.nb_classes:
                    self._rebuild_for_new_classes(ckpt_nc)

                ckpt_names = loaded.get("names")
                effective_nc = ckpt_nc if ckpt_nc is not None else self.nb_classes
                if ckpt_names is not None:
                    self.names = self._sanitize_names(ckpt_names, effective_nc)
            else:
                state_dict = loaded

            self.model.load_state_dict(state_dict, strict=self._strict_loading())
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model weights from {model_path}: {e}"
            ) from e

    # =========================================================================
    # Public API
    # =========================================================================

    def get_available_layer_names(self) -> List[str]:
        """Get list of available layer names."""
        return sorted(self._get_available_layers().keys())

    @property
    def _runner(self):
        if not hasattr(self, "_runner_instance") or self._runner_instance is None:
            from .inference import InferenceRunner

            self._runner_instance = InferenceRunner(self)
        return self._runner_instance

    def __call__(self, source=None, **kwargs):
        return self._runner(source, **kwargs)

    def predict(self, *args, **kwargs) -> Union[Results, List[Results]]:
        """Alias for __call__ method."""
        return self(*args, **kwargs)

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export model to deployment format.

        Args:
            format: Target format ("onnx", "torchscript", "tensorrt",
                "openvino", "ncnn").
            **kwargs: Format-specific parameters forwarded to the exporter.

        Returns:
            Path to the exported model file.
        """
        from libreyolo.export import BaseExporter

        return BaseExporter.create(format, self)(**kwargs)

    def val(
        self,
        data: str | None = None,
        batch: int = 16,
        imgsz: int | None = None,
        conf: float = 0.001,
        iou: float = 0.6,
        device: str | None = None,
        split: str = "val",
        save_json: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> Dict:
        """Run validation on a dataset.

        Args:
            data: Path to data.yaml file.
            batch: Batch size.
            imgsz: Image size (defaults to model's native input size).
            conf: Confidence threshold.
            iou: IoU threshold for NMS.
            device: Device to use (default: same as model).
            split: Dataset split ("val", "test").
            save_json: Save predictions in COCO JSON format.
            verbose: Print detailed metrics.

        Returns:
            Dictionary with metrics/precision, metrics/recall,
            metrics/mAP50, metrics/mAP50-95.
        """
        from libreyolo.validation import DetectionValidator, ValidationConfig

        if imgsz is None:
            imgsz = self._get_input_size()

        config = ValidationConfig(
            data=data,
            batch_size=batch,
            imgsz=imgsz,
            conf_thres=conf,
            iou_thres=iou,
            device=device or str(self.device),
            split=split,
            save_json=save_json,
            verbose=verbose,
            **kwargs,
        )

        validator = DetectionValidator(model=self, config=config)
        return validator()
