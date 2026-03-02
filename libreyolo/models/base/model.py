"""
Base model class for LibreYOLO model wrappers.

Provides shared functionality for all YOLO model variants.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from PIL import Image

from ...utils.general import COCO_CLASSES
from ...utils.image_loader import ImageInput
from ...utils.results import Results



class BaseModel(ABC):
    """
    Abstract base class for LibreYOLO model wrappers.

    Provides shared functionality for inference, saving, and tiling
    across all YOLO model variants.

    Subclasses must implement the abstract methods to provide model-specific
    behavior for initialization, forward pass, and postprocessing.
    """

    val_preprocessor_class = None

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
    # ABSTRACT METHODS - Must be implemented by subclasses
    # =========================================================================

    @abstractmethod
    def _init_model(self) -> nn.Module:
        """Initialize and return the neural network model."""
        pass

    @abstractmethod
    def _get_available_layers(self) -> Dict[str, nn.Module]:
        """Return mapping of layer names to module objects."""
        pass

    @abstractmethod
    def _get_valid_sizes(self) -> List[str]:
        """Return list of valid size codes for this model."""
        pass

    @abstractmethod
    def _get_model_name(self) -> str:
        """Return the model name for metadata."""
        pass

    @abstractmethod
    def _get_input_size(self) -> int:
        """Return the input size for this model."""
        pass

    @staticmethod
    @abstractmethod
    def _get_preprocess_numpy():
        """Return the ``preprocess_numpy(img_rgb_hwc, input_size)`` callable for this model family."""
        pass

    @abstractmethod
    def _preprocess(
        self, image: ImageInput, color_format: str = "auto", input_size: Optional[int] = None
    ) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int], float]:
        """Preprocess image for inference.

        Args:
            image: Input image.
            color_format: Color format hint.
            input_size: Override input size (None = model default).

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

    def _get_val_preprocessor(self, img_size: int = None):
        """
        Return the validation preprocessor for this model.

        Args:
            img_size: Target image size. Defaults to model's native input size.

        Returns:
            A preprocessor instance with __call__(img, targets, input_size).
        """
        if img_size is None:
            img_size = self._get_input_size()
        cls = self.val_preprocessor_class
        if cls is None:
            from libreyolo.validation.preprocessors import StandardValPreprocessor
            cls = StandardValPreprocessor
        return cls(img_size=(img_size, img_size))

    # =========================================================================
    # SHARED IMPLEMENTATION
    # =========================================================================

    def __init__(
        self,
        model_path: Union[str, dict, None],
        size: str,
        nb_classes: int = 80,
        device: str = "auto",
        **kwargs,
    ):
        """
        Initialize the model.

        Args:
            model_path: Path to weights file, pre-loaded state_dict, or None
                for random initialization (fresh model for training).
            size: Model size variant.
            nb_classes: Number of classes (default: 80 for COCO).
            device: Device for inference ("auto", "cuda", "mps", "cpu").
            **kwargs: Additional model-specific arguments.
        """
        # Validate size
        valid_sizes = self._get_valid_sizes()
        if size not in valid_sizes:
            raise ValueError(
                f"Invalid size: {size}. Must be one of: {', '.join(valid_sizes)}"
            )

        # Resolve device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Store parameters
        self.size = size
        self.nb_classes = nb_classes

        # Build names dict (matches Ultralytics model.names)
        if nb_classes == 80:
            self.names: Dict[int, str] = {i: n for i, n in enumerate(COCO_CLASSES)}
        else:
            self.names: Dict[int, str] = {i: f"class_{i}" for i in range(nb_classes)}

        # Store extra kwargs for subclass use
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Initialize model (implemented by subclass)
        self.model = self._init_model()

        # Load weights (or skip for fresh model)
        if model_path is None:
            self.model_path = None
        elif isinstance(model_path, dict):
            self.model_path = None
            self.model.load_state_dict(
                self._prepare_state_dict(model_path), strict=self._strict_loading()
            )
        else:
            self.model_path = model_path

        # Fresh models start in train mode; loaded models in eval mode
        if model_path is None:
            self.model.train()
        else:
            self.model.eval()
        self.model.to(self.device)

    def _rebuild_for_new_classes(self, new_nb_classes: int):
        """Rebuild model with a new class count, preserving weights where shapes match.

        Used when training on a dataset with a different number of classes
        than the model was initialized with. Backbone/neck weights are preserved;
        head weights (which depend on nb_classes) are reinitialized.

        Args:
            new_nb_classes: The new number of classes.
        """
        old_state = self.model.state_dict()
        self.nb_classes = new_nb_classes
        self.model = self._init_model()

        # Transfer weights with matching shapes (backbone/neck preserved, head reinitialized)
        new_state = self.model.state_dict()
        for key in old_state:
            if key in new_state and old_state[key].shape == new_state[key].shape:
                new_state[key] = old_state[key]

        self.model.load_state_dict(new_state)
        self.model.to(self.device)

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        """Transform state dict keys before loading.

        Override in subclasses that need to remap legacy key names.
        """
        return state_dict

    def _strict_loading(self) -> bool:
        """Return whether to use strict mode when loading weights.

        Override in subclasses that need non-strict loading.
        """
        return True

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

        Handles both raw state_dicts and training checkpoint dicts
        ({"model": state_dict, "optimizer": ..., "epoch": ...}).

        If the checkpoint contains model metadata (nc, names, size),
        auto-rebuilds the model architecture to match before loading weights.
        This enables loading fine-tuned checkpoints trained on different
        numbers of classes.

        Also handles:
        - DDP 'module.' prefix stripping
        - Cross-family checkpoint warnings
        - Names dict sanitization (string keys, gaps, nc mismatch)
        """
        logger = logging.getLogger(__name__)

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

                # Strip DDP 'module.' prefix if present
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

                # Auto-rebuild model if checkpoint has different nc
                ckpt_nc = loaded.get("nc")
                if ckpt_nc is not None and ckpt_nc != self.nb_classes:
                    self._rebuild_for_new_classes(ckpt_nc)

                # Restore and sanitize class names from checkpoint
                ckpt_names = loaded.get("names")
                effective_nc = ckpt_nc if ckpt_nc is not None else self.nb_classes
                if ckpt_names is not None:
                    self.names = self._sanitize_names(ckpt_names, effective_nc)
            else:
                state_dict = loaded

            state_dict = self._prepare_state_dict(state_dict)
            self.model.load_state_dict(state_dict, strict=self._strict_loading())
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model weights from {model_path}: {e}"
            ) from e

    def get_available_layer_names(self) -> List[str]:
        """Get list of available layer names."""
        return sorted(self._get_available_layers().keys())

    # =========================================================================
    # INFERENCE - delegates to InferenceRunner
    # =========================================================================

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

    # =========================================================================
    # EXPORT & VALIDATION - delegate to external modules
    # =========================================================================

    def export(self, format: str = "onnx", **kwargs) -> str:
        """Export model to deployment format.

        Args:
            format: Target format ("onnx", "torchscript", "tensorrt",
                "openvino", "ncnn").
            **kwargs: Format-specific parameters forwarded to the exporter.
                Common: output_path, imgsz, half, int8, batch, device, verbose.
                ONNX: opset, simplify, dynamic.
                TensorRT: workspace, hardware_compatibility, gpu_device, trt_config.
                INT8: data, fraction.

        Returns:
            Path to the exported model file.

        Example::

            model.export(format="onnx")
            model.export(format="tensorrt", half=True)
            model.export(format="tensorrt", int8=True, data="coco8.yaml")
        """
        from libreyolo.export import BaseExporter

        return BaseExporter.create(format, self)(**kwargs)

    def val(
        self,
        data: str = None,
        batch: int = 16,
        imgsz: int = None,
        conf: float = 0.001,
        iou: float = 0.6,
        device: str = None,
        split: str = "val",
        save_json: bool = False,
        verbose: bool = True,
        **kwargs,
    ) -> Dict:
        """
        Run validation on a dataset.

        Computes standard object detection metrics including mAP50, mAP50-95,
        precision, and recall.

        Args:
            data: Path to data.yaml file containing dataset configuration.
            batch: Batch size for validation.
            imgsz: Image size for validation. Defaults to model's native input size.
            conf: Confidence threshold. Use low value (0.001) for mAP calculation.
            iou: IoU threshold for NMS.
            device: Device to use (default: same as model).
            split: Dataset split to validate on ("val", "test").
            save_json: Save predictions in COCO JSON format.
            verbose: Print detailed metrics.
            **kwargs: Additional arguments passed to ValidationConfig.

        Returns:
            Dictionary with validation metrics:
                - metrics/precision: Mean precision at conf threshold
                - metrics/recall: Mean recall at conf threshold
                - metrics/mAP50: Mean AP at IoU=0.50
                - metrics/mAP50-95: Mean AP across IoU 0.50-0.95

        Example:
            >>> model = LibreYOLO("weights/LibreYOLOXs.pt")
            >>> results = model.val(data="coco8.yaml", batch=16)
            >>> print(f"mAP50-95: {results['metrics/mAP50-95']:.3f}")
        """
        from libreyolo.validation import DetectionValidator, ValidationConfig

        # Use model's native input size if not specified
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
