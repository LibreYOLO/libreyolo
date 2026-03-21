"""
Base model class for LibreYOLO model wrappers.

Provides shared functionality for all YOLO model variants.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, Dict, Generator, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
from PIL import Image

from ...utils.general import COCO_CLASSES
from ...utils.image_loader import ImageInput
from ...utils.results import Results

from typing import Generator
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
            self.model.load_state_dict(model_path, strict=self._strict_loading())
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
    def _filename_regex(cls) -> Optional[re.Pattern]:
        """Compile regex for matching weight filenames with optional task suffix."""
        if not cls.INPUT_SIZES or not cls.FILENAME_PREFIX:
            return None
        sizes_pattern = "".join(cls.INPUT_SIZES.keys())
        prefix = cls.FILENAME_PREFIX.lower()
        ext = re.escape(cls.WEIGHT_EXT)
        return re.compile(rf"{prefix}([{sizes_pattern}])(-seg)?{ext}")

    @classmethod
    def detect_size_from_filename(cls, filename: str) -> Optional[str]:
        """Extract model size from a weight filename."""
        pattern = cls._filename_regex()
        if pattern is None:
            return None
        m = pattern.search(filename.lower())
        return m.group(1) if m else None

    @classmethod
    def detect_task_from_filename(cls, filename: str) -> Optional[str]:
        """Extract task suffix from a weight filename (e.g. 'seg')."""
        pattern = cls._filename_regex()
        if pattern is None:
            return None
        m = pattern.search(filename.lower())
        if m and m.group(2):
            return m.group(2).lstrip("-")
        return None

    @classmethod
    def get_download_url(cls, filename: str) -> Optional[str]:
        """Return the Hugging Face download URL for the given weight filename."""
        size = cls.detect_size_from_filename(filename)
        if size is None:
            return None
        task = cls.detect_task_from_filename(filename)
        suffix = f"-{task}" if task else ""
        name = f"{cls.FILENAME_PREFIX}{size}{suffix}"
        return f"https://huggingface.co/LibreYOLO/{name}/resolve/main/{name}{cls.WEIGHT_EXT}"

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

    def __call__(
        self, source=None, **kwargs
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        return self._runner(source, **kwargs)

    def predict(
        self, *args, **kwargs
    ) -> Union[Results, List[Results], Generator[Results, None, None]]:
        """Alias for __call__ method."""
        return self(*args, **kwargs)

    def track(
        self,
        source: str | Path,
        *,
        track_conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        save: bool = False,
        output_path: Optional[str] = None,
        tracker_config=None,
        **tracker_kwargs,
    ) -> Generator[Results, None, None]:
        """Track objects across video frames.

        Runs detection on each frame and associates detections across time
        using the ByteTrack algorithm. Yields one Results per frame with
        ``track_id`` set.

        Args:
            source: Path to a video file.
            track_conf: Confidence threshold for the tracker's first
                association stage (``track_high_thresh``). The detector
                runs at the lower ``track_low_thresh`` internally so
                ByteTrack can use low-confidence detections for recovery.
            iou: IoU threshold for NMS during detection.
            imgsz: Override input image size.
            classes: Filter to specific class IDs.
            max_det: Maximum detections per frame.
            save: If True, save annotated frames (with bounding boxes and
                track IDs) as images to *output_path*.
            output_path: Directory to save annotated frames. Defaults to
                ``runs/track/<video_stem>/``.
            tracker_config: A ``TrackConfig`` instance, or None to build
                one from **tracker_kwargs.
            **tracker_kwargs: Forwarded to ``TrackConfig.from_kwargs``.

        Yields:
            Results with ``track_id`` attribute set as an (N,) int tensor.
        """
        import cv2
        import numpy as np

        from ...tracking import ByteTracker, TrackConfig
        from ...utils.drawing import draw_boxes

        if tracker_config is None:
            tracker_config = TrackConfig.from_kwargs(**tracker_kwargs)

        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"Video file not found: {source}")

        # ByteTrack needs to see low-confidence detections.
        effective_conf = tracker_config.track_low_thresh

        # Resolve save directory.
        output_dir = None
        if save:
            if output_path is not None:
                output_dir = Path(output_path)
            else:
                video_stem = Path(source).stem
                from ...utils.general import increment_path

                output_dir = increment_path(
                    Path("runs") / "track" / video_stem, exist_ok=False
                )
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving tracked frames to {output_dir}/")

        tracker = ByteTracker(config=tracker_config)
        cap = cv2.VideoCapture(str(source))
        frame_idx = 0

        try:
            while cap.isOpened():
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                # Convert BGR → RGB numpy array for the detector.
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                result = self._runner(
                    frame_rgb,
                    conf=effective_conf,
                    iou=iou,
                    imgsz=imgsz,
                    classes=classes,
                    max_det=max_det,
                    color_format="rgb",
                )

                tracked = tracker.update(result)

                if save and output_dir is not None:
                    img_pil = Image.fromarray(frame_rgb)
                    tid_list = (
                        tracked.track_id.tolist()
                        if tracked.track_id is not None and len(tracked) > 0
                        else None
                    )
                    if len(tracked) > 0:
                        annotated = draw_boxes(
                            img_pil,
                            tracked.boxes.xyxy.tolist(),
                            tracked.boxes.conf.tolist(),
                            tracked.boxes.cls.tolist(),
                            class_names=self.names,
                            track_ids=tid_list,
                        )
                    else:
                        annotated = img_pil
                    annotated.save(output_dir / f"frame_{frame_idx:06d}.jpg")

                frame_idx += 1
                yield tracked
        finally:
            cap.release()

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
