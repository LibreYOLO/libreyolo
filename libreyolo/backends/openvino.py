"""OpenVINO inference backend for LibreYOLO."""

import logging
from pathlib import Path
from typing import Dict

import numpy as np

from ..tasks import normalize_supported_tasks, normalize_task, resolve_task
from ..utils.serialization import warn_on_metadata_schema_version
from .base import BaseBackend, ImageSize, _read_metadata_imgsz

logger = logging.getLogger(__name__)


class OpenVINOBackend(BaseBackend):
    """OpenVINO inference backend for LibreYOLO models.

    Args:
        model_dir: Path to the OpenVINO model directory (containing model.xml,
            model.bin, and optionally metadata.yaml).
        nb_classes: Number of classes (default: auto-detected from metadata, fallback 80).
        device: Device for inference. "auto" (default) uses CPU. "gpu"/"cuda" uses GPU.

    Example:
        >>> model = OpenVINOBackend("exported_model_dir/")
        >>> result = model("image.jpg", save=True)
        >>> print(result.boxes.xyxy)
    """

    def __init__(
        self,
        model_dir: str | Path,
        nb_classes: int | None = None,
        device: str = "auto",
        task: str | None = None,
    ):
        try:
            import openvino as ov
        except ImportError as e:
            raise ImportError(
                "OpenVINO inference requires openvino. "
                "Install with: pip install openvino"
            ) from e

        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"OpenVINO model directory not found: {model_dir}")

        xml_path = model_dir / "model.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"model.xml not found in {model_dir}")

        explicit_task = task
        model_family = None
        model_size = None
        task = "detect"
        default_task = "detect"
        supported_tasks = ("detect",)
        imgsz = 640
        resolved_nb_classes = nb_classes if nb_classes is not None else 80
        names = self.build_names(resolved_nb_classes)

        metadata_path = model_dir / "metadata.yaml"
        if metadata_path.exists():
            (
                model_family,
                model_size,
                metadata_task,
                supported_tasks,
                default_task,
                imgsz,
                resolved_nb_classes,
                names,
            ) = self._read_metadata(metadata_path, nb_classes)
            task = resolve_task(
                explicit_task=explicit_task,
                checkpoint_task=metadata_task,
                default_task=default_task,
                supported_tasks=supported_tasks,
            )
        else:
            task = resolve_task(
                explicit_task=explicit_task,
                default_task=default_task,
                supported_tasks=supported_tasks,
            )

        # Map device strings to OpenVINO format
        device_lower = device.lower() if device else "auto"
        if device_lower in ("auto", "cpu"):
            ov_device = "CPU"
            resolved_device = "cpu"
        elif device_lower in ("gpu", "cuda"):
            ov_device = "GPU"
            resolved_device = "gpu"
        else:
            ov_device = device.upper()
            resolved_device = device_lower

        core = ov.Core()
        ov_model = core.read_model(str(xml_path))
        self._dynamic_batch_axis = self._detect_dynamic_batch_axis(ov_model)
        self.compiled_model = core.compile_model(ov_model, ov_device)

        static_imgsz = self._read_static_input_imgsz(ov_model)
        if static_imgsz is not None:
            imgsz = static_imgsz

        super().__init__(
            model_path=str(model_dir),
            nb_classes=resolved_nb_classes,
            device=resolved_device,
            imgsz=imgsz,
            model_family=model_family,
            names=names,
            model_size=model_size,
            task=task,
            supported_tasks=supported_tasks,
            default_task=default_task,
        )

    @staticmethod
    def _detect_dynamic_batch_axis(ov_model) -> bool:
        """True when the model input declares a dynamic batch dimension.

        OpenVINO models converted from dynamic-axes ONNX keep the symbolic
        batch dim and accept stacked (N, C, H, W) blobs as-is.
        """
        try:
            return bool(ov_model.inputs[0].get_partial_shape()[0].is_dynamic)
        except Exception:
            return False

    def _supports_batched_inference(self) -> bool:
        return self._dynamic_batch_axis and not getattr(self, "embedded_nms", False)

    @staticmethod
    def _read_static_input_imgsz(ov_model) -> ImageSize | None:
        try:
            input_shape = ov_model.inputs[0].shape
        except RuntimeError:
            # Some converted models keep a dynamic input shape. In that case,
            # use the exported metadata size read before compiling.
            return None

        if len(input_shape) != 4:
            return None
        h, w = input_shape[2], input_shape[3]
        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
            return h if h == w else (h, w)
        return None

    @staticmethod
    def _read_metadata(metadata_path: Path, nb_classes_override: int | None = None):
        """Read metadata from metadata.yaml file.

        Returns:
            Tuple of (model_family, model_size, task, supported_tasks, default_task, imgsz, nb_classes, names).
        """
        import yaml

        with open(metadata_path) as f:
            meta = yaml.safe_load(f) or {}
        warn_on_metadata_schema_version(
            meta,
            artifact=f"OpenVINO metadata sidecar {metadata_path}",
            logger=logger,
        )

        model_family = meta.get("model_family")
        model_size = meta.get("model_size") or meta.get("size")
        default_task = normalize_task(meta.get("default_task"), default="detect")
        task = normalize_task(meta.get("task"), default=default_task)
        supported_tasks = normalize_supported_tasks(meta.get("supported_tasks", (task,)))
        imgsz = (
            _read_metadata_imgsz(
                meta,
                model_family,
                artifact=f"OpenVINO metadata sidecar {metadata_path}",
            )
            or 640
        )

        if nb_classes_override is not None:
            nb_classes = nb_classes_override
        elif "nb_classes" in meta or "nc" in meta:
            nb_classes = int(meta.get("nb_classes", meta.get("nc")))
        else:
            nb_classes = 80

        if "names" in meta and nb_classes_override is None:
            names: Dict[int, str] = {int(k): v for k, v in meta["names"].items()}
        else:
            names = BaseBackend.build_names(nb_classes)

        return model_family, model_size, task, supported_tasks, default_task, imgsz, nb_classes, names

    def _run_inference(self, blob: np.ndarray) -> list:
        """Run OpenVINO inference."""
        result = self.compiled_model(blob)
        return [result[output] for output in self.compiled_model.outputs]
