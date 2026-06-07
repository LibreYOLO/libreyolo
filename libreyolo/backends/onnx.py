"""ONNX runtime inference backend for LibreYOLO."""

import logging
from pathlib import Path

import numpy as np

from ..tasks import normalize_supported_tasks, normalize_task, resolve_task
from ..utils.general import COCO_CLASSES
from ..utils.serialization import warn_on_metadata_schema_version
from .base import BaseBackend, ImageSize, MetadataImageSizeError, _read_metadata_imgsz

logger = logging.getLogger(__name__)


class OnnxBackend(BaseBackend):
    """ONNX runtime inference backend for LibreYOLO models.

    Args:
        onnx_path: Path to the ONNX model file.
        nb_classes: Number of classes (default: 80 for COCO).
        device: Device for inference. "auto" (default) uses CUDA if available, else CPU.

    Example:
        >>> model = OnnxBackend("model.onnx")
        >>> result = model("image.jpg", save=True)
        >>> print(result.boxes.xyxy)
    """

    def __init__(
        self,
        onnx_path: str,
        nb_classes: int = 80,
        device: str = "auto",
        task: str | None = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "ONNX inference requires onnxruntime. "
                "Install with: pip install onnxruntime"
            ) from e

        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        available_providers = ort.get_available_providers()
        if device == "auto":
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                resolved_device = "cuda"
            else:
                providers = ["CPUExecutionProvider"]
                resolved_device = "cpu"
        elif device in ("cuda", "gpu"):
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            resolved_device = (
                "cuda" if "CUDAExecutionProvider" in available_providers else "cpu"
            )
        else:
            providers = ["CPUExecutionProvider"]
            resolved_device = "cpu"

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        try:
            runtime_metadata = dict(
                self.session.get_modelmeta().custom_metadata_map or {}
            )
        except Exception:
            runtime_metadata = {}

        (
            model_family,
            model_size,
            metadata_task,
            supported_tasks,
            default_task,
            names,
            embedded_nms,
            metadata_imgsz,
        ) = self._read_onnx_metadata(
            onnx_path, nb_classes, runtime_metadata=runtime_metadata
        )
        # Models exported with nms=True emit final (1, max_det, 6) detections.
        # Newer YOLO9 ONNX exports also include a raw auxiliary output so the
        # LibreYOLO backend can apply native clipping/NMS for non-square images.
        self.embedded_nms = embedded_nms
        self.embedded_nms_raw_output_index = (
            self.output_names.index("raw")
            if embedded_nms and "raw" in self.output_names
            else None
        )
        input_shape = self.session.get_inputs()[0].shape
        static_imgsz = self._read_static_input_imgsz(input_shape)
        if static_imgsz is not None:
            imgsz = static_imgsz
        elif metadata_imgsz is not None:
            imgsz = metadata_imgsz
        else:
            imgsz = 640  # dynamic shape without metadata; use default
        resolved_task = resolve_task(
            explicit_task=task,
            checkpoint_task=metadata_task,
            default_task=default_task,
            supported_tasks=supported_tasks,
        )

        super().__init__(
            model_path=onnx_path,
            nb_classes=nb_classes if names is None else len(names),
            device=resolved_device,
            imgsz=imgsz,
            model_family=model_family,
            names=names if names is not None else self.build_names(nb_classes),
            model_size=model_size,
            task=resolved_task,
            supported_tasks=supported_tasks,
            default_task=default_task,
        )

    @staticmethod
    def _read_static_input_imgsz(input_shape) -> ImageSize | None:
        if len(input_shape) != 4:
            return None
        h, w = input_shape[2], input_shape[3]
        if not isinstance(h, int) or not isinstance(w, int) or h <= 0 or w <= 0:
            return None
        return h if h == w else (h, w)

    @staticmethod
    def _read_onnx_metadata(
        onnx_path: str,
        default_nb_classes: int,
        runtime_metadata: dict | None = None,
    ):
        """Read libreyolo metadata embedded in an ONNX model file.

        Returns:
            Tuple of (model_family, model_size, task, supported_tasks,
            default_task, names, embedded_nms, imgsz).
        """
        model_family = None
        model_size = None
        task = "detect"
        default_task = "detect"
        supported_tasks = ("detect",)
        names = None
        imgsz = None
        embedded_nms = False
        try:
            meta = dict(runtime_metadata or {})
            if not meta:
                import onnx

                model_proto = onnx.load(onnx_path)
                meta = {p.key: p.value for p in model_proto.metadata_props}
            warn_on_metadata_schema_version(
                meta,
                artifact=f"ONNX metadata for {onnx_path}",
                logger=logger,
            )

            if "model_family" in meta:
                model_family = meta["model_family"]
            if "model_size" in meta or "size" in meta:
                model_size = meta.get("model_size") or meta.get("size")
            imgsz = _read_metadata_imgsz(
                meta,
                model_family,
                artifact=f"ONNX metadata for {onnx_path}",
            )
            if "default_task" in meta:
                default_task = normalize_task(meta["default_task"], default="detect")
            if "task" in meta:
                task = normalize_task(meta["task"], default=default_task)
            elif meta.get("segmentation") == "true":
                task = "segment"
            if "supported_tasks" in meta:
                supported_tasks = normalize_supported_tasks(meta["supported_tasks"])
            else:
                supported_tasks = normalize_supported_tasks((task,))

            if "names" in meta:
                import json

                names_raw = json.loads(meta["names"])
                names = {int(k): v for k, v in names_raw.items()}

            if ("nb_classes" in meta or "nc" in meta) and names is None:
                nc = int(meta.get("nb_classes", meta.get("nc")))
                if nc == 80:
                    names = {i: n for i, n in enumerate(COCO_CLASSES)}
                else:
                    names = {i: f"class_{i}" for i in range(nc)}

            embedded_nms = str(meta.get("nms", "")).lower() == "true"
        except (NotImplementedError, MetadataImageSizeError):
            raise
        except Exception as e:
            logger.warning("Failed to read ONNX metadata from %s: %s", onnx_path, e)

        return (
            model_family,
            model_size,
            task,
            supported_tasks,
            default_task,
            names,
            embedded_nms,
            imgsz,
        )

    def _run_inference(self, blob: np.ndarray) -> list:
        """Run ONNX Runtime inference."""
        return self.session.run(None, {self.input_name: blob})
