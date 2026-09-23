"""ONNX runtime inference backend for LibreYOLO."""

import json
import logging
from pathlib import Path

import numpy as np

from ..tasks import resolve_task
from ..utils.serialization import (
    reject_unsupported_input_kind,
    warn_on_metadata_schema_version,
)
from .base import (
    classify_eval_kwargs,
    BaseBackend,
    ImageSize,
    MetadataImageSizeError,
    _read_pose_metadata,
)
from .metadata import parse_export_metadata

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
        elif device in ("cuda", "gpu") or device.startswith("cuda:"):
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                resolved_device = "cuda"
            else:
                providers = ["CPUExecutionProvider"]
                resolved_device = "cpu"
                logger.warning(
                    "Requested device %r but CUDAExecutionProvider is not "
                    "available; falling back to CPU.",
                    device,
                )
        else:
            providers = ["CPUExecutionProvider"]
            resolved_device = "cpu"
            logger.warning(
                "Requested device %r is not supported by the ONNX backend; "
                "falling back to CPU.",
                device,
            )

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
        pose_metadata = self._read_onnx_pose_metadata(
            onnx_path, runtime_metadata=runtime_metadata
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
        # Dynamic-batch exports carry a symbolic dim ("batch") or None at
        # axis 0; static exports carry an int.
        self._dynamic_batch_axis = bool(input_shape) and not isinstance(
            input_shape[0], int
        )
        # Faster R-CNN keeps its resize and normalization inside the graph.
        # Record dynamic H/W separately from dynamic batch so the shared
        # backend can pass original-resolution pixels through unchanged.
        self._dynamic_spatial_axes = len(input_shape) == 4 and (
            not isinstance(input_shape[2], int) or not isinstance(input_shape[3], int)
        )
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

        from ..utils.event_histogram import validate_input_profile

        self.input_profile = validate_input_profile(
            json.loads(runtime_metadata["input_profile"])
            if "input_profile" in runtime_metadata else None,
            family=model_family,
            task=resolved_task,
        )
        self.input_initialization = runtime_metadata.get("input_initialization")
        self.letterbox_pad = runtime_metadata.get("letterbox_pad", "topleft")
        if self.input_profile is not None:
            if self.letterbox_pad not in {"center", "topleft"}:
                raise ValueError("Invalid histogram letterbox_pad metadata")
            if len(input_shape) != 4 or input_shape[1] != 2:
                raise ValueError("Histogram ONNX graph must consume NCHW with two channels")
            if self.input_initialization not in {"random", "rgb_mean"}:
                raise ValueError("Histogram ONNX metadata requires input_initialization")
        elif len(input_shape) == 4 and input_shape[1] == 2:
            raise ValueError("Two-channel ONNX graph is missing input_profile metadata")

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
            **classify_eval_kwargs(runtime_metadata),
            num_bins=(
                int(runtime_metadata["num_bins"])
                if runtime_metadata.get("num_bins")
                else None
            ),
            bin_width_deg=(
                float(runtime_metadata["bin_width_deg"])
                if runtime_metadata.get("bin_width_deg")
                else None
            ),
            offset_deg=(
                float(runtime_metadata["offset_deg"])
                if runtime_metadata.get("offset_deg")
                else None
            ),
            **pose_metadata,
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
            reject_unsupported_input_kind(
                meta, artifact=f"ONNX metadata for {onnx_path}"
            )
            parsed = parse_export_metadata(
                meta,
                artifact=f"ONNX metadata for {onnx_path}",
                default_nb_classes=default_nb_classes,
            )
        except (NotImplementedError, MetadataImageSizeError):
            raise
        except Exception as e:
            logger.warning("Failed to read ONNX metadata from %s: %s", onnx_path, e)
            return (
                None,
                None,
                "detect",
                ("detect",),
                "detect",
                None,
                False,
                None,
            )

        return (
            parsed.model_family,
            parsed.model_size,
            parsed.task,
            parsed.supported_tasks,
            parsed.default_task,
            parsed.names
            if "names" in meta or "nc" in meta or "nb_classes" in meta
            else None,
            parsed.embedded_nms,
            parsed.imgsz,
        )

    @staticmethod
    def _read_onnx_pose_metadata(
        onnx_path: str,
        runtime_metadata: dict | None = None,
    ) -> dict:
        try:
            meta = dict(runtime_metadata or {})
            if not meta:
                import onnx

                model_proto = onnx.load(onnx_path)
                meta = {p.key: p.value for p in model_proto.metadata_props}
        except Exception as e:
            logger.warning(
                "Failed to read ONNX pose metadata from %s: %s", onnx_path, e
            )
            return {}

        return _read_pose_metadata(meta)

    def _supports_batched_inference(self) -> bool:
        # Embedded-NMS graphs are exported batch-1; everything else with a
        # dynamic batch axis accepts stacked blobs directly.
        return self._dynamic_batch_axis and not self.embedded_nms

    def _run_inference(self, blob: np.ndarray) -> list:
        """Run ONNX Runtime inference."""
        return self.session.run(None, {self.input_name: blob})
