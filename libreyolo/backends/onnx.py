"""ONNX runtime inference backend for LibreYOLO."""

import logging
from pathlib import Path

import numpy as np

from ..utils.general import COCO_CLASSES
from .base import BaseBackend

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

    def __init__(self, onnx_path: str, nb_classes: int = 80, device: str = "auto"):
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

        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 4 and isinstance(input_shape[2], int):
            imgsz = input_shape[2]
        else:
            imgsz = 640  # dynamic shape; use default

        model_family, model_size, names = self._read_onnx_metadata(
            onnx_path, nb_classes
        )

        super().__init__(
            model_path=onnx_path,
            nb_classes=nb_classes if names is None else len(names),
            device=resolved_device,
            imgsz=imgsz,
            model_family=model_family,
            names=names if names is not None else self.build_names(nb_classes),
            model_size=model_size,
        )

    @staticmethod
    def _read_onnx_metadata(onnx_path: str, default_nb_classes: int):
        """Read libreyolo metadata embedded in an ONNX model file.

        Returns:
            Tuple of (model_family, names_dict_or_None).
        """
        model_family = None
        model_size = None
        names = None
        try:
            import onnx

            model_proto = onnx.load(onnx_path)
            meta = {p.key: p.value for p in model_proto.metadata_props}

            if "model_family" in meta:
                model_family = meta["model_family"]
            if "model_size" in meta:
                model_size = meta["model_size"]

            if "names" in meta:
                import json

                names_raw = json.loads(meta["names"])
                names = {int(k): v for k, v in names_raw.items()}

            if "nb_classes" in meta and names is None:
                nc = int(meta["nb_classes"])
                if nc == 80:
                    names = {i: n for i, n in enumerate(COCO_CLASSES)}
                else:
                    names = {i: f"class_{i}" for i in range(nc)}
        except Exception as e:
            logger.warning("Failed to read ONNX metadata from %s: %s", onnx_path, e)

        return model_family, model_size, names

    def _run_inference(self, blob: np.ndarray) -> list:
        """Run ONNX Runtime inference."""
        return self.session.run(None, {self.input_name: blob})
