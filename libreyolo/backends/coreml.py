"""CoreML inference backend for LibreYOLO. macOS only.

Loads .mlpackage models produced by libreyolo.export.coreml and runs inference
via coremltools.models.MLModel. Mirrors OnnxBackend's public surface so the
rest of LibreYOLO (Results, drawing, etc.) sees the same interface.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from ..utils.general import COCO_CLASSES
from .base import BaseBackend

logger = logging.getLogger(__name__)


def _to_compute_unit(compute_units: str):
    """Same mapping as the exporter — duplicated to avoid pulling export deps in."""
    import coremltools as ct

    key = compute_units.lower()
    mapping = {
        "all": ct.ComputeUnit.ALL,
        "cpu_and_gpu": ct.ComputeUnit.CPU_AND_GPU,
        "cpu_and_ne": ct.ComputeUnit.CPU_AND_NE,
        "cpu_only": ct.ComputeUnit.CPU_ONLY,
    }
    if key not in mapping:
        raise ValueError(
            f"Invalid compute_units {compute_units!r}. "
            f"Must be one of: {sorted(mapping)}"
        )
    return mapping[key]


class CoreMLBackend(BaseBackend):
    """CoreML inference backend (macOS only).

    Args:
        model_path: Path to a .mlpackage directory.
        nb_classes: Number of classes (default: 80, overridden by metadata if present).
        device: Ignored — CoreML routes via compute_units instead.
        compute_units: 'all' | 'cpu_and_gpu' | 'cpu_and_ne' | 'cpu_only'. Default 'all'.
    """

    def __init__(
        self,
        model_path: str,
        nb_classes: int = 80,
        device: str = "auto",
        compute_units: str = "all",
    ):
        if sys.platform != "darwin":
            raise RuntimeError(
                "CoreML inference requires macOS. "
                f"Current platform: {sys.platform}."
            )
        try:
            import coremltools as ct
        except ImportError as e:
            raise ImportError(
                "CoreML inference requires coremltools. "
                "Install with: pip install libreyolo[coreml]"
            ) from e

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"CoreML model not found: {model_path}")

        self.model = ct.models.MLModel(
            str(path), compute_units=_to_compute_unit(compute_units)
        )

        meta = dict(self.model.user_defined_metadata) if self.model.user_defined_metadata else {}
        model_family, names, imgsz, has_embedded_nms = self._parse_metadata(meta, nb_classes)

        self._has_embedded_nms = has_embedded_nms

        super().__init__(
            model_path=str(path),
            nb_classes=len(names) if names else nb_classes,
            device="coreml",
            imgsz=imgsz,
            model_family=model_family,
            names=names if names else self.build_names(nb_classes),
        )

    @staticmethod
    def _parse_metadata(meta: dict, default_nb_classes: int):
        model_family: Optional[str] = meta.get("model_family") or None
        names: Optional[dict] = None
        imgsz = 640
        has_embedded_nms = False

        if "names" in meta:
            try:
                raw = json.loads(meta["names"])
                names = {int(k): v for k, v in raw.items()}
            except (ValueError, TypeError) as e:
                logger.warning("Failed to parse names metadata: %s", e)

        if names is None and meta.get("nb_classes"):
            try:
                nc = int(meta["nb_classes"])
                names = (
                    {i: n for i, n in enumerate(COCO_CLASSES)}
                    if nc == 80
                    else {i: f"class_{i}" for i in range(nc)}
                )
            except ValueError:
                pass

        if "imgsz" in meta:
            try:
                imgsz = int(meta["imgsz"])
            except ValueError:
                pass

        # If the model has 'confidence'/'coordinates' outputs (post-NMS pipeline),
        # we should not run python-side NMS again.
        try:
            output_names = {o.name for o in meta.get("__output_descriptions__", [])}
        except Exception:
            output_names = set()
        has_embedded_nms = output_names == {"confidence", "coordinates"}

        return model_family, names, imgsz, has_embedded_nms

    def _run_inference(self, blob: np.ndarray) -> list:
        """Run CoreML inference on a (1, C, H, W) preprocessed float blob.

        The exported model expects a CoreML ImageType input (uint8 PIL image,
        normalization baked in). We undo the libreyolo preprocess normalization
        to reconstruct a uint8 PIL image, then feed it.
        """
        if blob.ndim != 4 or blob.shape[0] != 1:
            raise ValueError(
                f"CoreMLBackend expects (1, C, H, W) blob; got {blob.shape}"
            )
        # Reverse the (x/255) normalization: blob is float in [0, 1].
        # CoreML model has scale=1/255 baked in, so it wants uint8 [0, 255].
        chw = blob[0]
        hwc = np.transpose(chw, (1, 2, 0))
        uint8 = np.clip(hwc * 255.0, 0, 255).astype(np.uint8)
        pil = Image.fromarray(uint8)

        out = self.model.predict({"image": pil})
        # Return in stable order — caller (BaseBackend postprocess) maps by index.
        return [np.asarray(v) for _, v in sorted(out.items())]