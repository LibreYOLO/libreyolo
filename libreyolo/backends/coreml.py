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

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


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
        spec = self.model.get_spec()
        self.output_names = [out.name for out in spec.description.output]

        meta = (
            dict(self.model.user_defined_metadata)
            if self.model.user_defined_metadata
            else {}
        )
        model_family, names, imgsz, has_embedded_nms = self._parse_metadata(
            meta,
            nb_classes,
            output_names=self.output_names,
        )

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
    def _parse_metadata(
        meta: dict,
        default_nb_classes: int,
        *,
        output_names: list[str] | None = None,
    ):
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

        has_embedded_nms = str(meta.get("nms", "")).lower() == "true"
        if output_names is not None:
            has_embedded_nms = has_embedded_nms or set(output_names) == {
                "confidence",
                "coordinates",
            }

        return model_family, names, imgsz, has_embedded_nms

    def _parse_outputs(
        self,
        all_outputs: list,
        effective_imgsz: int,
        original_size: tuple,
        conf: float,
        ratio: float = 1.0,
    ):
        if self._has_embedded_nms:
            return self._parse_embedded_nms(
                all_outputs, effective_imgsz, original_size, conf, ratio=ratio
            )
        return super()._parse_outputs(
            all_outputs, effective_imgsz, original_size, conf, ratio=ratio
        )

    def _parse_embedded_nms(
        self,
        all_outputs: list,
        effective_imgsz: int,
        original_size: tuple,
        conf: float,
        ratio: float = 1.0,
    ):
        output_by_name = {
            name: np.asarray(value)
            for name, value in zip(self.output_names, all_outputs)
        }
        confidence = output_by_name.get("confidence")
        coordinates = output_by_name.get("coordinates")
        if confidence is None or coordinates is None:
            raise RuntimeError(
                "CoreML embedded NMS output must include confidence and coordinates"
            )

        if confidence.ndim == 3:
            confidence = confidence[0]
        if coordinates.ndim == 3:
            coordinates = coordinates[0]

        max_scores = np.max(confidence, axis=1)
        class_ids = np.argmax(confidence, axis=1)
        mask = max_scores > conf
        boxes_raw = coordinates[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(boxes_raw) == 0:
            return np.empty((0, 4)), max_scores, class_ids

        orig_w, orig_h = original_size
        cx, cy, w, h = (
            boxes_raw[:, 0],
            boxes_raw[:, 1],
            boxes_raw[:, 2],
            boxes_raw[:, 3],
        )
        boxes = np.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
            axis=1,
        )

        family = (self.model_family or "").lower()
        if family == "yolox":
            boxes /= ratio
        elif family == "rtdetr":
            boxes[:, [0, 2]] *= orig_w
            boxes[:, [1, 3]] *= orig_h
        else:
            scale_x = orig_w / effective_imgsz
            scale_y = orig_h / effective_imgsz
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        return boxes, max_scores, class_ids

    def _run_inference(self, blob: np.ndarray) -> list:
        """Run CoreML inference on a (1, C, H, W) preprocessed float blob.

        The exported model takes a canonical RGB uint8 image; family-specific
        transforms (e.g. YOLOX BGR/0-255, RF-DETR ImageNet normalization) are
        baked into the .mlpackage by the exporter wrapper. Here we reconstruct
        a canonical RGB uint8 PIL image from each family's preprocessed blob.
        """
        if blob.ndim != 4 or blob.shape[0] != 1:
            raise ValueError(
                f"CoreMLBackend expects (1, C, H, W) blob; got {blob.shape}"
            )

        family = (self.model_family or "").lower()
        chw = blob[0]

        if family == "yolox":
            # blob is BGR float in [0, 255] → swap channels to RGB.
            hwc = np.transpose(chw, (1, 2, 0))[..., ::-1]
            arr = hwc
        elif family == "rfdetr":
            # blob is RGB float, ImageNet-normalized → un-normalize to [0, 1] then ×255.
            mean = np.array(_IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
            std = np.array(_IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
            chw_unnorm = chw * std + mean
            hwc = np.transpose(chw_unnorm, (1, 2, 0))
            arr = hwc * 255.0
        else:
            # yolo9, rtdetr, default: RGB float in [0, 1] → ×255.
            hwc = np.transpose(chw, (1, 2, 0))
            arr = hwc * 255.0

        uint8 = np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))
        pil = Image.fromarray(uint8)

        out = self.model.predict({"image": pil})
        return [np.asarray(out[name]) for name in self.output_names if name in out]
