"""TensorFlow Lite inference backend for LibreYOLO exports."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..tasks import normalize_supported_tasks, normalize_task, resolve_task
from ..utils.general import COCO_CLASSES
from ..utils.serialization import warn_on_metadata_schema_version
from .base import BaseBackend, _read_metadata_imgsz, _read_pose_metadata, classify_eval_kwargs

logger = logging.getLogger(__name__)

# Must match ``libreyolo.export.tflite.SPLIT_OUTPUT_LAYOUT``; kept here so the
# runtime does not import the (torch-based) export package.
SPLIT_OUTPUT_LAYOUT = "boxes_norm_scores"


class TFLiteBackend(BaseBackend):
    """Run `.tflite` artifacts through the LiteRT interpreter."""

    def __init__(
        self,
        model_path: str,
        nb_classes: int | None = None,
        device: str = "auto",
        task: str | None = None,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"TFLite model not found: {model_path}")
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError as exc:
            raise ImportError(
                "TFLite inference requires ai-edge-litert. "
                "Install with: pip install libreyolo[tflite]"
            ) from exc

        metadata_path = Path(str(path) + ".json")
        metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else {}
        )
        warn_on_metadata_schema_version(
            metadata,
            artifact=f"TFLite metadata for {model_path}",
            logger=logger,
        )
        family = metadata.get("model_family")
        model_size = metadata.get("model_size") or metadata.get("size")
        default_task = normalize_task(metadata.get("default_task"), default="detect")
        metadata_task = normalize_task(metadata.get("task"), default=default_task)
        supported_tasks = normalize_supported_tasks(
            metadata.get("supported_tasks", (metadata_task,))
        )
        resolved_task = resolve_task(
            explicit_task=task,
            checkpoint_task=metadata_task,
            default_task=default_task,
            supported_tasks=supported_tasks,
        )
        resolved_nc = int(
            nb_classes
            if nb_classes is not None
            else metadata.get("nc", metadata.get("nb_classes", 80))
        )
        names_raw = metadata.get("names")
        if isinstance(names_raw, str):
            names_raw = json.loads(names_raw)
        if isinstance(names_raw, dict):
            names = {int(key): value for key, value in names_raw.items()}
        elif resolved_nc == 80:
            names = {index: name for index, name in enumerate(COCO_CLASSES)}
        else:
            names = self.build_names(resolved_nc)
        imgsz = (
            _read_metadata_imgsz(
                metadata, family, artifact=f"TFLite metadata for {model_path}"
            )
            or 640
        )

        self._output_layout = metadata.get("output_layout")
        self._canvas_hw = (
            int(metadata.get("imgsz_h") or imgsz),
            int(metadata.get("imgsz_w") or imgsz),
        ) if isinstance(imgsz, int) else tuple(int(v) for v in imgsz)

        self.interpreter = Interpreter(model_path=str(path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        if len(self.input_details) != 1:
            raise ValueError(
                f"TFLite backend expects one image input, got {len(self.input_details)}."
            )
        super().__init__(
            model_path=str(path),
            nb_classes=resolved_nc,
            device="cpu",
            imgsz=imgsz,
            model_family=family,
            names=names,
            model_size=model_size,
            task=resolved_task,
            supported_tasks=supported_tasks,
            default_task=default_task,
            **classify_eval_kwargs(metadata),
            **_read_pose_metadata(metadata),
        )

    @staticmethod
    def _quantize(array: np.ndarray, detail: dict) -> np.ndarray:
        dtype = detail["dtype"]
        if np.issubdtype(dtype, np.floating):
            return array.astype(dtype, copy=False)
        scale, zero = detail.get("quantization", (0.0, 0))
        if not scale:
            raise ValueError("Quantized TFLite input is missing scale metadata.")
        limits = np.iinfo(dtype)
        return np.clip(np.rint(array / scale + zero), limits.min, limits.max).astype(
            dtype
        )

    @staticmethod
    def _dequantize(array: np.ndarray, detail: dict) -> np.ndarray:
        if np.issubdtype(array.dtype, np.floating):
            return array
        scale, zero = detail.get("quantization", (0.0, 0))
        return (array.astype(np.float32) - float(zero)) * float(scale or 1.0)

    def _run_inference(self, blob: np.ndarray) -> list:
        detail = self.input_details[0]
        expected = tuple(int(value) for value in detail["shape"])
        value = blob
        if len(expected) == 4 and expected[-1] == 3 and blob.shape[1] == 3:
            value = np.transpose(blob, (0, 2, 3, 1))
        value = self._quantize(np.ascontiguousarray(value), detail)
        self.interpreter.set_tensor(detail["index"], value)
        self.interpreter.invoke()
        outputs = []
        for output_detail in self.output_details:
            output = self._dequantize(
                self.interpreter.get_tensor(output_detail["index"]), output_detail
            )
            # LiteRT does not reliably preserve useful output names across the
            # ONNX conversion path, so fixed-family exports currently recover
            # layout from rank and channel counts. Keep these checks task- and
            # family-scoped: a coincidental spatial dimension equal to ``nc``
            # must not affect unrelated outputs. Prefer explicit sidecar layout
            # metadata once the converter can emit it consistently.
            if (
                self.task in {"semantic", "point", "depth", "matte", "edge"}
                and output.ndim == 4
                and output.shape[-1] in {1, self.nb_classes, self.nb_classes + 1}
            ):
                output = np.transpose(output, (0, 3, 1, 2))
            if (
                self.task == "detect"
                and self.model_family
                in {
                    "yolo1",
                    "yolo2",
                    "yolo3",
                    "yolo4",
                    "yolo7",
                    "yolo9",
                    "yolo9_e2e",
                    "yolo9_p2",
                }
                and output.ndim == 3
                and output.shape[-1] == self.nb_classes + 4
            ):
                output = np.transpose(output, (0, 2, 1))
            if (
                self.model_family == "yolonas"
                and self.task in {"detect", "pose"}
                and output.ndim == 3
                and output.shape[1] in {4, self.nb_classes}
            ):
                output = np.transpose(output, (0, 2, 1))
            outputs.append(np.ascontiguousarray(output))
        if getattr(self, "_output_layout", None) == SPLIT_OUTPUT_LAYOUT:
            return [self._merge_split_outputs(outputs)]
        return outputs

    def _merge_split_outputs(self, outputs: list) -> np.ndarray:
        """Rebuild the family's pixel-space tensor from the split INT8 layout.

        The graph emits normalized boxes ``(B, 4, N)`` and scores ``(B, N, C)``
        as separate tensors so each keeps its own int8 scale. LiteRT does not
        keep output order or names reliably, so the boxes are the rank-3
        output with 4 rows.
        """
        if len(outputs) != 2:
            raise ValueError(
                f"TFLite output layout {SPLIT_OUTPUT_LAYOUT!r} expects 2 outputs, "
                f"got {len(outputs)}."
            )
        box_first = outputs[0].ndim == 3 and outputs[0].shape[1] == 4
        boxes, scores = outputs if box_first else outputs[::-1]
        if boxes.ndim != 3 or boxes.shape[1] != 4 or scores.shape[1] != boxes.shape[2]:
            raise ValueError(
                "Unexpected TFLite split outputs: "
                f"{[tuple(o.shape) for o in outputs]}."
            )
        canvas_h, canvas_w = self._canvas_hw
        scale = np.array([canvas_w, canvas_h, canvas_w, canvas_h], dtype=np.float32)
        pixels = np.transpose(boxes, (0, 2, 1)) * scale
        merged = np.concatenate([pixels, scores.astype(np.float32)], axis=-1)
        if self.model_family == "yolo9":
            # YOLO9 parsing expects the channel-first (B, 4 + nc, N) layout.
            merged = np.transpose(merged, (0, 2, 1))
        return np.ascontiguousarray(merged)


__all__ = ["TFLiteBackend"]
