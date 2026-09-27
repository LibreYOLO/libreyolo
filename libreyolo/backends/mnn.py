"""CPU inference backend for LibreYOLO MNN exports.

The backend uses the public ``MNN.nn`` Module API. No MNN source code is
vendored or adapted here.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, cast

import numpy as np

from ..tasks import normalize_supported_tasks, normalize_task, resolve_task
from ..utils.serialization import warn_on_metadata_schema_version
from .base import BaseBackend, ImageSize, _read_metadata_imgsz

logger = logging.getLogger(__name__)

_SUPPORTED_FAMILIES = {
    "yolo9",
    "rfdetr",
    "yolo9_e2e",
    "yolo9_p2",
    "ec",
    "rtdetr",
    "rtdetrv2",
    "rtdetrv4",
    "dfine",
    "deim",
    "deimv2",
    "yolonas",
}


def _load_sidecar(model_path: Path) -> dict:
    sidecar_path = Path(f"{model_path}.json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"MNN metadata sidecar not found: {sidecar_path}")
    try:
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid MNN metadata sidecar: {sidecar_path}") from exc
    if not isinstance(metadata, dict):
        raise TypeError(
            f"MNN metadata sidecar must contain a JSON object: {sidecar_path}"
        )
    return metadata


def _required_string_list(metadata: dict, key: str) -> list[str]:
    value = metadata.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise ValueError(f"MNN metadata must declare a non-empty {key} list.")
    return [str(item) for item in value]


class MNNBackend(BaseBackend):
    """Run a fixed-shape FP32 ``.mnn`` artifact on the MNN CPU runtime."""

    fixed_input_shape = True

    def __init__(
        self,
        model_path: str,
        nb_classes: int | None = None,
        device: str = "auto",
        task: str | None = None,
    ) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"MNN model not found: {model_path}")
        try:
            import MNN
        except (ImportError, OSError) as exc:
            raise ImportError(
                "MNN inference requires the optional MNN runtime. "
                "Install with: pip install libreyolo[mnn]"
            ) from exc
        mnn_api = cast(Any, MNN)

        if str(device).lower() not in {"auto", "cpu"}:
            logger.warning("MNN exports run on CPU; ignoring device=%r.", device)

        metadata = _load_sidecar(path)
        warn_on_metadata_schema_version(
            metadata,
            artifact=f"MNN metadata for {model_path}",
            logger=logger,
        )
        if str(metadata.get("format", "")).lower() != "mnn":
            raise ValueError("MNN metadata must declare format='mnn'.")
        if not isinstance(metadata.get("mnn_version"), str) or not metadata.get(
            "mnn_version"
        ):
            raise ValueError("MNN metadata must declare the exporter mnn_version.")
        if str(metadata.get("mnn_backend", "")).lower() != "cpu":
            raise ValueError("MNN v1 requires metadata mnn_backend='cpu'.")
        if metadata.get("dynamic") not in {False, "false", "False"}:
            raise ValueError("MNN v1 requires fixed-shape metadata dynamic=false.")
        if str(metadata.get("precision", "")).lower() != "fp32":
            raise ValueError("MNN v1 requires metadata precision='fp32'.")

        model_family = str(metadata.get("model_family", "")).lower()
        if model_family not in _SUPPORTED_FAMILIES:
            raise ValueError(
                "MNN v1 has no runtime contract for this model family; "
                f"got model_family={model_family or 'missing'!r}."
            )
        model_size = metadata.get("model_size") or metadata.get("size")
        if not isinstance(model_size, str) or not model_size:
            raise ValueError("MNN metadata must declare model size.")

        default_task = (
            normalize_task(metadata.get("default_task"), default="detect") or "detect"
        )
        metadata_task = (
            normalize_task(metadata.get("task"), default=default_task) or default_task
        )
        supported_tasks = normalize_supported_tasks(
            metadata.get("supported_tasks", (metadata_task,))
        )
        resolved_task = resolve_task(
            explicit_task=task,
            checkpoint_task=metadata_task,
            default_task=default_task,
            supported_tasks=supported_tasks,
        )
        if resolved_task != "detect":
            raise ValueError(
                f"MNN v1 supports detection exports only, got task={resolved_task!r}."
            )

        input_names = _required_string_list(metadata, "mnn_input_names")
        output_names = _required_string_list(metadata, "mnn_output_names")
        if len(input_names) != 1:
            raise ValueError(
                f"MNN backend expects one image input, got {len(input_names)}."
            )
        input_shape = metadata.get("mnn_input_shape")
        if (
            not isinstance(input_shape, list)
            or len(input_shape) != 4
            or not all(isinstance(value, int) and value > 0 for value in input_shape)
        ):
            raise ValueError(
                "MNN metadata must declare a fixed positive NCHW mnn_input_shape."
            )
        artifact_batch = int(metadata.get("mnn_batch", input_shape[0]))
        if artifact_batch != input_shape[0]:
            raise ValueError(
                "MNN batch metadata does not match mnn_input_shape: "
                f"{artifact_batch} != {input_shape[0]}."
            )

        imgsz = _read_metadata_imgsz(
            metadata, model_family, artifact=f"MNN metadata for {model_path}"
        )
        if imgsz is None:
            raise ValueError("MNN metadata must declare a fixed input size.")
        expected_h, expected_w = (
            (imgsz, imgsz) if isinstance(imgsz, int) else (imgsz[0], imgsz[1])
        )
        if input_shape[1:] != [3, int(expected_h), int(expected_w)]:
            raise ValueError(
                "MNN input shape disagrees with image-size metadata: "
                f"shape={input_shape}, imgsz={(expected_h, expected_w)}."
            )

        names_raw = metadata.get("names")
        if isinstance(names_raw, str):
            names_raw = json.loads(names_raw)
        if not isinstance(names_raw, dict) or not names_raw:
            raise ValueError("MNN metadata must declare the exported class names.")
        names = {int(key): str(value) for key, value in names_raw.items()}
        metadata_nc = int(metadata.get("nc", metadata.get("nb_classes", len(names))))
        expected_name_keys = set(range(metadata_nc))
        if set(names) != expected_name_keys:
            raise ValueError(
                "MNN class-name metadata keys must cover the range 0..nc-1: "
                f"got keys={sorted(names)} for nc={metadata_nc}."
            )
        if nb_classes is not None and int(nb_classes) != metadata_nc:
            raise ValueError(
                "MNN nb_classes override does not match the fixed artifact metadata: "
                f"{int(nb_classes)} != {metadata_nc}."
            )
        resolved_nc = metadata_nc

        thread_count = min(max(os.cpu_count() or 1, 1), 4)
        runtime_manager = mnn_api.nn.create_runtime_manager(
            ({"backend": 0, "precision": 1, "numThread": thread_count},)
        )
        try:
            module = mnn_api.nn.load_module_from_file(
                str(path),
                input_names,
                output_names,
                runtime_manager=runtime_manager,
                dynamic=False,
                shape_mutable=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load MNN model {model_path}: {exc}") from exc

        self._MNN = mnn_api
        self.runtime_manager = runtime_manager
        self.module = module
        self.input_names = input_names
        self.output_names = output_names
        self.input_shape = tuple(input_shape)
        self._artifact_batch = artifact_batch

        super().__init__(
            model_path=str(path),
            nb_classes=resolved_nc,
            device="cpu",
            imgsz=imgsz,
            model_family=model_family,
            names=names,
            model_size=model_size,
            task=resolved_task,
            supported_tasks=supported_tasks,
            default_task=default_task,
            letterbox_pad=metadata.get("letterbox_pad"),
        )

    def _resolve_predict_imgsz(self, imgsz: ImageSize | None = None) -> ImageSize:
        effective = super()._resolve_predict_imgsz(imgsz)
        if effective != self.imgsz:
            raise ValueError(
                "MNN v1 artifacts use a fixed input size; "
                f"expected imgsz={self.imgsz}, got {effective}."
            )
        return effective

    def _supports_batched_inference(self) -> bool:
        return self._artifact_batch > 1

    def _process_in_batches(
        self,
        images,
        batch=1,
        save=False,
        output_path=None,
        conf=0.25,
        iou=0.45,
        imgsz=None,
        classes=None,
        max_det=300,
        color_format="auto",
        start_idx=0,
    ):
        return super()._process_in_batches(
            images,
            batch=min(max(int(batch), 1), self._artifact_batch),
            save=save,
            output_path=output_path,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            classes=classes,
            max_det=max_det,
            color_format=color_format,
            start_idx=start_idx,
        )

    def _run_inference(self, blob: np.ndarray) -> list[np.ndarray]:
        value = np.ascontiguousarray(blob, dtype=np.float32)
        if value.ndim != 4 or tuple(value.shape[1:]) != self.input_shape[1:]:
            raise ValueError(
                f"MNN input must have shape (N, {self.input_shape[1:]}); "
                f"got {value.shape}."
            )
        requested_batch = int(value.shape[0])
        if requested_batch < 1 or requested_batch > self._artifact_batch:
            raise ValueError(
                f"MNN artifact accepts at most batch {self._artifact_batch}, "
                f"got {requested_batch}."
            )
        if requested_batch < self._artifact_batch:
            padding = np.zeros(
                (self._artifact_batch - requested_batch, *value.shape[1:]),
                dtype=np.float32,
            )
            value = np.concatenate((value, padding), axis=0)

        input_var = self._MNN.expr.const(
            value,
            list(value.shape),
            self._MNN.expr.NCHW,
            self._MNN.expr.float,
        )
        try:
            raw_outputs = self.module.forward([input_var])
        except Exception as exc:
            raise RuntimeError(f"MNN CPU inference failed: {exc}") from exc
        if not isinstance(raw_outputs, (list, tuple)):
            raw_outputs = [raw_outputs]
        if len(raw_outputs) != len(self.output_names):
            raise RuntimeError(
                "MNN runtime output count does not match metadata: "
                f"expected {len(self.output_names)}, got {len(raw_outputs)}."
            )

        outputs = []
        for output in raw_outputs:
            converted = self._MNN.expr.convert(output, self._MNN.expr.NCHW)
            array = np.asarray(converted.read())
            shape = tuple(int(value) for value in converted.shape)
            if shape and array.shape != shape:
                array = array.reshape(shape)
            outputs.append(
                np.ascontiguousarray(array[:requested_batch], dtype=np.float32)
            )
        return outputs


__all__ = ["MNNBackend"]
