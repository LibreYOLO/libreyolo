"""TorchScript inference backend for LibreYOLO."""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import torch

from ..tasks import normalize_supported_tasks, normalize_task, resolve_task
from ..utils.general import COCO_CLASSES
from ..utils.serialization import (
    reject_unsupported_input_kind,
    warn_on_metadata_schema_version,
)
from .base import (
    classify_eval_kwargs,
    BaseBackend,
    _read_metadata_imgsz,
    _read_pose_metadata,
    _read_runtime_metadata,
)

logger = logging.getLogger(__name__)


def _graph_float_dtype(module, metadata: dict) -> torch.dtype | None:
    """Float dtype the traced graph was built with (its weights' dtype)."""
    for tensor in itertools.chain(module.parameters(), module.buffers()):
        if tensor.is_floating_point():
            return tensor.dtype
    if str(metadata.get("precision", "")).lower() == "fp16":
        return torch.float16
    return None


def _output_to_numpy(output: torch.Tensor) -> np.ndarray:
    output = output.detach()
    if output.dtype in (torch.float16, torch.bfloat16):
        output = output.float()
    return output.cpu().numpy()


class TorchScriptBackend(BaseBackend):
    """TorchScript inference backend for LibreYOLO models."""

    def __init__(
        self,
        model_path: str,
        nb_classes: int | None = None,
        device: str = "auto",
        task: str | None = None,
    ):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"TorchScript model not found: {model_path}")

        if device == "auto":
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = device

        map_location = torch.device(resolved_device)
        extra_files = {"libreyolo_metadata.json": ""}
        self.model = torch.jit.load(
            model_path, map_location=map_location, _extra_files=extra_files
        )
        self.model.eval()

        metadata = {}
        raw_meta = extra_files.get("libreyolo_metadata.json", "")
        if raw_meta:
            metadata = json.loads(raw_meta)
        # export(half=True) traces a float16 graph; preprocessing yields
        # float32, so inputs are cast to the graph's float dtype.
        self._input_float_dtype = _graph_float_dtype(self.model, metadata)
        warn_on_metadata_schema_version(
            metadata,
            artifact=f"TorchScript metadata for {model_path}",
            logger=logger,
        )
        reject_unsupported_input_kind(
            metadata, artifact=f"TorchScript metadata for {model_path}"
        )

        input_size = 640
        model_family = metadata.get("model_family")
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
        metadata_imgsz = _read_metadata_imgsz(
            metadata,
            model_family,
            artifact=f"TorchScript metadata for {model_path}",
        )
        if metadata_imgsz is not None:
            input_size = metadata_imgsz
        pose_metadata = _read_pose_metadata(metadata)
        runtime_metadata = _read_runtime_metadata(metadata)

        if nb_classes is not None:
            resolved_nb_classes = nb_classes
        elif "nb_classes" in metadata or "nc" in metadata:
            resolved_nb_classes = int(metadata.get("nb_classes", metadata.get("nc")))
        else:
            resolved_nb_classes = 80

        if "names" in metadata:
            names_raw = metadata["names"]
            if isinstance(names_raw, str):
                names_raw = json.loads(names_raw)
            names = {int(k): v for k, v in names_raw.items()}
        elif resolved_nb_classes == 80:
            names = {i: n for i, n in enumerate(COCO_CLASSES)}
        else:
            names = self.build_names(resolved_nb_classes)

        super().__init__(
            model_path=model_path,
            nb_classes=resolved_nb_classes,
            device=resolved_device,
            imgsz=input_size,
            model_family=model_family,
            names=names,
            model_size=model_size,
            task=resolved_task,
            supported_tasks=supported_tasks,
            default_task=default_task,
            **classify_eval_kwargs(runtime_metadata),
            letterbox_pad=runtime_metadata.get("letterbox_pad"),
            num_bins=runtime_metadata.get("num_bins"),
            bin_width_deg=runtime_metadata.get("bin_width_deg"),
            offset_deg=runtime_metadata.get("offset_deg"),
            **pose_metadata,
        )

    def _run_inference(self, blob: np.ndarray) -> list:
        tensor = torch.from_numpy(blob).to(self.device)
        input_dtype = getattr(self, "_input_float_dtype", None)
        if input_dtype is not None and tensor.is_floating_point():
            tensor = tensor.to(input_dtype)
        with torch.no_grad():
            outputs = self.model(tensor)

        if isinstance(outputs, torch.Tensor):
            return [_output_to_numpy(outputs)]

        if isinstance(outputs, (tuple, list)):
            out_list = []
            for out in outputs:
                if isinstance(out, torch.Tensor):
                    out_list.append(_output_to_numpy(out))
                else:
                    raise TypeError(
                        f"Unsupported TorchScript output element type: {type(out)!r}"
                    )
            return out_list

        raise TypeError(f"Unsupported TorchScript output type: {type(outputs)!r}")
