"""ExecuTorch inference backend for LibreYOLO."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from ..tasks import normalize_supported_tasks, normalize_task, resolve_task
from ..utils.general import COCO_CLASSES
from ..utils.serialization import warn_on_metadata_schema_version
from .base import BaseBackend, _read_metadata_imgsz, _read_pose_metadata, classify_eval_kwargs

logger = logging.getLogger(__name__)


class ExecuTorchBackend(BaseBackend):
    """Run a fixed-shape ``.pte`` program through the ExecuTorch Python runtime."""

    fixed_input_shape = True

    def __init__(
        self,
        model_path: str,
        nb_classes: int | None = None,
        device: str = "auto",
        task: str | None = None,
    ):
        program_path = Path(model_path)
        if not program_path.exists():
            raise FileNotFoundError(f"ExecuTorch program not found: {model_path}")

        try:
            from executorch.runtime import Runtime
        except (ImportError, OSError) as exc:
            raise ImportError(
                "ExecuTorch inference requires the optional runtime. "
                "Install it with: pip install libreyolo[executorch]"
            ) from exc

        if str(device).lower() not in {"auto", "cpu"}:
            logger.warning(
                "ExecuTorch XNNPACK programs run on CPU; ignoring device=%r.", device
            )

        sidecar_path = Path(f"{program_path}.json")
        if not sidecar_path.exists():
            raise FileNotFoundError(
                f"ExecuTorch metadata sidecar not found: {sidecar_path}"
            )
        try:
            metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid ExecuTorch metadata sidecar: {sidecar_path}"
            ) from exc
        if not isinstance(metadata, dict):
            raise TypeError(
                f"ExecuTorch metadata sidecar must contain a JSON object: {sidecar_path}"
            )
        warn_on_metadata_schema_version(
            metadata,
            artifact=f"ExecuTorch metadata for {model_path}",
            logger=logger,
        )

        delegate = str(metadata.get("executorch_delegate", "")).lower()
        if delegate != "xnnpack":
            raise ValueError(
                "This LibreYOLO version only supports ExecuTorch programs "
                f"exported for XNNPACK, got {delegate or 'missing delegate metadata'!r}."
            )
        dynamic = metadata.get("dynamic")
        if dynamic not in {False, "false", "False"}:
            raise ValueError(
                "ExecuTorch v1 requires metadata dynamic=false, "
                f"got {dynamic!r}."
            )
        precision = str(metadata.get("precision", "")).lower()
        if precision != "fp32":
            raise ValueError(
                "ExecuTorch v1 requires metadata precision='fp32', "
                f"got {precision or 'missing precision metadata'!r}."
            )

        runtime = Runtime.get()
        if not runtime.backend_registry.is_available("XnnpackBackend"):
            raise RuntimeError(
                "The installed ExecuTorch runtime does not provide "
                "XnnpackBackend, but this program requires it."
            )
        # Loading from bytes avoids keeping the .pte file memory-mapped and
        # locked for the backend lifetime on Windows.
        self.program = runtime.load_program(program_path.read_bytes())
        if "forward" not in self.program.method_names:
            raise ValueError("ExecuTorch program does not contain a 'forward' method.")
        self.method = self.program.load_method("forward")

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
        imgsz = _read_metadata_imgsz(
            metadata,
            model_family,
            artifact=f"ExecuTorch metadata for {model_path}",
        )
        if imgsz is None:
            raise ValueError("ExecuTorch metadata must declare a fixed input size.")

        if nb_classes is not None:
            resolved_nb_classes = int(nb_classes)
        elif "nc" in metadata or "nb_classes" in metadata:
            resolved_nb_classes = int(metadata.get("nc", metadata.get("nb_classes")))
        else:
            resolved_nb_classes = 80

        names_raw = metadata.get("names")
        if isinstance(names_raw, str):
            names_raw = json.loads(names_raw)
        if isinstance(names_raw, dict):
            names = {int(key): str(value) for key, value in names_raw.items()}
        elif resolved_nb_classes == 80:
            names = {i: name for i, name in enumerate(COCO_CLASSES)}
        else:
            names = self.build_names(resolved_nb_classes)

        super().__init__(
            model_path=str(program_path),
            nb_classes=resolved_nb_classes,
            device="cpu",
            imgsz=imgsz,
            model_family=model_family,
            names=names,
            model_size=model_size,
            task=resolved_task,
            supported_tasks=supported_tasks,
            default_task=default_task,
            **classify_eval_kwargs(metadata),
            **_read_pose_metadata(metadata),
        )

    def _run_inference(self, blob: np.ndarray) -> list[np.ndarray]:
        if blob.dtype != np.float32:
            blob = blob.astype(np.float32, copy=False)
        tensor = torch.from_numpy(np.ascontiguousarray(blob)).cpu()
        with torch.no_grad():
            outputs = self.method.execute((tensor,))
        if not isinstance(outputs, (list, tuple)):
            raise TypeError(
                "ExecuTorch forward must return a sequence of tensors, "
                f"got {type(outputs)!r}."
            )

        result = []
        for output in outputs:
            if not isinstance(output, torch.Tensor):
                raise TypeError(
                    "ExecuTorch forward returned a non-tensor output: "
                    f"{type(output)!r}."
                )
            result.append(np.ascontiguousarray(output.detach().cpu().numpy()))
        return result
