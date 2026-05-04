"""TorchScript inference backend for LibreYOLO."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..tasks import normalize_supported_tasks, normalize_task, resolve_task
from ..utils.general import COCO_CLASSES
from .base import BaseBackend


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

        input_size = 640
        model_family = metadata.get("model_family")
        model_size = metadata.get("model_size")
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
        if "imgsz" in metadata:
            input_size = int(metadata["imgsz"])

        if nb_classes is not None:
            resolved_nb_classes = nb_classes
        elif "nb_classes" in metadata:
            resolved_nb_classes = int(metadata["nb_classes"])
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
        )

    def _run_inference(self, blob: np.ndarray) -> list:
        tensor = torch.from_numpy(blob).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)

        if isinstance(outputs, torch.Tensor):
            return [outputs.detach().cpu().numpy()]

        if isinstance(outputs, (tuple, list)):
            out_list = []
            for out in outputs:
                if isinstance(out, torch.Tensor):
                    out_list.append(out.detach().cpu().numpy())
                else:
                    raise TypeError(
                        f"Unsupported TorchScript output element type: {type(out)!r}"
                    )
            return out_list

        raise TypeError(f"Unsupported TorchScript output type: {type(outputs)!r}")
