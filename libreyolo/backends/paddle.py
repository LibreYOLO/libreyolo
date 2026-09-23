"""PaddlePaddle CPU inference backend for LibreYOLO exports."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml

from ..tasks import normalize_supported_tasks, normalize_task, resolve_task
from ..utils.general import COCO_CLASSES
from ..utils.serialization import warn_on_metadata_schema_version
from .base import (
    classify_eval_kwargs,
    BaseBackend,
    _read_metadata_imgsz,
    _read_pose_metadata,
    _read_runtime_metadata,
)

logger = logging.getLogger(__name__)


class PaddleBackend(BaseBackend):
    """Run a LibreYOLO Paddle inference directory on the CPU."""

    def __init__(
        self,
        model_dir: str | Path,
        nb_classes: int | None = None,
        device: str = "auto",
        task: str | None = None,
    ) -> None:
        path = Path(model_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Paddle model directory not found: {path}")
        model_file = path / "model.pdmodel"
        params_file = path / "model.pdiparams"
        for required in (model_file, params_file):
            if not required.is_file():
                raise FileNotFoundError(f"Paddle model file not found: {required}")
        if str(device or "auto").lower() not in {"auto", "cpu"}:
            raise ValueError(
                "The LibreYOLO Paddle backend currently supports CPU inference "
                f"only; got device={device!r}."
            )

        try:
            import paddle.inference as paddle_infer
        except ImportError as exc:
            raise ImportError(
                "Paddle inference requires paddlepaddle. Install with: "
                "pip install libreyolo[paddle]"
            ) from exc

        metadata_path = path / "metadata.yaml"
        metadata = (
            yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
            if metadata_path.is_file()
            else {}
        )
        warn_on_metadata_schema_version(
            metadata,
            artifact=f"Paddle metadata for {path}",
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
        if isinstance(names_raw, dict) and nb_classes is None:
            names = {int(key): value for key, value in names_raw.items()}
        elif resolved_nc == 80:
            names = {index: name for index, name in enumerate(COCO_CLASSES)}
        else:
            names = self.build_names(resolved_nc)
        imgsz = (
            _read_metadata_imgsz(
                metadata,
                family,
                artifact=f"Paddle metadata for {path}",
            )
            or 640
        )

        config = paddle_infer.Config(str(model_file), str(params_file))
        config.disable_gpu()
        # The Paddle 2.6 CPU fusion pipeline can crash while optimizing the
        # large gather/scatter graphs emitted for deformable attention. Keep
        # the portable, unfused static graph used by parity validation.
        config.disable_mkldnn()
        config.switch_ir_optim(False)
        config.enable_memory_optim()
        self.predictor = paddle_infer.create_predictor(config)
        self.input_names = tuple(self.predictor.get_input_names())
        self.output_names = tuple(self.predictor.get_output_names())
        if len(self.input_names) != 1:
            raise ValueError(
                f"Paddle backend expects one image input, got {len(self.input_names)}."
            )

        runtime_metadata = _read_runtime_metadata(metadata)
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
            **classify_eval_kwargs(runtime_metadata),
            num_bins=runtime_metadata.get("num_bins"),
            bin_width_deg=runtime_metadata.get("bin_width_deg"),
            offset_deg=runtime_metadata.get("offset_deg"),
            **_read_pose_metadata(metadata),
        )

    def _run_inference(self, blob: np.ndarray) -> list[np.ndarray]:
        value = np.ascontiguousarray(blob, dtype=np.float32)
        input_handle = self.predictor.get_input_handle(self.input_names[0])
        input_handle.reshape(value.shape)
        input_handle.copy_from_cpu(value)
        ran = self.predictor.run()
        if ran is False:
            raise RuntimeError("Paddle inference execution failed.")
        return [
            self.predictor.get_output_handle(name).copy_to_cpu()
            for name in self.output_names
        ]


__all__ = ["PaddleBackend"]
