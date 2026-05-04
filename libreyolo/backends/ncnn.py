"""ncnn inference backend for LibreYOLO."""

from pathlib import Path
from typing import Dict

import numpy as np

from ..tasks import normalize_supported_tasks, normalize_task, resolve_task
from .base import BaseBackend


class NcnnBackend(BaseBackend):
    """ncnn inference backend for LibreYOLO models.

    Args:
        model_dir: Path to the ncnn model directory (containing model.ncnn.param,
            model.ncnn.bin, and optionally metadata.yaml).
        nb_classes: Number of classes (default: auto-detected from metadata, fallback 80).
        device: Device for inference. "auto" (default) uses CPU. "gpu"/"cuda" uses
            Vulkan GPU if available.

    Example:
        >>> model = NcnnBackend("exported_model_dir/")
        >>> result = model("image.jpg", save=True)
        >>> print(result.boxes.xyxy)
    """

    def __init__(
        self,
        model_dir: str | Path,
        nb_classes: int | None = None,
        device: str = "auto",
        task: str | None = None,
    ):
        try:
            import ncnn as _ncnn
        except ImportError as e:
            raise ImportError(
                "ncnn inference requires the ncnn package. "
                "Install with: pip install ncnn"
            ) from e

        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"ncnn model directory not found: {model_dir}")

        param_path = model_dir / "model.ncnn.param"
        bin_path = model_dir / "model.ncnn.bin"
        if not param_path.exists():
            raise FileNotFoundError(f"model.ncnn.param not found in {model_dir}")
        if not bin_path.exists():
            raise FileNotFoundError(f"model.ncnn.bin not found in {model_dir}")

        explicit_task = task
        model_family = None
        model_size = None
        task = "detect"
        default_task = "detect"
        supported_tasks = ("detect",)
        imgsz = 640
        resolved_nb_classes = nb_classes if nb_classes is not None else 80
        names = self.build_names(resolved_nb_classes)

        metadata_path = model_dir / "metadata.yaml"
        if metadata_path.exists():
            (
                model_family,
                model_size,
                metadata_task,
                supported_tasks,
                default_task,
                imgsz,
                resolved_nb_classes,
                names,
            ) = self._read_metadata(metadata_path, nb_classes)
            task = resolve_task(
                explicit_task=explicit_task,
                checkpoint_task=metadata_task,
                default_task=default_task,
                supported_tasks=supported_tasks,
            )
        else:
            task = resolve_task(
                explicit_task=explicit_task,
                default_task=default_task,
                supported_tasks=supported_tasks,
            )

        # Map device strings
        device_lower = device.lower() if device else "auto"
        if device_lower in ("auto", "cpu"):
            resolved_device = "cpu"
            use_vulkan = False
        elif device_lower in ("gpu", "cuda"):
            resolved_device = "gpu"
            use_vulkan = True
        else:
            resolved_device = device_lower
            use_vulkan = False

        self.net = _ncnn.Net()
        if use_vulkan and hasattr(_ncnn, "build_with_gpu") and _ncnn.build_with_gpu:
            self.net.opt.use_vulkan_compute = True
        self.net.load_param(str(param_path))
        self.net.load_model(str(bin_path))

        input_names_fn = getattr(self.net, "input_names", None)
        output_names_fn = getattr(self.net, "output_names", None)
        if callable(input_names_fn) and callable(output_names_fn):
            self._input_names = list(input_names_fn())
            self._output_names = list(output_names_fn())
        else:
            self._input_names, self._output_names = self._discover_blob_names(
                param_path
            )

        super().__init__(
            model_path=str(model_dir),
            nb_classes=resolved_nb_classes,
            device=resolved_device,
            imgsz=imgsz,
            model_family=model_family,
            names=names,
            model_size=model_size,
            task=task,
            supported_tasks=supported_tasks,
            default_task=default_task,
        )

    @staticmethod
    def _discover_blob_names(param_path: Path):
        """Read the .param file to discover input and output blob names.

        Falls back to 'in0'/'out0' convention if parsing fails.
        """
        input_names = []
        output_names = []
        try:
            with open(param_path) as f:
                lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                layer_type = parts[0]
                if layer_type == "Input" and len(parts) >= 4:
                    input_names.append(parts[-1])
        except Exception:
            pass

        if not input_names:
            input_names = ["in0"]
        if not output_names:
            output_names = ["out0"]
        return input_names, output_names

    @staticmethod
    def _read_metadata(metadata_path: Path, nb_classes_override: int | None = None):
        """Read metadata from metadata.yaml file.

        Returns:
            Tuple of (model_family, model_size, task, supported_tasks, default_task, imgsz, nb_classes, names).
        """
        import yaml

        with open(metadata_path) as f:
            meta = yaml.safe_load(f) or {}

        model_family = meta.get("model_family")
        model_size = meta.get("model_size")
        default_task = normalize_task(meta.get("default_task"), default="detect")
        task = normalize_task(meta.get("task"), default=default_task)
        supported_tasks = normalize_supported_tasks(meta.get("supported_tasks", (task,)))
        imgsz = int(meta["imgsz"]) if "imgsz" in meta else 640

        if nb_classes_override is not None:
            nb_classes = nb_classes_override
        elif "nb_classes" in meta:
            nb_classes = int(meta["nb_classes"])
        else:
            nb_classes = 80

        if "names" in meta and nb_classes_override is None:
            names: Dict[int, str] = {int(k): v for k, v in meta["names"].items()}
        else:
            names = BaseBackend.build_names(nb_classes)

        return model_family, model_size, task, supported_tasks, default_task, imgsz, nb_classes, names

    def _run_inference(self, blob: np.ndarray) -> list:
        """Run ncnn inference."""
        import ncnn as _ncnn

        # ncnn.Mat expects a C-contiguous (C, H, W) float32 array.
        # blob[0] is a view with non-standard strides (from removing
        # the batch dim of a permuted tensor), so we must make a
        # contiguous copy; otherwise ncnn reads scrambled channel data.
        input_data = np.ascontiguousarray(blob[0])
        mat_in = _ncnn.Mat(input_data)

        ex = self.net.create_extractor()
        ex.input(self._input_names[0], mat_in)

        all_outputs = []
        for out_name in self._output_names:
            ret, mat_out = ex.extract(out_name)
            if ret != 0:
                for fallback in ("out0", "output", "output0"):
                    ret, mat_out = ex.extract(fallback)
                    if ret == 0:
                        break
                if ret != 0:
                    raise RuntimeError(
                        f"Failed to extract output '{out_name}' from ncnn model"
                    )
            all_outputs.append(np.array(mat_out).reshape(1, *np.array(mat_out).shape))

        # YOLO-NAS exports two outputs (scores, boxes) from pnnx as out1/out0.
        # Reorder them to [boxes, scores] so the shared backend parser can stay
        # aligned with the ONNX/TorchScript/OpenVINO conventions.
        if self.model_family == "yolonas" and len(all_outputs) == 2:
            first, second = all_outputs
            if first.shape[-1] != 4 and second.shape[-1] == 4:
                all_outputs = [second, first]

        return all_outputs
