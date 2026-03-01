"""
TensorRT inference backend for LIBREYOLO.

Provides GPU-accelerated inference using TensorRT engines exported from LibreYOLO models.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .base import BaseBackend


class TensorRTBackend(BaseBackend):
    """
    TensorRT inference backend for LIBREYOLO models.

    Provides the same API as LIBREYOLOX/LIBREYOLO9 but uses TensorRT
    for GPU-accelerated inference.

    Args:
        engine_path: Path to the TensorRT engine file (.engine).
            If a JSON sidecar file exists at ``<engine_path>.json``, model
            metadata (nb_classes, class names, model family, etc.) is loaded
            from it automatically.
        nb_classes: Number of classes. When ``None`` (default), uses the value
            from the sidecar file if available, otherwise defaults to 80.
        device: Device for inference. Must be "cuda" or "auto" (TensorRT requires GPU).

    Example:
        >>> model = TensorRTBackend("model.engine")
        >>> result = model("image.jpg", save=True)
        >>> print(result.boxes.xyxy)
    """

    def __init__(self, engine_path: str, nb_classes: int = None, device: str = "auto"):
        try:
            import tensorrt as trt
        except ImportError as e:
            raise ImportError(
                "TensorRT inference requires tensorrt. "
                "Install with: pip install tensorrt"
            ) from e

        if not torch.cuda.is_available():
            raise RuntimeError(
                "TensorRT requires CUDA. No CUDA-capable GPU detected."
            )

        if not Path(engine_path).exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        # Load metadata sidecar if available
        sidecar_path = Path(str(engine_path) + ".json")
        self._metadata = {}
        if sidecar_path.exists():
            with open(sidecar_path) as f:
                self._metadata = json.load(f)

        # Priority: explicit arg > sidecar > default (80)
        resolved_nb_classes = (
            nb_classes
            if nb_classes is not None
            else self._metadata.get("nb_classes", 80)
        )
        model_family = self._metadata.get("model_family")
        self._sidecar_size = self._metadata.get("model_size")

        # Build names dict: sidecar names when available, else COCO / generic
        sidecar_names = self._metadata.get("names")
        if sidecar_names is not None and nb_classes is None:
            names: Dict[int, str] = {int(k): v for k, v in sidecar_names.items()}
        else:
            names = self.build_names(resolved_nb_classes)

        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            engine_data = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        # Get input/output tensor info
        self.input_name = None
        self.output_names: List[str] = []
        self.input_shape = None
        self.output_shapes: Dict[str, Tuple] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            mode = self.engine.get_tensor_mode(name)

            if mode == trt.TensorIOMode.INPUT:
                self.input_name = name
                self.input_shape = tuple(shape)
            else:
                self.output_names.append(name)
                self.output_shapes[name] = tuple(shape)

        if self.input_name is None:
            raise RuntimeError("No input tensor found in TensorRT engine")

        # Extract input size from shape (batch, channels, height, width)
        imgsz = self.input_shape[2]  # Assuming square input

        # Detect batch capability: -1 means dynamic batch
        self._dynamic_batch = self.input_shape[0] == -1
        self._max_batch = 1 if self._dynamic_batch else self.input_shape[0]

        # Allocate CUDA memory for inputs/outputs (batch=1 initially)
        self._allocate_buffers()

        # Detect model family from output shapes if not in sidecar
        if model_family is None:
            model_family = self._detect_model_family()

        super().__init__(
            model_path=engine_path,
            nb_classes=resolved_nb_classes,
            device="cuda",
            imgsz=imgsz,
            model_family=model_family,
            names=names,
        )

    # ------------------------------------------------------------------
    # TensorRT-specific internals
    # ------------------------------------------------------------------

    def _allocate_buffers(self, batch_size: int = 1):
        """Allocate CUDA memory for input and output tensors."""
        self.inputs = {}
        self.outputs = {}
        self.bindings = []
        self.stream = torch.cuda.Stream()
        self._current_batch = batch_size

        def _resolve_shape(shape):
            return tuple(batch_size if d == -1 else d for d in shape)

        resolved_input = _resolve_shape(self.input_shape)
        input_size = int(np.prod(resolved_input))
        self.inputs[self.input_name] = torch.zeros(
            input_size, dtype=torch.float32, device="cuda"
        )

        for name in self.output_names:
            shape = _resolve_shape(self.output_shapes[name])
            size = int(np.prod(shape))
            self.outputs[name] = torch.zeros(
                size, dtype=torch.float32, device="cuda"
            )

    def _detect_model_family(self) -> Optional[str]:
        """Detect model family from output shapes when sidecar metadata is absent."""
        if "output" in self.output_shapes:
            shape = self.output_shapes["output"]
            if len(shape) == 3 and shape[2] == 4 and len(self.output_names) == 2:
                return "rfdetr"
            elif len(shape) == 3:
                return "yolo9"
            elif len(shape) == 4:
                return "yolox"
        else:
            yolox_outputs = [n for n in self.output_names if n.startswith("cat_")]
            if yolox_outputs:
                return "yolox"
        return None

    def _infer(self, input_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Run TensorRT inference.

        Args:
            input_array: Input tensor of shape (B, C, H, W) or (C, H, W).

        Returns:
            Dict mapping output tensor names to numpy arrays.
        """
        input_array = np.ascontiguousarray(input_array, dtype=np.float32)

        if input_array.ndim == 3:
            input_array = input_array[np.newaxis]
        actual_batch = input_array.shape[0]

        if actual_batch != self._current_batch:
            self._allocate_buffers(actual_batch)

        if self._dynamic_batch:
            _, c, h, w = self.input_shape
            self.context.set_input_shape(
                self.input_name, (actual_batch, c, h, w)
            )

        input_tensor = torch.from_numpy(input_array).cuda().flatten()
        self.inputs[self.input_name].copy_(input_tensor)

        self.context.set_tensor_address(
            self.input_name, self.inputs[self.input_name].data_ptr()
        )
        for name in self.output_names:
            self.context.set_tensor_address(
                name, self.outputs[name].data_ptr()
            )

        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        results = {}
        for name in self.output_names:
            shape = tuple(
                actual_batch if d == -1 else d
                for d in self.output_shapes[name]
            )
            output = self.outputs[name].cpu().numpy().reshape(shape)
            results[name] = output

        return results

    # ------------------------------------------------------------------
    # BaseBackend interface
    # ------------------------------------------------------------------

    def _run_inference(self, blob: np.ndarray) -> list:
        """Run TensorRT inference and return outputs as a list."""
        outputs_dict = self._infer(blob)
        return [outputs_dict[name] for name in self.output_names]

    def _process_in_batches(
        self,
        image_paths: List,
        batch: int = 1,
        save: bool = False,
        output_path: str = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
    ) -> list:
        """Process multiple images with GPU batching when possible.

        When batch > 1 and the engine supports it (dynamic batch or static
        batch >= requested), images are preprocessed, stacked, and inferred
        together in a single forward pass. Otherwise falls back to sequential
        processing.
        """
        can_batch = batch > 1 and (
            self._dynamic_batch or self._max_batch >= batch
        )
        if not can_batch:
            return super()._process_in_batches(
                image_paths,
                batch=batch,
                save=save,
                output_path=output_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                classes=classes,
                max_det=max_det,
                color_format=color_format,
            )

        effective_imgsz = imgsz if imgsz is not None else self.imgsz
        results = []

        for i in range(0, len(image_paths), batch):
            chunk_paths = image_paths[i : i + batch]

            # Preprocess all images in the chunk
            tensors = []
            preprocess_info = []
            for path in chunk_paths:
                preprocess_out = self._preprocess(
                    path, effective_imgsz, color_format
                )
                if len(preprocess_out) == 4:
                    tensor, orig_img, orig_size, ratio = preprocess_out
                else:
                    tensor, orig_img, orig_size = preprocess_out
                    ratio = None
                tensors.append(tensor)
                preprocess_info.append((orig_img, orig_size, ratio, path))

            # Stack into a single (B, C, H, W) tensor and run inference
            batched_input = np.concatenate(
                [t.numpy() for t in tensors], axis=0
            )
            batch_outputs = self._infer(batched_input)

            # Split per-image and postprocess using base class methods
            for idx, (orig_img, orig_size, ratio, path) in enumerate(
                preprocess_info
            ):
                per_image = [
                    batch_outputs[name][idx : idx + 1]
                    for name in self.output_names
                ]

                # Set YOLOX ratio for this specific image
                if ratio is not None:
                    self._yolox_ratio = ratio

                boxes, max_scores, class_ids = self._parse_outputs(
                    per_image, effective_imgsz, orig_size, conf
                )

                orig_w, orig_h = orig_size
                orig_shape = (orig_h, orig_w)
                result = self._build_result(
                    boxes,
                    max_scores,
                    class_ids,
                    orig_shape=orig_shape,
                    image_path=path,
                    iou=iou,
                    classes=classes,
                    max_det=max_det,
                )

                if save:
                    self._save_annotated(result, orig_img, path, output_path)

                results.append(result)

        return results

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @property
    def size(self) -> str:
        """Return model size from sidecar metadata or engine filename."""
        if self._sidecar_size is not None:
            return self._sidecar_size
        stem = Path(self.model_path).stem.lower()
        for s in ["nano", "tiny", "s", "m", "l", "x", "t", "c"]:
            if s in stem:
                return s
        return "unknown"

    def _get_model_name(self) -> str:
        """Return model name for compatibility."""
        if self.model_family == "yolo9":
            return "yolo9"
        elif self.model_family == "yolox":
            return "yolox"
        elif self.model_family == "rfdetr":
            return "rfdetr"
        return "libreyolo"

    def _get_input_size(self) -> int:
        """Return model input size."""
        return self.imgsz
