"""
TensorRT inference backend for LIBREYOLO.

Provides GPU-accelerated inference using TensorRT engines exported from LibreYOLO models.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

from .utils import preprocess_image as preprocess_yolov9, draw_boxes, get_safe_stem, COCO_CLASSES, nms
from .image_loader import ImageLoader
from .results import Boxes, Results

# Import YOLOX-specific preprocessing (BGR, 0-255 range)
from ..yolox.utils import preprocess_image as preprocess_yolox


class LIBREYOLOTensorRT:
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
        >>> model = LIBREYOLOTensorRT("model.engine")
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

        self.model_path = engine_path
        self.device = "cuda"

        # Load metadata sidecar if available
        sidecar_path = Path(str(engine_path) + ".json")
        self._metadata = {}
        if sidecar_path.exists():
            with open(sidecar_path) as f:
                self._metadata = json.load(f)

        # Priority: explicit arg > sidecar > default (80)
        self.nb_classes = nb_classes if nb_classes is not None else self._metadata.get("nb_classes", 80)
        self.model_family = self._metadata.get("model_family")
        self._sidecar_size = self._metadata.get("model_size")

        # Build names dict: sidecar names when available, else COCO / generic
        sidecar_names = self._metadata.get("names")
        if sidecar_names is not None and nb_classes is None:
            self.names: Dict[int, str] = {int(k): v for k, v in sidecar_names.items()}
        elif self.nb_classes == 80:
            self.names: Dict[int, str] = {i: n for i, n in enumerate(COCO_CLASSES)}
        else:
            self.names: Dict[int, str] = {i: f"class_{i}" for i in range(self.nb_classes)}

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
        self.output_names = []
        self.input_shape = None
        self.output_shapes = {}

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
        self.input_size = self.input_shape[2]  # Assuming square input

        # Detect batch capability: -1 means dynamic batch
        self._dynamic_batch = self.input_shape[0] == -1
        self._max_batch = 1 if self._dynamic_batch else self.input_shape[0]

        # Allocate CUDA memory for inputs/outputs (batch=1 initially)
        self._allocate_buffers()

        # Determine model type from output shapes
        self._detect_model_type()

    def _allocate_buffers(self, batch_size: int = 1):
        """Allocate CUDA memory for input and output tensors."""
        self.inputs = {}
        self.outputs = {}
        self.bindings = []
        self.stream = torch.cuda.Stream()
        self._current_batch = batch_size

        # For dynamic batch, replace -1 with the requested batch size
        def _resolve_shape(shape):
            return tuple(batch_size if d == -1 else d for d in shape)

        # Allocate input buffer
        resolved_input = _resolve_shape(self.input_shape)
        input_size = int(np.prod(resolved_input))
        self.inputs[self.input_name] = torch.zeros(
            input_size, dtype=torch.float32, device="cuda"
        )

        # Allocate output buffers
        for name in self.output_names:
            shape = _resolve_shape(self.output_shapes[name])
            size = int(np.prod(shape))
            self.outputs[name] = torch.zeros(size, dtype=torch.float32, device="cuda")

    def _detect_model_type(self):
        """Detect model type (YOLOX vs YOLOv9) from sidecar metadata or output shapes."""
        # Use sidecar metadata when available
        if self.model_family is not None:
            family = self.model_family.upper()
            if "YOLOX" in family:
                self.model_type = "yolox"
                self.main_output = None
                return
            if "9" in family or "YOLO9" in family:
                self.model_type = "yolov9"
                self.main_output = "output" if "output" in self.output_shapes else (
                    self.output_names[0] if self.output_names else None
                )
                return

        # Fall back to shape-based heuristic
        if "output" in self.output_shapes:
            shape = self.output_shapes["output"]
            if len(shape) == 3:
                self.model_type = "yolov9"
                self.main_output = "output"
            elif len(shape) == 4:
                self.model_type = "yolox"
                self.main_output = None
        else:
            yolox_outputs = [n for n in self.output_names if n.startswith("cat_")]
            if yolox_outputs:
                self.model_type = "yolox"
                self.main_output = None
            else:
                self.model_type = "unknown"
                self.main_output = self.output_names[0] if self.output_names else None

    def __call__(
        self,
        source: Union[str, Path, Image.Image, np.ndarray] = None,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        save: bool = False,
        batch: int = 1,
        output_path: str = None,
        color_format: str = "auto",
    ) -> Union[Results, List[Results]]:
        """
        Run inference on an image or directory of images.

        Args:
            source: Input image path, PIL Image, or numpy array.
            conf: Confidence threshold.
            iou: IoU threshold for NMS.
            imgsz: Input size override (None = use engine's input size).
            classes: Filter to specific class IDs.
            max_det: Maximum detections per image.
            save: If True, saves annotated image to disk.
            batch: Batch size for directory processing.
            output_path: Optional path to save the annotated image.
            color_format: Color format hint.

        Returns:
            Results instance or list of Results.
        """
        if isinstance(source, (str, Path)) and Path(source).is_dir():
            image_paths = ImageLoader.collect_images(source)
            if not image_paths:
                return []
            return self._process_in_batches(
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

        return self._predict_single(
            source, save=save, output_path=output_path,
            conf=conf, iou=iou, imgsz=imgsz,
            classes=classes, max_det=max_det,
            color_format=color_format,
        )

    def _process_in_batches(
        self,
        image_paths: List[Path],
        batch: int = 1,
        save: bool = False,
        output_path: str = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
    ) -> List[Results]:
        """Process multiple images in batches.

        When batch > 1 and the engine supports it (dynamic batch or static
        batch >= requested), images are preprocessed, stacked, and inferred
        together in a single forward pass. Otherwise falls back to sequential
        processing.
        """
        # Fall back to sequential when batching isn't useful or possible
        can_batch = batch > 1 and (self._dynamic_batch or self._max_batch >= batch)
        if not can_batch:
            results = []
            for path in image_paths:
                results.append(
                    self._predict_single(
                        path, save=save, output_path=output_path,
                        conf=conf, iou=iou, imgsz=imgsz,
                        classes=classes, max_det=max_det,
                        color_format=color_format,
                    )
                )
            return results

        effective_imgsz = imgsz if imgsz is not None else self.input_size
        results = []

        # Process in chunks of `batch`
        for i in range(0, len(image_paths), batch):
            chunk_paths = image_paths[i : i + batch]

            # Preprocess all images in the chunk
            tensors = []
            preprocess_info = []  # (original_img, original_size, ratio, path)
            for path in chunk_paths:
                if self.model_type == "yolox":
                    tensor, orig_img, orig_size, ratio = preprocess_yolox(
                        path, input_size=effective_imgsz, color_format=color_format
                    )
                else:
                    tensor, orig_img, orig_size = preprocess_yolov9(
                        path, input_size=effective_imgsz, color_format=color_format
                    )
                    ratio = None
                tensors.append(tensor)
                preprocess_info.append((orig_img, orig_size, ratio, path))

            # Stack into a single (B, C, H, W) tensor and run inference
            batched_input = np.concatenate([t.numpy() for t in tensors], axis=0)
            batch_outputs = self._infer(batched_input)

            # Split per-image and postprocess
            for idx, (orig_img, orig_size, ratio, path) in enumerate(preprocess_info):
                per_image_outputs = {}
                for name, arr in batch_outputs.items():
                    per_image_outputs[name] = arr[idx : idx + 1]

                result = self._postprocess_single(
                    per_image_outputs,
                    original_img=orig_img,
                    original_size=orig_size,
                    ratio=ratio,
                    image_path=path,
                    effective_imgsz=effective_imgsz,
                    conf=conf,
                    iou=iou,
                    classes=classes,
                    max_det=max_det,
                    save=save,
                    output_path=output_path,
                )
                results.append(result)

        return results

    def _predict_single(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        save: bool = False,
        output_path: str = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        max_det: int = 300,
        color_format: str = "auto",
    ) -> Results:
        """Run inference on a single image."""
        image_path = image if isinstance(image, (str, Path)) else None
        effective_imgsz = imgsz if imgsz is not None else self.input_size

        if self.model_type == "yolox":
            input_tensor, original_img, original_size, ratio = preprocess_yolox(
                image, input_size=effective_imgsz, color_format=color_format
            )
        else:
            input_tensor, original_img, original_size = preprocess_yolov9(
                image, input_size=effective_imgsz, color_format=color_format
            )
            ratio = None

        outputs = self._infer(input_tensor.numpy())

        return self._postprocess_single(
            outputs,
            original_img=original_img,
            original_size=original_size,
            ratio=ratio,
            image_path=image_path,
            effective_imgsz=effective_imgsz,
            conf=conf,
            iou=iou,
            classes=classes,
            max_det=max_det,
            save=save,
            output_path=output_path,
        )

    def _postprocess_single(
        self,
        outputs: Dict[str, np.ndarray],
        *,
        original_img,
        original_size: Tuple[int, int],
        ratio,
        image_path,
        effective_imgsz: int,
        conf: float,
        iou: float,
        classes: Optional[List[int]],
        max_det: int,
        save: bool,
        output_path: Optional[str],
    ) -> Results:
        """Decode raw model outputs into a Results object for one image."""
        if self.model_type == "yolov9":
            boxes, scores, class_ids = self._postprocess_yolov9(outputs, conf)
        elif self.model_type == "yolox":
            boxes, scores, class_ids = self._postprocess_yolox(outputs, conf)
        else:
            boxes, scores, class_ids = self._postprocess_generic(outputs, conf)

        orig_w, orig_h = original_size
        orig_shape = (orig_h, orig_w)

        if len(boxes) == 0:
            result = Results(
                boxes=Boxes(
                    torch.zeros((0, 4), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.float32),
                ),
                orig_shape=orig_shape,
                path=str(image_path) if image_path else None,
                names=self.names,
            )
        else:
            if self.model_type == "yolox" and ratio is not None:
                boxes = boxes / ratio
            else:
                scale_x = orig_w / effective_imgsz
                scale_y = orig_h / effective_imgsz
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y

            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            scores_t = torch.tensor(scores, dtype=torch.float32)
            class_ids_t = torch.tensor(class_ids, dtype=torch.int64)

            keep = nms(boxes_t, scores_t, iou)
            boxes_t = boxes_t[keep]
            scores_t = scores_t[keep]
            class_ids_t = class_ids_t[keep]

            if len(boxes_t) > max_det:
                top_indices = torch.argsort(scores_t, descending=True)[:max_det]
                boxes_t = boxes_t[top_indices]
                scores_t = scores_t[top_indices]
                class_ids_t = class_ids_t[top_indices]

            if classes is not None and len(boxes_t) > 0:
                cls_mask = torch.zeros(len(class_ids_t), dtype=torch.bool)
                for cid in classes:
                    cls_mask |= class_ids_t == cid
                boxes_t = boxes_t[cls_mask]
                scores_t = scores_t[cls_mask]
                class_ids_t = class_ids_t[cls_mask]

            result = Results(
                boxes=Boxes(boxes_t, scores_t, class_ids_t.float()),
                orig_shape=orig_shape,
                path=str(image_path) if image_path else None,
                names=self.names,
            )

        if save:
            if len(result) > 0:
                annotated_img = draw_boxes(
                    original_img,
                    result.boxes.xyxy.tolist(),
                    result.boxes.conf.tolist(),
                    result.boxes.cls.tolist(),
                )
            else:
                annotated_img = original_img

            if output_path:
                final_path = Path(output_path)
                final_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                stem = get_safe_stem(image_path) if image_path else "inference"
                ext = Path(image_path).suffix if image_path else ".jpg"
                model_tag = Path(self.model_path).stem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_dir = Path("runs/detections")
                save_dir.mkdir(parents=True, exist_ok=True)
                final_path = save_dir / f"{stem}_{model_tag}_{timestamp}{ext}"

            annotated_img.save(final_path)
            result.saved_path = str(final_path)

        return result

    def _infer(self, input_array: np.ndarray) -> Dict[str, np.ndarray]:
        """Run TensorRT inference.

        Args:
            input_array: Input tensor of shape (B, C, H, W) or (C, H, W).
        """
        # Ensure input is contiguous and correct dtype
        input_array = np.ascontiguousarray(input_array, dtype=np.float32)

        # Determine actual batch size from input
        if input_array.ndim == 3:
            input_array = input_array[np.newaxis]
        actual_batch = input_array.shape[0]

        # Reallocate buffers if batch size changed
        if actual_batch != self._current_batch:
            self._allocate_buffers(actual_batch)

        # For dynamic batch engines, set the actual input shape
        if self._dynamic_batch:
            _, c, h, w = self.input_shape
            self.context.set_input_shape(self.input_name, (actual_batch, c, h, w))

        # Copy input to GPU
        input_tensor = torch.from_numpy(input_array).cuda().flatten()
        self.inputs[self.input_name].copy_(input_tensor)

        # Set tensor addresses
        self.context.set_tensor_address(
            self.input_name, self.inputs[self.input_name].data_ptr()
        )
        for name in self.output_names:
            self.context.set_tensor_address(name, self.outputs[name].data_ptr())

        # Run inference
        self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()

        # Copy outputs to CPU, resolving -1 dims with actual batch size
        results = {}
        for name in self.output_names:
            shape = tuple(
                actual_batch if d == -1 else d for d in self.output_shapes[name]
            )
            output = self.outputs[name].cpu().numpy().reshape(shape)
            results[name] = output

        return results

    def _postprocess_yolov9(
        self, outputs: Dict[str, np.ndarray], conf_thres: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Postprocess YOLOv9 outputs."""
        # YOLOv9 output: (1, 84, 8400) -> 4 box coords (xyxy) + 80 class scores
        output = outputs.get("output", outputs[self.output_names[0]])

        # Shape: (1, 84, N) where N is number of predictions
        if output.shape[1] == 84:
            # Format: (1, 84, N) -> transpose to (N, 84)
            output = output[0].T  # (N, 84)
        elif output.shape[2] == 84:
            # Format: (1, N, 84)
            output = output[0]  # (N, 84)
        else:
            # Try to handle other formats
            output = output.reshape(-1, output.shape[-1])

        # Extract boxes (already in xyxy format) and class scores
        boxes = output[:, :4]  # xyxy format - no conversion needed
        class_scores = output[:, 4:]

        # Get max score and class per detection
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Apply confidence threshold
        mask = max_scores > conf_thres
        boxes = boxes[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        return boxes, max_scores, class_ids

    def _postprocess_yolox(
        self, outputs: Dict[str, np.ndarray], conf_thres: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Postprocess YOLOX outputs (multi-scale feature maps)."""
        # YOLOX has 3 outputs at different scales
        # Each output: (1, 85, H, W) where 85 = 4 (box) + 1 (obj) + 80 (cls)

        all_boxes = []
        all_scores = []
        all_class_ids = []

        # Strides for different scales
        strides = [8, 16, 32]

        # Sort outputs by spatial size (largest first)
        sorted_outputs = sorted(
            [(name, outputs[name]) for name in self.output_names if name in outputs],
            key=lambda x: x[1].shape[2] * x[1].shape[3],
            reverse=True
        )

        for idx, (name, output) in enumerate(sorted_outputs):
            if len(output.shape) != 4:
                continue

            stride = strides[idx] if idx < len(strides) else strides[-1]
            batch, channels, h, w = output.shape

            # Reshape: (1, 85, H, W) -> (H*W, 85)
            output = output[0].transpose(1, 2, 0).reshape(-1, channels)

            # Extract components
            # YOLOX format: x, y, w, h, obj_conf, class_scores...
            xy = output[:, :2]
            wh = output[:, 2:4]
            obj_conf = output[:, 4:5]
            class_scores = output[:, 5:]

            # Create grid
            yv, xv = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            grid = np.stack([xv.flatten(), yv.flatten()], axis=1)

            # Decode boxes
            xy = (xy + grid) * stride
            wh = np.exp(wh) * stride

            # Convert to xyxy
            x1 = xy[:, 0] - wh[:, 0] / 2
            y1 = xy[:, 1] - wh[:, 1] / 2
            x2 = xy[:, 0] + wh[:, 0] / 2
            y2 = xy[:, 1] + wh[:, 1] / 2
            boxes = np.stack([x1, y1, x2, y2], axis=1)

            # Compute final scores
            scores = (obj_conf * class_scores)
            max_scores = np.max(scores, axis=1)
            class_ids = np.argmax(scores, axis=1)

            # Apply confidence threshold
            mask = max_scores > conf_thres
            all_boxes.append(boxes[mask])
            all_scores.append(max_scores[mask])
            all_class_ids.append(class_ids[mask])

        if all_boxes:
            boxes = np.concatenate(all_boxes, axis=0)
            scores = np.concatenate(all_scores, axis=0)
            class_ids = np.concatenate(all_class_ids, axis=0)
        else:
            boxes = np.zeros((0, 4))
            scores = np.zeros((0,))
            class_ids = np.zeros((0,), dtype=np.int64)

        return boxes, scores, class_ids

    def _postprocess_generic(
        self, outputs: Dict[str, np.ndarray], conf_thres: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generic postprocessing for unknown model types."""
        # Try to find the main output
        if self.main_output and self.main_output in outputs:
            output = outputs[self.main_output]
        else:
            output = outputs[self.output_names[0]]

        # Flatten to 2D if needed
        if len(output.shape) > 2:
            output = output.reshape(-1, output.shape[-1])
        elif len(output.shape) == 1:
            output = output.reshape(1, -1)

        # Assume format: boxes (4) + scores (N classes)
        boxes = output[:, :4]
        scores = output[:, 4:]

        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        mask = max_scores > conf_thres
        return boxes[mask], max_scores[mask], class_ids[mask]

    def predict(self, *args, **kwargs) -> Union[Results, List[Results]]:
        """Alias for __call__ method."""
        return self(*args, **kwargs)

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
        if self.model_type == "yolov9":
            return "LIBREYOLO9"
        elif self.model_type == "yolox":
            return "LIBREYOLOX"
        return "LIBREYOLO"

    def _get_input_size(self) -> int:
        """Return model input size."""
        return self.input_size
