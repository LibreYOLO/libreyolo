"""
ncnn inference backend for LIBREYOLO.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from .utils import preprocess_image, draw_boxes, get_safe_stem, COCO_CLASSES
from .image_loader import ImageLoader
from .results import Boxes, Results
from ..yolox.utils import preprocess_image as yolox_preprocess_image


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list:
    """Numpy-based Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]

    return keep


class LIBREYOLONCNN:
    """
    ncnn inference backend for LIBREYOLO models.

    Provides the same API as LIBREYOLOX/LIBREYOLO9 but uses ncnn
    instead of PyTorch for inference.

    Args:
        model_dir: Path to the ncnn model directory (containing model.ncnn.param,
            model.ncnn.bin, and optionally metadata.yaml).
        nb_classes: Number of classes (default: auto-detected from metadata, fallback 80).
        device: Device for inference. "auto" (default) uses CPU. "gpu"/"cuda" uses
            Vulkan GPU if available.

    Example:
        >>> model = LIBREYOLONCNN("exported_model_dir/")
        >>> result = model("image.jpg", save=True)
        >>> print(result.boxes.xyxy)
    """

    def __init__(self, model_dir: str, nb_classes: int = None, device: str = "auto"):
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

        self.model_path = str(model_dir)
        self.model_family = None
        self.imgsz = 640

        # Load metadata from metadata.yaml if present
        metadata_path = model_dir / "metadata.yaml"
        if metadata_path.exists():
            self._read_metadata(metadata_path, nb_classes)
        else:
            self.nb_classes = nb_classes if nb_classes is not None else 80
            if self.nb_classes == 80:
                self.names: Dict[int, str] = {i: n for i, n in enumerate(COCO_CLASSES)}
            else:
                self.names: Dict[int, str] = {i: f"class_{i}" for i in range(self.nb_classes)}

        # Map device strings
        device_lower = device.lower() if device else "auto"
        if device_lower in ("auto", "cpu"):
            self.device = "cpu"
            use_vulkan = False
        elif device_lower in ("gpu", "cuda"):
            self.device = "gpu"
            use_vulkan = True
        else:
            self.device = device_lower
            use_vulkan = False

        # Load ncnn model
        self.net = _ncnn.Net()
        if use_vulkan and hasattr(_ncnn, "build_with_gpu") and _ncnn.build_with_gpu:
            self.net.opt.use_vulkan_compute = True
        self.net.load_param(str(param_path))
        self.net.load_model(str(bin_path))

        # Discover input/output blob names from the param file
        self._input_names, self._output_names = self._discover_blob_names(param_path)

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
                    # Input layer: Input <name> 0 1 <blob_name>
                    input_names.append(parts[-1])
                elif layer_type in (
                    "Softmax", "Sigmoid", "Permute", "Reshape",
                    "InnerProduct", "Convolution",
                ):
                    # Last layer output is typically the model output
                    # We'll rely on the fallback mostly
                    pass
        except Exception:
            pass

        if not input_names:
            input_names = ["in0"]
        if not output_names:
            output_names = ["out0"]
        return input_names, output_names

    def _read_metadata(self, metadata_path: Path, nb_classes_override: int = None) -> None:
        """Read metadata from metadata.yaml file."""
        import yaml

        with open(metadata_path) as f:
            meta = yaml.safe_load(f)

        if meta is None:
            meta = {}

        self.model_family = meta.get("model_family")

        if "imgsz" in meta:
            self.imgsz = int(meta["imgsz"])

        if nb_classes_override is not None:
            self.nb_classes = nb_classes_override
        elif "nb_classes" in meta:
            self.nb_classes = int(meta["nb_classes"])
        else:
            self.nb_classes = 80

        if "names" in meta and nb_classes_override is None:
            self.names: Dict[int, str] = {int(k): v for k, v in meta["names"].items()}
        elif self.nb_classes == 80:
            self.names: Dict[int, str] = {i: n for i, n in enumerate(COCO_CLASSES)}
        else:
            self.names: Dict[int, str] = {i: f"class_{i}" for i in range(self.nb_classes)}

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
            source: Input image or directory.
            conf: Confidence threshold.
            iou: IoU threshold for NMS.
            imgsz: Input size override (None = use model's input size).
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
        """Process multiple images in batches."""
        results = []
        for i in range(0, len(image_paths), batch):
            chunk = image_paths[i:i + batch]
            for path in chunk:
                results.append(
                    self._predict_single(
                        path, save=save, output_path=output_path,
                        conf=conf, iou=iou, imgsz=imgsz,
                        classes=classes, max_det=max_det,
                        color_format=color_format,
                    )
                )
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
        import ncnn as _ncnn

        image_path = image if isinstance(image, (str, Path)) else None
        effective_imgsz = imgsz if imgsz is not None else self.imgsz

        # Use model-family-specific preprocessing
        if self.model_family == "LIBREYOLOX":
            input_tensor, original_img, original_size, ratio = yolox_preprocess_image(
                image, input_size=effective_imgsz, color_format=color_format
            )
            self._yolox_ratio = ratio
        elif self.model_family == "LIBREYOLORFDETR":
            input_tensor, original_img, original_size = self._preprocess_rfdetr(
                image, effective_imgsz, color_format
            )
        else:
            input_tensor, original_img, original_size = preprocess_image(
                image, input_size=effective_imgsz, color_format=color_format
            )

        blob = input_tensor.numpy()

        # Run ncnn inference
        # ncnn.Mat expects a C-contiguous (C, H, W) float32 array.
        # blob[0] is a view with non-standard strides (from removing
        # the batch dim of a permuted tensor), so we must make a
        # contiguous copy; otherwise ncnn reads scrambled channel data.
        input_data = np.ascontiguousarray(blob[0])
        mat_in = _ncnn.Mat(input_data)

        ex = self.net.create_extractor()
        ex.input(self._input_names[0], mat_in)

        # Extract outputs
        all_outputs = []
        for out_name in self._output_names:
            ret, mat_out = ex.extract(out_name)
            if ret != 0:
                # Try common fallback names
                for fallback in ("out0", "output", "output0"):
                    ret, mat_out = ex.extract(fallback)
                    if ret == 0:
                        break
                if ret != 0:
                    raise RuntimeError(
                        f"Failed to extract output '{out_name}' from ncnn model"
                    )
            all_outputs.append(np.array(mat_out).reshape(1, *np.array(mat_out).shape))

        # Parse outputs based on model family
        boxes, max_scores, class_ids = self._parse_outputs(
            all_outputs, effective_imgsz, original_size, conf
        )

        # orig_shape for Ultralytics: (H, W)
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
            # Apply NMS
            keep = _nms(boxes, max_scores, iou)
            boxes, max_scores, class_ids = boxes[keep], max_scores[keep], class_ids[keep]

            # Limit to max_det
            if len(boxes) > max_det:
                top_indices = np.argsort(max_scores)[::-1][:max_det]
                boxes, max_scores, class_ids = boxes[top_indices], max_scores[top_indices], class_ids[top_indices]

            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            conf_t = torch.tensor(max_scores, dtype=torch.float32)
            cls_t = torch.tensor(class_ids, dtype=torch.float32)

            # Apply classes filter
            if classes is not None and len(boxes_t) > 0:
                cls_mask = torch.zeros(len(cls_t), dtype=torch.bool)
                for cid in classes:
                    cls_mask |= cls_t == cid
                boxes_t = boxes_t[cls_mask]
                conf_t = conf_t[cls_mask]
                cls_t = cls_t[cls_mask]

            result = Results(
                boxes=Boxes(boxes_t, conf_t, cls_t),
                orig_shape=orig_shape,
                path=str(image_path) if image_path else None,
                names=self.names,
            )

        # Save annotated image if requested
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
                model_tag = Path(self.model_path).name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_dir = Path("runs/detections")
                save_dir.mkdir(parents=True, exist_ok=True)
                final_path = save_dir / f"{stem}_{model_tag}_{timestamp}{ext}"

            annotated_img.save(final_path)
            result.saved_path = str(final_path)

        return result

    # ------------------------------------------------------------------
    # Model-family preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess_rfdetr(image, input_size, color_format):
        """RF-DETR preprocessing: direct resize + ImageNet normalization."""
        img = ImageLoader.load(image, color_format=color_format)
        original_size = img.size  # (W, H)
        original_img = img.copy()

        img_resized = img.resize((input_size, input_size), Image.Resampling.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std

        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        return img_tensor, original_img, original_size

    # ------------------------------------------------------------------
    # Model-family output parsers
    # ------------------------------------------------------------------

    def _parse_outputs(
        self,
        all_outputs: list,
        effective_imgsz: int,
        original_size: tuple,
        conf: float,
    ):
        """Parse ncnn outputs into (boxes_xyxy, scores, class_ids) based on model family."""
        orig_w, orig_h = original_size

        if self.model_family == "LIBREYOLOX":
            return self._parse_yolox(all_outputs, effective_imgsz, orig_w, orig_h, conf)
        elif self.model_family == "LIBREYOLORFDETR":
            return self._parse_rfdetr(all_outputs, orig_w, orig_h, conf)
        else:
            return self._parse_yolov9(all_outputs, effective_imgsz, orig_w, orig_h, conf)

    def _parse_yolox(self, all_outputs, effective_imgsz, orig_w, orig_h, conf):
        """Parse YOLOX output: (B, N, 5+nc) with cxcywh + objectness + class_scores."""
        outputs = all_outputs[0][0]  # (N, 5+nc)

        cx, cy, w, h = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]
        objectness = outputs[:, 4]
        class_scores = outputs[:, 5:]

        max_class_scores = np.max(class_scores, axis=1)
        max_scores = objectness * max_class_scores
        class_ids = np.argmax(class_scores, axis=1)

        mask = max_scores > conf
        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        max_scores, class_ids = max_scores[mask], class_ids[mask]

        if len(max_scores) == 0:
            return np.empty((0, 4)), max_scores, class_ids

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        ratio = getattr(self, '_yolox_ratio', 1.0)
        boxes /= ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        return boxes, max_scores, class_ids

    def _parse_yolov9(self, all_outputs, effective_imgsz, orig_w, orig_h, conf):
        """Parse YOLOv9 output: (B, 4+nc, N) with xyxy + class_scores."""
        outputs = all_outputs[0][0].T  # (N, 4+nc)

        boxes = outputs[:, :4]
        scores = outputs[:, 4:]

        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        mask = max_scores > conf
        boxes, max_scores, class_ids = boxes[mask], max_scores[mask], class_ids[mask]

        if len(boxes) == 0:
            return boxes, max_scores, class_ids

        scale_x = orig_w / effective_imgsz
        scale_y = orig_h / effective_imgsz
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        return boxes, max_scores, class_ids

    def _parse_rfdetr(self, all_outputs, orig_w, orig_h, conf):
        """Parse RF-DETR output: two tensors - boxes (B,300,4) cxcywh [0,1] + logits (B,300,nc)."""
        boxes_raw = all_outputs[0][0]  # (300, 4) normalized cxcywh
        logits = all_outputs[1][0]     # (300, nc) raw logits

        scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64))).astype(np.float32)

        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        mask = max_scores > conf
        boxes_raw, max_scores, class_ids = boxes_raw[mask], max_scores[mask], class_ids[mask]

        if len(boxes_raw) == 0:
            return boxes_raw, max_scores, class_ids

        cx, cy, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
        x1 = (cx - w / 2) * orig_w
        y1 = (cy - h / 2) * orig_h
        x2 = (cx + w / 2) * orig_w
        y2 = (cy + h / 2) * orig_h
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        return boxes, max_scores, class_ids

    def predict(self, *args, **kwargs) -> Union[Results, List[Results]]:
        """Alias for __call__ method."""
        return self(*args, **kwargs)
