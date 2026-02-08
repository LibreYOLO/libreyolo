"""
ONNX runtime inference backend for LIBREYOLO.
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


class LIBREYOLOOnnx:
    """
    ONNX runtime inference backend for LIBREYOLO models.

    Provides the same API as LIBREYOLOX/LIBREYOLO9 but uses ONNX Runtime
    instead of PyTorch for inference.

    Args:
        onnx_path: Path to the ONNX model file.
        nb_classes: Number of classes (default: 80 for COCO).
        device: Device for inference. "auto" (default) uses CUDA if available, else CPU.

    Example:
        >>> model = LIBREYOLOOnnx("model.onnx")
        >>> result = model("image.jpg", save=True)
        >>> print(result.boxes.xyxy)
    """

    def __init__(self, onnx_path: str, nb_classes: int = 80, device: str = "auto"):
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "ONNX inference requires onnxruntime. "
                "Install with: pip install onnxruntime"
            ) from e

        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self.model_path = onnx_path
        self.nb_classes = nb_classes

        # Build names dict (matches Ultralytics model.names)
        if nb_classes == 80:
            self.names: Dict[int, str] = {i: n for i, n in enumerate(COCO_CLASSES)}
        else:
            self.names: Dict[int, str] = {i: f"class_{i}" for i in range(nb_classes)}

        # Resolve device and set providers
        available_providers = ort.get_available_providers()
        if device == "auto":
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.device = "cuda"
            else:
                providers = ["CPUExecutionProvider"]
                self.device = "cpu"
        elif device in ("cuda", "gpu"):
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            self.device = "cuda" if "CUDAExecutionProvider" in available_providers else "cpu"
        else:
            providers = ["CPUExecutionProvider"]
            self.device = "cpu"

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

        # Read expected input size from model (fallback to 640)
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 4 and isinstance(input_shape[2], int):
            self.imgsz = input_shape[2]
        else:
            self.imgsz = 640

        # Try to read libreyolo metadata from ONNX model
        self._read_onnx_metadata(onnx_path)

    def _read_onnx_metadata(self, onnx_path: str) -> None:
        """Read libreyolo metadata embedded in an ONNX model file.

        Silently falls back to constructor args if metadata is missing
        or the onnx package is not installed.
        """
        self.model_family = None
        try:
            import onnx

            model_proto = onnx.load(onnx_path)
            meta = {p.key: p.value for p in model_proto.metadata_props}

            if "model_family" in meta:
                self.model_family = meta["model_family"]

            if "names" in meta:
                import json

                names_raw = json.loads(meta["names"])
                self.names = {int(k): v for k, v in names_raw.items()}

            if "nb_classes" in meta:
                self.nb_classes = int(meta["nb_classes"])
                # Rebuild names if nb_classes changed but names wasn't present
                if "names" not in meta:
                    if self.nb_classes == 80:
                        self.names = {i: n for i, n in enumerate(COCO_CLASSES)}
                    else:
                        self.names = {
                            i: f"class_{i}" for i in range(self.nb_classes)
                        }
        except Exception:
            pass

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
            imgsz: Input size override (None = 640).
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
        image_path = image if isinstance(image, (str, Path)) else None
        effective_imgsz = imgsz if imgsz is not None else self.imgsz

        # Use model-family-specific preprocessing
        if self.model_family == "LIBREYOLOX":
            # YOLOX: letterbox + 0-255 range + BGR
            input_tensor, original_img, original_size, ratio = yolox_preprocess_image(
                image, input_size=effective_imgsz, color_format=color_format
            )
            self._yolox_ratio = ratio
        elif self.model_family == "LIBREYOLORFDETR":
            # RF-DETR: direct resize + ImageNet normalization
            input_tensor, original_img, original_size = self._preprocess_rfdetr(
                image, effective_imgsz, color_format
            )
        else:
            input_tensor, original_img, original_size = preprocess_image(
                image, input_size=effective_imgsz, color_format=color_format
            )

        blob = input_tensor.numpy()

        # Run ONNX inference
        all_outputs = self.session.run(None, {self.input_name: blob})

        # Parse outputs based on model family
        boxes, max_scores, class_ids = self._parse_outputs(
            all_outputs, effective_imgsz, original_size, conf
        )

        # orig_shape for Ultralytics: (H, W)
        orig_w, orig_h = original_size  # preprocess_image returns (W, H)
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
                model_tag = Path(self.model_path).stem
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
        """Parse ONNX outputs into (boxes_xyxy, scores, class_ids) based on model family.

        Returns filtered arrays in original image coordinates.
        """
        orig_w, orig_h = original_size

        if self.model_family == "LIBREYOLOX":
            return self._parse_yolox(all_outputs, effective_imgsz, orig_w, orig_h, conf)
        elif self.model_family == "LIBREYOLORFDETR":
            return self._parse_rfdetr(all_outputs, orig_w, orig_h, conf)
        else:
            # Default / YOLOv9: single tensor (B, 4+nc, N)
            return self._parse_yolov9(all_outputs, effective_imgsz, orig_w, orig_h, conf)

    def _parse_yolox(self, all_outputs, effective_imgsz, orig_w, orig_h, conf):
        """Parse YOLOX output: (B, N, 5+nc) with cxcywh + objectness + class_scores."""
        outputs = all_outputs[0][0]  # (N, 5+nc)

        # Split: 4 box coords (cxcywh), 1 objectness, nc class scores
        cx, cy, w, h = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]
        objectness = outputs[:, 4]
        class_scores = outputs[:, 5:]

        # Confidence = objectness * max class score
        max_class_scores = np.max(class_scores, axis=1)
        max_scores = objectness * max_class_scores
        class_ids = np.argmax(class_scores, axis=1)

        # Apply confidence threshold
        mask = max_scores > conf
        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        max_scores, class_ids = max_scores[mask], class_ids[mask]

        if len(max_scores) == 0:
            return np.empty((0, 4)), max_scores, class_ids

        # Convert cxcywh to xyxy
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Scale from letterboxed input to original image using ratio
        ratio = getattr(self, '_yolox_ratio', 1.0)
        boxes /= ratio
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

        return boxes, max_scores, class_ids

    def _parse_yolov9(self, all_outputs, effective_imgsz, orig_w, orig_h, conf):
        """Parse YOLOv9 output: (B, 4+nc, N) with xyxy + class_scores."""
        # Transpose from (4+nc, N) to (N, 4+nc)
        outputs = all_outputs[0][0].T  # (N, 4+nc)

        boxes = outputs[:, :4]  # xyxy in input space
        scores = outputs[:, 4:]

        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        # Apply confidence threshold
        mask = max_scores > conf
        boxes, max_scores, class_ids = boxes[mask], max_scores[mask], class_ids[mask]

        if len(boxes) == 0:
            return boxes, max_scores, class_ids

        # Scale from input space to original image
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

        # Sigmoid to get probabilities
        scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64))).astype(np.float32)

        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        # Apply confidence threshold
        mask = max_scores > conf
        boxes_raw, max_scores, class_ids = boxes_raw[mask], max_scores[mask], class_ids[mask]

        if len(boxes_raw) == 0:
            return boxes_raw, max_scores, class_ids

        # Convert cxcywh [0,1] to xyxy in original image coordinates
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
