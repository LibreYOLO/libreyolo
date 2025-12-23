"""
ONNX runtime inference backend for LIBREYOLO.
"""

from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from .utils import preprocess_image, draw_boxes


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
    
    Provides the same API as LIBREYOLO8/LIBREYOLO11 but uses ONNX Runtime
    instead of PyTorch for inference.
    
    Args:
        onnx_path: Path to the ONNX model file.
        nb_classes: Number of classes (default: 80 for COCO).
        device: Device for inference. "auto" (default) uses CUDA if available, else CPU.
    
    Example:
        >>> model = LIBREYOLOOnnx("model.onnx")
        >>> detections = model("image.jpg", save=True)
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
    
    def __call__(
        self,
        image: Union[str, Image.Image, np.ndarray],
        save: bool = False,
        output_path: str = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> dict:
        """
        Run inference on an image.
        
        Args:
            image: Input image (file path, PIL Image, or numpy array).
            save: If True, saves annotated image to disk.
            output_path: Optional path to save the annotated image.
            conf_thres: Confidence threshold (default: 0.25).
            iou_thres: IoU threshold for NMS (default: 0.45).
        
        Returns:
            Dictionary with boxes, scores, classes, and num_detections.
        """
        image_path = image if isinstance(image, str) else None
        
        # Preprocess (reuse existing utility, convert tensor to numpy)
        input_tensor, original_img, original_size = preprocess_image(image, input_size=640)
        blob = input_tensor.numpy()
        
        # Run ONNX inference
        outputs = self.session.run(None, {self.input_name: blob})[0][0]  # (N, 84)
        
        # Parse outputs: first 4 = xyxy boxes, rest = class scores
        boxes = outputs[:, :4]
        scores = outputs[:, 4:]
        
        # Get max score and class per detection
        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        
        # Apply confidence threshold
        mask = max_scores > conf_thres
        boxes, max_scores, class_ids = boxes[mask], max_scores[mask], class_ids[mask]
        
        if len(boxes) == 0:
            return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}
        
        # Scale boxes to original image size
        scale_x = original_size[0] / 640
        scale_y = original_size[1] / 640
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        # Clip to image bounds
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_size[0])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_size[1])
        
        # Apply NMS
        keep = _nms(boxes, max_scores, iou_thres)
        boxes, max_scores, class_ids = boxes[keep], max_scores[keep], class_ids[keep]
        
        detections = {
            "boxes": boxes.tolist(),
            "scores": max_scores.tolist(),
            "classes": class_ids.tolist(),
            "num_detections": len(boxes),
        }
        
        # Save annotated image if requested
        if save:
            if detections["num_detections"] > 0:
                annotated_img = draw_boxes(
                    original_img,
                    detections["boxes"],
                    detections["scores"],
                    detections["classes"],
                )
            else:
                annotated_img = original_img
            
            if output_path:
                final_path = Path(output_path)
                final_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                stem = Path(image_path).stem if image_path else "inference"
                ext = Path(image_path).suffix if image_path else ".jpg"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = Path("runs/detections")
                save_dir.mkdir(parents=True, exist_ok=True)
                final_path = save_dir / f"{stem}_{timestamp}{ext}"
            
            annotated_img.save(final_path)
            detections["saved_path"] = str(final_path)
        
        return detections
    
    def predict(
        self,
        image: Union[str, Image.Image, np.ndarray],
        save: bool = False,
        output_path: str = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ) -> dict:
        """Alias for __call__ method."""
        return self(image=image, save=save, output_path=output_path, conf_thres=conf_thres, iou_thres=iou_thres)

