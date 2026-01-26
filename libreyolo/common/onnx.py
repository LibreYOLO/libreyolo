"""
ONNX runtime inference backend for LIBREYOLO.
"""

from datetime import datetime
from pathlib import Path
from typing import Union, List

import numpy as np
from PIL import Image

from .utils import preprocess_image, draw_boxes, get_safe_stem
from .image_loader import ImageLoader


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
        image: Union[str, Path, Image.Image, np.ndarray],
        save: bool = False,
        output_path: str = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        color_format: str = "auto",
        batch_size: int = 1
    ) -> Union[dict, List[dict]]:
        """
        Run inference on an image or directory of images.
        
        Args:
            image: Input image or directory (file path, directory path, PIL Image, or numpy array).
            save: If True, saves annotated image to disk.
            output_path: Optional path to save the annotated image.
            conf_thres: Confidence threshold (default: 0.25).
            iou_thres: IoU threshold for NMS (default: 0.45).
            color_format: Color format hint for NumPy/OpenCV arrays ("auto", "rgb", "bgr").
            batch_size: Number of images to process per batch when handling multiple
                images (e.g., directories). Currently used for chunking at the Python
                level; true batched model inference is planned for future versions.
                Default: 1 (process one image at a time).
        
        Returns:
            For single image: Dictionary with boxes, scores, classes, source, and num_detections.
            For directory: List of dictionaries, one per image processed.
        """
        if isinstance(image, (str, Path)) and Path(image).is_dir():
            image_paths = ImageLoader.collect_images(image)
            if not image_paths:
                return []
            return self._process_in_batches(
                image_paths,
                batch_size=batch_size,
                save=save,
                output_path=output_path,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                color_format=color_format
            )
        
        return self._predict_single(image, save, output_path, conf_thres, iou_thres, color_format)
    
    def _process_in_batches(
        self,
        image_paths: List[Path],
        batch_size: int = 1,
        save: bool = False,
        output_path: str = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        color_format: str = "auto"
    ) -> List[dict]:
        """
        Process multiple images, respecting batch_size for chunking.
        
        This method provides the scaffolding for batch processing. Currently, it
        processes images sequentially within each batch chunk. Future versions
        will implement true batched model inference for improved throughput.
        
        Args:
            image_paths: List of image paths to process.
            batch_size: Number of images per batch chunk.
            save: If True, saves annotated images.
            output_path: Optional output path for saved images.
            conf_thres: Confidence threshold.
            iou_thres: IoU threshold for NMS.
            color_format: Color format hint.
        
        Returns:
            List of detection dictionaries, one per image.
        """
        results = []
        for i in range(0, len(image_paths), batch_size):
            chunk = image_paths[i:i + batch_size]
            # TODO: Implement _predict_batch() for true batched model inference
            # For now, process images sequentially within each chunk
            for path in chunk:
                results.append(
                    self._predict_single(path, save, output_path, conf_thres, iou_thres, color_format)
                )
        return results
    
    def _predict_single(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        save: bool = False,
        output_path: str = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        color_format: str = "auto"
    ) -> dict:
        """
        Run inference on a single image.
        
        This is the internal implementation for single-image inference.
        Use __call__ for the public API which also supports directories.
        """
        image_path = image if isinstance(image, (str, Path)) else None
        
        input_tensor, original_img, original_size = preprocess_image(image, input_size=640, color_format=color_format)
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
            return {"boxes": [], "scores": [], "classes": [], "num_detections": 0, "source": str(image_path) if image_path else None}
        
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
            "source": str(image_path) if image_path else None,
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
                stem = get_safe_stem(image_path) if image_path else "inference"
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
        image: Union[str, Path, Image.Image, np.ndarray],
        save: bool = False,
        output_path: str = None,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        color_format: str = "auto",
        batch_size: int = 1
    ) -> Union[dict, List[dict]]:
        """
        Alias for __call__ method.
        
        Args:
            image: Input image or directory.
            save: If True, saves annotated image to disk.
            output_path: Optional path to save the annotated image.
            conf_thres: Confidence threshold (default: 0.25).
            iou_thres: IoU threshold for NMS (default: 0.45).
            color_format: Color format hint for NumPy/OpenCV arrays.
            batch_size: Number of images to process per batch when handling multiple
                images (e.g., directories). Default: 1.
        
        Returns:
            For single image: Dictionary containing detection results.
            For directory: List of dictionaries, one per image processed.
        """
        return self(image=image, save=save, output_path=output_path, conf_thres=conf_thres, iou_thres=iou_thres, color_format=color_format, batch_size=batch_size)

