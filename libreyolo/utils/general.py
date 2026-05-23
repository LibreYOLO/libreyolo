"""Shared general utility functions."""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union
from urllib.parse import urlparse

import torch
import torchvision.ops

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================


def increment_path(
    path: Union[str, Path], exist_ok: bool = False, sep: str = "", mkdir: bool = False
) -> Path:
    """
    Return an incremented path if it already exists.

    E.g. runs/detect/predict -> runs/detect/predict2 -> runs/detect/predict3, etc.

    Args:
        path: Base path to increment.
        exist_ok: If True, return the path as-is even if it exists.
        sep: Separator between base name and number (default: "").
        mkdir: Create the directory if True.

    Returns:
        Incremented Path.
    """
    path = Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"
            if not Path(p).exists():
                break
        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path


# COCO class names (80 classes)
COCO_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


# =============================================================================
# Box Utilities
# =============================================================================


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2).

    Args:
        boxes: Boxes in cxcywh format (..., 4)

    Returns:
        Boxes in xyxy format (..., 4)
    """
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


# =============================================================================
# Path Utilities
# =============================================================================

_save_dir_cache: Dict[str, Path] = {}


def get_safe_stem(path: Union[str, Path]) -> str:
    path_str = str(path)
    if path_str.startswith(("http://", "https://", "s3://", "gs://")):
        parsed = urlparse(path_str)
        filename = Path(parsed.path).name
        return Path(filename).stem if filename else "inference"
    return Path(path_str).stem


def resolve_save_path(
    output_path: Union[str, Path, None],
    image_path: Union[str, Path, None],
    prefix: str = "",
    ext: str = "jpg",
    default_dir: str = "runs/detect",
    exist_ok: bool = False,
) -> Path:
    """
    Generate a save path handling both directory and file output paths.

    Uses an auto-incrementing directory scheme: runs/detect/predict,
    runs/detect/predict2, etc. The original filename is preserved.
    Within a single process, all images are saved to the same directory.
    Duplicate filenames from different input folders will overwrite.

    Args:
        output_path: User-provided output path (file or directory) or None
        image_path: Source image path for deriving filename
        prefix: Optional prefix for the filename (e.g., "tiled_")
        ext: File extension without dot (default: "jpg")
        default_dir: Default directory if output_path is None
        exist_ok: If True, reuse existing predict/ directory without incrementing

    Returns:
        Resolved Path object ready for saving
    """
    # Get filename from image path or use default
    if image_path is not None:
        stem = get_safe_stem(image_path)
    else:
        stem = "inference"

    filename = f"{prefix}{stem}.{ext}"

    if output_path is None:
        if default_dir not in _save_dir_cache:
            _save_dir_cache[default_dir] = increment_path(
                Path(default_dir) / "predict", exist_ok=exist_ok, mkdir=True
            )
        return _save_dir_cache[default_dir] / filename

    save_path = Path(output_path)

    if save_path.suffix == "":
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path / filename
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        return save_path


def log_saved_result(result, save_path: Union[str, Path]) -> str:
    """Attach and log the path where an inference result was saved."""
    saved_path = str(save_path)
    result.saved_path = saved_path
    logger.info("Results saved to %s", saved_path)
    return saved_path


# =============================================================================
# Image Tiling
# =============================================================================


def get_slice_bboxes(
    image_width: int,
    image_height: int,
    slice_size: int = 640,
    overlap_ratio: float = 0.2,
) -> List[Tuple[int, int, int, int]]:
    """
    Generate tile coordinates for slicing a large image.

    Args:
        image_width: Width of the original image.
        image_height: Height of the original image.
        slice_size: Size of each square tile (default: 640).
        overlap_ratio: Fractional overlap between tiles (default: 0.2).

    Returns:
        List of (x1, y1, x2, y2) tuples representing tile coordinates.
    """
    slices = []
    overlap = int(slice_size * overlap_ratio)
    step = slice_size - overlap

    y = 0
    while y < image_height:
        x = 0
        while x < image_width:
            x2 = min(x + slice_size, image_width)
            y2 = min(y + slice_size, image_height)
            # Ensure full tile size when near edges by adjusting start position
            x1 = max(0, x2 - slice_size) if x2 == image_width else x
            y1 = max(0, y2 - slice_size) if y2 == image_height else y
            slices.append((x1, y1, x2, y2))
            x += step
            if x2 == image_width:
                break
        y += step
        if y2 == image_height:
            break
    return slices


# =============================================================================
# Detection Post-processing
# =============================================================================


def make_anchors(
    feats: List[torch.Tensor], strides: List[int], grid_cell_offset: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate anchor points from feature map sizes.

    Args:
        feats: List of feature tensors from different scales
        strides: List of stride values corresponding to each feature map
        grid_cell_offset: Offset for grid cell centers (default: 0.5)

    Returns:
        Tuple of (anchor_points, stride_tensor)
    """
    anchor_points = []
    stride_tensor = []

    for feat, stride in zip(feats, strides):
        _, _, h, w = feat.shape
        dtype, device = feat.dtype, feat.device

        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))

    return torch.cat(anchor_points), torch.cat(stride_tensor)


def postprocess_detections(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    input_size: int = 640,
    original_size: Tuple[int, int] | None = None,
    max_det: int = 300,
    letterbox: bool = False,
) -> dict:
    """
    Shared post-processing pipeline for object detection outputs.

    This function handles the common post-processing steps:
    - Scale boxes to original image size
    - Clip boxes to image boundaries
    - Filter invalid boxes (zero/negative area)
    - Apply per-class NMS
    - Limit to max detections

    Args:
        boxes: Decoded boxes in xyxy format (N, 4)
        scores: Confidence scores after sigmoid (N,)
        class_ids: Class indices (N,)
        conf_thres: Confidence threshold (already applied before calling)
        iou_thres: IoU threshold for NMS
        input_size: Model input size for scaling
        original_size: Original image size (width, height)
        max_det: Maximum number of detections
        letterbox: If True, use letterbox-inverse scaling (aspect-preserving).
            If False, use independent x/y scaling (simple resize).

    Returns:
        Dictionary with boxes, scores, classes, num_detections
    """
    if len(boxes) == 0:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    # Scale boxes to original image size
    if original_size is not None:
        if letterbox:
            # Letterbox inverse: r = min(input/orig_h, input/orig_w)
            orig_w, orig_h = original_size
            r = min(input_size / orig_h, input_size / orig_w)
            boxes[:, :4] = boxes[:, :4] / r
        else:
            # Simple resize: independent x/y scaling
            scale_x = original_size[0] / input_size
            scale_y = original_size[1] / input_size
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, original_size[0])
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, original_size[1])

        # Filter zero/negative-area boxes
        box_widths = boxes[:, 2] - boxes[:, 0]
        box_heights = boxes[:, 3] - boxes[:, 1]
        valid_mask = (box_widths > 0) & (box_heights > 0)

        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            class_ids = class_ids[valid_mask]

    if len(boxes) == 0:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    # Drop NaN/Inf boxes — batched_nms has undefined behaviour on non-finite
    # values and can return wrong indices or raise on CUDA.
    finite_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
    if not finite_mask.all():
        boxes = boxes[finite_mask]
        scores = scores[finite_mask]
        class_ids = class_ids[finite_mask]

    if len(boxes) == 0:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    # Cast to fp32 — batched_nms applies a per-class offset of (boxes.max() + 1)
    # which overflows fp16 (max 65504) for typical letterbox coords × num_classes
    # and silently merges classes that should stay separate. Detectron2 carries
    # this same wrapper for the same reason. scores is cast too because
    # torchvision.ops.nms requires matching dtypes for boxes and scores.
    if boxes.dtype == torch.float16:
        boxes = boxes.float()
        scores = scores.float()

    # Per-class NMS — single batched dispatch instead of one kernel per class.
    # Equivalent to the previous ``for cls in unique_classes: ops.nms(...)``
    # loop (same class-offset trick, same CUDA kernel) but avoids the ~80-
    # iteration Python loop on COCO, especially during validation which forces
    # conf_thres=0.0 and so cannot rely on the conf prefilter to shrink classes.
    keep_indices = torchvision.ops.batched_nms(boxes, scores, class_ids, iou_thres)

    if len(keep_indices) == 0:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    # Use topk rather than relying on batched_nms output order, which is an
    # undocumented implementation detail that could change across torchvision versions.
    if len(keep_indices) > max_det:
        _, top_indices = torch.topk(scores[keep_indices], max_det)
        keep_indices = keep_indices[top_indices]

    final_boxes = boxes[keep_indices].cpu().numpy()
    final_scores = scores[keep_indices].cpu().numpy()
    final_classes = class_ids[keep_indices].cpu().numpy()

    return {
        "boxes": final_boxes.tolist(),
        "scores": final_scores.tolist(),
        "classes": final_classes.tolist(),
        "num_detections": len(final_boxes),
    }
