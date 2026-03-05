"""Shared general utility functions."""

from pathlib import Path
from typing import Dict, List, Tuple, Union
from urllib.parse import urlparse

import torch


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


def nms(
    boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.45
) -> torch.Tensor:
    """
    Non-Maximum Suppression using torch operations.

    Args:
        boxes: Boxes in xyxy format (N, 4)
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold for suppression

    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    # Filter out boxes with NaN or Inf values
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
    if not valid_mask.any():
        return torch.tensor([], dtype=torch.long, device=boxes.device)

    if not valid_mask.all():
        valid_indices = torch.where(valid_mask)[0]
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
    else:
        valid_indices = None

    _, order = scores.sort(0, descending=True)
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i.item())

        if len(order) == 1:
            break

        box_i = boxes[i]
        boxes_remaining = boxes[order[1:]]

        x1_i, y1_i, x2_i, y2_i = box_i
        x1_r, y1_r, x2_r, y2_r = (
            boxes_remaining[:, 0],
            boxes_remaining[:, 1],
            boxes_remaining[:, 2],
            boxes_remaining[:, 3],
        )

        x1_inter = torch.max(x1_i, x1_r)
        y1_inter = torch.max(y1_i, y1_r)
        x2_inter = torch.min(x2_i, x2_r)
        y2_inter = torch.min(y2_i, y2_r)

        inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(
            y2_inter - y1_inter, min=0
        )

        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        area_r = (x2_r - x1_r) * (y2_r - y1_r)
        union_area = area_i + area_r - inter_area

        iou = inter_area / (union_area + 1e-7)
        order = order[1:][iou < iou_threshold]

    keep_tensor = torch.tensor(keep, dtype=torch.long, device=boxes.device)

    # Map back to original indices if we filtered invalid boxes
    if valid_indices is not None:
        keep_tensor = valid_indices[keep_tensor]

    return keep_tensor


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

    # Per-class NMS
    try:
        import torchvision.ops

        use_torchvision_nms = True
    except ImportError:
        use_torchvision_nms = False

    unique_classes = torch.unique(class_ids)
    keep_indices_list = []

    for cls in unique_classes:
        cls_mask = class_ids == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        if len(cls_boxes) == 0:
            continue

        if use_torchvision_nms:
            max_wh = 7680.0
            boxes_for_nms = cls_boxes + cls.float() * max_wh
            cls_keep = torchvision.ops.nms(boxes_for_nms, cls_scores, iou_thres)
        else:
            cls_keep = nms(cls_boxes, cls_scores, iou_thres)

        cls_indices = torch.where(cls_mask)[0]
        keep_indices_list.append(cls_indices[cls_keep])

    if len(keep_indices_list) == 0:
        return {"boxes": [], "scores": [], "classes": [], "num_detections": 0}

    keep_indices = torch.cat(keep_indices_list)

    if len(keep_indices) > max_det:
        final_scores_temp = scores[keep_indices]
        _, top_indices = torch.topk(final_scores_temp, max_det)
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


def postprocess_batch(
    batch_boxes: torch.Tensor,
    batch_scores: torch.Tensor,
    batch_class_ids: torch.Tensor,
    batch_indices: torch.Tensor,
    batch_size: int,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    input_size: int = 640,
    original_sizes: List[Tuple[int, int]] | None = None,
    max_det: int = 300,
    device: torch.device | None = None,
) -> List[Dict]:
    """
    Batched post-processing for efficient validation.

    Processes all detections from a batch at once using GPU-accelerated operations,
    then splits results per image. This is much faster than processing images
    one at a time in a Python loop.

    Args:
        batch_boxes: All boxes from batch (N_total, 4) in xyxy format
        batch_scores: All scores from batch (N_total,)
        batch_class_ids: All class IDs from batch (N_total,)
        batch_indices: Image index for each detection (N_total,) - which image each box belongs to
        batch_size: Number of images in batch
        conf_thres: Confidence threshold (already applied if boxes are pre-filtered)
        iou_thres: IoU threshold for NMS
        input_size: Model input size for scaling
        original_sizes: List of (width, height) tuples for each image
        max_det: Maximum detections per image
        device: Device for computations

    Returns:
        List of detection dicts, one per image in batch
    """
    try:
        import torchvision.ops

        has_torchvision = True
    except ImportError:
        has_torchvision = False

    results = []

    if len(batch_boxes) == 0:
        for _ in range(batch_size):
            results.append(
                {
                    "boxes": torch.zeros((0, 4), device=device),
                    "scores": torch.zeros(0, device=device),
                    "classes": torch.zeros(0, dtype=torch.int64, device=device),
                    "num_detections": 0,
                }
            )
        return results

    # Batched NMS using class offsets: offset boxes by batch_idx and class_id
    # to prevent cross-image and cross-class suppression in a single call
    if has_torchvision:
        max_wh = 7680.0  # max expected image dimension
        max_batch_offset = max_wh * 100  # large offset between batches

        combined_idx = (
            batch_indices.float() * max_batch_offset + batch_class_ids.float() * max_wh
        )
        boxes_for_nms = batch_boxes + combined_idx.unsqueeze(1)
        keep = torchvision.ops.nms(boxes_for_nms, batch_scores, iou_thres)

        batch_boxes = batch_boxes[keep]
        batch_scores = batch_scores[keep]
        batch_class_ids = batch_class_ids[keep]
        batch_indices = batch_indices[keep]

    # Split results by image
    for img_idx in range(batch_size):
        img_mask = batch_indices == img_idx

        if not img_mask.any():
            results.append(
                {
                    "boxes": torch.zeros((0, 4), device=device),
                    "scores": torch.zeros(0, device=device),
                    "classes": torch.zeros(0, dtype=torch.int64, device=device),
                    "num_detections": 0,
                }
            )
            continue

        img_boxes = batch_boxes[img_mask].detach()
        img_scores = batch_scores[img_mask].detach()
        img_classes = batch_class_ids[img_mask].detach()

        if original_sizes is not None and img_idx < len(original_sizes):
            orig_w, orig_h = original_sizes[img_idx]
            scale_x = orig_w / input_size
            scale_y = orig_h / input_size
            img_boxes = img_boxes.clone()
            img_boxes[:, [0, 2]] *= scale_x
            img_boxes[:, [1, 3]] *= scale_y
            img_boxes[:, [0, 2]] = torch.clamp(img_boxes[:, [0, 2]], 0, orig_w)
            img_boxes[:, [1, 3]] = torch.clamp(img_boxes[:, [1, 3]], 0, orig_h)

        if len(img_boxes) > max_det:
            _, top_k = torch.topk(img_scores, max_det)
            img_boxes = img_boxes[top_k]
            img_scores = img_scores[top_k]
            img_classes = img_classes[top_k]

        results.append(
            {
                "boxes": img_boxes,
                "scores": img_scores,
                "classes": img_classes,
                "num_detections": len(img_boxes),
            }
        )

    return results
