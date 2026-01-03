"""
Shared utility functions.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Union
from urllib.parse import urlparse
from PIL import Image, ImageDraw, ImageFont
import colorsys

from .image_loader import ImageLoader, ImageInput


def get_slice_bboxes(
    image_width: int,
    image_height: int,
    slice_size: int = 640,
    overlap_ratio: float = 0.2
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


def draw_tile_grid(img: Image.Image, tile_coords: List[Tuple[int, int, int, int]], line_color: str = "#FF0000", line_width: int = 3) -> Image.Image:
    """
    Draw grid lines on an image to visualize tile boundaries.

    Args:
        img: PIL Image to draw on.
        tile_coords: List of (x1, y1, x2, y2) tuples representing tile coordinates.
        line_color: Color of the grid lines (default: red).
        line_width: Width of the grid lines in pixels (default: 3).

    Returns:
        PIL Image with grid lines drawn.
    """
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    # Scale line width based on image size
    max_dim = max(img.size)
    scale_factor = max_dim / 640.0
    scaled_width = max(2, min(int(line_width * scale_factor), 10))

    for x1, y1, x2, y2 in tile_coords:
        # Draw rectangle for each tile
        draw.rectangle([x1, y1, x2, y2], outline=line_color, width=scaled_width)

    return img_draw


def get_safe_stem(path: Union[str, Path]) -> str:
    path_str = str(path)
    if path_str.startswith(("http://", "https://", "s3://", "gs://")):
        parsed = urlparse(path_str)
        filename = Path(parsed.path).name
        return Path(filename).stem if filename else "inference"
    return Path(path_str).stem

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_class_color(class_id: int) -> str:
    """
    Get a unique color for a class ID.
    Uses a hash-based approach to generate consistent colors.
    """
    # Generate colors using HSV color space for better distribution
    hue = (class_id * 137.508) % 360 / 360.0  # Golden angle approximation
    saturation = 0.7 + (class_id % 3) * 0.1  # Vary saturation
    value = 0.8 + (class_id % 2) * 0.15  # Vary brightness
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"

def preprocess_image(
    image: ImageInput,
    input_size: int = 640,
    color_format: str = "auto"
) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
    """
    Preprocess image for model inference.
    
    Args:
        image: Input image. Supported types:
            - str: Local file path or URL (http/https/s3/gs)
            - pathlib.Path: Local file path
            - PIL.Image: PIL Image object
            - np.ndarray: NumPy array (HWC or CHW, RGB or BGR)
            - torch.Tensor: PyTorch tensor (CHW or NCHW)
            - bytes: Raw image bytes
            - io.BytesIO: BytesIO object containing image data
        input_size: Target size for resizing (default: 640)
        color_format: Color format hint for NumPy/OpenCV arrays.
            - "auto": Auto-detect (default)
            - "rgb": Input is RGB format
            - "bgr": Input is BGR format (e.g., OpenCV)
        
    Returns:
        Tuple of (preprocessed_tensor, original_image, original_size)
    """
    # Use unified ImageLoader to handle all input types
    img = ImageLoader.load(image, color_format=color_format)
    
    original_size = img.size  # (width, height)
    original_img = img.copy()
    
    # Resize to input_size (simple resize, maintain aspect ratio could be added)
    img_resized = img.resize((input_size, input_size), Image.Resampling.BILINEAR)
    
    # Convert to numpy and normalize
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Convert to tensor: HWC -> CHW -> add batch dimension
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, original_img, original_size

def draw_boxes(img: Image.Image, boxes: List, scores: List, classes: List, class_names: List = None) -> Image.Image:
    """
    Draw bounding boxes on image with class-specific colors.

    Box thickness and font size scale automatically based on image dimensions
    for better visibility on both small and large images.

    Args:
        img: PIL Image to draw on
        boxes: List of boxes in xyxy format
        scores: List of confidence scores
        classes: List of class IDs
        class_names: Optional list of class names (default: COCO_CLASSES)

    Returns:
        Annotated PIL Image
    """
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    # Use COCO classes if not provided
    if class_names is None:
        class_names = COCO_CLASSES

    # Calculate scaling factor based on image size
    # Use the larger dimension to determine scale
    img_width, img_height = img.size
    max_dim = max(img_width, img_height)

    # Scale factor: base thickness/font at 640px, scales up for larger images
    # Minimum thickness of 2, scales up to ~6 for 2000px+ images
    scale_factor = max_dim / 640.0
    box_thickness = max(2, min(int(2 * scale_factor), 8))

    # Font size scales similarly: base 12px at 640px
    font_size = max(12, min(int(12 * scale_factor), 36))

    # Try to load a font with scaled size, fallback to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        try:
            # Try common Linux fonts
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            font = ImageFont.load_default()

    # Label padding scales with font size
    label_padding = max(2, int(2 * scale_factor))

    for box, score, cls_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        cls_id_int = int(cls_id)

        # Get class-specific color
        color = get_class_color(cls_id_int)

        # Draw box with class color and scaled thickness
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_thickness)

        # Prepare label
        if class_names and cls_id_int < len(class_names):
            label = f"{class_names[cls_id_int]}: {score:.2f}"
        else:
            label = f"Class {cls_id_int}: {score:.2f}"

        # Get text size
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw label background with class color (scaled padding)
        draw.rectangle([x1, y1 - text_height - label_padding * 2, x1 + text_width + label_padding * 2, y1], fill=color)

        # Draw label text
        draw.text((x1 + label_padding, y1 - text_height - label_padding), label, fill="white", font=font)

    return img_draw


# =============================================================================
# Shared Model Utilities
# =============================================================================

def resolve_save_path(
    output_path: Union[str, Path, None],
    image_path: Union[str, Path, None],
    prefix: str = "",
    ext: str = "jpg",
    default_dir: str = "runs/detections"
) -> Path:
    """
    Generate a save path handling both directory and file output paths.

    Args:
        output_path: User-provided output path (file or directory) or None
        image_path: Source image path for deriving filename
        prefix: Optional prefix for the filename (e.g., "tiled_")
        ext: File extension without dot (default: "jpg")
        default_dir: Default directory if output_path is None

    Returns:
        Resolved Path object ready for saving
    """
    from datetime import datetime

    # Get stem from image path or use default
    if image_path is not None:
        stem = get_safe_stem(image_path)
    else:
        stem = "inference"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}{stem}_{timestamp}.{ext}"

    if output_path is None:
        save_dir = Path(default_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / filename

    save_path = Path(output_path)

    if save_path.suffix == "":
        # output_path is a directory
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path / filename
    else:
        # output_path is a file
        save_path.parent.mkdir(parents=True, exist_ok=True)
        return save_path


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.45) -> torch.Tensor:
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

    # Sort by scores (descending)
    _, order = scores.sort(0, descending=True)
    keep = []

    while len(order) > 0:
        # Keep the box with highest score
        i = order[0]
        keep.append(i.item())

        if len(order) == 1:
            break

        # Calculate IoU with remaining boxes
        box_i = boxes[i]
        boxes_remaining = boxes[order[1:]]

        # Calculate intersection
        x1_i, y1_i, x2_i, y2_i = box_i
        x1_r, y1_r, x2_r, y2_r = boxes_remaining[:, 0], boxes_remaining[:, 1], boxes_remaining[:, 2], boxes_remaining[:, 3]

        x1_inter = torch.max(x1_i, x1_r)
        y1_inter = torch.max(y1_i, y1_r)
        x2_inter = torch.min(x2_i, x2_r)
        y2_inter = torch.min(y2_i, y2_r)

        inter_area = torch.clamp(x2_inter - x1_inter, min=0) * torch.clamp(y2_inter - y1_inter, min=0)

        # Calculate union
        area_i = (x2_i - x1_i) * (y2_i - y1_i)
        area_r = (x2_r - x1_r) * (y2_r - y1_r)
        union_area = area_i + area_r - inter_area

        # Calculate IoU
        iou = inter_area / (union_area + 1e-7)

        # Keep boxes with IoU < threshold
        order = order[1:][iou < iou_threshold]

    keep_tensor = torch.tensor(keep, dtype=torch.long, device=boxes.device)

    # Map back to original indices if we filtered out invalid boxes
    if valid_indices is not None:
        keep_tensor = valid_indices[keep_tensor]

    return keep_tensor


def make_anchors(
    feats: List[torch.Tensor],
    strides: List[int],
    grid_cell_offset: float = 0.5
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

    for i, (feat, stride) in enumerate(zip(feats, strides)):
        _, _, h, w = feat.shape
        dtype, device = feat.dtype, feat.device

        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx, indexing='ij')
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
    original_size: Tuple[int, int] = None,
    max_det: int = 300
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
        
    Returns:
        Dictionary with boxes, scores, classes, num_detections
    """
    if len(boxes) == 0:
        return {
            "boxes": [],
            "scores": [],
            "classes": [],
            "num_detections": 0
        }
    
    # Scale boxes to original image size
    if original_size is not None:
        scale_x = original_size[0] / input_size
        scale_y = original_size[1] / input_size
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        
        # Clip to image bounds
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], 0, original_size[0])
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], 0, original_size[1])
        
        # Filter invalid boxes
        box_widths = boxes[:, 2] - boxes[:, 0]
        box_heights = boxes[:, 3] - boxes[:, 1]
        valid_mask = (box_widths > 0) & (box_heights > 0)
        
        if not valid_mask.all():
            boxes = boxes[valid_mask]
            scores = scores[valid_mask]
            class_ids = class_ids[valid_mask]
    
    if len(boxes) == 0:
        return {
            "boxes": [],
            "scores": [],
            "classes": [],
            "num_detections": 0
        }
    
    # Apply per-class NMS
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
        return {
            "boxes": [],
            "scores": [],
            "classes": [],
            "num_detections": 0
        }
    
    keep_indices = torch.cat(keep_indices_list)
    
    # Limit to max detections
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
        "num_detections": len(final_boxes)
    }

