"""
Utility functions for preprocessing, postprocessing, and visualization.
"""

import torch
import numpy as np
from typing import Union, Tuple, Dict, List
from PIL import Image, ImageDraw, ImageFont

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
    import colorsys
    hue = (class_id * 137.508) % 360 / 360.0  # Golden angle approximation
    saturation = 0.7 + (class_id % 3) * 0.1  # Vary saturation
    value = 0.8 + (class_id % 2) * 0.15  # Vary brightness
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"


def preprocess_image(image: Union[str, Image.Image, np.ndarray], input_size: int = 640) -> Tuple[torch.Tensor, Image.Image, Tuple[int, int]]:
    """
    Preprocess image for model inference.
    
    Args:
        image: Input image as file path (str), PIL Image, or numpy array
        input_size: Target size for resizing (default: 640)
        
    Returns:
        Tuple of (preprocessed_tensor, original_image, original_size)
    """
    # Load image
    if isinstance(image, str):
        img = Image.open(image).convert('RGB')
    elif isinstance(image, Image.Image):
        img = image.convert('RGB')
    elif isinstance(image, np.ndarray):
        img = Image.fromarray(image).convert('RGB')
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    original_size = img.size  # (width, height)
    original_img = img.copy()
    
    # Resize to input_size (simple resize, maintain aspect ratio could be added)
    img_resized = img.resize((input_size, input_size), Image.Resampling.BILINEAR)
    
    # Convert to numpy and normalize
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    
    # Convert to tensor: HWC -> CHW -> add batch dimension
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, original_img, original_size


def make_anchors(feats: List[torch.Tensor], strides: List[int], grid_cell_offset: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
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


def decode_boxes(box_preds: torch.Tensor, anchors: torch.Tensor, stride_tensor: torch.Tensor) -> torch.Tensor:
    """
    Decode box predictions to xyxy coordinates.
    
    Args:
        box_preds: Box predictions [l, t, r, b] distances from anchors (B, N, 4)
        anchors: Anchor points (N, 2)
        stride_tensor: Stride values (N, 1)
        
    Returns:
        Decoded boxes in xyxy format (B, N, 4)
    """
    # box_preds shape: (B, N, 4) where 4 = [l, t, r, b]
    # anchors shape: (N, 2) where 2 = [x, y]
    # stride_tensor shape: (N, 1)
    
    # Decode: xyxy = [x - l, y - t, x + r, y + b] * stride
    x1 = (anchors[:, 0:1] - box_preds[0, :, 0:1]) * stride_tensor[:, 0:1]
    y1 = (anchors[:, 1:2] - box_preds[0, :, 1:2]) * stride_tensor[:, 0:1]
    x2 = (anchors[:, 0:1] + box_preds[0, :, 2:3]) * stride_tensor[:, 0:1]
    y2 = (anchors[:, 1:2] + box_preds[0, :, 3:4]) * stride_tensor[:, 0:1]
    
    decoded_boxes = torch.cat([x1, y1, x2, y2], dim=1)
    return decoded_boxes


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
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def postprocess(output: Dict, conf_thres: float = 0.25, iou_thres: float = 0.45, input_size: int = 640, original_size: Tuple[int, int] = None) -> Dict:
    """
    Postprocess model outputs to get final detections.
    
    Args:
        output: Model output dictionary with 'x8', 'x16', 'x32' keys
        conf_thres: Confidence threshold (default: 0.25)
        iou_thres: IoU threshold for NMS (default: 0.45)
        input_size: Input image size (default: 640)
        original_size: Original image size (width, height) for scaling
        
    Returns:
        Dictionary with boxes, scores, classes, num_detections
    """
    # Collect outputs from the 3 heads
    box_layers = [output['x8']['box'], output['x16']['box'], output['x32']['box']]
    cls_layers = [output['x8']['cls'], output['x16']['cls'], output['x32']['cls']]
    strides = [8, 16, 32]
    
    # Generate anchors
    anchors, stride_tensor = make_anchors(box_layers, strides)
    
    # Flatten and concatenate predictions
    # Box: (Batch, 4, H, W) -> (Batch, H*W, 4)
    box_preds = torch.cat([x.flatten(2).permute(0, 2, 1) for x in box_layers], dim=1)
    # Cls: (Batch, 80, H, W) -> (Batch, H*W, 80)
    cls_preds = torch.cat([x.flatten(2).permute(0, 2, 1) for x in cls_layers], dim=1)
    
    # Decode boxes
    decoded_boxes = decode_boxes(box_preds, anchors, stride_tensor)
    
    # Apply confidence threshold
    scores = cls_preds[0].sigmoid()  # (N, 80)
    max_scores, class_ids = torch.max(scores, dim=1)
    
    mask = max_scores > conf_thres
    if not mask.any():
        return {
            "boxes": [],
            "scores": [],
            "classes": [],
            "num_detections": 0
        }
    
    valid_boxes = decoded_boxes[mask]
    valid_scores = max_scores[mask]
    valid_classes = class_ids[mask]
    
    # Scale boxes to original image size if provided
    if original_size is not None:
        scale_x = original_size[0] / input_size
        scale_y = original_size[1] / input_size
        valid_boxes[:, [0, 2]] *= scale_x
        valid_boxes[:, [1, 3]] *= scale_y
    
    # Apply NMS
    keep_indices = nms(valid_boxes, valid_scores, iou_thres)
    
    final_boxes = valid_boxes[keep_indices].cpu().numpy()
    final_scores = valid_scores[keep_indices].cpu().numpy()
    final_classes = valid_classes[keep_indices].cpu().numpy()
    
    return {
        "boxes": final_boxes.tolist(),
        "scores": final_scores.tolist(),
        "classes": final_classes.tolist(),
        "num_detections": len(final_boxes)
    }


def draw_boxes(img: Image.Image, boxes: List, scores: List, classes: List, class_names: List = None) -> Image.Image:
    """
    Draw bounding boxes on image with class-specific colors.
    
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
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font = ImageFont.load_default()
    
    for box, score, cls_id in zip(boxes, scores, classes):
        x1, y1, x2, y2 = box
        cls_id_int = int(cls_id)
        
        # Get class-specific color
        color = get_class_color(cls_id_int)
        
        # Draw box with class color
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Prepare label
        if class_names and cls_id_int < len(class_names):
            label = f"{class_names[cls_id_int]}: {score:.2f}"
        else:
            label = f"Class {cls_id_int}: {score:.2f}"
        
        # Get text size
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw label background with class color
        draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
        
        # Draw label text
        draw.text((x1 + 2, y1 - text_height - 2), label, fill="white", font=font)
    
    return img_draw

