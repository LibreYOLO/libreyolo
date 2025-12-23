"""
Shared utility functions.
"""
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Union
from PIL import Image, ImageDraw, ImageFont
import colorsys

from .image_loader import ImageLoader, ImageInput

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

