"""
Drawing utility functions for visualization.
"""

from typing import Tuple, List
from PIL import Image, ImageDraw, ImageFont
import colorsys

from .general import COCO_CLASSES


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
    return f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"


def draw_boxes(
    img: Image.Image,
    boxes: List,
    scores: List,
    classes: List,
    class_names: List | None = None,
) -> Image.Image:
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
    except OSError:
        try:
            # Try common Linux fonts
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except OSError:
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
        draw.rectangle(
            [
                x1,
                y1 - text_height - label_padding * 2,
                x1 + text_width + label_padding * 2,
                y1,
            ],
            fill=color,
        )

        # Draw label text
        draw.text(
            (x1 + label_padding, y1 - text_height - label_padding),
            label,
            fill="white",
            font=font,
        )

    return img_draw


def draw_tile_grid(
    img: Image.Image,
    tile_coords: List[Tuple[int, int, int, int]],
    line_color: str = "#FF0000",
    line_width: int = 3,
) -> Image.Image:
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
