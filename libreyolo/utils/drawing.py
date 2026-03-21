"""Drawing utility functions for visualization."""

import colorsys
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .general import COCO_CLASSES


def _get_class_color_rgb(class_id: int) -> Tuple[int, int, int]:
    """Get a unique, consistent color for a class ID as (R, G, B) ints."""
    hue = (class_id * 137.508) % 360 / 360.0  # golden angle approximation
    saturation = 0.7 + (class_id % 3) * 0.1
    value = 0.8 + (class_id % 2) * 0.15
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return int(r * 255), int(g * 255), int(b * 255)


def get_class_color(class_id: int) -> str:
    """Get a unique, consistent color for a class ID as hex string."""
    r, g, b = _get_class_color_rgb(class_id)
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_boxes(
    img: Image.Image,
    boxes: List,
    scores: List,
    classes: List,
    class_names: List[str] | Dict[int, str] | None = None,
    track_ids: List | None = None,
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
        class_names: Optional class-name container, either a list indexed by class
            ID or a dict mapping class ID to class name (default: COCO_CLASSES)
        track_ids: Optional list of track IDs. When provided, each box is
            colored by its track ID and the label includes ``ID:<n>``.

    Returns:
        Annotated PIL Image
    """
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    if class_names is None:
        class_names = COCO_CLASSES

    # Scale factor: base sizes at 640px, scales up for larger images
    img_width, img_height = img.size
    max_dim = max(img_width, img_height)
    scale_factor = max_dim / 640.0
    box_thickness = max(2, int(2 * scale_factor))
    font_size = max(12, int(12 * scale_factor))

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except OSError:
        try:
            # Linux fallback
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except OSError:
            font = ImageFont.load_default()

    label_padding = max(2, int(2 * scale_factor))

    _track_ids = track_ids or [None] * len(boxes)

    for box, score, cls_id, tid in zip(boxes, scores, classes, _track_ids):
        x1, y1, x2, y2 = box
        cls_id_int = int(cls_id)

        # Color by track ID when tracking, otherwise by class ID.
        color = get_class_color(int(tid)) if tid is not None else get_class_color(cls_id_int)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_thickness)

        # Tracking mode: short two-tone label  "#23 0.87"
        # Detection mode: full label           "person: 0.87"
        if tid is not None:
            id_text = f"#{int(tid)}"
            conf_text = f" {score:.2f}"
            # Measure both parts separately for two-tone rendering.
            id_bbox = draw.textbbox((0, 0), id_text, font=font)
            full_label = id_text + conf_text
            full_bbox = draw.textbbox((0, 0), full_label, font=font)
            text_width = full_bbox[2] - full_bbox[0]
            text_height = full_bbox[3] - full_bbox[1]
            id_width = id_bbox[2] - id_bbox[0]
        else:
            class_name = None
            if isinstance(class_names, dict):
                class_name = class_names.get(cls_id_int)
            elif class_names and cls_id_int < len(class_names):
                class_name = class_names[cls_id_int]

            if class_name is not None:
                full_label = f"{class_name}: {score:.2f}"
            else:
                full_label = f"Class {cls_id_int}: {score:.2f}"
            full_bbox = draw.textbbox((0, 0), full_label, font=font)
            text_width = full_bbox[2] - full_bbox[0]
            text_height = full_bbox[3] - full_bbox[1]

        # Check if label fits above box; if not, draw inside
        outside = y1 >= text_height + label_padding * 2

        # Clamp label x to stay within image bounds
        label_x = min(x1, img_width - text_width - label_padding * 2)
        label_x = max(0, label_x)

        if outside:
            bg_y0 = y1 - text_height - label_padding * 2
            bg_y1 = y1
            text_y = y1 - text_height - label_padding
        else:
            bg_y0 = y1
            bg_y1 = y1 + text_height + label_padding * 2
            text_y = y1 + label_padding

        draw.rectangle(
            [label_x, bg_y0, label_x + text_width + label_padding * 2, bg_y1],
            fill=color,
        )

        if tid is not None:
            # Two-tone: track ID in yellow, confidence in white
            draw.text(
                (label_x + label_padding, text_y),
                id_text,
                fill="#FFFF00",
                font=font,
            )
            draw.text(
                (label_x + label_padding + id_width, text_y),
                conf_text,
                fill="#DDDDDD",
                font=font,
            )
        else:
            draw.text(
                (label_x + label_padding, text_y),
                full_label,
                fill="white",
                font=font,
            )

    return img_draw


def draw_masks(
    img: Image.Image,
    masks: np.ndarray,
    classes: List,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Draw semi-transparent instance segmentation masks on image.

    Args:
        img: PIL Image to draw on.
        masks: (N, H, W) boolean numpy array of instance masks.
        classes: List of class IDs (one per mask).
        alpha: Mask opacity (0 = transparent, 1 = opaque).

    Returns:
        Annotated PIL Image with mask overlays.
    """
    img_draw = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", img_draw.size, (0, 0, 0, 0))

    alpha_int = int(alpha * 255)

    for mask, cls_id in zip(masks, classes):
        r, g, b = _get_class_color_rgb(int(cls_id))

        # Create colored mask layer
        mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        mask_rgba[mask > 0] = (r, g, b, alpha_int)

        mask_img = Image.fromarray(mask_rgba, mode="RGBA")
        overlay = Image.alpha_composite(overlay, mask_img)

    result = Image.alpha_composite(img_draw, overlay)
    return result.convert("RGB")


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

    max_dim = max(img.size)
    scale_factor = max_dim / 640.0
    scaled_width = max(2, min(int(line_width * scale_factor), 10))

    for x1, y1, x2, y2 in tile_coords:
        draw.rectangle([x1, y1, x2, y2], outline=line_color, width=scaled_width)

    return img_draw
