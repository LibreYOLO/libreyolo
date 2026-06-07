"""Drawing utility functions for visualization."""

import colorsys
import math
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .general import COCO_CLASSES


@lru_cache(maxsize=16)
def _get_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load and cache a font at the given size."""
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except OSError:
        try:
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except OSError:
            return ImageFont.load_default()


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

    font = _get_font(font_size)

    label_padding = max(2, int(2 * scale_factor))

    _track_ids = track_ids or [None] * len(boxes)

    for box, score, cls_id, tid in zip(boxes, scores, classes, _track_ids):
        x1, y1, x2, y2 = box
        cls_id_int = int(cls_id)

        # Color by track ID when tracking, otherwise by class ID.
        color = (
            get_class_color(int(tid))
            if tid is not None
            else get_class_color(cls_id_int)
        )

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


def _xywhr_to_points(row: Sequence[float]) -> List[Tuple[float, float]]:
    cx, cy, w, h, angle = (float(v) for v in row[:5])
    half_w = w / 2.0
    half_h = h / 2.0
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    offsets = (
        (-half_w, -half_h),
        (half_w, -half_h),
        (half_w, half_h),
        (-half_w, half_h),
    )
    return [
        (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)
        for dx, dy in offsets
    ]


def draw_obb(
    img: Image.Image,
    obb: Sequence[Sequence[float]],
    scores: Sequence[float],
    classes: Sequence[float],
    class_names: List[str] | Dict[int, str] | None = None,
    track_ids: Sequence[float] | None = None,
) -> Image.Image:
    """Draw oriented bounding boxes as rotated polygons."""
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    if class_names is None:
        class_names = COCO_CLASSES

    img_width, img_height = img.size
    max_dim = max(img_width, img_height)
    scale_factor = max_dim / 640.0
    box_thickness = max(2, int(2 * scale_factor))
    font_size = max(12, int(12 * scale_factor))
    label_padding = max(2, int(2 * scale_factor))
    font = _get_font(font_size)
    _track_ids = list(track_ids) if track_ids is not None else [None] * len(obb)

    for row, score, cls_id, tid in zip(obb, scores, classes, _track_ids):
        cls_id_int = int(cls_id)
        color = (
            get_class_color(int(tid))
            if tid is not None
            else get_class_color(cls_id_int)
        )

        points = _xywhr_to_points(row)
        draw.line(points + [points[0]], fill=color, width=box_thickness)

        if tid is not None:
            label = f"#{int(tid)} {float(score):.2f}"
        else:
            class_name = None
            if isinstance(class_names, dict):
                class_name = class_names.get(cls_id_int)
            elif class_names and cls_id_int < len(class_names):
                class_name = class_names[cls_id_int]
            label = (
                f"{class_name}: {float(score):.2f}"
                if class_name is not None
                else f"Class {cls_id_int}: {float(score):.2f}"
            )

        full_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = full_bbox[2] - full_bbox[0]
        text_height = full_bbox[3] - full_bbox[1]
        x_values = [p[0] for p in points]
        y_values = [p[1] for p in points]
        x1 = min(x_values)
        y1 = min(y_values)

        outside = y1 >= text_height + label_padding * 2
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
        draw.text(
            (label_x + label_padding, text_y),
            label,
            fill="white",
            font=font,
        )

    return img_draw


def draw_points(
    img: Image.Image,
    points: Sequence[Sequence[float]],
    scores: Sequence[float],
    classes: Sequence[float],
    class_names: List[str] | Dict[int, str] | None = None,
) -> Image.Image:
    """Draw point-localization predictions as labeled centroids."""
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    if class_names is None:
        class_names = COCO_CLASSES

    max_dim = max(img.size)
    scale = max_dim / 640.0
    radius = max(3, int(round(4 * scale)))
    stroke = max(2, int(round(2 * scale)))
    font = _get_font(max(12, int(12 * scale)))
    label_padding = max(2, int(2 * scale))

    for point, score, cls_id in zip(points, scores, classes):
        x, y = float(point[0]), float(point[1])
        cls_id_int = int(cls_id)
        color = get_class_color(cls_id_int)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline=(0, 0, 0),
            width=stroke,
        )
        draw.line([(x - radius * 1.5, y), (x + radius * 1.5, y)], fill=(0, 0, 0), width=1)
        draw.line([(x, y - radius * 1.5), (x, y + radius * 1.5)], fill=(0, 0, 0), width=1)

        if isinstance(class_names, dict):
            class_name = class_names.get(cls_id_int)
        elif class_names and cls_id_int < len(class_names):
            class_name = class_names[cls_id_int]
        else:
            class_name = None
        label = (
            f"{class_name}: {float(score):.2f}"
            if class_name is not None
            else f"Class {cls_id_int}: {float(score):.2f}"
        )

        full_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = full_bbox[2] - full_bbox[0]
        text_height = full_bbox[3] - full_bbox[1]
        label_x = min(max(0, x + radius + label_padding), img.width - text_width - label_padding * 2)
        label_y = min(max(0, y - text_height / 2 - label_padding), img.height - text_height - label_padding * 2)
        draw.rectangle(
            [
                label_x,
                label_y,
                label_x + text_width + label_padding * 2,
                label_y + text_height + label_padding * 2,
            ],
            fill=color,
        )
        draw.text(
            (label_x + label_padding, label_y + label_padding),
            label,
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


# COCO 17-keypoint skeleton + colors (matches super-gradients defaults).
COCO_KEYPOINT_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)
COCO_KEYPOINT_COLOR: Tuple[int, int, int] = (51, 153, 255)
COCO_EDGE_COLOR: Tuple[int, int, int] = (255, 128, 0)


def draw_keypoints(
    img: Image.Image,
    keypoints: np.ndarray,
    edges: Tuple[Tuple[int, int], ...] = COCO_KEYPOINT_EDGES,
    point_color: Tuple[int, int, int] = COCO_KEYPOINT_COLOR,
    edge_color: Tuple[int, int, int] = COCO_EDGE_COLOR,
    point_radius: int | None = None,
    edge_width: int | None = None,
    conf_thres: float = 0.5,
) -> Image.Image:
    """Draw keypoints + skeleton edges for one or more instances.

    Args:
        img: PIL image to draw on.
        keypoints: ``(N, K, 2)`` or ``(N, K, 3)`` array. The third channel,
            when present, is per-keypoint confidence; keypoints with
            ``conf < conf_thres`` are skipped.
        edges: Pairs of keypoint indices to connect.
        point_color: RGB color for keypoint dots.
        edge_color: RGB color for skeleton edges.
        point_radius: Dot radius in pixels (auto-scaled if None).
        edge_width: Edge line width in pixels (auto-scaled if None).
        conf_thres: Per-keypoint confidence cutoff for visibility.
    """
    arr = np.asarray(keypoints)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.size == 0:
        return img

    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    img_diag = (img.width ** 2 + img.height ** 2) ** 0.5
    if point_radius is None:
        point_radius = max(2, int(round(img_diag / 400)))
    if edge_width is None:
        edge_width = max(1, int(round(img_diag / 600)))

    has_conf = arr.shape[-1] >= 3

    for instance in arr:
        visible = (
            instance[:, 2] >= conf_thres if has_conf
            else np.ones(instance.shape[0], dtype=bool)
        )
        for a, b in edges:
            if a >= len(instance) or b >= len(instance):
                continue
            if not (visible[a] and visible[b]):
                continue
            xa, ya = float(instance[a, 0]), float(instance[a, 1])
            xb, yb = float(instance[b, 0]), float(instance[b, 1])
            draw.line([(xa, ya), (xb, yb)], fill=edge_color, width=edge_width)
        for k, (x, y) in enumerate(instance[:, :2]):
            if not visible[k]:
                continue
            cx, cy = float(x), float(y)
            draw.ellipse(
                [cx - point_radius, cy - point_radius,
                 cx + point_radius, cy + point_radius],
                fill=point_color,
                outline=(0, 0, 0),
            )
    return img_draw


def draw_gaze_arrows(
    img: Image.Image,
    boxes: Sequence[Sequence[float]],
    pitch_rad: Sequence[float],
    yaw_rad: Sequence[float],
    color: Tuple[int, int, int] = (0, 200, 255),
    arrow_length_ratio: float = 0.6,
    arrow_thickness: int | None = None,
) -> Image.Image:
    """Draw a gaze direction arrow per face on the image.

    The arrow originates at the face bbox center and points in the gaze
    direction. The 2D projection is the standard appearance-based-gaze
    formula: ``dx = -L * sin(yaw) * cos(pitch)``, ``dy = -L * sin(pitch)``
    (horizontal displacement driven by yaw, vertical by pitch).

    Args:
        img: PIL Image (RGB) to draw on.
        boxes: Iterable of ``(x1, y1, x2, y2)`` face boxes — one per face.
        pitch_rad: Per-face pitch angle in radians.
        yaw_rad: Per-face yaw angle in radians.
        color: RGB tuple for the arrow.
        arrow_length_ratio: Arrow length as a fraction of the face bbox's
            smaller side. Default 0.6 — long enough to read, short enough to
            stay inside crowded scenes.
        arrow_thickness: Optional override for line thickness. When None,
            scales with image size to stay legible on both webcams and 4K.

    Returns:
        New PIL Image with arrows drawn on top of the input.
    """
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    max_dim = max(img.size)
    scale = max_dim / 640.0
    thickness = (
        arrow_thickness if arrow_thickness is not None else max(2, int(3 * scale))
    )

    head_color = color

    for box, pitch, yaw in zip(boxes, pitch_rad, yaw_rad):
        x1, y1, x2, y2 = (float(v) for v in box)
        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        cx = x1 + w / 2.0
        cy = y1 + h / 2.0
        length = arrow_length_ratio * min(w, h)
        # Standard gaze projection: yaw drives horizontal, pitch drives vertical.
        dx = -length * math.sin(float(yaw)) * math.cos(float(pitch))
        dy = -length * math.sin(float(pitch))
        ex = cx + dx
        ey = cy + dy

        draw.line([(cx, cy), (ex, ey)], fill=color, width=thickness)

        # Arrowhead: two short segments rotated ±25° from the shaft direction.
        head_len = max(6.0, length * 0.18)
        shaft_angle = math.atan2(dy, dx)
        for side in (1, -1):
            angle = shaft_angle + side * math.radians(150.0)
            hx = ex + head_len * math.cos(angle)
            hy = ey + head_len * math.sin(angle)
            draw.line([(ex, ey), (hx, hy)], fill=head_color, width=thickness)

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

    max_dim = max(img.size)
    scale_factor = max_dim / 640.0
    scaled_width = max(2, min(int(line_width * scale_factor), 10))

    for x1, y1, x2, y2 in tile_coords:
        draw.rectangle([x1, y1, x2, y2], outline=line_color, width=scaled_width)

    return img_draw
