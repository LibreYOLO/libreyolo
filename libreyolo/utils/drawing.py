"""Drawing utility functions for visualization."""

import colorsys
import math
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .general import COCO_CLASSES

FONT_CANDIDATES = (
    "arial.ttf",
    "segoeui.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)


@lru_cache(maxsize=16)
def _get_font(font_size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load and cache a font at the given size."""
    for font in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font, font_size)
        except OSError:
            continue

    try:
        return ImageFont.load_default(size=font_size)
    except TypeError:
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
        draw.line(
            [(x - radius * 1.5, y), (x + radius * 1.5, y)], fill=(0, 0, 0), width=1
        )
        draw.line(
            [(x, y - radius * 1.5), (x, y + radius * 1.5)], fill=(0, 0, 0), width=1
        )

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
        label_x = min(
            max(0, x + radius + label_padding),
            img.width - text_width - label_padding * 2,
        )
        label_y = min(
            max(0, y - text_height / 2 - label_padding),
            img.height - text_height - label_padding * 2,
        )
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


# Fonts that can render CJK glyphs, tried in order for OCR transcripts.
# The default label fonts (Arial/DejaVu) draw tofu boxes for Chinese and
# Japanese text, which PP-OCR transcripts routinely contain.
CJK_FONT_CANDIDATES = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)

_warned_no_cjk_font = False


@lru_cache(maxsize=8)
def _get_cjk_font(font_size: int) -> ImageFont.FreeTypeFont | None:
    """Load and cache a CJK-capable font, or ``None`` if the system has none."""
    for font in CJK_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(font, font_size)
        except OSError:
            continue
    return None


def _text_needs_cjk(text: str) -> bool:
    return any(
        "⺀" <= ch <= "鿿"
        or "぀" <= ch <= "ヿ"
        or "豈" <= ch <= "﫿"
        or "＀" <= ch <= "￯"
        for ch in text
    )


def draw_ocr_regions(
    img: Image.Image,
    polygons: Sequence,
    texts: Sequence[str],
    scores: Sequence[float],
) -> Image.Image:
    """Draw OCR text-region polygons and render each transcript nearby.

    Transcripts containing CJK characters need a CJK-capable font; when the
    system has none, boxes are still drawn and a single warning is logged.
    """
    global _warned_no_cjk_font
    import logging

    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    max_dim = max(img.size)
    scale = max_dim / 640.0
    stroke = max(2, int(round(2 * scale)))
    font_size = max(12, int(14 * scale))
    plain_font = _get_font(font_size)
    cjk_font = _get_cjk_font(font_size)
    label_padding = max(2, int(2 * scale))
    color = get_class_color(0)

    for polygon, text, score in zip(polygons, texts, scores):
        pts = [(float(p[0]), float(p[1])) for p in polygon]
        draw.polygon(pts, outline=color, width=stroke)

        label = f"{text} {float(score):.2f}" if text else f"{float(score):.2f}"
        font = plain_font
        if _text_needs_cjk(label):
            if cjk_font is not None:
                font = cjk_font
            else:
                if not _warned_no_cjk_font:
                    logging.getLogger(__name__).warning(
                        "No CJK-capable font found on this system; OCR polygons "
                        "are drawn but CJK transcripts are omitted from the overlay."
                    )
                    _warned_no_cjk_font = True
                continue

        full_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = full_bbox[2] - full_bbox[0]
        text_height = full_bbox[3] - full_bbox[1]
        top_left_x = min(p[0] for p in pts)
        top_left_y = min(p[1] for p in pts)
        label_x = min(
            max(0, top_left_x), max(0, img.width - text_width - label_padding * 2)
        )
        label_y = top_left_y - text_height - label_padding * 2
        if label_y < 0:
            label_y = top_left_y
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


def draw_semantic_mask(
    img: Image.Image,
    semantic_mask: np.ndarray,
    alpha: float = 0.55,
    ignore_index: int = 255,
) -> Image.Image:
    """
    Overlay a dense semantic class map on an image.

    Args:
        img: PIL Image to draw on.
        semantic_mask: (H, W) integer numpy array of per-pixel class IDs.
        alpha: Overlay opacity (0 = transparent, 1 = opaque).
        ignore_index: Class value left unpainted.

    Returns:
        Annotated PIL Image with the class-color overlay.
    """
    mask = np.asarray(semantic_mask)
    if mask.shape[:2] != (img.height, img.width):
        mask_img = Image.fromarray(mask.astype(np.int32), mode="I")
        mask_img = mask_img.resize((img.width, img.height), Image.NEAREST)
        mask = np.asarray(mask_img)

    img_draw = img.copy().convert("RGBA")
    overlay = np.zeros((img.height, img.width, 4), dtype=np.uint8)
    alpha_int = int(alpha * 255)
    for cls_id in np.unique(mask):
        cls_id = int(cls_id)
        if cls_id == ignore_index:
            continue
        r, g, b = _get_class_color_rgb(cls_id)
        overlay[mask == cls_id] = (r, g, b, alpha_int)

    result = Image.alpha_composite(img_draw, Image.fromarray(overlay, mode="RGBA"))
    return result.convert("RGB")


def draw_panoptic(
    img: Image.Image,
    panoptic_map: np.ndarray,
    segments_info: List[Dict],
    class_names: Dict[int, str] | None = None,
    alpha: float = 0.55,
    ignore_index: int = 0,
) -> Image.Image:
    """
    Overlay a dense panoptic segment-id map on an image.

    Thing segments are colored per segment id (so touching instances of the
    same class stay distinguishable); stuff segments are colored per category.
    Segments covering at least 0.5% of the image get a class-name label at
    their centroid.

    Args:
        img: PIL Image to draw on.
        panoptic_map: (H, W) integer numpy array of per-pixel segment IDs.
        segments_info: One dict per segment with at least ``id``,
            ``category_id``, and ``isthing``.
        class_names: Optional mapping of category ID to class name.
        alpha: Overlay opacity (0 = transparent, 1 = opaque).
        ignore_index: Segment ID left unpainted (COCO convention: 0 = void).

    Returns:
        Annotated PIL Image with the segment-color overlay and labels.
    """
    seg_map = np.asarray(panoptic_map)
    if seg_map.shape[:2] != (img.height, img.width):
        seg_img = Image.fromarray(seg_map.astype(np.int32), mode="I")
        seg_img = seg_img.resize((img.width, img.height), Image.NEAREST)
        seg_map = np.asarray(seg_img)

    img_draw = img.copy().convert("RGBA")
    overlay = np.zeros((img.height, img.width, 4), dtype=np.uint8)
    alpha_int = int(alpha * 255)
    info_by_id = {int(seg["id"]): seg for seg in segments_info}

    labels: List[Tuple[str, Tuple[int, int], Tuple[int, int, int]]] = []
    min_label_area = 0.005 * seg_map.size
    for seg_id in np.unique(seg_map):
        seg_id = int(seg_id)
        if seg_id == ignore_index:
            continue
        seg = info_by_id.get(seg_id)
        if seg is not None and not seg.get("isthing", True):
            color = _get_class_color_rgb(int(seg["category_id"]))
        else:
            # Things (and unlisted segments) vary by segment id so adjacent
            # instances of one class do not blend together.
            color = _get_class_color_rgb(seg_id * 3 + 1)
        region = seg_map == seg_id
        overlay[region] = (*color, alpha_int)

        if seg is not None and class_names and region.sum() >= min_label_area:
            name = class_names.get(int(seg["category_id"]))
            if name:
                ys, xs = np.nonzero(region)
                labels.append((str(name), (int(xs.mean()), int(ys.mean())), color))

    result = Image.alpha_composite(img_draw, Image.fromarray(overlay, mode="RGBA"))
    result = result.convert("RGB")

    if labels:
        draw = ImageDraw.Draw(result)
        font_size = max(12, min(img.width, img.height) // 40)
        font = _get_font(font_size)
        for name, (cx, cy), color in labels:
            bbox = draw.textbbox((0, 0), name, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = max(0, min(img.width - tw, cx - tw // 2))
            y = max(0, min(img.height - th, cy - th // 2))
            draw.rectangle([x - 3, y - 2, x + tw + 3, y + th + 2], fill=(15, 23, 42))
            draw.text((x, y), name, fill="white", font=font)

    return result


# Anchor colors for the depth colormap, near (warm) to far (cold). Linear
# interpolation between anchors gives a smooth ramp without a matplotlib
# dependency.
_DEPTH_COLOR_ANCHORS: Tuple[Tuple[int, int, int], ...] = (
    (122, 4, 3),
    (228, 65, 26),
    (249, 152, 40),
    (164, 252, 60),
    (58, 222, 130),
    (32, 144, 222),
    (64, 67, 166),
    (48, 18, 59),
)


@lru_cache(maxsize=1)
def _depth_colormap_lut() -> np.ndarray:
    """256-entry RGB lookup table interpolated between depth anchors."""
    anchors = np.asarray(_DEPTH_COLOR_ANCHORS, dtype=np.float64)
    positions = np.linspace(0.0, 1.0, len(anchors))
    samples = np.linspace(0.0, 1.0, 256)
    lut = np.stack(
        [np.interp(samples, positions, anchors[:, c]) for c in range(3)], axis=1
    )
    return lut.round().astype(np.uint8)


def draw_depth_map(
    img: Image.Image,
    depth_map: np.ndarray,
    alpha: float = 1.0,
) -> Image.Image:
    """Render a relative inverse-depth map as a colormapped image."""
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.shape[:2] != (img.height, img.width):
        depth_img = Image.fromarray(depth, mode="F")
        depth_img = depth_img.resize((img.width, img.height), Image.BILINEAR)
        depth = np.asarray(depth_img, dtype=np.float32)

    finite = np.isfinite(depth)
    normalized = np.zeros_like(depth, dtype=np.float32)
    if finite.any():
        values = depth[finite]
        lo = float(values.min())
        hi = float(values.max())
        if hi - lo > 0:
            normalized[finite] = (values - lo) / (hi - lo)
    # Higher values are closer; index 0 of the LUT is the near anchor.
    indices = ((1.0 - normalized) * 255).round().astype(np.uint8)
    colored = _depth_colormap_lut()[indices]
    colored[~finite] = 0

    result = Image.fromarray(colored, mode="RGB")
    if alpha < 1.0:
        result = Image.blend(img.convert("RGB"), result, alpha)
    return result


def draw_edge_map(
    img: Image.Image,
    edge_map: np.ndarray,
) -> Image.Image:
    """Render edge probabilities as inverted grayscale."""
    edges = np.asarray(edge_map, dtype=np.float32)
    if edges.ndim != 2:
        raise ValueError(f"expected an (H, W) edge map, got {edges.shape}")
    if edges.shape != (img.height, img.width):
        edge_image = Image.fromarray(edges, mode="F").resize(
            (img.width, img.height),
            Image.BILINEAR,
        )
        edges = np.asarray(edge_image, dtype=np.float32)
    grayscale = np.rint((1.0 - np.clip(edges, 0.0, 1.0)) * 255.0).astype(np.uint8)
    return Image.fromarray(grayscale, mode="L").convert("RGB")


def draw_normal_map(
    img: Image.Image,
    normal_map: np.ndarray,
    alpha: float = 1.0,
) -> Image.Image:
    """Render OpenCV-frame surface normals with the canonical RGB mapping.

    The payload stays as float vectors; this function alone applies
    ``rgb = (normal + 1) / 2``. If resizing is needed, vector components are
    interpolated independently and then renormalized.
    """
    normals = np.asarray(normal_map, dtype=np.float32)
    if normals.ndim != 3 or normals.shape[-1] != 3:
        raise ValueError(f"expected (H, W, 3) normal map but got shape {normals.shape}")
    if normals.shape[:2] != (img.height, img.width):
        components = []
        for component in range(3):
            component_img = Image.fromarray(normals[..., component], mode="F")
            component_img = component_img.resize(
                (img.width, img.height), Image.BILINEAR
            )
            components.append(np.asarray(component_img, dtype=np.float32))
        normals = np.stack(components, axis=-1)

    finite = np.isfinite(normals).all(axis=-1)
    norms = np.linalg.norm(np.where(finite[..., None], normals, 0.0), axis=-1)
    valid = finite & (norms > 1e-12)
    normalized = np.zeros_like(normals, dtype=np.float32)
    normalized[valid] = normals[valid] / norms[valid, None]

    colored = np.clip((normalized + 1.0) * 127.5, 0.0, 255.0).round()
    colored = colored.astype(np.uint8)
    colored[~valid] = 0
    result = Image.fromarray(colored, mode="RGB")
    if alpha < 1.0:
        result = Image.blend(img.convert("RGB"), result, alpha)
    return result


def _checkerboard(
    height: int,
    width: int,
    tile: int = 16,
    light: int = 200,
    dark: int = 154,
) -> np.ndarray:
    """Build an ``(H, W, 3)`` uint8 checkerboard, the standard transparency backdrop."""
    ys = (np.arange(height) // tile)[:, None]
    xs = (np.arange(width) // tile)[None, :]
    board = np.where((ys + xs) % 2 == 0, light, dark).astype(np.uint8)
    return np.repeat(board[:, :, None], 3, axis=2)


def draw_matte(
    img: Image.Image,
    matte: np.ndarray,
    tile: int = 16,
) -> Image.Image:
    """Preview a soft alpha matte by compositing the cutout over a checkerboard.

    Args:
        img: Source PIL image.
        matte: ``(H, W)`` float alpha in ``[0, 1]`` (foreground opacity).
        tile: Checkerboard tile size in pixels.

    Returns:
        RGB PIL image: foreground kept, background replaced by a checkerboard so
        the transparency (and soft hair/fur edges) is visible at a glance.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    alpha = np.asarray(matte, dtype=np.float32)
    if alpha.shape[:2] != (h, w):
        alpha_img = Image.fromarray(alpha, mode="F").resize((w, h), Image.BILINEAR)
        alpha = np.asarray(alpha_img, dtype=np.float32)
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
    board = _checkerboard(h, w, tile=tile).astype(np.float32)
    composite = rgb * alpha + board * (1.0 - alpha)
    return Image.fromarray(np.clip(composite, 0, 255).astype(np.uint8), mode="RGB")


# COCO 17-keypoint skeleton + colors (matches super-gradients defaults).
COCO_KEYPOINT_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
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

    img_diag = (img.width**2 + img.height**2) ** 0.5
    if point_radius is None:
        point_radius = max(2, int(round(img_diag / 400)))
    if edge_width is None:
        edge_width = max(1, int(round(img_diag / 600)))

    has_conf = arr.shape[-1] >= 3

    for instance in arr:
        visible = (
            instance[:, 2] >= conf_thres
            if has_conf
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
                [
                    cx - point_radius,
                    cy - point_radius,
                    cx + point_radius,
                    cy + point_radius,
                ],
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


# MHR-70 keypoint skeleton. Indices 0-16 follow the COCO body ordering
# exactly, 17-22 are the feet, 23-62 are the two hands and 63-69 are extra
# anatomical landmarks; only the body, feet and neck links are drawn, because
# finger edges collapse into noise at whole-image scale.
MHR70_SKELETON_EDGES: Tuple[Tuple[int, int], ...] = COCO_KEYPOINT_EDGES + (
    (15, 17),
    (15, 18),
    (15, 19),  # left ankle to big toe, small toe, heel
    (16, 20),
    (16, 21),
    (16, 22),  # right ankle to big toe, small toe, heel
    (63, 5),
    (63, 6),  # neck to shoulders
)
MESH_VERTEX_COLOR: Tuple[int, int, int] = (120, 200, 255)
# Neutral clay, the convention for body-mesh overlays: light enough to read
# against dark clothing, desaturated enough not to compete with the photo.
MESH_SURFACE_COLOR: Tuple[int, int, int] = (200, 202, 210)
# Light direction in camera space, pointing from the scene toward the viewer
# and slightly up-left, which is what makes limbs read as rounded.
_MESH_LIGHT_DIR = np.array([-0.35, -0.55, -0.75], dtype=np.float32)


def render_mesh_surface(
    img: Image.Image,
    vertices2d: np.ndarray,
    vertices3d: np.ndarray,
    faces: np.ndarray,
    color: Tuple[int, int, int] = MESH_SURFACE_COLOR,
    alpha: float = 0.9,
    ambient: float = 0.22,
    specular: float = 0.30,
    shininess: float = 16.0,
    shading: str = "diffuse",
) -> Image.Image:
    """Rasterize shaded body-mesh surfaces over an image.

    A small painter's-algorithm renderer: back-facing triangles are culled,
    the rest are sorted far-to-near and filled. This keeps a real surface
    render available without a GPU rasterizer dependency such as pyrender or
    PyTorch3D, neither of which installs cleanly everywhere LibreYOLO runs.

    Shading normals are computed from the **metric camera-space** vertices,
    not from screen coordinates. Mixing pixel units with metric depth makes
    every normal point at the camera, which flattens the render into what
    looks like a silhouette.

    Args:
        img: PIL image to draw on.
        vertices2d: ``(N, V, 2)`` projected vertices in pixels.
        vertices3d: ``(N, V, 3)`` camera-space metric vertices. Drives both
            the shading normals and the far-to-near draw order.
        faces: ``(F, 3)`` vertex indices, shared by every person.
        color: Base RGB of the surface, used by ``shading="diffuse"``.
        alpha: Blend weight of the rendered surface over the photo.
        ambient: Fraction of base color present in unlit areas.
        specular: Strength of the highlight that conveys curvature.
        shininess: Highlight tightness; larger is glossier.
        shading: ``"diffuse"`` for lit clay, or ``"normal"`` to colour each
            face by its normal direction, the convention papers use when the
            point is to show surface orientation rather than realism.
    """
    verts2d = np.asarray(vertices2d, dtype=np.float32)
    verts3d = np.asarray(vertices3d, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    if verts2d.ndim == 2:
        verts2d = verts2d[None, ...]
    if verts3d.ndim == 2:
        verts3d = verts3d[None, ...]
    if verts2d.size == 0 or faces.size == 0:
        return img.convert("RGB")

    overlay = img.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    base = np.asarray(color, dtype=np.float32)
    light = _MESH_LIGHT_DIR / np.linalg.norm(_MESH_LIGHT_DIR)

    for person_xy, person_xyz in zip(verts2d, verts3d):
        tri2d = np.stack([person_xy[faces[:, i]] for i in range(3)], axis=1)
        tri3d = np.stack([person_xyz[faces[:, i]] for i in range(3)], axis=1)

        # True geometric normals, in metres, from the camera-space mesh.
        normals = np.cross(tri3d[:, 1] - tri3d[:, 0], tri3d[:, 2] - tri3d[:, 0])
        norm_len = np.linalg.norm(normals, axis=1)
        valid = norm_len > 1e-12
        normals[valid] /= norm_len[valid, None]

        centroid = tri3d.mean(axis=1)
        view_len = np.linalg.norm(centroid, axis=1)
        view = centroid / np.maximum(view_len, 1e-9)[:, None]

        # A face is visible when its normal opposes the viewing direction.
        # Orient normals toward the camera so shading never depends on the
        # mesh's winding convention.
        facing = np.einsum("ij,ij->i", normals, view)
        normals[facing > 0] *= -1
        front = valid & (np.abs(facing) > 1e-6)
        if not front.any():
            continue
        # Screen-space winding is the reliable visibility test for a closed
        # body: keep the consistently-wound half.
        edge1 = tri2d[:, 1] - tri2d[:, 0]
        edge2 = tri2d[:, 2] - tri2d[:, 0]
        signed_area = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
        for sign in (-1.0, 1.0):
            candidate = front & (signed_area * sign > 0)
            if candidate.sum() > front.sum() * 0.25:
                front = candidate
                break

        n = normals[front]
        v = view[front]
        if shading == "normal":
            colors = np.clip((n * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        else:
            diffuse = np.clip(n @ light, 0.0, 1.0)
            # Blinn-Phong highlight: the half-vector between light and viewer.
            half = light - v
            half /= np.maximum(np.linalg.norm(half, axis=1), 1e-9)[:, None]
            spec = np.power(
                np.clip(np.einsum("ij,ij->i", n, half), 0.0, 1.0), shininess
            )
            shade = ambient + (1.0 - ambient) * diffuse
            lit = base[None, :] * shade[:, None] + 255.0 * specular * spec[:, None]
            colors = np.clip(lit, 0, 255).astype(np.uint8)

        tri_front = tri2d[front]
        depth_front = tri3d[front, :, 2].mean(axis=1)
        # Painter's algorithm: farthest first, so nearer surfaces overwrite.
        for idx in np.argsort(-depth_front):
            a, b, c = tri_front[idx]
            draw.polygon(
                [(a[0], a[1]), (b[0], b[1]), (c[0], c[1])],
                fill=tuple(int(value) for value in colors[idx]),
            )

    if alpha >= 1.0:
        return overlay
    return Image.blend(img.convert("RGB"), overlay, alpha)


def draw_mesh(
    img: Image.Image,
    joints2d: np.ndarray | None = None,
    vertices2d: np.ndarray | None = None,
    faces: np.ndarray | None = None,
    vertices3d: np.ndarray | None = None,
    edges: Tuple[Tuple[int, int], ...] = MHR70_SKELETON_EDGES,
    vertex_color: Tuple[int, int, int] = MESH_VERTEX_COLOR,
    surface_color: Tuple[int, int, int] = MESH_SURFACE_COLOR,
    max_vertices: int = 1200,
    surface_alpha: float = 0.9,
    shading: str = "diffuse",
    draw_skeleton: bool = False,
) -> Image.Image:
    """Overlay body meshes on an image.

    Renders a shaded surface when the topology and depths are available, which
    is what a body mesh is meant to look like. Falls back to a decimated vertex
    scatter when only projected points are on hand, so a parameters-only result
    still shows something.

    Args:
        img: PIL image to draw on.
        joints2d: ``(N, K, 2)`` projected keypoints in pixels, or None.
        vertices2d: ``(N, V, 2)`` projected mesh vertices in pixels, or None.
        faces: ``(F, 3)`` shared mesh topology; enables surface rendering.
        vertices3d: ``(N, V, 3)`` camera-space metric vertices; enables
            surface rendering, shading normals and correct draw order.
        edges: Pairs of keypoint indices to connect.
        vertex_color: RGB color for the fallback vertex scatter.
        surface_color: Base RGB of the rendered surface.
        max_vertices: Per-person cap on scattered vertices in fallback mode.
        surface_alpha: Blend weight of the rendered surface.
        shading: ``"diffuse"`` or ``"normal"``; see ``render_mesh_surface``.
        draw_skeleton: Also draw the joint skeleton. Off by default: over a
            solid surface it mostly adds clutter.
    """
    img_draw = img.convert("RGB")

    if vertices2d is not None and faces is not None and vertices3d is not None:
        img_draw = render_mesh_surface(
            img_draw,
            vertices2d,
            vertices3d,
            faces,
            color=surface_color,
            alpha=surface_alpha,
            shading=shading,
        )
    elif vertices2d is not None:
        verts = np.asarray(vertices2d, dtype=np.float32)
        if verts.ndim == 2:
            verts = verts[None, ...]
        if verts.size:
            overlay = img_draw.copy()
            draw = ImageDraw.Draw(overlay)
            img_diag = (img.width**2 + img.height**2) ** 0.5
            radius = max(1, int(round(img_diag / 900)))
            for person in verts:
                if len(person) > max_vertices:
                    step = int(np.ceil(len(person) / max_vertices))
                    person = person[::step]
                for x, y in person:
                    cx, cy = float(x), float(y)
                    draw.ellipse(
                        [cx - radius, cy - radius, cx + radius, cy + radius],
                        fill=vertex_color,
                    )
            img_draw = Image.blend(img_draw, overlay, 0.55)

    if draw_skeleton and joints2d is not None:
        joints = np.asarray(joints2d, dtype=np.float32)
        if joints.size:
            img_draw = draw_keypoints(img_draw, joints, edges=edges)

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


def draw_boxes3d(image, boxes3d, near_clip=0.01):
    """Draw calibrated cuboids, clipping each edge at the camera near plane."""
    from PIL import ImageDraw

    if boxes3d.intrinsics is None:
        raise ValueError("3D drawing requires original-image intrinsics.")
    if not np.isfinite(near_clip) or near_clip <= 0:
        raise ValueError("near_clip must be positive and finite.")
    payload = boxes3d.numpy()
    calibration = np.asarray(payload.intrinsics, dtype=np.float64)
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    edges = (
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    )
    for corners in payload.corners:
        for start, end in edges:
            a, b = corners[start].copy(), corners[end].copy()
            if a[2] < near_clip and b[2] < near_clip:
                continue
            if a[2] < near_clip:
                a += (b - a) * ((near_clip - a[2]) / (b[2] - a[2]))
            elif b[2] < near_clip:
                b += (a - b) * ((near_clip - b[2]) / (a[2] - b[2]))
            projected = np.stack((a, b)) @ calibration.T
            xy = projected[:, :2] / projected[:, 2:3]
            # Bound the coordinates passed to Pillow's integer rasterizer.
            xy = np.clip(xy, -1e6, 1e6)
            draw.line([tuple(xy[0]), tuple(xy[1])], fill=(0, 220, 120), width=2)
    return canvas
