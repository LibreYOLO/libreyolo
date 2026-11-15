# SPDX-License-Identifier: Apache-2.0
# Adapted from the GLM-OCR SDK (https://github.com/zai-org/GLM-OCR, commit
# 0288f96bb9e78cb439883eaa725eeae90bd177c5, Apache-2.0, Copyright (c) Z.ai /
# zai-org): glmocr/utils/layout_postprocess_utils.py (iou, nms,
# check_containment, unclip_boxes, apply_layout_postprocess),
# glmocr/layout/layout_detector.py (region mapping to 0-1000 boxes) and the
# layout section of glmocr/config.yaml. See NOTICE.
"""PP-DocLayoutV3 layout stage of the GLM-OCR pipeline.

The network and its raw post-processing (score threshold, reading-order sort)
run through the native transformers ``PPDocLayoutV3ForObjectDetection`` and
``PPDocLayoutV3ImageProcessor`` classes, as the GLM-OCR SDK does. This module
reproduces what the SDK does next: class-aware NMS, the large-image filter,
the per-class containment merge, the reading-order sort, box clamping, the
label to task routing and the 0-1000 normalized boxes the SDK crops from.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Label list of the released PP-DocLayoutV3 (``glmocr/config.yaml`` id2label).
LAYOUT_LABELS: Tuple[str, ...] = (
    "abstract",
    "algorithm",
    "aside_text",
    "chart",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footer_image",
    "footnote",
    "formula_number",
    "header",
    "header_image",
    "image",
    "inline_formula",
    "number",
    "paragraph_title",
    "reference",
    "reference_content",
    "seal",
    "table",
    "text",
    "vertical_text",
    "vision_footnote",
)

# SDK defaults (``pipeline.layout`` in ``glmocr/config.yaml``).
LAYOUT_THRESHOLD = 0.3
LAYOUT_UNCLIP_RATIO = (1.0, 1.0)
LAYOUT_MERGE_MODES: Dict[int, str] = {i: "large" for i in range(len(LAYOUT_LABELS))}
LAYOUT_MERGE_MODES[LAYOUT_LABELS.index("reference")] = "small"

# Which recognition prompt each layout class gets; ``skip`` regions are kept
# as pictures without recognition and ``abandon`` regions are dropped.
LABEL_TASK_MAPPING: Dict[str, Tuple[str, ...]] = {
    "text": (
        "abstract",
        "algorithm",
        "content",
        "doc_title",
        "figure_title",
        "paragraph_title",
        "reference_content",
        "text",
        "vertical_text",
        "vision_footnote",
        "seal",
        "formula_number",
    ),
    "table": ("table",),
    "formula": ("display_formula", "inline_formula"),
    "skip": ("chart", "image"),
    "abandon": (
        "header",
        "footer",
        "number",
        "footnote",
        "aside_text",
        "reference",
        "footer_image",
        "header_image",
    ),
}


def task_for_label(label: str) -> Optional[str]:
    """The SDK's first matching task for ``label`` (``None`` when unmapped)."""
    for task, labels in LABEL_TASK_MAPPING.items():
        if label in labels:
            return task
    return None


def iou(box1, box2) -> float:
    """IoU with the ``+1`` pixel convention of the SDK."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    x1_i = max(x1, x1_p)
    y1_i = max(y1, y1_p)
    x2_i = min(x2, x2_p)
    y2_i = min(y2, y2_p)
    inter_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    return inter_area / float(box1_area + box2_area - inter_area)


def nms(boxes: np.ndarray, iou_same: float = 0.6, iou_diff: float = 0.95) -> List[int]:
    """Greedy NMS over ``[cls, score, x1, y1, x2, y2]`` rows, class-aware
    thresholds. Returns the kept row indices in descending score order."""
    scores = boxes[:, 1]
    indices = np.argsort(scores)[::-1]
    selected = []
    while len(indices) > 0:
        current = indices[0]
        current_class = boxes[current][0]
        current_coords = boxes[current][2:]
        selected.append(current)
        indices = indices[1:]
        kept = []
        for i in indices:
            threshold = iou_same if current_class == boxes[i][0] else iou_diff
            if iou(current_coords, boxes[i][2:]) < threshold:
                kept.append(i)
        indices = kept
    return selected


def is_contained(box1, box2) -> bool:
    """True when at least 80% of ``box1`` lies inside ``box2``."""
    _, _, x1, y1, x2, y2 = box1
    _, _, x1_p, y1_p, x2_p, y2_p = box2
    box1_area = (x2 - x1) * (y2 - y1)
    inter_w = max(0, min(x2, x2_p) - max(x1, x1_p))
    inter_h = max(0, min(y2, y2_p) - max(y1, y1_p))
    ratio = inter_w * inter_h / box1_area if box1_area > 0 else 0
    return ratio >= 0.8


def check_containment(
    boxes: np.ndarray,
    preserve_indices=None,
    category_index: Optional[int] = None,
    mode: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(contains_other, contained_by_other)`` flags per row."""
    n = len(boxes)
    contains_other = np.zeros(n, dtype=int)
    contained_by_other = np.zeros(n, dtype=int)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if preserve_indices is not None and boxes[i][0] in preserve_indices:
                continue
            if category_index is not None and mode is not None:
                if mode == "large" and boxes[j][0] == category_index:
                    if is_contained(boxes[i], boxes[j]):
                        contained_by_other[i] = 1
                        contains_other[j] = 1
                if mode == "small" and boxes[i][0] == category_index:
                    if is_contained(boxes[i], boxes[j]):
                        contained_by_other[i] = 1
                        contains_other[j] = 1
            elif is_contained(boxes[i], boxes[j]):
                contained_by_other[i] = 1
                contains_other[j] = 1
    return contains_other, contained_by_other


def unclip_boxes(boxes: np.ndarray, unclip_ratio) -> np.ndarray:
    """Scale each box about its center by ``(width_ratio, height_ratio)``."""
    if unclip_ratio is None:
        return boxes
    widths = boxes[:, 4] - boxes[:, 2]
    heights = boxes[:, 5] - boxes[:, 3]
    new_w = widths * unclip_ratio[0]
    new_h = heights * unclip_ratio[1]
    center_x = boxes[:, 2] + widths / 2
    center_y = boxes[:, 3] + heights / 2
    expanded = np.column_stack(
        (
            boxes[:, 0],
            boxes[:, 1],
            center_x - new_w / 2,
            center_y - new_h / 2,
            center_x + new_w / 2,
            center_y + new_h / 2,
        )
    )
    if boxes.shape[1] > 6:
        expanded = np.column_stack((expanded, boxes[:, 6:]))
    return expanded


def postprocess_layout(
    rows: np.ndarray,
    image_size: Tuple[int, int],
    *,
    nms_enabled: bool = True,
    merge_modes: Optional[Dict[int, str]] = None,
    unclip_ratio: Optional[Sequence[float]] = LAYOUT_UNCLIP_RATIO,
) -> List[dict]:
    """SDK layout post-processing for one page.

    Args:
        rows: ``(N, 7)`` array ``[cls_id, score, x1, y1, x2, y2, order]`` in
            pixels, as the transformers post-processor returns them.
        image_size: ``(width, height)`` of the page.

    Returns:
        Regions in reading order, ``{"cls_id", "label", "score",
        "coordinate"}`` with integer ``xyxy`` pixel coordinates.
    """
    merge_modes = LAYOUT_MERGE_MODES if merge_modes is None else merge_modes
    boxes = np.asarray(rows, dtype=np.float64).reshape(-1, 7)
    if len(boxes) == 0:
        return []
    if nms_enabled:
        boxes = boxes[nms(boxes[:, :6], iou_same=0.6, iou_diff=0.98)]

    width, height = image_size
    if len(boxes) > 1:
        # Drop picture regions that cover (almost) the whole page.
        area_thres = 0.82 if width > height else 0.93
        image_index = LAYOUT_LABELS.index("image")
        kept = []
        for box in boxes:
            label_index, _, xmin, ymin, xmax, ymax = box[:6]
            if label_index == image_index:
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(width, xmax), min(height, ymax)
                if (xmax - xmin) * (ymax - ymin) <= area_thres * width * height:
                    kept.append(box)
            else:
                kept.append(box)
        if kept:
            boxes = np.array(kept)

    if merge_modes:
        preserve = {LAYOUT_LABELS.index(name) for name in ("image", "seal", "chart")}
        keep = np.ones(len(boxes), dtype=bool)
        for category_index, mode in merge_modes.items():
            if mode not in ("union", "large", "small"):
                raise ValueError(
                    f"merge mode must be 'union', 'large' or 'small', got {mode!r}"
                )
            if mode == "union":
                continue
            contains_other, contained_by_other = check_containment(
                boxes[:, :6], preserve, category_index, mode=mode
            )
            if mode == "large":
                keep &= contained_by_other == 0
            else:
                keep &= (contains_other == 0) | (contained_by_other == 1)
        boxes = boxes[keep]

    if len(boxes) == 0:
        return []
    boxes = boxes[np.argsort(boxes[:, 6])]
    if unclip_ratio:
        boxes = unclip_boxes(boxes, unclip_ratio)

    regions = []
    for row in boxes:
        cls_id = int(row[0])
        x1 = max(0, min(float(row[2]), width))
        y1 = max(0, min(float(row[3]), height))
        x2 = max(0, min(float(row[4]), width))
        y2 = max(0, min(float(row[5]), height))
        if x1 >= x2 or y1 >= y2:
            continue
        regions.append(
            {
                "cls_id": cls_id,
                "label": LAYOUT_LABELS[cls_id]
                if 0 <= cls_id < len(LAYOUT_LABELS)
                else f"class_{cls_id}",
                "score": float(row[1]),
                "coordinate": [int(x1), int(y1), int(x2), int(y2)],
            }
        )
    return regions


def to_regions(layout_boxes: List[dict], image_size: Tuple[int, int]) -> List[dict]:
    """Route layout boxes to recognition tasks, as the SDK detector does.

    Drops ``abandon`` (headers, footers, page numbers, ...) and unmapped
    classes, and adds ``bbox_2d``: the box on the SDK's 0-1000 integer grid,
    which is what the SDK crops from and what its formatter compares.
    """
    width, height = image_size
    regions = []
    for box in layout_boxes:
        task = task_for_label(box["label"])
        if task is None or task == "abandon":
            continue
        x1, y1, x2, y2 = box["coordinate"]
        regions.append(
            {
                "index": len(regions),
                "label": box["label"],
                "score": float(box["score"]),
                "bbox_2d": [
                    int(float(x1) / width * 1000),
                    int(float(y1) / height * 1000),
                    int(float(x2) / width * 1000),
                    int(float(y2) / height * 1000),
                ],
                "task_type": task,
            }
        )
    return regions


def bbox_to_pixels(bbox_2d: Sequence[int], image_size: Tuple[int, int]) -> List[int]:
    """De-normalize a 0-1000 box to the pixel crop the SDK reads."""
    width, height = image_size
    x1, y1, x2, y2 = bbox_2d
    return [
        int(x1 * width / 1000),
        int(y1 * height / 1000),
        int(x2 * width / 1000),
        int(y2 * height / 1000),
    ]


def raw_rows(result: dict) -> np.ndarray:
    """``[cls, score, x1, y1, x2, y2, order]`` rows from one transformers
    ``post_process_object_detection`` result."""
    scores = result["scores"].detach().cpu().float().numpy()
    if len(scores) == 0:
        return np.zeros((0, 7))
    labels = result["labels"].detach().cpu().numpy()
    boxes = result["boxes"].detach().cpu().float().numpy()
    order = result["order_seq"].detach().cpu().numpy()
    return np.array(
        [
            [int(labels[i]), float(scores[i]), *boxes[i].tolist(), int(order[i])]
            for i in range(len(scores))
        ],
        dtype=np.float64,
    )
