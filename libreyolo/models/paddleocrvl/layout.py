# SPDX-License-Identifier: Apache-2.0
# Adapted from PaddleX (https://github.com/PaddlePaddle/PaddleX, commit
# 4c968d30b9d0d4a693d9eaaef0881c20f8d7d0e0, Apache-2.0, Copyright (c) 2024
# PaddlePaddle Authors): paddlex/inference/models/object_detection/processors.py
# (DetPostProcess, nms, check_containment), paddlex/configs/pipelines/
# PaddleOCR-VL.yaml (thresholds, merge modes) and paddlex/inference/pipelines/
# paddleocr_vl/uilts.py (filter_overlap_boxes). The reading-order vote follows
# transformers' PPDocLayoutV2ImageProcessor (Apache-2.0). See NOTICE.
"""PP-DocLayoutV2 layout stage: preprocessing and the upstream post-processor.

The network itself runs through the native transformers
``PPDocLayoutV2ForObjectDetection`` class. This module reproduces the
PaddleOCR-VL layout settings around it: the 800x800 bicubic resize, the
per-class score thresholds, layout NMS, the large-image filter, the
"large" containment merge for titles/formulas/charts, the reading-order sort,
and the pipeline's overlap filter.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch

LAYOUT_INPUT_SIZE = 800

# Label list of the released PP-DocLayoutV2 (``PP-DocLayoutV2/inference.yml``).
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

# Per-class score thresholds of the PaddleOCR-VL pipeline config.
LAYOUT_THRESHOLDS: Dict[int, float] = {i: 0.5 for i in range(len(LAYOUT_LABELS))}
LAYOUT_THRESHOLDS.update({5: 0.4, 6: 0.4, 15: 0.4, 17: 0.4, 20: 0.45, 22: 0.4, 23: 0.4})

# Classes whose boxes swallow the boxes they contain ("large" merge mode);
# every other class keeps both ("union").
LAYOUT_LARGE_MERGE: Tuple[int, ...] = (3, 5, 6, 15, 17)

LAYOUT_UNCLIP_RATIO = (1.0, 1.0)


def preprocess_layout(rgb: np.ndarray) -> torch.Tensor:
    """RGB uint8 ``(H, W, 3)`` page to the ``(1, 3, 800, 800)`` float input."""
    resized = cv2.resize(
        rgb,
        (LAYOUT_INPUT_SIZE, LAYOUT_INPUT_SIZE),
        interpolation=cv2.INTER_CUBIC,
    )
    tensor = torch.from_numpy(np.ascontiguousarray(resized)).permute(2, 0, 1)
    return (tensor.float() / 255.0).unsqueeze(0)


def order_sequence(order_logits: torch.Tensor) -> torch.Tensor:
    """Reading-order rank of each query from the pointer-network logits."""
    scores = torch.sigmoid(order_logits)
    _, n, _ = scores.shape
    votes = scores.triu(diagonal=1).sum(dim=1) + (1.0 - scores.transpose(1, 2)).tril(
        diagonal=-1
    ).sum(dim=1)
    pointers = torch.argsort(votes, dim=1)
    seq = torch.empty_like(pointers)
    ranks = torch.arange(n, device=pointers.device, dtype=pointers.dtype).expand(
        pointers.shape[0], -1
    )
    seq.scatter_(1, pointers, ranks)
    return seq


def raw_layout_boxes(
    logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    order_logits: torch.Tensor,
    image_size: Tuple[int, int],
) -> np.ndarray:
    """Decode one image's raw outputs to ``(K, 7)`` rows.

    Columns are ``class_id, score, x1, y1, x2, y2, order`` in original-image
    pixels, top-K over (query, class) pairs like the exported upstream graph.
    """
    width, height = image_size
    logits = logits[:1].float()
    order = order_sequence(order_logits[:1].float())
    centers, dims = pred_boxes[:1].float().split(2, dim=-1)
    boxes = torch.cat([centers - 0.5 * dims, centers + 0.5 * dims], dim=-1)
    boxes = boxes * boxes.new_tensor([width, height, width, height])
    num_queries, num_classes = logits.shape[1], logits.shape[2]
    scores, index = torch.topk(logits.sigmoid().flatten(1), num_queries, dim=-1)
    labels = index % num_classes
    query = index // num_classes
    boxes = boxes.gather(1, query.unsqueeze(-1).repeat(1, 1, 4))
    order = order.gather(1, query)
    rows = torch.cat(
        [
            labels[0, :, None].float(),
            scores[0, :, None],
            boxes[0],
            order[0, :, None].float(),
        ],
        dim=1,
    )
    return rows.cpu().numpy().astype(np.float64)


def _iou_plus_one(a: Sequence[float], b: Sequence[float]) -> float:
    # Upstream measures areas with the +1 pixel convention.
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1 + 1) * max(0.0, y2 - y1 + 1)
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return inter / float(area_a + area_b - inter)


def layout_nms(
    rows: np.ndarray, iou_same: float = 0.6, iou_diff: float = 0.98
) -> List[int]:
    """Class-aware NMS: a looser threshold across different classes."""
    indices = list(np.argsort(rows[:, 1])[::-1])
    keep: List[int] = []
    while indices:
        current = indices.pop(0)
        keep.append(int(current))
        remaining = []
        for i in indices:
            same = rows[current, 0] == rows[i, 0]
            if _iou_plus_one(rows[current, 2:6], rows[i, 2:6]) < (
                iou_same if same else iou_diff
            ):
                remaining.append(i)
        indices = remaining
    return keep


def _is_contained(inner: Sequence[float], outer: Sequence[float]) -> bool:
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    w = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    h = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return (w * h / area if area > 0 else 0.0) >= 0.9


def _contained_by_class(rows: np.ndarray, category: int) -> np.ndarray:
    """Mask of rows contained (>= 90 percent) in a box of ``category``."""
    n = len(rows)
    contained = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if (
                i != j
                and rows[j, 0] == category
                and _is_contained(rows[i, 2:6], rows[j, 2:6])
            ):
                contained[i] = True
    return contained


def postprocess_layout(
    rows: np.ndarray,
    image_size: Tuple[int, int],
    threshold: Dict[int, float] | float | None = None,
    nms: bool = True,
    large_merge: Sequence[int] = LAYOUT_LARGE_MERGE,
    unclip_ratio: Tuple[float, float] | None = LAYOUT_UNCLIP_RATIO,
    labels: Sequence[str] = LAYOUT_LABELS,
) -> List[dict]:
    """Upstream layout post-processing on ``raw_layout_boxes`` rows.

    Returns ``{"cls_id", "label", "score", "coordinate"}`` dicts in reading
    order, with coordinates clipped to the image.
    """
    width, height = image_size
    threshold = LAYOUT_THRESHOLDS if threshold is None else threshold
    rows = np.asarray(rows, dtype=np.float64).reshape(-1, 7)
    if isinstance(threshold, dict):
        keep = np.array(
            [row[1] > threshold.get(int(row[0]), 0.5) for row in rows], dtype=bool
        )
    else:
        keep = rows[:, 1] > float(threshold)
    rows = rows[keep & (rows[:, 0] > -1)]
    if len(rows) and nms:
        rows = rows[layout_nms(rows)]

    # Drop a page-sized "image" box (a full-page scan detected as one figure).
    if len(rows) > 1 and "image" in labels:
        image_id = labels.index("image")
        area_thres = 0.82 if width > height else 0.93
        page_area = width * height
        kept = []
        for row in rows:
            if row[0] == image_id:
                x1, y1 = max(0.0, row[2]), max(0.0, row[3])
                x2, y2 = min(float(width), row[4]), min(float(height), row[5])
                if (x2 - x1) * (y2 - y1) > area_thres * page_area:
                    continue
            kept.append(row)
        if kept:
            rows = np.asarray(kept)

    if len(rows) and large_merge:
        mask = np.ones(len(rows), dtype=bool)
        for category in large_merge:
            mask &= ~_contained_by_class(rows, int(category))
        rows = rows[mask]
    if len(rows) == 0:
        return []

    # Same (unstable) sort as upstream so ties between classes of one query
    # resolve identically.
    rows = rows[np.argsort(rows[:, 6])][:, :6]

    if unclip_ratio is not None:
        w = rows[:, 4] - rows[:, 2]
        h = rows[:, 5] - rows[:, 3]
        cx = rows[:, 2] + w / 2
        cy = rows[:, 3] + h / 2
        nw, nh = w * unclip_ratio[0], h * unclip_ratio[1]
        rows = np.column_stack(
            (rows[:, 0], rows[:, 1], cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2)
        )

    boxes = []
    for row in rows:
        x1, y1 = max(0.0, row[2]), max(0.0, row[3])
        x2, y2 = min(float(width), row[4]), min(float(height), row[5])
        if x2 <= x1 or y2 <= y1:
            continue
        cls_id = int(row[0])
        boxes.append(
            {
                "cls_id": cls_id,
                "label": labels[cls_id],
                "score": float(row[1]),
                "coordinate": [x1, y1, x2, y2],
            }
        )
    return boxes


def filter_overlap_boxes(boxes: List[dict]) -> List[dict]:
    """Pipeline overlap filter applied before recognition.

    Drops ``reference`` boxes and boxes thinner than 6 px; when two boxes
    overlap by more than 70 percent of the smaller one, keeps the larger
    (inline formulas lose at 50 percent), except that image/table/seal/chart
    boxes are not removed by a box of another class.
    """
    boxes = [b for b in boxes if b["label"] != "reference"]
    if not boxes:
        return []
    coords = np.array([b["coordinate"] for b in boxes], dtype=np.float64)
    widths = coords[:, 2] - coords[:, 0]
    heights = coords[:, 3] - coords[:, 1]
    areas = np.abs(widths * heights)
    ix1 = np.maximum(coords[:, None, 0], coords[None, :, 0])
    iy1 = np.maximum(coords[:, None, 1], coords[None, :, 1])
    ix2 = np.minimum(coords[:, None, 2], coords[None, :, 2])
    iy2 = np.minimum(coords[:, None, 3], coords[None, :, 3])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    small = np.minimum(areas[:, None], areas[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        overlap = np.where(small > 0, inter / small, 0.0)

    dropped = set()
    for i in range(len(boxes)):
        if widths[i] < 6 or heights[i] < 6:
            dropped.add(i)
        for j in range(i + 1, len(boxes)):
            if i in dropped or j in dropped:
                continue
            ratio = overlap[i, j]
            li, lj = boxes[i]["label"], boxes[j]["label"]
            if "inline_formula" in (li, lj):
                if ratio > 0.5:
                    if li == "inline_formula":
                        dropped.add(i)
                    if lj == "inline_formula":
                        dropped.add(j)
                    continue
            if ratio > 0.7:
                pair = {li, lj}
                if pair & {"image", "table", "seal", "chart"} and len(pair) > 1:
                    if "table" not in pair or pair <= {
                        "table",
                        "image",
                        "seal",
                        "chart",
                    }:
                        continue
                dropped.add(j if areas[i] >= areas[j] else i)
    return [b for k, b in enumerate(boxes) if k not in dropped]


__all__ = [
    "LAYOUT_LABELS",
    "LAYOUT_THRESHOLDS",
    "LAYOUT_LARGE_MERGE",
    "preprocess_layout",
    "order_sequence",
    "raw_layout_boxes",
    "postprocess_layout",
    "layout_nms",
    "filter_overlap_boxes",
]
