"""Association utilities for multi-object tracking.

All functions operate on numpy arrays — the tracker runs in numpy-land.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment as scipy_lsa

if TYPE_CHECKING:
    from .strack import STrack


def bbox_iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of bounding boxes.

    Args:
        bboxes_a: (N, 4) array in xyxy format.
        bboxes_b: (M, 4) array in xyxy format.

    Returns:
        (N, M) IoU matrix.
    """
    x1 = np.maximum(bboxes_a[:, 0:1], bboxes_b[:, 0])
    y1 = np.maximum(bboxes_a[:, 1:2], bboxes_b[:, 1])
    x2 = np.minimum(bboxes_a[:, 2:3], bboxes_b[:, 2])
    y2 = np.minimum(bboxes_a[:, 3:4], bboxes_b[:, 3])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])

    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / np.maximum(union, 1e-6)


def iou_distance(tracks: list[STrack], detections: np.ndarray) -> np.ndarray:
    """Compute cost matrix as 1 - IoU between tracks and detections.

    Args:
        tracks: List of STrack instances.
        detections: (M, 4) array of detection bboxes in xyxy format.

    Returns:
        (N, M) cost matrix where N = len(tracks).
    """
    if len(tracks) == 0 or len(detections) == 0:
        return np.empty((len(tracks), len(detections)), dtype=np.float64)

    track_bboxes = np.array([t.xyxy for t in tracks], dtype=np.float64)
    return 1.0 - bbox_iou_batch(track_bboxes, detections)


def fuse_score(cost_matrix: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Fuse detection confidence scores into the IoU cost matrix.

    Formula: fused = 1 - (iou_similarity * detection_score)

    Args:
        cost_matrix: (N, M) IoU-based cost matrix.
        scores: (M,) detection confidence scores.

    Returns:
        (N, M) fused cost matrix.
    """
    iou_sim = 1.0 - cost_matrix
    fused_sim = iou_sim * scores[None, :]
    return 1.0 - fused_sim


def linear_assignment(
    cost_matrix: np.ndarray,
    thresh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve the linear assignment problem and filter by threshold.

    Args:
        cost_matrix: (N, M) cost matrix.
        thresh: Maximum cost for a valid assignment.

    Returns:
        matches: (K, 2) array of (row, col) index pairs.
        unmatched_a: (P,) array of unmatched row indices.
        unmatched_b: (Q,) array of unmatched col indices.
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )

    row_indices, col_indices = scipy_lsa(cost_matrix)

    matches = []
    unmatched_a = set(range(cost_matrix.shape[0]))
    unmatched_b = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_indices, col_indices):
        if cost_matrix[r, c] > thresh:
            continue
        matches.append([r, c])
        unmatched_a.discard(r)
        unmatched_b.discard(c)

    matches_arr = np.array(matches, dtype=int) if matches else np.empty((0, 2), dtype=int)
    return (
        matches_arr,
        np.array(sorted(unmatched_a), dtype=int),
        np.array(sorted(unmatched_b), dtype=int),
    )
