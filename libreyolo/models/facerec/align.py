"""5-point face alignment for the embedding (face-recognition) task.

ArcFace-style recognition heads expect a face warped onto a canonical 112x112
template via a *similarity* transform (rotation + uniform scale + translation,
no shear) estimated from 5 facial landmarks. We implement Umeyama's closed-form
least-squares similarity estimate in pure numpy and warp with cv2 — no
scikit-image dependency. When landmarks are unavailable we fall back to a
center-square crop of the face box, which is lower quality but keeps the
pipeline running.

The Umeyama algorithm (S. Umeyama, IEEE PAMI 1991) and the canonical 5-point
template are standard published math/data, implemented here from first
principles.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Canonical ArcFace destination landmarks for a 112x112 aligned crop, in the
# standard 5-point order: [right-eye, left-eye, nose, right-mouth, left-mouth]
# expressed in image coordinates (index 0 sits on the image's left).
ARCFACE_DST_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float64,
)


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform mapping ``src`` onto ``dst``.

    Returns a ``(2, 3)`` affine matrix ``M`` (uniform scale + rotation +
    translation) such that ``dst ~= src @ M[:, :2].T + M[:, 2]``.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, dim = src.shape

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    cov = dst_demean.T @ src_demean / n
    U, S, Vt = np.linalg.svd(cov)

    d = np.ones(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[-1] = -1.0
    R = U @ np.diag(d) @ Vt

    var_src = (src_demean ** 2).sum() / n
    scale = (S * d).sum() / var_src if var_src > 0 else 1.0

    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = scale * R
    M[:, 2] = dst_mean - scale * (R @ src_mean)
    return M


def estimate_norm(landmarks5: np.ndarray, image_size: int = 112) -> np.ndarray:
    """Affine ``(2, 3)`` mapping 5 landmarks onto the canonical template."""
    template = ARCFACE_DST_112.copy()
    if image_size != 112:
        template = template * (image_size / 112.0)
    return umeyama_similarity(np.asarray(landmarks5, dtype=np.float64), template)


def _center_crop_resize(image_rgb: np.ndarray, box: Sequence[float], size: int) -> np.ndarray:
    """Fallback when no landmarks: square-crop the face box and resize."""
    import cv2

    h, w = image_rgb.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = max(x2 - x1, y2 - y1) / 2.0
    x1 = int(max(0, round(cx - half)))
    y1 = int(max(0, round(cy - half)))
    x2 = int(min(w, round(cx + half)))
    y2 = int(min(h, round(cy + half)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Degenerate face box for crop: {(x1, y1, x2, y2)}")
    crop = image_rgb[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size))


def align_face(
    image_rgb: np.ndarray,
    box: Sequence[float],
    landmarks: Optional[np.ndarray] = None,
    image_size: int = 112,
) -> np.ndarray:
    """Return a canonical ``image_size`` aligned face crop (RGB uint8).

    Uses a 5-point similarity warp when ``landmarks`` of shape ``(5, 2)`` are
    given, otherwise a center-square crop of ``box``.
    """
    import cv2

    if landmarks is not None:
        lm = np.asarray(landmarks, dtype=np.float64)
        if lm.shape == (5, 2):
            M = estimate_norm(lm, image_size)
            return cv2.warpAffine(
                image_rgb, M.astype(np.float32), (image_size, image_size), borderValue=0
            )
    return _center_crop_resize(image_rgb, box, image_size)
