"""Property tests for the geometric augmentation knobs.

These complement the byte-exact golden fixtures in ``test_augment_parity.py``
by pinning invariants that hold regardless of platform: the perspective knob
is a no-op when zero, vertical flip is an involution, a full turn of the rot90
helper is the identity, and the OBB angle remap agrees with a brute-force
corner rotation.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np
import pytest

from libreyolo.data.augment.geometry import (
    MAX_PERSPECTIVE_DISTORTION,
    PERSPECTIVE_TO_DISTORTION,
    apply_affine_to_bboxes,
    get_affine_matrix,
    mirror_vertical,
    random_affine,
    rot90_image_boxes,
)
from libreyolo.data.obb import (
    normalize_obb_angle,
    xywhr_iou,
    xywhr_to_corners,
    corners_to_xywhr,
)

pytestmark = pytest.mark.unit


def test_perspective_zero_matches_affine_matrix_and_rng():
    """perspective=0.0 returns the historical 2x3 affine and draws no extra RNG."""
    random.seed(4242)
    m_zero, scale_zero = get_affine_matrix((64, 64), 10.0, 0.1, 0.1, 10.0, perspective=0.0)
    tail_zero = random.random()

    random.seed(4242)
    m_default, scale_default = get_affine_matrix((64, 64), 10.0, 0.1, 0.1, 10.0)
    tail_default = random.random()

    assert m_zero.shape == (2, 3)
    assert np.array_equal(m_zero, m_default)
    assert scale_zero == scale_default
    # Identical follow-up draw proves the same number of RNG values was consumed.
    assert tail_zero == tail_default


def _affine_3x3(target_size, seed):
    """The 2x3 affine for ``seed`` lifted to 3x3, and its scale."""
    random.seed(seed)
    m, scale = get_affine_matrix(target_size, 10.0, 0.1, (0.5, 1.5), 2.0)
    return np.vstack([m, [0.0, 0.0, 1.0]]), scale


def _canvas_corners(width, height):
    return np.array(
        [[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]], dtype=float
    )


def test_perspective_zero_random_affine_matches_warp_affine():
    """random_affine with perspective=0.0 is the plain cv2.warpAffine path."""
    rng = np.random.RandomState(0)
    img = rng.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    boxes = np.array([[5.0, 6.0, 40.0, 50.0, 1.0]], dtype=np.float32)

    random.seed(11)
    out_img, out_boxes = random_affine(
        img,
        boxes.copy(),
        target_size=(64, 64),
        degrees=10.0,
        translate=0.1,
        scales=0.1,
        shear=10.0,
        perspective=0.0,
    )
    tail = random.random()

    random.seed(11)
    M, _ = get_affine_matrix((64, 64), 10.0, 0.1, 0.1, 10.0)
    assert random.random() == tail  # six draws, nothing more
    expected = cv2.warpAffine(img, M, dsize=(64, 64), borderValue=(114, 114, 114))
    assert np.array_equal(out_img, expected)
    assert (out_boxes[:, :4] >= 0).all() and (out_boxes[:, :4] <= 64).all()


def test_perspective_nonzero_is_3x3_homography():
    random.seed(7)
    M, _scale = get_affine_matrix((80, 64), 10.0, 0.1, 0.1, 10.0, perspective=5e-4)
    assert M.shape == (3, 3)
    assert M.dtype == np.float64
    # The bottom row is non-trivial (projective), unlike an affine's [0, 0, 1].
    assert not np.allclose(M[2, :2], 0.0)


def test_perspective_reuses_affine_draws_then_draws_eight_more():
    """M = P @ A: A is the affine from the same six draws; P costs eight draws."""
    size = (96, 64)
    A, scale_affine = _affine_3x3(size, seed=99)
    tail_affine = [random.random() for _ in range(9)]

    random.seed(99)
    M, scale_persp = get_affine_matrix(size, 10.0, 0.1, (0.5, 1.5), 2.0, perspective=1e-3)
    tail_persp = random.random()

    assert scale_persp == scale_affine
    # Exactly eight extra draws (x and y offset of each of the four corners).
    assert tail_persp == tail_affine[8]

    # Factoring the affine back out leaves the pure corner-displacement
    # homography: it moves every canvas corner inward by at most
    # perspective * 100 * half the canvas side, per axis.
    P = M @ np.linalg.inv(A)
    width, height = size
    warped = _canvas_corners(width, height) @ P.T
    warped = warped[:, :2] / warped[:, 2:3]
    inward = np.abs(warped - _canvas_corners(width, height)[:, :2])
    assert (inward[:, 0] <= 0.1 * width / 2 + 1e-3).all()
    assert (inward[:, 1] <= 0.1 * height / 2 + 1e-3).all()
    assert (warped >= -1e-3).all()
    assert (warped[:, 0] <= width + 1e-3).all()
    assert (warped[:, 1] <= height + 1e-3).all()
    assert inward.max() > 0.0


def test_perspective_strength_scales_and_is_capped():
    """Same seed: corner offsets grow linearly with the knob up to the cap."""
    size = (640, 640)
    corners = _canvas_corners(*size)

    def offsets(perspective):
        A, _ = _affine_3x3(size, seed=5)
        random.seed(5)
        M, _ = get_affine_matrix(size, 10.0, 0.1, (0.5, 1.5), 2.0, perspective=perspective)
        warped = corners @ (M @ np.linalg.inv(A)).T
        return warped[:, :2] / warped[:, 2:3] - corners[:, :2]

    np.testing.assert_allclose(offsets(1e-3), 2 * offsets(5e-4), atol=1e-2)
    assert np.abs(offsets(1e-3)).max() <= 32.0 + 1e-3
    # distortion_scale is capped at MAX_PERSPECTIVE_DISTORTION (0.2).
    assert MAX_PERSPECTIVE_DISTORTION == 0.2
    assert PERSPECTIVE_TO_DISTORTION == 100.0
    np.testing.assert_allclose(offsets(1.0), offsets(2e-3), atol=1e-6)


def test_perspective_negative_raises():
    with pytest.raises(ValueError, match="non-negative"):
        get_affine_matrix((64, 64), 10.0, 0.1, 0.1, 10.0, perspective=-1e-3)


@pytest.mark.parametrize("size", [(320, 320), (640, 640), (1280, 1280), (1280, 320)])
@pytest.mark.parametrize("perspective", [1e-4, 1e-3, 1.0])
def test_perspective_homography_is_well_conditioned(size, perspective):
    """No fold-over and no near-zero denominator on the canvas, many seeds."""
    width, height = size
    corners = _canvas_corners(width, height)
    # Denominator floor of the corner-displacement homography on the canvas:
    # 0.81 at distortion_scale 0.1 (perspective=0.001), 0.63 at the 0.2 cap.
    floor = 0.8 if perspective <= 1e-3 else 0.59
    for seed in range(300):
        A, _ = _affine_3x3(size, seed)
        random.seed(seed)
        M, _ = get_affine_matrix(size, 10.0, 0.1, (0.5, 1.5), 2.0, perspective=perspective)
        assert np.isfinite(M).all()

        P = M @ np.linalg.inv(A)
        P = P / P[2, 2]
        # Forward denominator is linear, so its canvas extremes are at corners.
        w = corners @ P[2]
        assert w.min() > floor and w.max() < 1.0 / floor

        # warpPerspective inverts M: the inverse denominator over the output
        # canvas must stay away from zero too.
        P_inv = np.linalg.inv(P)
        P_inv = P_inv / P_inv[2, 2]
        w_inv = corners @ P_inv[2]
        assert w_inv.min() > floor and w_inv.max() < 1.0 / floor

        # Orientation preserved everywhere on the canvas (no fold-over): the
        # Jacobian determinant of x -> P x has the sign of det(P) / w**3.
        assert np.linalg.det(P) > 0
        # The warped canvas is a convex quadrilateral with positive area.
        quad = corners @ P.T
        quad = quad[:, :2] / quad[:, 2:3]
        edges = np.roll(quad, -1, axis=0) - quad
        nxt = np.roll(edges, -1, axis=0)
        cross = edges[:, 0] * nxt[:, 1] - edges[:, 1] * nxt[:, 0]
        assert (cross > 0).all()
        # The full matrix keeps the affine's orientation and is invertible.
        assert np.linalg.det(M) * np.linalg.det(A) > 0
        assert np.linalg.cond(P) < 1e8


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("perspective", [1e-3, 2e-3])
def test_perspective_warped_box_bounds_warped_rectangle(seed, perspective):
    """Image and boxes go through the same homography."""
    size = (320, 256)
    width, height = size
    box = np.array([[110.0, 90.0, 200.0, 160.0, 3.0]], dtype=np.float32)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(img, (110, 90), (199, 159), (255, 255, 255), thickness=-1)

    random.seed(seed)
    out_img, out_boxes = random_affine(
        img,
        box.copy(),
        target_size=size,
        degrees=10.0,
        translate=0.1,
        scales=(0.8, 1.2),
        shear=2.0,
        perspective=perspective,
    )
    assert out_img.shape == (height, width, 3)
    assert out_boxes[0, 4] == 3.0

    ys, xs = np.nonzero(out_img[:, :, 0] > 200)
    assert len(xs) > 0
    x1, y1, x2, y2 = out_boxes[0, :4]
    assert x2 > x1 and y2 > y1
    # The warped box contains the warped rectangle (interpolation tolerance)...
    assert xs.min() >= x1 - 1.5 and xs.max() <= x2 + 1.5
    assert ys.min() >= y1 - 1.5 and ys.max() <= y2 + 1.5
    # ...and is tight on any side that was not clipped by the canvas.
    if 0 < x1:
        assert xs.min() <= x1 + 2.5
    if x2 < width:
        assert xs.max() >= x2 - 3.5
    if 0 < y1:
        assert ys.min() <= y1 + 2.5
    if y2 < height:
        assert ys.max() >= y2 - 3.5


def test_perspective_keeps_centered_content_on_canvas():
    """At perspective=0.001 with no affine jitter the image stays on canvas."""
    size = (640, 640)
    for seed in range(100):
        random.seed(seed)
        M, _ = get_affine_matrix(size, 0.0, 0.0, (1.0, 1.0), 0.0, perspective=1e-3)
        warped = _canvas_corners(*size) @ M.T
        warped = warped[:, :2] / warped[:, 2:3]
        assert (warped >= -1e-3).all() and (warped <= 640 + 1e-3).all()
        # Each corner moved by at most 5% of the canvas side per axis.
        assert np.abs(warped - _canvas_corners(*size)[:, :2]).max() <= 32.0 + 1e-3


def test_apply_affine_to_bboxes_handles_corner_behind_horizon():
    """A corner with a non-positive homogeneous coordinate does not fold over."""
    M = np.eye(3)
    M[2, 0] = -1.0 / 100.0  # horizon at x = 100
    boxes = np.array([[50.0, 10.0, 150.0, 40.0, 0.0]], dtype=np.float32)
    out = apply_affine_to_bboxes(boxes.copy(), (64, 64), M, 1.0)
    assert np.isfinite(out).all()
    # x = 50 maps to 100 -> clipped to 64; the far side runs to +infinity.
    assert out[0, 0] == 64.0 and out[0, 2] == 64.0


def test_random_affine_perspective_runs_and_clips_boxes():
    rng = np.random.RandomState(0)
    img = rng.randint(0, 255, (80, 64, 3), dtype=np.uint8)
    boxes = np.array(
        [[5.0, 6.0, 40.0, 50.0, 1.0], [10.0, 12.0, 30.0, 35.0, 0.0]],
        dtype=np.float32,
    )
    random.seed(1)
    out_img, out_boxes = random_affine(
        img,
        boxes.copy(),
        target_size=(64, 64),
        degrees=10.0,
        translate=0.1,
        scales=0.1,
        shear=10.0,
        perspective=1e-3,
    )
    assert out_img.shape == (64, 64, 3)
    assert (out_boxes[:, :4] >= 0).all()
    assert (out_boxes[:, [0, 2]] <= 64).all()
    assert (out_boxes[:, [1, 3]] <= 64).all()


def test_mirror_vertical_twice_is_identity():
    rng = np.random.RandomState(3)
    img = rng.randint(0, 255, (48, 72, 3), dtype=np.uint8)
    boxes = np.array([[4.0, 6.0, 20.0, 30.0], [10.0, 2.0, 60.0, 40.0]], dtype=np.float32)

    once_img, once_boxes = mirror_vertical(img.copy(), boxes.copy(), prob=1.0)
    twice_img, twice_boxes = mirror_vertical(once_img, once_boxes, prob=1.0)

    assert np.array_equal(twice_img, img)
    assert np.allclose(twice_boxes, boxes)


def test_rot90_k4_is_identity():
    rng = np.random.RandomState(5)
    img = rng.randint(0, 255, (37, 53, 3), dtype=np.uint8)
    boxes = np.array([[3.0, 4.0, 20.0, 25.0], [8.0, 10.0, 40.0, 30.0]], dtype=np.float32)

    out_img, out_boxes = rot90_image_boxes(img.copy(), boxes.copy(), k=4)
    assert np.array_equal(out_img, img)
    assert np.allclose(out_boxes, boxes)


def _rotate_point_k(x, y, k, width, height):
    """Brute-force apply k CCW quarter turns, matching rot90_image_boxes' T."""
    cur_w, cur_h = width, height
    for _ in range(k % 4):
        x, y = y, cur_w - x
        cur_w, cur_h = cur_h, cur_w
    return x, y


@pytest.mark.parametrize("k", [1, 2, 3])
def test_obb_rot90_matches_bruteforce_corner_rotation(k):
    """The proxy-box + angle remap equals refitting brute-force-rotated corners."""
    width, height = 100, 80
    cx, cy, w, h, r = 30.0, 25.0, 16.0, 6.0, 0.4  # canonical (w > h)

    # Model path: rotate the horizontal proxy box and turn the angle by k*90.
    proxy = np.array(
        [[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]], dtype=np.float32
    )
    img = np.zeros((height, width, 3), dtype=np.uint8)
    _rot_img, new_proxy = rot90_image_boxes(img, proxy.copy(), k)
    ncx = (new_proxy[0, 0] + new_proxy[0, 2]) / 2
    ncy = (new_proxy[0, 1] + new_proxy[0, 3]) / 2
    nw = new_proxy[0, 2] - new_proxy[0, 0]
    nh = new_proxy[0, 3] - new_proxy[0, 1]
    new_angle = normalize_obb_angle(r + k * math.pi / 2)
    model_xywhr = np.array([ncx, ncy, nw, nh, new_angle], dtype=np.float32)

    # Brute force: rotate the true rectangle corners, then refit canonically.
    corners = xywhr_to_corners(np.array([cx, cy, w, h, r], dtype=np.float32))
    rotated_corners = np.array(
        [_rotate_point_k(px, py, k, width, height) for px, py in corners],
        dtype=np.float32,
    )
    brute_xywhr = corners_to_xywhr(rotated_corners)

    # Same rectangle: centers coincide and IoU is ~1.
    assert model_xywhr[0] == pytest.approx(brute_xywhr[0], abs=1e-3)
    assert model_xywhr[1] == pytest.approx(brute_xywhr[1], abs=1e-3)
    assert xywhr_iou(model_xywhr, brute_xywhr) > 0.999
