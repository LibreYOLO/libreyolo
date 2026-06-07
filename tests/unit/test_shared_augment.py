import numpy as np
import pytest

from libreyolo.data import augment as shared
from libreyolo.data.dataset import DenseMaskRing
from libreyolo.models.yolo9 import transforms as yolo9_transforms
from libreyolo.training import augment as legacy

pytestmark = pytest.mark.unit


def test_legacy_training_augment_reexports_shared_primitives():
    assert legacy.adjust_box_anns is shared.adjust_box_anns
    assert legacy.apply_affine_to_bboxes is shared.apply_affine_to_bboxes
    assert legacy.augment_hsv is shared.augment_hsv
    assert legacy.cxcywh2xyxy is shared.cxcywh2xyxy
    assert legacy.get_affine_matrix is shared.get_affine_matrix
    assert legacy.get_aug_params is shared.get_aug_params
    assert legacy.get_mosaic_coordinate is shared.get_mosaic_coordinate
    assert legacy.mirror is shared.mirror
    assert legacy.random_affine is shared.random_affine
    assert legacy.xyxy2cxcywh is shared.xyxy2cxcywh


def test_yolo9_transform_uses_shared_leaf_primitives():
    assert yolo9_transforms.augment_hsv is shared.augment_hsv
    assert yolo9_transforms.get_mosaic_coordinate is shared.get_mosaic_coordinate
    assert yolo9_transforms.mirror is shared.mirror


def test_shared_mosaic_coordinate_geometry():
    mosaic = np.zeros((20, 20, 3), dtype=np.uint8)
    input_h, input_w = 10, 10
    xc, yc, w, h = 8, 7, 6, 5

    coords = [
        shared.get_mosaic_coordinate(mosaic, i, xc, yc, w, h, input_h, input_w)
        for i in range(4)
    ]

    assert coords[0] == ((2, 2, 8, 7), (0, 0, 6, 5))
    assert coords[1] == ((8, 2, 14, 7), (0, 0, 6, 5))
    assert coords[2] == ((2, 7, 8, 12), (0, 0, 6, 5))
    assert coords[3] == ((8, 7, 14, 12), (0, 0, 6, 5))


def test_shared_segment_geometry_helpers():
    segments = [
        [
            np.array(
                [[1.0, 2.0], [4.0, 2.0], [4.0, 5.0], [1.0, 5.0]],
                dtype=np.float32,
            )
        ],
        [
            np.array(
                [[6.0, 1.0], [8.0, 1.0], [8.0, 3.0], [6.0, 3.0]],
                dtype=np.float32,
            )
        ],
    ]

    copied = shared.copy_segments(segments)
    assert copied is not segments
    assert copied[0][0] is not segments[0][0]
    np.testing.assert_array_equal(copied[0][0], segments[0][0])

    transformed = shared.transform_segments(
        segments, scale=2.0, padw=1.0, padh=-1.0, width=10, height=10
    )
    np.testing.assert_array_equal(
        transformed[0][0],
        np.array([[3.0, 3.0], [9.0, 3.0], [9.0, 9.0], [3.0, 9.0]], dtype=np.float32),
    )

    flipped = shared.flip_segments_lr(segments, width=10)
    np.testing.assert_array_equal(
        flipped[0][0],
        np.array([[9.0, 2.0], [6.0, 2.0], [6.0, 5.0], [9.0, 5.0]], dtype=np.float32),
    )

    kept = shared.filter_segments(segments, np.array([False, True]))
    assert len(kept) == 1
    np.testing.assert_array_equal(kept[0][0], segments[1][0])

    masks = shared.rasterize_segments(
        [segments[0]], image_shape=(10, 10), mask_shape=(5, 5), max_masks=2
    )
    assert masks.shape == (2, 5, 5)
    assert masks[0].sum() > 0
    assert masks[1].sum() == 0


def test_transform_segments_moves_dense_mask_to_scaled_padded_canvas():
    dense = np.zeros((4, 6), dtype=np.uint8)
    dense[1:3, 2:5] = 1
    ring = np.array(
        [[2.0, 1.0], [5.0, 1.0], [5.0, 3.0], [2.0, 3.0]],
        dtype=np.float32,
    )

    transformed = shared.transform_segments(
        [[DenseMaskRing(ring, dense)]],
        scale=2.0,
        padw=1.0,
        padh=3.0,
        width=14,
        height=12,
    )

    moved = transformed[0][0]
    np.testing.assert_array_equal(
        moved,
        np.array(
            [[5.0, 5.0], [11.0, 5.0], [11.0, 9.0], [5.0, 9.0]],
            dtype=np.float32,
        ),
    )
    moved_dense = getattr(moved, "dense_mask", None)
    assert moved_dense is not None
    assert moved_dense.shape == (12, 14)
    assert moved_dense[5:9, 5:11].sum() == 24
    assert moved_dense[:3, :5].sum() == 0

    no_canvas = shared.transform_segments([[DenseMaskRing(ring, dense)]], scale=2.0)
    assert getattr(no_canvas[0][0], "dense_mask", None) is None
