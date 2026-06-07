import numpy as np

from libreyolo.data import augment as shared
from libreyolo.models.yolo9 import transforms as yolo9_transforms
from libreyolo.training import augment as legacy


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
