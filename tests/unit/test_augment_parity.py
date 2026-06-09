"""Golden parity tests for every train-time augmentation pipeline.

These tests pin the exact numerical behavior of each family's augmentation
pipeline (seeded RNG, fixed synthetic inputs) against committed ``.npz``
fixtures. They exist so the shared-augmentation refactor cannot silently
change training behavior: any change to RNG call order, BGR/RGB handling,
normalization, padding, or label packing flips them red.

Regenerate fixtures (only when a behavior change is intentional):

    python tests/unit/test_augment_parity.py --generate
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "augment_golden"
SEED = 20260610


def _seed_all():
    random.seed(SEED)
    np.random.seed(SEED)
    try:
        import torch

        torch.manual_seed(SEED)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Deterministic synthetic inputs
# ---------------------------------------------------------------------------


def _image(h=97, w=131, key=0):
    rng = np.random.RandomState(1000 + key)
    return rng.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _boxes_xyxy_cls(key=0):
    """Pixel xyxy + class column, incl. a near-border and a small box."""
    boxes = {
        0: np.array(
            [
                [10.0, 12.0, 60.0, 70.0, 1.0],
                [0.5, 0.5, 25.0, 30.0, 0.0],
                [100.0, 60.0, 130.0, 96.0, 2.0],
                [50.0, 50.0, 52.5, 53.0, 1.0],
            ],
            dtype=np.float32,
        ),
        1: np.array(
            [
                [5.0, 20.0, 80.0, 90.0, 0.0],
                [40.0, 10.0, 120.0, 50.0, 2.0],
            ],
            dtype=np.float32,
        ),
        2: np.array([[20.0, 30.0, 110.0, 85.0, 1.0]], dtype=np.float32),
        3: np.array(
            [
                [15.0, 5.0, 45.0, 40.0, 0.0],
                [60.0, 55.0, 125.0, 92.0, 1.0],
                [30.0, 60.0, 75.0, 95.0, 2.0],
            ],
            dtype=np.float32,
        ),
    }
    return boxes[key].copy()


def _obb_targets():
    """Pixel xyxy + class + angle (radians)."""
    t = _boxes_xyxy_cls(0)
    angles = np.array([[0.3], [-0.9], [1.2], [0.0]], dtype=np.float32)
    return np.hstack([t, angles]).astype(np.float32)


def _segments(key=0, with_dense=False):
    """Per-instance polygon rings matching _boxes_xyxy_cls(key) rows."""
    boxes = _boxes_xyxy_cls(key)
    segments = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b[:4]
        # A pentagon-ish ring inside the box so rasterization is nontrivial.
        ring = np.array(
            [
                [x1, y1],
                [x2, y1],
                [x2, (y1 + y2) / 2],
                [(x1 + x2) / 2, y2],
                [x1, y2],
            ],
            dtype=np.float32,
        )
        if with_dense and i == 0:
            import cv2

            from libreyolo.data.dataset import DenseMaskRing

            mask = np.zeros((97, 131), dtype=np.uint8)
            cv2.fillPoly(mask, [np.round(ring).astype(np.int32)], color=1)
            segments.append([DenseMaskRing(ring, mask)])
        else:
            segments.append([ring])
    return segments


def _pose_inputs(num_keypoints=5):
    """Normalized cxcywh boxes, classes, and (N, K, 3) normalized keypoints."""
    bboxes = np.array(
        [
            [0.30, 0.40, 0.25, 0.35],
            [0.70, 0.55, 0.30, 0.40],
            [0.50, 0.85, 0.20, 0.22],
        ],
        dtype=np.float32,
    )
    cls = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    rng = np.random.RandomState(7)
    kpts = np.zeros((3, num_keypoints, 3), dtype=np.float32)
    for i, b in enumerate(bboxes):
        cx, cy, w, h = b
        kpts[i, :, 0] = np.clip(cx + (rng.rand(num_keypoints) - 0.5) * w, 0, 1)
        kpts[i, :, 1] = np.clip(cy + (rng.rand(num_keypoints) - 0.5) * h, 0, 1)
        kpts[i, :, 2] = rng.choice([0.0, 1.0, 2.0], size=num_keypoints, p=[0.15, 0.25, 0.6])
    return bboxes, cls, kpts


FLIP_IDX_5 = [0, 2, 1, 4, 3]


class _StubDetDataset:
    """Minimal dataset exposing the pull_item/load_anno contract."""

    def __init__(self, n=6, with_segments=False, dense_first=False):
        self.n = n
        self.with_segments = with_segments
        self.dense_first = dense_first

    def __len__(self):
        return self.n

    def load_anno(self, idx):
        return _boxes_xyxy_cls(idx % 4)

    def pull_item(self, idx):
        img = _image(key=idx % 4)
        label = _boxes_xyxy_cls(idx % 4)
        if self.with_segments:
            segments = _segments(idx % 4, with_dense=self.dense_first and idx % 4 == 0)
            return img, label, (img.shape[0], img.shape[1]), idx, segments
        return img, label, (img.shape[0], img.shape[1]), idx


# ---------------------------------------------------------------------------
# Pipeline cases. Each returns a dict of arrays to pin.
# ---------------------------------------------------------------------------


def _case_yolox_train():
    from libreyolo.training.augment import TrainTransform

    t = TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0)
    _seed_all()
    img, labels = t(_image(), _boxes_xyxy_cls(0), (64, 64))
    return {"img": img, "labels": labels}


def _case_yolox_train_empty():
    from libreyolo.training.augment import TrainTransform

    t = TrainTransform(max_labels=50)
    _seed_all()
    img, labels = t(_image(), np.zeros((0, 5), dtype=np.float32), (64, 64))
    return {"img": img, "labels": labels}


def _case_yolox_val():
    from libreyolo.training.augment import ValTransform

    t = ValTransform()
    _seed_all()
    img, labels = t(_image(), None, (64, 64))
    return {"img": img, "labels": labels}


def _case_yolox_mosaic_mixup():
    from libreyolo.training.augment import MosaicMixupDataset, TrainTransform

    ds = MosaicMixupDataset(
        _StubDetDataset(),
        img_size=(64, 64),
        mosaic=True,
        preproc=TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0),
        degrees=10.0,
        translate=0.1,
        mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5),
        shear=2.0,
        enable_mixup=True,
        mosaic_prob=1.0,
        mixup_prob=1.0,
    )
    _seed_all()
    img, labels, _info, _id = ds[1]
    return {"img": img, "labels": labels}


def _case_yolonas_train():
    from libreyolo.models.yolonas.transforms import YOLONASTrainTransform

    t = YOLONASTrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=0.5)
    _seed_all()
    img, labels = t(_image(), _boxes_xyxy_cls(0), (64, 64))
    return {"img": img, "labels": labels}


def _case_yolonas_affine_mixup():
    from libreyolo.models.yolonas.transforms import (
        YOLONASAffineMixupDataset,
        YOLONASTrainTransform,
    )

    ds = YOLONASAffineMixupDataset(
        _StubDetDataset(),
        img_size=(64, 64),
        preproc=YOLONASTrainTransform(max_labels=50),
        degrees=0.0,
        translate=0.25,
        mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5),
        shear=0.0,
        enable_mixup=True,
        mixup_prob=1.0,
    )
    _seed_all()
    img, labels, _info, _id = ds[2]
    return {"img": img, "labels": labels}


def _case_yolo9_train_det():
    from libreyolo.models.yolo9.transforms import YOLO9TrainTransform

    t = YOLO9TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0)
    _seed_all()
    img, labels = t(_image(), _boxes_xyxy_cls(0), (64, 64))
    return {"img": img, "labels": labels}


def _case_yolo9_train_obb():
    from libreyolo.models.yolo9.transforms import YOLO9TrainTransform

    t = YOLO9TrainTransform(
        max_labels=50,
        flip_prob=0.5,
        vertical_flip_prob=0.5,
        hsv_prob=1.0,
        output_label_dim=6,
    )
    _seed_all()
    img, labels = t(_image(), _obb_targets(), (64, 64))
    return {"img": img, "labels": labels}


def _case_yolo9_train_seg():
    from libreyolo.models.yolo9.transforms import YOLO9TrainTransform

    t = YOLO9TrainTransform(max_labels=50, flip_prob=1.0, hsv_prob=1.0)
    _seed_all()
    img, labels, masks = t(
        _image(), _boxes_xyxy_cls(0), (64, 64), segments=_segments(0, with_dense=True)
    )
    return {"img": img, "labels": labels, "masks": masks}


def _case_yolo9_mosaic_det():
    from libreyolo.models.yolo9.transforms import (
        YOLO9MosaicMixupDataset,
        YOLO9TrainTransform,
    )

    ds = YOLO9MosaicMixupDataset(
        _StubDetDataset(),
        img_size=(64, 64),
        preproc=YOLO9TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0),
        mosaic_prob=1.0,
    )
    _seed_all()
    img, labels, _info, _id = ds[0]
    return {"img": img, "labels": labels}


def _case_yolo9_mosaic_seg():
    from libreyolo.models.yolo9.transforms import (
        YOLO9MosaicMixupDataset,
        YOLO9TrainTransform,
    )

    ds = YOLO9MosaicMixupDataset(
        _StubDetDataset(with_segments=True, dense_first=True),
        img_size=(64, 64),
        preproc=YOLO9TrainTransform(max_labels=50, flip_prob=0.5, hsv_prob=1.0),
        mosaic_prob=1.0,
    )
    _seed_all()
    img, labels, _info, _id, masks = ds[0]
    return {"img": img, "labels": labels, "masks": masks}


def _case_yolo9_val():
    from libreyolo.models.yolo9.transforms import YOLO9ValTransform

    t = YOLO9ValTransform()
    _seed_all()
    img, labels = t(_image(), None, (64, 64))
    return {"img": img, "labels": labels}


def _case_deim_train():
    from libreyolo.models.deim.transforms import DEIMTrainTransform

    t = DEIMTrainTransform(max_labels=20, flip_prob=0.5, imgsz=64)
    _seed_all()
    img, labels = t(_image(), _boxes_xyxy_cls(0), (64, 64))
    return {"img": img, "labels": labels}


def _case_deim_train_weak_only():
    from libreyolo.models.deim.transforms import DEIMTrainTransform

    t = DEIMTrainTransform(max_labels=20, flip_prob=0.5, imgsz=64, strong_augs=False)
    _seed_all()
    img, labels = t(_image(), _boxes_xyxy_cls(1), (64, 64))
    return {"img": img, "labels": labels}


def _case_deim_imagenet_norm():
    from libreyolo.models.deim.transforms import DEIMTrainTransform

    t = DEIMTrainTransform(max_labels=20, imgsz=64, imagenet_norm=True)
    _seed_all()
    img, labels = t(_image(key=2), _boxes_xyxy_cls(2), (64, 64))
    return {"img": img, "labels": labels}


def _case_deim_multiscale_collate():
    from libreyolo.models.deim.transforms import (
        DEIMMultiScaleCollate,
        DEIMTrainTransform,
    )

    t = DEIMTrainTransform(max_labels=20, imgsz=64)
    _seed_all()
    item0 = t(_image(key=0), _boxes_xyxy_cls(0), (64, 64))
    item1 = t(_image(key=1), _boxes_xyxy_cls(1), (64, 64))
    collate = DEIMMultiScaleCollate(base_size=64, base_size_repeat=3, stop_epoch=10)
    collate.set_epoch(0)
    imgs, labels, _infos, _ids = collate(
        [(item0[0], item0[1], (97, 131), 0), (item1[0], item1[1], (97, 131), 1)]
    )
    return {"imgs": imgs.numpy(), "labels": labels.numpy()}


def _case_dfine_train():
    from libreyolo.models.dfine.transforms import DFINETrainTransform

    t = DFINETrainTransform(max_labels=20, flip_prob=0.5, imgsz=64)
    _seed_all()
    img, labels = t(_image(), _boxes_xyxy_cls(0), (64, 64))
    return {"img": img, "labels": labels}


def _case_rfdetr_seg():
    from libreyolo.models.rfdetr.seg_transforms import RFDETRSegTransform

    t = RFDETRSegTransform(
        max_labels=20,
        flip_prob=1.0,
        imgsz=64,
        crop_resize_prob=1.0,
        crop_intermediate_sizes=(80, 96),
        crop_min_size=48,
        crop_max_size=96,
    )
    _seed_all()
    img, labels, masks = t(
        _image(), _boxes_xyxy_cls(0), (64, 64), segments=_segments(0, with_dense=True)
    )
    return {"img": img, "labels": labels, "masks": masks}


def _case_rfdetr_seg_no_crop():
    from libreyolo.models.rfdetr.seg_transforms import RFDETRSegTransform

    t = RFDETRSegTransform(max_labels=20, flip_prob=0.5, imgsz=64)
    _seed_all()
    img, labels, masks = t(
        _image(key=3), _boxes_xyxy_cls(3), (64, 64), segments=_segments(3)
    )
    return {"img": img, "labels": labels, "masks": masks}


def _case_rfdetr_det():
    from libreyolo.models.rfdetr.seg_transforms import RFDETRDetTransform

    t = RFDETRDetTransform(max_labels=20, flip_prob=0.5, imgsz=64)
    _seed_all()
    img, labels = t(_image(), _boxes_xyxy_cls(0), (64, 64))
    return {"img": img, "labels": labels}


def _case_rfdetr_det_obb():
    from libreyolo.models.rfdetr.seg_transforms import RFDETRDetTransform

    t = RFDETRDetTransform(max_labels=20, flip_prob=1.0, imgsz=64, target_dim=6)
    _seed_all()
    img, labels = t(_image(), _obb_targets(), (64, 64))
    return {"img": img, "labels": labels}


def _case_rfdetr_pose():
    from libreyolo.models.rfdetr.pose_transforms import RFDETRPoseTransform

    bboxes, cls, kpts = _pose_inputs()
    t = RFDETRPoseTransform(
        num_keypoints=5,
        flip_idx=FLIP_IDX_5,
        max_labels=20,
        flip_prob=1.0,
        imgsz=64,
        crop_resize_prob=1.0,
        crop_intermediate_sizes=(80, 96),
        crop_min_size=48,
        crop_max_size=96,
    )
    _seed_all()
    img, target = t(_image(), bboxes, cls, kpts, (64, 64))
    return {"img": img, "target": target}


def _case_yolo9_pose_train():
    from libreyolo.models.yolo9.pose_transforms import YOLO9PoseTrainTransform

    bboxes, cls, kpts = _pose_inputs()
    t = YOLO9PoseTrainTransform(
        num_keypoints=5, flip_idx=FLIP_IDX_5, max_labels=20, flip_prob=0.5
    )
    _seed_all()
    img, target = t(_image(), bboxes, cls, kpts, (64, 64))
    return {"img": img, "target": target}


def _case_yolo9_pose_val():
    from libreyolo.models.yolo9.pose_transforms import YOLO9PoseValTransform

    bboxes, cls, kpts = _pose_inputs()
    t = YOLO9PoseValTransform(num_keypoints=5, max_labels=20)
    _seed_all()
    img, target = t(_image(), bboxes, cls, kpts, (64, 64))
    return {"img": img, "target": target}


def _case_yolonas_pose_train():
    from libreyolo.models.yolonas.pose_transforms import YOLONASPoseTrainTransform

    bboxes, cls, kpts = _pose_inputs()
    t = YOLONASPoseTrainTransform(
        num_keypoints=5, flip_idx=FLIP_IDX_5, max_labels=20
    )
    _seed_all()
    img, target = t(_image(), bboxes, cls, kpts, (64, 64))
    return {"img": img, "target": target}


def _case_yolonas_pose_val():
    from libreyolo.models.yolonas.pose_transforms import YOLONASPoseValTransform

    bboxes, cls, kpts = _pose_inputs()
    t = YOLONASPoseValTransform(num_keypoints=5, max_labels=20)
    _seed_all()
    img, target = t(_image(), bboxes, cls, kpts, (64, 64))
    return {"img": img, "target": target}


def _case_ec_pose_train():
    from libreyolo.models.ec.pose_transforms import ECPoseTrainTransform

    bboxes, cls, kpts = _pose_inputs()
    t = ECPoseTrainTransform(num_keypoints=5, flip_idx=FLIP_IDX_5, max_labels=20)
    _seed_all()
    img, target = t(_image(), bboxes, cls, kpts, (64, 64))
    return {"img": img, "target": target}


def _case_ec_pose_val():
    from libreyolo.models.ec.pose_transforms import ECPoseValTransform

    bboxes, cls, kpts = _pose_inputs()
    t = ECPoseValTransform(num_keypoints=5, max_labels=20)
    _seed_all()
    img, target = t(_image(), bboxes, cls, kpts, (64, 64))
    return {"img": img, "target": target}


CASES = {
    "yolox_train": _case_yolox_train,
    "yolox_train_empty": _case_yolox_train_empty,
    "yolox_val": _case_yolox_val,
    "yolox_mosaic_mixup": _case_yolox_mosaic_mixup,
    "yolonas_train": _case_yolonas_train,
    "yolonas_affine_mixup": _case_yolonas_affine_mixup,
    "yolo9_train_det": _case_yolo9_train_det,
    "yolo9_train_obb": _case_yolo9_train_obb,
    "yolo9_train_seg": _case_yolo9_train_seg,
    "yolo9_mosaic_det": _case_yolo9_mosaic_det,
    "yolo9_mosaic_seg": _case_yolo9_mosaic_seg,
    "yolo9_val": _case_yolo9_val,
    "deim_train": _case_deim_train,
    "deim_train_weak_only": _case_deim_train_weak_only,
    "deim_imagenet_norm": _case_deim_imagenet_norm,
    "deim_multiscale_collate": _case_deim_multiscale_collate,
    "dfine_train": _case_dfine_train,
    "rfdetr_seg": _case_rfdetr_seg,
    "rfdetr_seg_no_crop": _case_rfdetr_seg_no_crop,
    "rfdetr_det": _case_rfdetr_det,
    "rfdetr_det_obb": _case_rfdetr_det_obb,
    "rfdetr_pose": _case_rfdetr_pose,
    "yolo9_pose_train": _case_yolo9_pose_train,
    "yolo9_pose_val": _case_yolo9_pose_val,
    "yolonas_pose_train": _case_yolonas_pose_train,
    "yolonas_pose_val": _case_yolonas_pose_val,
    "ec_pose_train": _case_ec_pose_train,
    "ec_pose_val": _case_ec_pose_val,
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_augment_parity(name):
    fixture = FIXTURE_DIR / f"{name}.npz"
    if not fixture.exists():
        pytest.fail(
            f"Missing golden fixture {fixture}. Run "
            f"`python tests/unit/test_augment_parity.py --generate`."
        )
    expected = np.load(fixture)
    actual = CASES[name]()
    assert set(expected.files) == set(actual), (
        f"Output keys changed for {name}: {sorted(actual)} vs {sorted(expected.files)}"
    )
    for key in expected.files:
        np.testing.assert_array_equal(
            np.asarray(actual[key]),
            expected[key],
            err_msg=f"{name}/{key} diverged from golden fixture",
        )


def _generate():
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in sorted(CASES.items()):
        out = fn()
        np.savez_compressed(FIXTURE_DIR / f"{name}.npz", **out)
        shapes = {k: tuple(np.asarray(v).shape) for k, v in out.items()}
        print(f"wrote {name}.npz {shapes}")


if __name__ == "__main__":
    if "--generate" in sys.argv:
        _generate()
    else:
        print(__doc__)
