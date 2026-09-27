"""YOLO ground-truth geometry must match in training and validation (#814).

The validation parser (``data/yolo_coco_api.py``) has always clamped boxes to
the image and dropped the ones left with no area. The YOLO txt training parser
did not, so a label crossing the border produced one box for training and a
different one for scoring.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from libreyolo.data.dataset import YOLODataset
from libreyolo.data.yolo_coco_api import create_yolo_coco_api

pytestmark = pytest.mark.unit

IMG = 160


def _make_dataset(tmp_path, rows, nc=1):
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True)
        (tmp_path / "labels" / split).mkdir(parents=True)
        cv2.imwrite(
            str(tmp_path / "images" / split / "0.jpg"),
            np.full((IMG, IMG, 3), 40, np.uint8),
        )
        (tmp_path / "labels" / split / "0.txt").write_text("\n".join(rows) + "\n")
    names = "\n".join(f"  {i}: c{i}" for i in range(nc))
    (tmp_path / "data.yaml").write_text(
        f"path: {tmp_path}\ntrain: images/train\nval: images/val\nnc: {nc}\nnames:\n{names}\n"
    )
    return tmp_path / "data.yaml"


def _train_boxes(root, **kwargs):
    ds = YOLODataset(data_dir=str(root), split="train", img_size=(IMG, IMG), **kwargs)
    return np.asarray(ds.load_anno(0))


def _val_boxes(yaml_path):
    api = create_yolo_coco_api(str(yaml_path), split="val")
    out = []
    for ann in api.anns.values():
        x, y, w, h = ann["bbox"]
        out.append([x, y, x + w, y + h])
    return np.asarray(sorted(out, key=lambda b: (b[0], b[1])), dtype=np.float32)


@pytest.mark.parametrize(
    "row,label",
    [
        # Polygon running off the left and top edges.
        ("0 -0.12 -0.05 0.30 -0.02 0.34 0.40 -0.08 0.36", "polygon"),
        # Plain box row whose extent crosses the right edge. Not polygon
        # specific: any out-of-range label hit this.
        ("0 0.95 0.50 0.30 0.40", "box"),
    ],
)
def test_border_crossing_label_matches_between_train_and_val(tmp_path, row, label):
    yaml_path = _make_dataset(tmp_path, [row])

    train = _train_boxes(tmp_path)[:, :4]
    val = _val_boxes(yaml_path)

    assert train.shape == (1, 4), f"{label}: expected one training box"
    np.testing.assert_allclose(train, val, atol=1e-4)
    assert train.min() >= 0.0
    assert train[:, [0, 2]].max() <= IMG
    assert train[:, [1, 3]].max() <= IMG


def test_label_entirely_outside_the_image_is_dropped_by_both_paths(tmp_path):
    yaml_path = _make_dataset(tmp_path, ["0 -0.50 0.50 0.20 0.20"])

    assert len(_train_boxes(tmp_path)) == 0
    assert len(_val_boxes(yaml_path)) == 0


def test_in_frame_labels_are_untouched(tmp_path):
    """Regression guard: clamping must not perturb ordinary labels."""
    yaml_path = _make_dataset(tmp_path, ["0 0.50 0.50 0.40 0.30"])

    train = _train_boxes(tmp_path)[:, :4]
    expected = np.array([[0.30 * IMG, 0.35 * IMG, 0.70 * IMG, 0.65 * IMG]], np.float32)

    np.testing.assert_allclose(train, expected, atol=1e-4)
    np.testing.assert_allclose(train, _val_boxes(yaml_path), atol=1e-4)


def test_dropping_a_row_keeps_segments_aligned_with_labels(tmp_path):
    """A dropped label must drop its segment, or every later row desyncs."""
    _make_dataset(
        tmp_path,
        [
            "0 -0.60 0.40 -0.40 0.40 -0.40 0.60 -0.60 0.60",  # dropped
            "0 0.30 0.30 0.20 0.20",  # kept
            "0 0.70 0.70 1.10 0.70 1.10 1.10 0.70 1.10",  # clipped
        ],
    )

    ds = YOLODataset(
        data_dir=str(tmp_path), split="train", img_size=(IMG, IMG), load_segments=True
    )
    boxes = np.asarray(ds.load_anno(0))
    segments = ds.load_segments_for_index(0) if hasattr(
        ds, "load_segments_for_index"
    ) else ds.segments[0]

    assert len(boxes) == 2
    assert len(segments) == len(boxes)

    # Each surviving segment must sit inside its own box, which is only true
    # if the pairing did not shift when the first row was dropped.
    for box, seg in zip(boxes, segments):
        ring = np.asarray(seg[0], dtype=np.float32).reshape(-1, 2)
        assert ring[:, 0].min() >= box[0] - 1e-3
        assert ring[:, 1].min() >= box[1] - 1e-3
        assert ring[:, 0].max() <= box[2] + 1e-3
        assert ring[:, 1].max() <= box[3] + 1e-3


@pytest.mark.parametrize(
    "row",
    [
        "0 nan nan 0.50 0.50",
        "0 0.50 0.50 inf 0.50",
        "0 nan 0.10 0.50 0.10 0.50 0.50",
    ],
)
def test_non_finite_coordinates_are_dropped_by_both_paths(tmp_path, row):
    """Corrupt numeric values must never become full-image targets."""
    yaml_path = _make_dataset(tmp_path, [row])

    with pytest.warns(UserWarning, match="coordinates must be finite"):
        assert len(_train_boxes(tmp_path)) == 0
    with pytest.warns(UserWarning, match="coordinates must be finite"):
        assert len(_val_boxes(yaml_path)) == 0


@pytest.mark.parametrize(
    "row",
    [
        "0 0.10 0.10 0.90 0.10 0.90",  # odd coordinate count
        "0 0.10 0.10 0.90 0.10 0.90 0.90 0.10",  # odd coordinate count
    ],
)
def test_malformed_polygons_are_dropped_by_both_paths(tmp_path, row):
    yaml_path = _make_dataset(tmp_path, [row])

    with pytest.warns(UserWarning, match="coordinate pairs"):
        assert len(_train_boxes(tmp_path)) == 0
    with pytest.warns(UserWarning, match="coordinate pairs"):
        assert len(_val_boxes(yaml_path)) == 0


def test_invalid_class_is_dropped_when_training_knows_class_count(tmp_path):
    yaml_path = _make_dataset(tmp_path, ["1 0.50 0.50 0.20 0.20"], nc=1)

    with pytest.warns(UserWarning, match="out of range"):
        assert len(_train_boxes(tmp_path, num_classes=1)) == 0
    with pytest.warns(UserWarning, match="out of range"):
        assert len(_val_boxes(yaml_path)) == 0


def test_clipped_polygon_reaches_flagship_transforms_with_visible_box(tmp_path):
    """YOLO9 and RF-DETR must receive the same clipped object extent."""
    from libreyolo.data.augment.rfdetr import RFDETRDetTransform
    from libreyolo.data.augment.yolo9 import YOLO9TrainTransform

    _make_dataset(
        tmp_path,
        ["0 -0.12 -0.05 0.30 -0.02 0.34 0.40 -0.08 0.36"],
    )
    targets = _train_boxes(tmp_path)
    image = np.zeros((IMG, IMG, 3), dtype=np.uint8)

    yolo9 = YOLO9TrainTransform(
        max_labels=2,
        flip_prob=0.0,
        hsv_prob=0.0,
    )
    _, yolo9_targets = yolo9(image.copy(), targets.copy(), (IMG, IMG))
    np.testing.assert_allclose(
        yolo9_targets[0],
        [0.0, 0.0, 0.0, 0.34, 0.40],
        atol=1e-5,
    )

    rfdetr = RFDETRDetTransform(
        max_labels=2,
        flip_prob=0.0,
        imgsz=IMG,
        imagenet_norm=False,
    )
    _, rfdetr_targets = rfdetr(image.copy(), targets.copy(), (IMG, IMG))
    np.testing.assert_allclose(
        rfdetr_targets[0],
        [0.0, 0.17 * IMG, 0.20 * IMG, 0.34 * IMG, 0.40 * IMG],
        atol=1e-4,
    )
