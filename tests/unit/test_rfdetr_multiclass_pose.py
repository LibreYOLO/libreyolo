"""RF-DETR GroupPose on multi-class pose datasets (#872).

Every class shares the ``kpt_shape`` skeleton; ``kpt_names`` keyed by class
narrows a class to fewer keypoints, and an empty list declares a box-only class.
The GroupPose schema is ``[0, count_0, count_1, ...]`` and contiguous class ``j``
is schema index ``j + 1``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo.data.pose_metadata import keypoints_per_class
from libreyolo.models.rfdetr.keypoints import (
    keypoint_schema_label_offset,
    map_labels_to_keypoint_schema,
)

pytestmark = [pytest.mark.unit, pytest.mark.rfdetr]


# ---------------------------------------------------------------------------
# Dataset contract: per-class keypoint counts
# ---------------------------------------------------------------------------
def test_keypoints_per_class_defaults_every_class_to_kpt_shape():
    assert keypoints_per_class({"kpt_shape": [4, 3]}, 3, 4) == [4, 4, 4]
    assert keypoints_per_class({"kpt_names": ["a", "b"]}, 2, 4) == [4, 4]


def test_keypoints_per_class_reads_kpt_names_by_index_and_name():
    cfg = {
        "names": ["marker", "blank", "plate"],
        "kpt_names": {"marker": ["tail", "head"], 1: [], "2": ["a", "b", "c"]},
    }
    assert keypoints_per_class(cfg, 3, 4) == [2, 0, 3]


def test_keypoints_per_class_keeps_unlisted_classes_at_full_count():
    cfg = {"names": {0: "a", 1: "b"}, "kpt_names": {0: ["x"]}}
    assert keypoints_per_class(cfg, 2, 4) == [1, 4]


def test_keypoints_per_class_prefers_digit_class_names_over_indices():
    cfg = {"names": ["7", "0"], "kpt_names": {"0": [], 0: ["x"]}}
    # String "0" is the name of class 1; int 0 is class index 0.
    assert keypoints_per_class(cfg, 2, 4) == [1, 0]
    assert keypoints_per_class({"names": ["a", "b"], "kpt_names": {"1": []}}, 2, 4) == [4, 0]


@pytest.mark.parametrize(
    "kpt_names, match",
    [
        ({"unknown": ["x"]}, "neither a class index nor a class name"),
        ({5: ["x"]}, "outside the dataset classes"),
        ({0: ["a", "b", "c", "d", "e"]}, "kpt_shape allows 4"),
        ({0: "tail"}, "must be a list"),
        ({0: ["x"], "a": []}, "more than once"),
    ],
)
def test_keypoints_per_class_rejects_invalid_kpt_names(kpt_names, match):
    cfg = {"names": ["a", "b"], "kpt_names": kpt_names}
    with pytest.raises(ValueError, match=match):
        keypoints_per_class(cfg, 2, 4)


# ---------------------------------------------------------------------------
# Label mapping: contiguous class j -> schema index j + 1
# ---------------------------------------------------------------------------
def test_label_mapping_keeps_zero_keypoint_classes_in_their_own_slot():
    schema = [0, 2, 0, 4]
    assert keypoint_schema_label_offset(schema) == 1
    mapped = map_labels_to_keypoint_schema(torch.tensor([0, 1, 2]), schema)
    assert mapped.tolist() == [1, 2, 3]


@pytest.mark.parametrize(
    "schema, labels, expected",
    [
        ([0, 17], [0], [1]),
        ([0, 17, 4], [0, 1], [1, 2]),
        ([17], [0], [0]),
        ([0, 17], [1], [1]),  # out of range passes through unchanged
    ],
)
def test_label_mapping_matches_previous_behavior_on_existing_schemas(schema, labels, expected):
    assert map_labels_to_keypoint_schema(torch.tensor(labels), schema).tolist() == expected


# ---------------------------------------------------------------------------
# PyTorch postprocess
# ---------------------------------------------------------------------------
def test_postprocess_keeps_zero_keypoint_class_and_pads_per_class():
    from libreyolo.postprocess.rfdetr import postprocess

    schema = [0, 2, 0, 4]
    num_queries, width = 3, len(schema)
    logits = torch.full((1, num_queries, width), -10.0)
    logits[0, 0, 1] = 6.0  # contiguous 0, two keypoints
    logits[0, 1, 2] = 5.0  # contiguous 1, no keypoints
    logits[0, 2, 3] = 4.0  # contiguous 2, four keypoints
    boxes = torch.full((1, num_queries, 4), 0.5)
    keypoints = torch.zeros(1, num_queries, width * max(schema), 8)
    keypoints[..., 0] = 0.25
    keypoints[..., 1] = 0.75
    keypoints[..., 2] = 2.0

    result = postprocess(
        {"pred_logits": logits, "pred_boxes": boxes, "pred_keypoints": keypoints},
        torch.tensor([[80.0, 120.0]]),
        num_select=3,
        num_keypoints_per_class=schema,
        trace_alpha=0.0,
    )[0]

    assert result["labels"].tolist() == [0, 1, 2]
    kpts = result["keypoints"]
    assert kpts.shape == (3, 4, 3)
    assert (kpts[0, :2, 2] > 0).all() and (kpts[0, 2:] == 0).all()
    assert (kpts[1] == 0).all()
    assert (kpts[2, :, 2] > 0).all()


def test_postprocess_drops_empty_slot_and_columns_past_schema():
    from libreyolo.postprocess.rfdetr import postprocess

    schema = [0, 2]
    logits = torch.full((1, 3, 3), -10.0)
    logits[0, 0, 0] = 6.0  # empty slot
    logits[0, 1, 2] = 5.0  # head column past the schema
    logits[0, 2, 1] = 4.0  # the only schema class
    keypoints = torch.zeros(1, 3, 2 * 2, 8)

    result = postprocess(
        {
            "pred_logits": logits,
            "pred_boxes": torch.full((1, 3, 4), 0.5),
            "pred_keypoints": keypoints,
        },
        torch.tensor([[80.0, 120.0]]),
        num_select=3,
        num_keypoints_per_class=schema,
        trace_alpha=0.0,
    )[0]

    assert result["labels"].tolist() == [0]


# ---------------------------------------------------------------------------
# Runtime (exported) backend postprocess
# ---------------------------------------------------------------------------
def test_backend_keeps_zero_keypoint_class():
    from libreyolo.backends.base import BaseBackend

    class _Backend(BaseBackend):
        def _run_inference(self, blob):
            raise NotImplementedError

    backend = _Backend(
        model_path="dummy",
        nb_classes=3,
        device="cpu",
        imgsz=192,
        model_family="rfdetr",
        names={0: "marker", 1: "blank", 2: "plate"},
        task="pose",
        supported_tasks=("detect", "pose"),
        num_keypoints_per_class=[0, 2, 0, 4],
    )
    boxes = np.array([[[0.5, 0.5, 0.2, 0.4]] * 2], dtype=np.float32)
    logits = np.full((1, 2, 4), -10.0, dtype=np.float32)
    logits[0, 0, 2] = 10.0  # contiguous 1, no keypoints
    logits[0, 1, 1] = 9.0  # contiguous 0, two keypoints
    keypoints = np.zeros((1, 2, 16, 8), dtype=np.float32)
    keypoints[0, 1, 4:6, :7] = [0.25, 0.5, 2.0, 0.0, 0.0, 1.0, 0.0]

    _, _, classes, _, _, parsed_keypoints = backend._parse_rfdetr(
        [boxes, logits, keypoints], orig_w=200, orig_h=100, conf=0.5
    )

    assert sorted(classes.tolist()) == [0, 1]
    assert parsed_keypoints.shape == (2, 4, 3)
    blank = parsed_keypoints[classes.tolist().index(1)]
    np.testing.assert_allclose(blank, 0.0)


# ---------------------------------------------------------------------------
# Train entry point
# ---------------------------------------------------------------------------
class _StopTraining(Exception):
    pass


def _write_pose_yaml(tmp_path, body: str):
    for split in ("train", "val"):
        (tmp_path / split / "images").mkdir(parents=True)
        (tmp_path / split / "labels").mkdir(parents=True)
    path = tmp_path / "data.yaml"
    path.write_text(f"path: {tmp_path}\ntrain: train/images\nval: val/images\n{body}")
    return path


def _capture_train(monkeypatch, data):
    import libreyolo.models.rfdetr.model as rfdetr_model

    captured = {}

    class _FakeTrainer:
        def __init__(self, model, **kwargs):
            captured.update(kwargs)
            raise _StopTraining

    monkeypatch.setattr(rfdetr_model, "RFDETRTrainer", _FakeTrainer)
    model = rfdetr_model.LibreRFDETR(task="pose", size="x", device="cpu")
    with pytest.raises(_StopTraining):
        model.train(data=str(data), epochs=1, imgsz=192)
    return model, captured


def test_train_accepts_multiclass_pose_with_per_class_keypoints(tmp_path, monkeypatch):
    data = _write_pose_yaml(
        tmp_path,
        "nc: 3\nnames: [marker, blank, plate]\nkpt_shape: [4, 3]\n"
        "kpt_names:\n  marker: [tail, head]\n  blank: []\n",
    )
    model, captured = _capture_train(monkeypatch, data)

    inner = model.model.model
    assert inner.get_num_keypoints_per_class() == [0, 2, 0, 4]
    assert model.model.args.num_keypoints_per_class == [0, 2, 0, 4]
    assert model.nb_classes == 3
    assert model.names == {0: "marker", 1: "blank", 2: "plate"}
    assert captured["num_classes"] == 3
    assert captured["num_keypoints"] == 4


def test_train_person_only_pose_keeps_the_existing_schema(tmp_path, monkeypatch):
    data = _write_pose_yaml(tmp_path, "nc: 1\nkpt_shape: [17, 3]\n")
    model, captured = _capture_train(monkeypatch, data)

    assert model.model.model.get_num_keypoints_per_class() == [0, 17]
    assert model.names == {0: "person"}
    assert captured["num_classes"] == 1


def test_train_rejects_dataset_without_any_keypoints(tmp_path, monkeypatch):
    import libreyolo.models.rfdetr.model as rfdetr_model

    data = _write_pose_yaml(
        tmp_path,
        "nc: 2\nnames: [a, b]\nkpt_shape: [2, 3]\nkpt_names:\n  a: []\n  b: []\n",
    )
    model = rfdetr_model.LibreRFDETR(task="pose", size="x", device="cpu")
    with pytest.raises(ValueError, match="at least one class with keypoints"):
        model.train(data=str(data), epochs=1, imgsz=192)


def test_train_rejects_single_cls_on_pose(tmp_path):
    import libreyolo.models.rfdetr.model as rfdetr_model

    data = _write_pose_yaml(tmp_path, "nc: 2\nnames: [a, b]\nkpt_shape: [2, 3]\n")
    model = rfdetr_model.LibreRFDETR(task="pose", size="x", device="cpu")
    with pytest.raises(ValueError, match="supported only for G0/G1 detection"):
        model.train(data=str(data), epochs=1, imgsz=192, single_cls=True)


def test_train_requires_names_for_multiclass_pose(tmp_path):
    import libreyolo.models.rfdetr.model as rfdetr_model

    data = _write_pose_yaml(tmp_path, "nc: 2\nkpt_shape: [2, 3]\n")
    model = rfdetr_model.LibreRFDETR(task="pose", size="x", device="cpu")
    with pytest.raises(ValueError, match="needs ``names``"):
        model.train(data=str(data), epochs=1, imgsz=192)


def test_trainer_setup_sizes_head_and_criterion_to_the_dataset_schema(tmp_path):
    """``RFDETRTrainer.on_setup`` applies the schema on its own, from a [0, 17] model."""
    from libreyolo.models.rfdetr.model import LibreRFDETR
    from libreyolo.models.rfdetr.trainer import RFDETRTrainer

    data = _write_pose_yaml(
        tmp_path,
        "nc: 3\nnames: [marker, blank, plate]\nkpt_shape: [4, 3]\n"
        "kpt_names:\n  marker: [tail, head]\n  blank: []\n",
    )
    model = LibreRFDETR(task="pose", size="x", device="cpu")
    assert model.model.model.get_num_keypoints_per_class() == [0, 17]

    trainer = RFDETRTrainer(
        model.model,
        wrapper_model=model,
        data=str(data),
        epochs=1,
        imgsz=192,
        device="cpu",
        size="x",
    )
    trainer.model.to(trainer.device)
    trainer.on_setup()

    inner = trainer.model.model
    assert trainer.config.num_classes == 3
    assert inner.get_num_keypoints_per_class() == [0, 2, 0, 4]
    assert inner.class_embed.out_features == 4
    assert trainer.model.args.num_keypoints_per_class == [0, 2, 0, 4]
    assert list(trainer.criterion.num_keypoints_per_class) == [0, 2, 0, 4]
    assert list(trainer.criterion.matcher.num_keypoints_per_class) == [0, 2, 0, 4]
    assert model.names == {0: "marker", 1: "blank", 2: "plate"}
    assert trainer._checkpoint_extra_metadata()["num_keypoints_per_class"] == [0, 2, 0, 4]


# ---------------------------------------------------------------------------
# No class uses every kpt_shape row
# ---------------------------------------------------------------------------
def test_loss_slices_targets_wider_than_the_schema_slots():
    from libreyolo.models.rfdetr.loss import SetCriterion
    from libreyolo.models.rfdetr.matcher import HungarianMatcher

    schema = [0, 2, 3]  # kpt_shape has 4 rows; no class uses all of them
    matcher = HungarianMatcher(num_keypoints_per_class=schema)
    criterion = SetCriterion(
        num_classes=3,
        matcher=matcher,
        weight_dict={},
        focal_alpha=0.25,
        losses=["keypoints"],
        group_detr=1,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=schema,
    )
    target_keypoints = torch.rand(2, 4, 3)
    target_keypoints[..., 2] = 2.0
    targets = [
        {
            "labels": torch.tensor([0, 1]),
            "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]]),
            "keypoints": target_keypoints,
        }
    ]
    outputs = {"pred_keypoints": torch.randn(1, 5, len(schema) * max(schema), 8)}
    indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]

    losses = criterion.loss_keypoints(outputs, targets, indices, num_boxes=2.0)

    assert all(torch.isfinite(v) for v in losses.values())
    assert float(losses["loss_keypoints_l1"]) > 0.0


def test_backend_pads_keypoints_to_the_declared_skeleton():
    from libreyolo.backends.base import BaseBackend

    class _Backend(BaseBackend):
        def _run_inference(self, blob):
            raise NotImplementedError

    backend = _Backend(
        model_path="dummy",
        nb_classes=1,
        device="cpu",
        imgsz=192,
        model_family="rfdetr",
        names={0: "person"},
        task="pose",
        supported_tasks=("detect", "pose"),
        num_keypoints=4,
        num_keypoints_per_class=[0, 3],
    )
    boxes = np.array([[[0.5, 0.5, 0.2, 0.4]]], dtype=np.float32)
    logits = np.array([[[-10.0, 10.0]]], dtype=np.float32)
    keypoints = np.zeros((1, 1, 6, 8), dtype=np.float32)
    keypoints[0, 0, 3:6, :7] = [0.25, 0.5, 2.0, 0.0, 0.0, 1.0, 0.0]

    _, _, classes, _, _, parsed_keypoints = backend._parse_rfdetr(
        [boxes, logits, keypoints], orig_w=200, orig_h=100, conf=0.5
    )

    assert classes.tolist() == [0]
    assert parsed_keypoints.shape == (1, 4, 3)
    assert (parsed_keypoints[0, :3, 2] > 0).all()
    np.testing.assert_allclose(parsed_keypoints[0, 3], 0.0)


# ---------------------------------------------------------------------------
# Box-only classes reach the loss; new class columns do not start "off"
# ---------------------------------------------------------------------------
def test_pose_targets_keep_box_only_class_instances():
    from libreyolo.data.augment.rfdetr import _build_target

    cls = np.array([0.0, 1.0, 1.0])
    boxes = np.array([[50.0, 50.0, 20.0, 20.0]] * 3)
    kpts = np.zeros((3, 4, 3))
    kpts[0, :2, 2] = 2.0  # class 0 has visible keypoints; both class-1 rows have none

    dropped = _build_target(cls, boxes, kpts, 4, 10)
    kept = _build_target(cls, boxes, kpts, 4, 10, np.array([1]))

    assert (dropped[:, 3] > 0).sum() == 1  # person-only behavior: no visible keypoint, no target
    assert (kept[:, 3] > 0).sum() == 3
    assert kept[:3, 0].tolist() == [0.0, 1.0, 1.0]


def test_grouppose_class_head_resize_copies_class_rows_not_the_empty_slot():
    from libreyolo.models.rfdetr.model import LibreRFDETR

    inner = LibreRFDETR(task="pose", size="x", device="cpu").model.model
    old = inner.class_embed
    empty_row, person_row = old.weight[0].clone(), old.weight[1].clone()

    inner.reinitialize_grouppose_class_head(4)

    new = inner.class_embed
    assert new.out_features == 4
    assert torch.equal(new.weight[0], empty_row)
    for column in (1, 2, 3):
        assert torch.equal(new.weight[column], person_row)
    for enc in inner.transformer.enc_out_class_embed:
        assert enc.out_features == 4
