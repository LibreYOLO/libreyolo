"""Unit coverage for the YOLO-NAS-R (OBB) task of the YOLO-NAS family."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from libreyolo.data.obb import xywhr_to_corners
from libreyolo.models.yolonas.model import LibreYOLONAS
from libreyolo.models.yolonas.nn import (
    LibreYOLONASModel,
    LibreYOLONASOBBModel,
    LibreYOLONASPoseModel,
)
from libreyolo.postprocess.obb_ops import (
    canonicalize_xywhr_tensor,
    rotated_nms_keep_indices,
    xywhr_to_xyxy,
)
from libreyolo.postprocess.yolonas import postprocess_obb


pytestmark = [pytest.mark.unit, pytest.mark.yolonas]


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def obb_model():
    return LibreYOLONASOBBModel(config="s", nb_classes=3).eval()


def test_obb_head_channel_layout(obb_model):
    head = obb_model.heads.head1
    reg_max = obb_model.heads.reg_max
    assert head.reg_pred.out_channels == 2 * (reg_max + 1)
    assert head.offset_pred.out_channels == 2
    assert head.rot_pred.out_channels == 1
    assert head.cls_pred.out_channels == 3
    # Upstream zero-initialises the offset branch so decoding starts at the
    # anchor centre.
    assert torch.count_nonzero(head.offset_pred.weight) == 0
    assert torch.count_nonzero(head.offset_pred.bias) == 0


def test_obb_forward_shapes_and_angle_range(obb_model):
    x = torch.zeros(1, 3, 256, 256)
    with torch.no_grad():
        (boxes, scores), raw = obb_model(x)
    anchors = (256 // 8) ** 2 + (256 // 16) ** 2 + (256 // 32) ** 2
    assert boxes.shape == (1, anchors, 5)
    assert scores.shape == (1, anchors, 3)
    assert raw["size_dist"].shape == (1, anchors, 2 * (obb_model.heads.reg_max + 1))
    assert raw["angles"].shape == (1, anchors, 1)
    assert raw["offsets"].shape == (1, anchors, 2)
    # (sigmoid(x) - 0.25) * pi, i.e. the upstream code's (-pi/4, 3*pi/4).
    angles = boxes[..., 4]
    assert (angles > -math.pi / 4).all() and (angles < 3 * math.pi / 4).all()
    assert (scores >= 0).all() and (scores <= 1).all()


def test_obb_replace_num_classes(obb_model):
    model = LibreYOLONASOBBModel(config="s", nb_classes=3).eval()
    model.replace_num_classes(7)
    with torch.no_grad():
        (_boxes, scores), _raw = model(torch.zeros(1, 3, 128, 128))
    assert scores.shape[-1] == 7
    assert model.nc == 7


def test_obb_export_path_returns_only_decoded(obb_model):
    """Traced graphs must emit exactly (boxes [B,N,5], scores [B,N,C])."""
    traced = torch.jit.trace(obb_model, torch.zeros(1, 3, 128, 128), strict=False)
    with torch.no_grad():
        out = traced(torch.zeros(1, 3, 128, 128))
    assert isinstance(out, (tuple, list)) and len(out) == 2
    assert out[0].shape[-1] == 5 and out[1].shape[-1] == 3


# ---------------------------------------------------------------------------
# Checkpoint / task discrimination
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def state_dicts():
    return {
        "obb": LibreYOLONASOBBModel(config="s", nb_classes=18).state_dict(),
        "detect": LibreYOLONASModel(config="s", nb_classes=80).state_dict(),
        "pose": LibreYOLONASPoseModel(
            config="s", num_keypoints=17, num_classes=1
        ).state_dict(),
    }


def test_family_still_claims_every_task(state_dicts):
    for sd in state_dicts.values():
        assert LibreYOLONAS.can_load(sd) is True


def test_obb_discriminator_is_exclusive(state_dicts):
    assert LibreYOLONAS.is_obb_state_dict(state_dicts["obb"]) is True
    assert LibreYOLONAS.is_obb_state_dict(state_dicts["detect"]) is False
    assert LibreYOLONAS.is_obb_state_dict(state_dicts["pose"]) is False
    # ... and the pose discriminator must not claim the rotated head.
    assert LibreYOLONAS.is_pose_state_dict(state_dicts["obb"]) is False


def test_checkpoint_task_resolution(state_dicts):
    assert LibreYOLONAS.detect_checkpoint_task(state_dicts["obb"]) == "obb"
    assert LibreYOLONAS.detect_checkpoint_task(state_dicts["pose"]) == "pose"
    assert LibreYOLONAS.detect_checkpoint_task(state_dicts["detect"]) is None


def test_obb_size_and_class_count_from_state_dict(state_dicts):
    assert LibreYOLONAS.detect_size(state_dicts["obb"]) == "s"
    assert LibreYOLONAS.detect_nb_classes(state_dicts["obb"]) == 18


@pytest.mark.parametrize("size", ["s", "m", "l"])
def test_obb_filename_and_download_route(size):
    canonical = f"LibreYOLONAS{size}-obb.pt"
    assert LibreYOLONAS.detect_size_from_filename(canonical) == size
    assert LibreYOLONAS.detect_task_from_filename(canonical) == "obb"
    assert LibreYOLONAS.get_download_url(canonical) == (
        f"https://d2gjn4b69gu75n.cloudfront.net/models/yolo_nas_r_{size}_dota2.pth"
    )

    native = f"yolo_nas_r_{size}_dota2.pth"
    assert LibreYOLONAS.detect_size_from_filename(native) == size
    assert LibreYOLONAS.detect_task_from_filename(native) == "obb"


def test_native_detect_and_pose_filenames_are_untouched():
    assert LibreYOLONAS.detect_task_from_filename("yolo_nas_s_coco.pth") != "obb"
    assert (
        LibreYOLONAS.detect_task_from_filename("yolo_nas_pose_s_coco_pose.pth")
        == "pose"
    )


def test_every_rotated_checkpoint_has_a_pinned_hash():
    for size in ("s", "m", "l"):
        name = f"yolo_nas_r_{size}_dota2.pth"
        assert name in LibreYOLONAS._DECI_CHECKPOINT_SHA256
        assert len(LibreYOLONAS._DECI_CHECKPOINT_SHA256[name]) == 64


def test_obb_task_metadata():
    assert "obb" in LibreYOLONAS.SUPPORTED_TASKS
    assert LibreYOLONAS.TASK_INPUT_SIZES["obb"] == {"s": 1024, "m": 1024, "l": 1024}
    # OBB does not ship the pose-only `n` size.
    assert "n" not in LibreYOLONAS.TASK_INPUT_SIZES["obb"]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def test_canonicalization_changes_representation_not_geometry():
    raw = torch.tensor(
        [
            [50.0, 60.0, 10.0, 30.0, 0.3],  # h > w -> must swap
            [50.0, 60.0, 30.0, 10.0, 2.5],  # angle outside [-pi/2, pi/2)
            [10.0, 10.0, 4.0, 4.0, -math.pi / 2],  # boundary
        ]
    )
    out = canonicalize_xywhr_tensor(raw)

    assert (out[:, 2] >= out[:, 3]).all()
    assert (out[:, 4] >= -math.pi / 2).all() and (out[:, 4] < math.pi / 2).all()
    assert torch.equal(out[:, :2], raw[:, :2])

    for before, after in zip(raw.numpy(), out.numpy()):
        poly_a = np.sort(xywhr_to_corners(before).reshape(-1))
        poly_b = np.sort(xywhr_to_corners(after).reshape(-1))
        assert np.allclose(poly_a, poly_b, atol=1e-4)


def test_xywhr_to_xyxy_matches_corner_extents():
    boxes = torch.tensor([[10.0, 20.0, 8.0, 4.0, 0.7], [0.0, 0.0, 6.0, 6.0, 0.0]])
    aabb = xywhr_to_xyxy(boxes).numpy()
    for row, box in zip(aabb, boxes.numpy()):
        corners = xywhr_to_corners(box)
        assert np.allclose(row[0], corners[:, 0].min(), atol=1e-4)
        assert np.allclose(row[1], corners[:, 1].min(), atol=1e-4)
        assert np.allclose(row[2], corners[:, 0].max(), atol=1e-4)
        assert np.allclose(row[3], corners[:, 1].max(), atol=1e-4)


def test_xywhr_to_xyxy_empty():
    assert xywhr_to_xyxy(torch.zeros((0, 5))).shape == (0, 4)


def test_rotated_nms_suppresses_within_class_only():
    boxes = torch.tensor(
        [
            [50.0, 50.0, 40.0, 20.0, 0.0],
            [51.0, 50.0, 40.0, 20.0, 0.0],  # near-duplicate, same class
            [51.0, 50.0, 40.0, 20.0, 0.0],  # near-duplicate, different class
            [300.0, 300.0, 10.0, 10.0, 0.0],  # disjoint
        ]
    )
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    classes = torch.tensor([0, 0, 1, 0])
    keep = rotated_nms_keep_indices(boxes, scores, classes, 0.5, 100).tolist()
    assert keep == [0, 2, 3]


def test_rotated_nms_drops_non_finite_rows():
    boxes = torch.tensor(
        [
            [float("nan"), 0.0, 4.0, 4.0, 0.0],
            [10.0, 10.0, 4.0, 4.0, 0.0],
        ]
    )
    scores = torch.tensor([0.99, 0.5])
    classes = torch.tensor([0, 0])
    assert rotated_nms_keep_indices(boxes, scores, classes, 0.5, 10).tolist() == [1]


def test_rotated_nms_respects_max_det():
    boxes = torch.tensor([[i * 100.0, 0.0, 5.0, 5.0, 0.0] for i in range(5)])
    scores = torch.linspace(0.9, 0.5, 5)
    classes = torch.zeros(5, dtype=torch.long)
    assert len(rotated_nms_keep_indices(boxes, scores, classes, 0.5, 2)) == 2


# ---------------------------------------------------------------------------
# Postprocess
# ---------------------------------------------------------------------------


def _fake_output(xywhr, scores):
    return {
        "boxes": torch.as_tensor(xywhr, dtype=torch.float32)[None],
        "scores": torch.as_tensor(scores, dtype=torch.float32)[None],
    }


def test_postprocess_obb_empty_below_threshold():
    out = postprocess_obb(
        _fake_output([[10.0, 10.0, 4.0, 2.0, 0.0]], [[0.01]]),
        conf_thres=0.5,
    )
    assert out["num_detections"] == 0
    assert out["obb"] == []


def test_postprocess_obb_contract_and_letterbox_inverse():
    # 1024 canvas, original 2048x1024 -> longest side rescale ratio 0.5,
    # bottom-right padded, so undoing it is a plain divide by the ratio.
    out = postprocess_obb(
        _fake_output(
            [[100.0, 200.0, 40.0, 20.0, 0.2], [700.0, 700.0, 10.0, 30.0, 0.0]],
            [[0.9, 0.1], [0.2, 0.8]],
        ),
        conf_thres=0.05,
        iou_thres=0.5,
        input_size=1024,
        original_size=(2048, 1024),
    )
    assert out["num_detections"] == 2
    obb = out["obb"].numpy()
    assert obb.shape == (2, 7)

    top = obb[0]
    assert np.allclose(top[:2], [200.0, 400.0], atol=1e-3)
    assert np.allclose(top[2:4], [80.0, 40.0], atol=1e-3)
    assert np.allclose(top[4], 0.2, atol=1e-5)  # uniform scale keeps the angle
    assert np.isclose(top[5], 0.9)
    assert top[6] == 0

    # Long-side canonicalization applied to the w<h row.
    second = obb[1]
    assert second[2] >= second[3]
    assert -math.pi / 2 <= second[4] < math.pi / 2

    aabb = out["boxes"].numpy()
    assert (aabb[:, 2] > aabb[:, 0]).all() and (aabb[:, 3] > aabb[:, 1]).all()
    assert (aabb[:, [0, 2]] <= 2048).all() and (aabb[:, [1, 3]] <= 1024).all()


def test_postprocess_obb_rejects_non_uniform_scaling():
    with pytest.raises(ValueError, match="aspect-preserving"):
        postprocess_obb(
            _fake_output([[10.0, 10.0, 4.0, 2.0, 0.0]], [[0.9]]),
            conf_thres=0.1,
            input_size=1024,
            original_size=(2048, 512),
            letterbox=False,
        )


def test_postprocess_obb_honours_max_det():
    xywhr = [[i * 200.0 + 10.0, 10.0, 8.0, 4.0, 0.0] for i in range(6)]
    scores = [[0.9 - i * 0.05] for i in range(6)]
    out = postprocess_obb(
        _fake_output(xywhr, scores), conf_thres=0.1, iou_thres=0.5, max_det=3
    )
    assert out["num_detections"] == 3


def test_postprocess_obb_prefilter_caps_candidates():
    n = 2500
    xywhr = [[float(i % 1000), 10.0, 4.0, 2.0, 0.0] for i in range(n)]
    scores = [[0.5 + (i / n) * 0.4] for i in range(n)]
    out = postprocess_obb(
        _fake_output(xywhr, scores),
        conf_thres=0.1,
        iou_thres=0.5,
        max_det=10,
        pre_nms_top_k=1000,
    )
    assert out["num_detections"] == 10


def test_postprocess_obb_accepts_traced_tuple_output():
    boxes = torch.tensor([[[10.0, 10.0, 6.0, 4.0, 0.0]]])
    scores = torch.tensor([[[0.9]]])
    out = postprocess_obb((boxes, scores), conf_thres=0.1)
    assert out["num_detections"] == 1


def test_postprocess_obb_rejects_unknown_output_shape():
    with pytest.raises(TypeError):
        postprocess_obb(object())


def test_autoconvert_reads_supergradients_processing_params_names():
    """Bare Deci checkpoints keep their real labels through auto-conversion."""
    from libreyolo.models.autoconvert import _checkpoint_names

    loaded = {
        "ema_net": {},
        "processing_params": {"class_names": ["plane", "ship", "harbor"]},
    }
    assert _checkpoint_names(loaded, nc=3) == ["plane", "ship", "harbor"]
    assert _checkpoint_names(loaded, nc=2) == ["plane", "ship"]
    # An explicit `names` key still wins.
    assert _checkpoint_names({**loaded, "names": ["a"]}, nc=1) == ["a"]


def test_backend_obb_parser_drops_non_finite_rows():
    """The numpy backend path must reject NaN/Inf rows like the native one."""
    from libreyolo.backends.base import BaseBackend

    boxes = np.array(
        [[[np.nan, 10.0, 8.0, 4.0, 0.0], [200.0, 100.0, 20.0, 10.0, 0.1]]],
        dtype=np.float32,
    )
    scores = np.array([[[0.99], [0.8]]], dtype=np.float32)
    parsed = BaseBackend._parse_yolonas_obb(
        [boxes, scores], 1024, 1024, 512, conf=0.1, iou=0.5, ratio=1.0
    )
    _boxes, out_scores, _cls, _masks, obb = parsed
    assert len(obb) == 1
    assert np.isfinite(obb).all()
    assert np.isclose(out_scores[0], 0.8)


def test_ui_summary_reports_obb_not_detect():
    """OBB results fill `boxes` too, so the UI must dispatch on `obb` first."""
    from libreyolo.ui.server import _summarize_result
    from libreyolo.utils.results import OBB, Boxes, Results

    result = Results(
        boxes=Boxes(
            np.zeros((2, 4), dtype=np.float32),
            np.zeros((2,), dtype=np.float32),
            np.zeros((2,), dtype=np.float32),
        ),
        obb=OBB(np.zeros((2, 7), dtype=np.float32), (100, 100)),
        orig_shape=(100, 100),
    )
    assert _summarize_result(result) == ("obb", "2 objects")
