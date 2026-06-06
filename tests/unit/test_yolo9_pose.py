"""Unit tests for YOLO9 pose support."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def test_yolo9_pose_wrapper_defaults_to_person():
    from libreyolo import LibreYOLO9

    model = LibreYOLO9(None, size="t", task="pose", device="cpu")
    assert model.task == "pose"
    assert model.nb_classes == 1
    assert model.names == {0: "person"}
    assert model.num_keypoints == 17


def test_yolo9_pose_warm_starts_from_detection_checkpoint(tmp_path):
    from libreyolo import LibreYOLO9
    from libreyolo.models.yolo9.nn import LibreYOLO9Model

    det_model = LibreYOLO9Model(config="t", nb_classes=80).eval()
    state = {key: value.detach().clone() for key, value in det_model.state_dict().items()}
    state["backbone.conv0.conv.weight"].fill_(0.125)
    ckpt_path = tmp_path / "LibreYOLO9t.pt"
    torch.save(
        {
            "model": state,
            "model_family": "yolo9",
            "task": "detect",
            "nc": 80,
            "names": {i: f"class_{i}" for i in range(80)},
        },
        ckpt_path,
    )

    model = LibreYOLO9(str(ckpt_path), size="t", task="pose", device="cpu")

    assert model.task == "pose"
    assert model.nb_classes == 1
    assert model.names == {0: "person"}
    assert model.model.head.nc == 1
    assert model.model.head.cv3[0][-1].out_channels == 1
    assert hasattr(model.model.head, "cv4")
    loaded_weight = model.model.state_dict()["backbone.conv0.conv.weight"]
    assert torch.allclose(loaded_weight, torch.full_like(loaded_weight, 0.125))


def test_ddetect_pose_forward_shapes():
    from libreyolo.models.yolo9.nn import DDetectPose

    head = DDetectPose(nc=1, ch=(64, 128, 256), reg_max=16, stride=(8, 16, 32))
    head.eval()
    x = [
        torch.randn(1, 64, 8, 8),
        torch.randn(1, 128, 4, 4),
        torch.randn(1, 256, 2, 2),
    ]
    decoded, raw, keypoints = head(x)
    assert decoded.shape == (1, 5, 84)
    assert len(raw) == 3
    assert keypoints.shape == (1, 84, 17, 3)


def test_yolo9_pose_model_forward_shapes():
    from libreyolo.models.yolo9.nn import LibreYOLO9Model

    model = LibreYOLO9Model(config="t", nb_classes=1, pose=True).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 64))
    assert out["predictions"].shape == (1, 5, 84)
    assert out["keypoints"].shape == (1, 84, 17, 3)


def test_yolo9_pose_loss_runs_and_is_finite():
    from libreyolo.models.yolo9.loss import YOLO9PoseLoss

    loss_fn = YOLO9PoseLoss(
        num_classes=1,
        reg_max=16,
        strides=[8, 16, 32],
        image_size=[64, 64],
        device=torch.device("cpu"),
    )
    preds = [
        torch.randn(1, 65, 8, 8),
        torch.randn(1, 65, 4, 4),
        torch.randn(1, 65, 2, 2),
    ]
    keypoints = torch.randn(1, 51, 84)
    targets = torch.zeros(1, 10, 56)
    targets[0, 0, 0:5] = torch.tensor([0.0, 32.0, 32.0, 20.0, 20.0])
    targets[0, 0, 5:] = torch.tensor([32.0, 32.0, 2.0] * 17)

    out = loss_fn(preds, targets, keypoints)
    assert torch.isfinite(out["total_loss"])
    for key in (
        "box_loss",
        "dfl_loss",
        "cls_loss",
        "pose_loss",
        "pose_l1_loss",
        "pose_vis_loss",
    ):
        assert torch.isfinite(out[key])


def test_yolo9_pose_loss_decode_matches_inference_decode():
    from libreyolo.models.yolo9.loss import YOLO9PoseLoss
    from libreyolo.models.yolo9.nn import DDetectPose

    head = DDetectPose(nc=1, ch=(64, 128, 256), reg_max=16, stride=(8, 16, 32))
    features = [
        torch.zeros(1, 64, 8, 8),
        torch.zeros(1, 128, 4, 4),
        torch.zeros(1, 256, 2, 2),
    ]
    head.eval()
    with torch.no_grad():
        _, raw, _ = head([feature.clone() for feature in features])
    head.anchors, head.strides = (
        x.transpose(0, 1) for x in head._make_anchors(raw, head.stride, 0.5)
    )
    keypoints = torch.zeros(1, head.nk, 84)
    keypoints[:, 0::3, :] = 0.25
    keypoints[:, 1::3, :] = -0.5
    keypoints[:, 2::3, :] = 0.75

    loss_fn = YOLO9PoseLoss(
        num_classes=1,
        reg_max=16,
        strides=[8, 16, 32],
        image_size=[64, 64],
        device=torch.device("cpu"),
    )
    loss_xy, loss_vis_logits = loss_fn._decode_keypoints_for_loss(keypoints)
    infer_kpts = head._decode_keypoints(keypoints)

    assert torch.allclose(loss_xy, infer_kpts[..., :2])
    assert torch.allclose(loss_vis_logits.sigmoid(), infer_kpts[..., 2])


def test_yolo9_pose_postprocess_filters_keypoints_in_lockstep():
    from libreyolo.models.yolo9.utils import postprocess

    predictions = torch.zeros(1, 5, 3)
    predictions[0, :4, :] = torch.tensor(
        [
            [10.0, 10.0, 10.0],
            [10.0, 10.0, 10.0],
            [30.0, 30.0, 10.0],
            [30.0, 30.0, 10.0],
        ]
    )
    predictions[0, 4, :] = torch.tensor([0.9, 0.8, 0.95])
    keypoints = torch.zeros(1, 3, 17, 3)
    keypoints[0, :, :, 0] = torch.tensor([1.0, 2.0, 3.0]).view(3, 1)
    keypoints[0, :, :, 1] = 5.0
    keypoints[0, :, :, 2] = 0.7

    out = postprocess(
        {"predictions": predictions, "keypoints": keypoints},
        conf_thres=0.25,
        iou_thres=0.45,
        input_size=64,
        original_size=(64, 64),
        max_det=10,
    )

    assert out["num_detections"] == 1
    assert out["keypoints"].shape == (1, 17, 3)
    # The degenerate third box is removed before NMS; NMS then keeps the
    # higher-scoring first box and its matching keypoints.
    assert torch.all(out["keypoints"][0, :, 0] == 1.0)
