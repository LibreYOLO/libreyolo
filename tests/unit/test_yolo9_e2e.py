"""Unit tests for the native YOLOv9 end-to-end (NMS-free) family."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.unit


def test_yolo9_e2e_is_registered_and_detects_filename():
    from libreyolo import LibreYOLO9E2E
    from libreyolo.models.base.model import BaseModel

    assert any(cls.__name__ == "LibreYOLO9E2E" for cls in BaseModel._registry)
    assert LibreYOLO9E2E.FAMILY == "yolo9_e2e"
    assert LibreYOLO9E2E.detect_size_from_filename("LibreYOLO9E2Et.pt") == "t"
    assert LibreYOLO9E2E.detect_size_from_filename("LibreYOLO9E2Ec.pt") == "c"


def test_yolo9_e2e_eval_forward_shapes():
    """Inference path uses only the one-to-one branch and returns the
    standard yolo9 prediction tensor (B, 4+nc, num_anchors)."""
    from libreyolo import LibreYOLO9E2E

    model = LibreYOLO9E2E(None, size="t", device="cpu")
    model.model.eval()
    with torch.no_grad():
        out = model.model(torch.zeros(1, 3, 640, 640))

    assert isinstance(out, dict)
    assert "predictions" in out
    # 4 box channels + 80 classes; 80*80 + 40*40 + 20*20 = 8400 anchors at 640
    assert out["predictions"].shape == (1, 84, 8400)
    assert len(out["raw_outputs"]) == 3


def test_yolo9_e2e_train_forward_returns_both_branches():
    """Training mode without targets returns the raw dual-branch maps so the
    loss can be computed externally. Each branch is a list of three FPN
    tensors of shape (B, 4*reg_max + nc, H, W)."""
    from libreyolo import LibreYOLO9E2E

    model = LibreYOLO9E2E(None, size="t", device="cpu")
    model.model.train()
    out = model.model(torch.zeros(1, 3, 640, 640))

    assert set(out.keys()) == {"one2many", "one2one"}
    expected_shapes = [(1, 144, 80, 80), (1, 144, 40, 40), (1, 144, 20, 20)]
    for branch in (out["one2many"], out["one2one"]):
        assert [t.shape for t in branch] == [torch.Size(s) for s in expected_shapes]


def test_yolo9_e2e_can_load_discriminator():
    """can_load must isolate E2E checkpoints (containing one2one_cv2/one2one_cv3)
    from regular YOLOv9 checkpoints. Both directions must be exclusive so the
    registry never resolves an ambiguous checkpoint."""
    from libreyolo import LibreYOLO9, LibreYOLO9E2E

    e2e_sd = LibreYOLO9E2E(None, size="t", device="cpu").model.state_dict()
    reg_sd = LibreYOLO9(None, size="t", device="cpu").model.state_dict()

    assert LibreYOLO9E2E.can_load(e2e_sd) is True
    assert LibreYOLO9E2E.can_load(reg_sd) is False
    assert LibreYOLO9.can_load(e2e_sd) is False
    assert LibreYOLO9.can_load(reg_sd) is True


def test_yolo9_e2e_factory_resolves_e2e_checkpoint(tmp_path):
    """Saving an E2E state_dict and loading via the LibreYOLO factory must
    resolve to LibreYOLO9E2E, not LibreYOLO9."""
    from libreyolo import LibreYOLO, LibreYOLO9E2E

    src = LibreYOLO9E2E(None, size="t", device="cpu")
    ckpt = tmp_path / "LibreYOLO9E2Et.pt"
    torch.save({"model": src.model.state_dict()}, ckpt)

    loaded = LibreYOLO(str(ckpt), device="cpu")
    assert loaded.FAMILY == "yolo9_e2e"
    assert loaded.size == "t"


def test_yolo9_e2e_detect_nb_classes():
    from libreyolo import LibreYOLO9E2E

    sd = LibreYOLO9E2E(None, size="t", device="cpu").model.state_dict()
    assert LibreYOLO9E2E.detect_nb_classes(sd) == 80


def test_yolo9_e2e_loss_runs_and_produces_finite_total():
    """Direct YOLO9E2ELoss call on synthetic dual-branch outputs returns a
    finite total_loss equal to one2many + one2one."""
    from libreyolo.models.yolo9_e2e.loss import YOLO9E2ELoss

    loss_fn = YOLO9E2ELoss(
        num_classes=80,
        reg_max=16,
        strides=[8, 16, 32],
        image_size=[640, 640],
        device=torch.device("cpu"),
    )

    one2many = [
        torch.randn(2, 144, 80, 80),
        torch.randn(2, 144, 40, 40),
        torch.randn(2, 144, 20, 20),
    ]
    one2one = [
        torch.randn(2, 144, 80, 80),
        torch.randn(2, 144, 40, 40),
        torch.randn(2, 144, 20, 20),
    ]
    targets = torch.zeros(2, 30, 5)
    targets[0, 0] = torch.tensor([3.0, 320.0, 240.0, 100.0, 80.0])
    targets[1, 0] = torch.tensor([1.0, 400.0, 320.0, 120.0, 100.0])

    out = loss_fn(one2many, one2one, targets)
    assert torch.isfinite(out["total_loss"])
    assert out["total_loss"].item() > 0
    for k in ("box_loss", "dfl_loss", "cls_loss"):
        assert torch.isfinite(out[k])


def test_yolo9_e2e_postprocess_topk_no_nms_caps_at_max_det():
    """postprocess uses top-K selection (no NMS) and respects max_det.
    Output is a flat list keyed by boxes/scores/classes/num_detections."""
    from libreyolo.models.yolo9_e2e.utils import postprocess

    nc = 80
    num_anchors = 8400
    # All anchors confident enough that top-K is the binding cap.
    predictions = torch.zeros(1, 4 + nc, num_anchors)
    predictions[:, :4, :] = torch.tensor([100.0, 100.0, 200.0, 200.0]).view(1, 4, 1)
    predictions[:, 4:, :] = 0.9  # uniformly high class scores

    out = postprocess(
        {"predictions": predictions},
        conf_thres=0.25,
        iou_thres=0.45,  # ignored; no NMS
        input_size=640,
        original_size=None,
        max_det=50,
    )

    assert out["num_detections"] == 50
    assert len(out["boxes"]) == 50
    assert len(out["scores"]) == 50
    assert len(out["classes"]) == 50
    # Scores must be sorted descending (top-K guarantees this).
    diffs = out["scores"][1:] - out["scores"][:-1]
    assert (diffs <= 1e-6).all()


def test_yolo9_e2e_postprocess_returns_empty_when_below_threshold():
    """Confidence threshold above all scores returns empty result."""
    from libreyolo.models.yolo9_e2e.utils import postprocess

    nc = 80
    num_anchors = 100
    predictions = torch.zeros(1, 4 + nc, num_anchors)
    predictions[:, 4:, :] = 0.1

    out = postprocess(
        {"predictions": predictions},
        conf_thres=0.5,
        iou_thres=0.45,
        input_size=640,
        original_size=None,
        max_det=300,
    )
    assert out["num_detections"] == 0
