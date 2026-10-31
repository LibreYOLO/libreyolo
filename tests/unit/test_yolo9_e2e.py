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
    assert LibreYOLO9E2E.SUPPORTED_TASKS == ("detect",)
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

    assert set(out.keys()) == {"one_to_many", "one_to_one"}
    expected_shapes = [(1, 144, 80, 80), (1, 144, 40, 40), (1, 144, 20, 20)]
    for branch in (out["one_to_many"], out["one_to_one"]):
        assert [t.shape for t in branch] == [torch.Size(s) for s in expected_shapes]


def _legacy_e2e_state_dict(state_dict):
    """Rename current E2E head keys to the pre-rename (legacy) spelling."""
    renames = (
        ("head.one_to_one_anchor_convs.", "head.one2one_cv2."),
        ("head.one_to_one_class_convs.", "head.one2one_cv3."),
        ("head.anchor_convs.", "head.cv2."),
        ("head.class_convs.", "head.cv3."),
    )
    legacy = {}
    for key, value in state_dict.items():
        for new, old in renames:
            if key.startswith(new):
                key = old + key[len(new):]
                break
        legacy[key] = value
    return legacy


def test_yolo9_e2e_can_load_discriminator():
    """can_load must isolate E2E checkpoints (one-to-one towers, legacy
    ``one2one_cv2``/``one2one_cv3`` or current ``one_to_one_*`` keys) from
    regular YOLOv9 checkpoints. Both directions must be exclusive so the
    registry never resolves an ambiguous checkpoint."""
    from libreyolo import LibreYOLO9, LibreYOLO9E2E

    e2e_sd = LibreYOLO9E2E(None, size="t", device="cpu").model.state_dict()
    reg_sd = LibreYOLO9(None, size="t", device="cpu").model.state_dict()
    legacy_sd = _legacy_e2e_state_dict(e2e_sd)
    assert any(k.startswith("head.one2one_cv3.") for k in legacy_sd)

    for sd in (e2e_sd, legacy_sd):
        assert LibreYOLO9E2E.can_load(sd) is True
        assert LibreYOLO9.can_load(sd) is False
    assert LibreYOLO9E2E.can_load(reg_sd) is False
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

    sd = LibreYOLO9E2E(None, size="t", device="cpu", nb_classes=7).model.state_dict()
    assert LibreYOLO9E2E.detect_nb_classes(sd) == 7
    assert LibreYOLO9E2E.detect_nb_classes(_legacy_e2e_state_dict(sd)) == 7
    # A lone one-to-one class output (either spelling) is enough.
    for key in ("head.one_to_one_class_convs.0.2.weight", "head.one2one_cv3.0.2.weight"):
        assert LibreYOLO9E2E.detect_nb_classes({key: torch.zeros(7, 64, 1, 1)}) == 7


def test_yolo9_e2e_legacy_checkpoint_loads_into_current_layout(tmp_path):
    """Legacy-keyed E2E checkpoints load through the shared key upgrade."""
    from libreyolo import LibreYOLO, LibreYOLO9E2E

    src = LibreYOLO9E2E(None, size="t", device="cpu", nb_classes=3)
    with torch.no_grad():
        for tensor in src.model.state_dict().values():
            if tensor.is_floating_point():
                tensor.add_(torch.rand_like(tensor))
    ckpt = tmp_path / "LibreYOLO9E2Et.pt"
    torch.save({"model": _legacy_e2e_state_dict(src.model.state_dict())}, ckpt)

    loaded = LibreYOLO(str(ckpt), device="cpu")
    assert loaded.FAMILY == "yolo9_e2e"
    assert loaded.nb_classes == 3
    expected = src.model.state_dict()
    actual = loaded.model.state_dict()
    assert list(actual) == list(expected)
    assert all(torch.equal(actual[k], expected[k]) for k in expected)


def test_yolo9_e2e_load_adopts_checkpoint_class_tower_width(tmp_path):
    """A 2-class fine-tune of COCO weights keeps 80-wide class towers, wider
    than a fresh 2-class build; both tower sets must adopt that width."""
    from libreyolo import LibreYOLO, LibreYOLO9E2E

    src = LibreYOLO9E2E(None, size="t", device="cpu")
    src._rebuild_for_new_classes(2)
    ckpt = tmp_path / "LibreYOLO9E2Et.pt"
    torch.save({"model": src.model.state_dict(), "nc": 2}, ckpt)

    loaded = LibreYOLO(str(ckpt), device="cpu")
    head = loaded.model.head
    for towers in (head.class_convs, head.one_to_one_class_convs):
        assert [tower[0].conv.out_channels for tower in towers] == [80, 80, 80]
        assert [tower[-1].out_channels for tower in towers] == [2, 2, 2]
    expected = src.model.state_dict()
    actual = loaded.model.state_dict()
    assert all(torch.equal(actual[k], expected[k]) for k in expected)


def test_yolo9_e2e_head_layout_and_initial_towers():
    """One-to-one towers follow the dense ones and start as exact copies."""
    from libreyolo.models.yolo9_e2e.nn import LibreYOLO9E2EModel

    head = LibreYOLO9E2EModel(config="t", nb_classes=5).head
    order = []
    for name, _ in head.named_parameters():
        group = name.split(".")[0]
        if group not in order:
            order.append(group)
    assert order == [
        "anchor_convs",
        "class_convs",
        "one_to_one_anchor_convs",
        "one_to_one_class_convs",
    ]
    state = head.state_dict()
    for key, value in state.items():
        if key.startswith(("anchor_convs.", "class_convs.")):
            assert torch.equal(value, state["one_to_one_" + key]), key
    for towers, bias in (
        (head.one_to_one_anchor_convs, 1.0),
        (head.one_to_one_class_convs, -10.0),
    ):
        for tower in towers:
            assert torch.all(tower[-1].bias == bias)


def test_yolo9_e2e_set_num_classes_updates_both_branches():
    from libreyolo.models.yolo9_e2e.nn import LibreYOLO9E2EModel

    head = LibreYOLO9E2EModel(config="t", nb_classes=80).head
    hidden = [tower[0].conv.out_channels for tower in head.one_to_one_class_convs]
    head.set_num_classes(3)

    assert head.num_classes == 3
    for towers in (head.class_convs, head.one_to_one_class_convs):
        assert [tower[-1].out_channels for tower in towers] == [3, 3, 3]
        assert all(torch.all(tower[-1].bias == -10.0) for tower in towers)
    assert [tower[0].conv.out_channels for tower in head.one_to_one_class_convs] == hidden

    head.eval()
    feats = [torch.zeros(1, c, s, s) for c, s in zip(head.in_channels, (8, 4, 2))]
    with torch.no_grad():
        decoded, raw = head(feats)
    assert decoded.shape == (1, 4 + 3, 84)
    assert raw[0].shape == (1, 4 * 16 + 3, 8, 8)


def test_yolo9_e2e_loss_runs_and_produces_finite_total():
    """Direct YOLO9E2ELoss call on synthetic dual-branch outputs returns a
    finite total_loss (one-to-many + one-to-one)."""
    from libreyolo.models.yolo9_e2e.loss import YOLO9E2ELoss

    loss_fn = YOLO9E2ELoss(
        num_classes=80,
        reg_max=16,
        strides=[8, 16, 32],
        image_size=[640, 640],
        device=torch.device("cpu"),
    )

    one_to_many = [
        torch.randn(2, 144, 80, 80),
        torch.randn(2, 144, 40, 40),
        torch.randn(2, 144, 20, 20),
    ]
    one_to_one = [
        torch.randn(2, 144, 80, 80),
        torch.randn(2, 144, 40, 40),
        torch.randn(2, 144, 20, 20),
    ]
    targets = torch.zeros(2, 30, 5)
    targets[0, 0] = torch.tensor([3.0, 320.0, 240.0, 100.0, 80.0])
    targets[1, 0] = torch.tensor([1.0, 400.0, 320.0, 120.0, 100.0])

    out = loss_fn(one_to_many, one_to_one, targets)
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


def test_yolo9_e2e_postprocess_ranks_anchor_class_pairs():
    """Top-K runs over (anchor, class) pairs: one anchor may yield two
    classes, and K = min(max_det, anchors * classes)."""
    from libreyolo.models.yolo9_e2e.utils import postprocess

    predictions = torch.zeros(1, 4 + 2, 3)
    predictions[0, :4] = torch.tensor(
        [[0.0, 10.0, 20.0], [0.0, 10.0, 20.0], [5.0, 15.0, 25.0], [5.0, 15.0, 25.0]]
    )
    predictions[0, 4:] = torch.tensor([[0.89, 0.92, 0.50], [0.91, 0.10, 0.99]])

    out = postprocess({"predictions": predictions}, conf_thres=0.3, max_det=300)
    assert out["num_detections"] == 5  # 6 pairs, 0.10 is below conf
    assert out["scores"].tolist() == pytest.approx([0.99, 0.92, 0.91, 0.89, 0.50])
    assert out["classes"].tolist() == [1, 0, 1, 0, 0]
    assert out["boxes"][2].tolist() == [0.0, 0.0, 5.0, 5.0]

    top1 = postprocess(predictions, conf_thres=0.3, max_det=1)
    assert top1["classes"].tolist() == [1]
    assert top1["boxes"].tolist() == [[20.0, 20.0, 25.0, 25.0]]


def test_yolo9_e2e_postprocess_defaults_to_letterbox_inverse():
    """E2E predict geometry matches YOLO9 letterboxed inputs."""
    from libreyolo.models.yolo9_e2e.utils import postprocess

    predictions = torch.zeros(1, 6, 1)
    predictions[0, :4, 0] = torch.tensor([0.0, 0.0, 320.0, 320.0])
    predictions[0, 4, 0] = 0.9

    out = postprocess(
        {"predictions": predictions},
        input_size=640,
        original_size=(1280, 960),
    )

    assert out["num_detections"] == 1
    torch.testing.assert_close(
        out["boxes"],
        torch.tensor([[0.0, 0.0, 640.0, 640.0]]),
    )


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
