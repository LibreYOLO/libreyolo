"""Unit tests for the LibreRTMDet family.

Smoke tests only — they verify the architecture builds at every size, the
forward returns the expected shape contract, and the postprocess returns the
expected dict schema. Numerical parity vs upstream is gated by the COCO val
mAP rather than a per-tensor diff (mmdet/mmcv aren't a runtime dep).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from libreyolo import LibreRTMDet
from libreyolo.models.rtmdet.nn import LibreRTMDetModel
from libreyolo.models.rtmdet.utils import (
    _distance2bbox,
    _make_grid_priors,
    postprocess,
)


SIZES = ["t", "s", "m", "l", "x"]

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("size", SIZES)
def test_build_and_forward(size):
    """Architecture builds and produces the expected per-level shapes."""
    model = LibreRTMDetModel(size=size, nc=80).eval()
    x = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        cls_scores, bbox_preds = model(x)

    expected_levels = [(80, 80), (40, 40), (20, 20)]
    assert len(cls_scores) == 3
    assert len(bbox_preds) == 3
    for cls, reg, (h, w) in zip(cls_scores, bbox_preds, expected_levels):
        assert cls.shape == (1, 80, h, w)
        assert reg.shape == (1, 4, h, w)


@pytest.mark.parametrize("size,exp_on_reg", [("t", False), ("s", False), ("m", True), ("l", True), ("x", True)])
def test_exp_on_reg_per_size(size, exp_on_reg):
    """tiny / s use linear reg, m / l / x use exp(reg). Empirically pinned to
    match the published COCO weight checkpoints."""
    model = LibreRTMDetModel(size=size, nc=80)
    assert model.head.exp_on_reg is exp_on_reg


def test_share_conv_aliasing():
    """share_conv=True aliases cls_convs[n][i].conv across levels.

    The state_dict will only persist one conv per stacked-conv index, but
    PyTorch leaves the aliased Parameters in place. Verify by checking
    Python-id equality of the .conv submodules.
    """
    model = LibreRTMDetModel(size="t", nc=80)
    head = model.head
    for i in range(head.stacked_convs):
        assert id(head.cls_convs[0][i].conv) == id(head.cls_convs[1][i].conv)
        assert id(head.cls_convs[0][i].conv) == id(head.cls_convs[2][i].conv)
        assert id(head.reg_convs[0][i].conv) == id(head.reg_convs[1][i].conv)


def test_grid_priors_corner_offset():
    """``_make_grid_priors`` uses MlvlPointGenerator(offset=0) — corners, not centers.

    For 640x640 input at stride 8, the very first prior is at (0, 0). Using
    offset=0.5 would put it at (4, 4). This is the trap the reviewer flagged.
    """
    fake = [torch.zeros(1, 1, 80, 80), torch.zeros(1, 1, 40, 40), torch.zeros(1, 1, 20, 20)]
    pts = _make_grid_priors(fake, [8, 16, 32])
    assert pts.shape == (8400, 2)
    assert pts[0].tolist() == [0.0, 0.0]
    # Second prior on level 0 is at (8, 0).
    assert pts[1].tolist() == [8.0, 0.0]
    # First prior on level 1 (stride 16) lands at index 6400.
    assert pts[6400].tolist() == [0.0, 0.0]


def test_distance2bbox():
    points = torch.tensor([[10.0, 20.0], [50.0, 60.0]])
    distance = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    boxes = _distance2bbox(points, distance)
    assert boxes.tolist() == [[9.0, 18.0, 13.0, 24.0], [45.0, 54.0, 57.0, 68.0]]


def test_postprocess_shape_contract():
    """Postprocess returns the canonical {boxes, scores, classes, num_detections} dict."""
    cls = (
        torch.zeros(1, 80, 80, 80) - 10.0,  # all near-zero post-sigmoid
        torch.zeros(1, 80, 40, 40) - 10.0,
        torch.zeros(1, 80, 20, 20) - 10.0,
    )
    reg = (
        torch.full((1, 4, 80, 80), 32.0),
        torch.full((1, 4, 40, 40), 32.0),
        torch.full((1, 4, 20, 20), 32.0),
    )
    out = postprocess((cls, reg), conf_thres=0.25, iou_thres=0.65, input_size=640)
    # All scores below 0.25, no detections.
    assert out["num_detections"] == 0
    assert out["boxes"] == []


def test_factory_registration():
    """LibreYOLO factory recognises a converted LibreRTMDet checkpoint."""
    # We test the discriminator classmethods directly, no I/O.
    fake_sd = {
        "head.rtm_cls.0.weight": torch.zeros(80, 96, 1, 1),
        "head.rtm_reg.0.weight": torch.zeros(4, 96, 1, 1),
        "backbone.stem.0.conv.weight": torch.zeros(12, 3, 3, 3),
    }
    assert LibreRTMDet.can_load(fake_sd)
    assert LibreRTMDet.detect_size(fake_sd) == "t"
    assert LibreRTMDet.detect_nb_classes(fake_sd) == 80


def test_export_mode_returns_flat_tensor():
    """In export mode the head returns a single (B, N, 4 + nc) tensor."""
    model = LibreRTMDetModel(size="t", nc=80).eval()
    model.head.export = True
    x = torch.zeros(1, 3, 640, 640)
    with torch.no_grad():
        out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 8400, 84)  # 80*80 + 40*40 + 20*20 = 8400; 4 box + 80 cls
