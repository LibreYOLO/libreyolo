"""Exact native-port parity against torchvision's official COCO models.

This network-gated acceptance test downloads the four official torchvision
v0.26.0 checkpoints. The implementation is BSD-3-Clause; the checkpoint
redistribution basis is documented separately in the Faster R-CNN provenance
notice. The test compares RPN outputs and pre-postprocess Fast R-CNN head
tensors before final detections because in-graph NMS could otherwise hide an
earlier numerical drift.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_mobilenet_v3_large_fpn,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.transforms.functional import pil_to_tensor

from libreyolo.models.faster_rcnn.nn import LibreFasterRCNNModel

pytestmark = [
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("LIBREYOLO_FASTER_RCNN_ACCEPTANCE") != "1",
        reason=(
            "set LIBREYOLO_FASTER_RCNN_ACCEPTANCE=1 to download the four "
            "official checkpoints and run exact parity"
        ),
    ),
]


CASES = {
    "n": (
        fasterrcnn_mobilenet_v3_large_320_fpn,
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT,
    ),
    "s": (
        fasterrcnn_mobilenet_v3_large_fpn,
        FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
    ),
    "m": (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
    ),
    "l": (
        fasterrcnn_resnet50_fpn_v2,
        FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
    ),
}


def _image_tensor() -> torch.Tensor:
    image_path = Path(__file__).parents[2] / "tests" / "data" / "parkour.jpg"
    with Image.open(image_path) as image:
        return pil_to_tensor(image.convert("RGB")).float() / 255.0


def _tensor_leaves(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _tensor_leaves(value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensor_leaves(item)
    else:
        raise TypeError(f"Unsupported parity value: {type(value).__name__}")


def _max_abs_diff(reference: Any, actual: Any) -> float:
    reference_tensors = list(_tensor_leaves(reference))
    actual_tensors = list(_tensor_leaves(actual))
    assert len(reference_tensors) == len(actual_tensors)
    maximum = 0.0
    for expected, observed in zip(reference_tensors, actual_tensors):
        assert expected.shape == observed.shape
        assert expected.dtype == observed.dtype
        if expected.numel():
            maximum = max(
                maximum,
                float((expected - observed).abs().max().item()),
            )
    return maximum


@pytest.mark.parametrize("size", ["n", "s", "m", "l"])
def test_torchvision_exact_parity(size):
    builder, weights = CASES[size]
    upstream = builder(weights=weights).eval()
    ours = LibreFasterRCNNModel(size=size, num_classes=91).eval()
    load_result = ours.load_state_dict(upstream.state_dict(), strict=True)
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    upstream.to(device)
    ours.to(device)
    image = _image_tensor().to(device)
    captured: dict[str, dict[str, Any]] = {"upstream": {}, "ours": {}}
    hooks = [
        upstream.rpn.head.register_forward_hook(
            lambda _module, _inputs, output: captured["upstream"].__setitem__(
                "rpn", output
            )
        ),
        upstream.roi_heads.box_predictor.register_forward_hook(
            lambda _module, _inputs, output: captured["upstream"].__setitem__(
                "roi", output
            )
        ),
        ours.rpn.head.register_forward_hook(
            lambda _module, _inputs, output: captured["ours"].__setitem__(
                "rpn", output
            )
        ),
        ours.roi_heads.box_predictor.register_forward_hook(
            lambda _module, _inputs, output: captured["ours"].__setitem__(
                "roi", output
            )
        ),
    ]
    try:
        with torch.inference_mode():
            expected = upstream([image])
            actual = ours([image])
    finally:
        for hook in hooks:
            hook.remove()

    assert expected[0]["boxes"].shape[0] > 0
    rpn_diff = _max_abs_diff(captured["upstream"]["rpn"], captured["ours"]["rpn"])
    roi_diff = _max_abs_diff(captured["upstream"]["roi"], captured["ours"]["roi"])
    detection_diff = _max_abs_diff(expected, actual)
    assert rpn_diff == 0.0, f"{size}: RPN max_abs_diff={rpn_diff}"
    assert roi_diff == 0.0, f"{size}: RoI head max_abs_diff={roi_diff}"
    assert detection_diff == 0.0, (
        f"{size}: final detection max_abs_diff={detection_diff}"
    )
