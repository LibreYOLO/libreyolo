"""Exact native-port parity against the official Mask R-CNN COCO model.

This network-gated acceptance test downloads the torchvision v0.26.0
ResNet-50-FPN v2 checkpoint. The implementation is BSD-3-Clause; the
checkpoint redistribution basis is documented separately in the family
provenance notice. The gate compares RPN tensors, box-head tensors, final box
outputs, and pre-sigmoid mask logits so final thresholding cannot hide drift.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.transforms.functional import pil_to_tensor

from libreyolo.models.mask_rcnn.nn import LibreMaskRCNNModel

pytestmark = [
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.skipif(
        os.environ.get("LIBREYOLO_MASK_RCNN_ACCEPTANCE") != "1",
        reason=(
            "set LIBREYOLO_MASK_RCNN_ACCEPTANCE=1 to download the official "
            "checkpoint and run exact parity"
        ),
    ),
]


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


def test_torchvision_r50_exact_box_and_mask_logit_parity():
    upstream = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    ).eval()
    ours = LibreMaskRCNNModel(size="r50", num_classes=91).eval()
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
                "box", output
            )
        ),
        upstream.roi_heads.mask_predictor.register_forward_hook(
            lambda _module, _inputs, output: captured["upstream"].__setitem__(
                "mask_logits", output
            )
        ),
        ours.rpn.head.register_forward_hook(
            lambda _module, _inputs, output: captured["ours"].__setitem__(
                "rpn", output
            )
        ),
        ours.roi_heads.box_predictor.register_forward_hook(
            lambda _module, _inputs, output: captured["ours"].__setitem__(
                "box", output
            )
        ),
        ours.roi_heads.mask_predictor.register_forward_hook(
            lambda _module, _inputs, output: captured["ours"].__setitem__(
                "mask_logits", output
            )
        ),
    ]
    try:
        with torch.inference_mode():
            expected = upstream([image])[0]
            actual = ours([image])[0]
    finally:
        for hook in hooks:
            hook.remove()

    expected_boxes = {
        key: expected[key] for key in ("boxes", "labels", "scores")
    }
    actual_boxes = {key: actual[key] for key in ("boxes", "labels", "scores")}
    assert expected_boxes["boxes"].shape[0] > 0

    rpn_diff = _max_abs_diff(captured["upstream"]["rpn"], captured["ours"]["rpn"])
    box_head_diff = _max_abs_diff(
        captured["upstream"]["box"], captured["ours"]["box"]
    )
    boxes_diff = _max_abs_diff(expected_boxes, actual_boxes)
    mask_logits_diff = _max_abs_diff(
        captured["upstream"]["mask_logits"],
        captured["ours"]["mask_logits"],
    )
    final_masks_diff = _max_abs_diff(expected["masks"], actual["masks"])
    print(
        "r50 parity: "
        f"rpn={rpn_diff} box_head={box_head_diff} "
        f"boxes={boxes_diff} raw_mask_logits={mask_logits_diff} "
        f"final_masks={final_masks_diff}"
    )
    assert rpn_diff == 0.0, f"r50 RPN max_abs_diff={rpn_diff}"
    assert box_head_diff == 0.0, f"r50 box head max_abs_diff={box_head_diff}"
    assert boxes_diff == 0.0, f"r50 boxes max_abs_diff={boxes_diff}"
    assert mask_logits_diff == 0.0, (
        f"r50 raw mask logits max_abs_diff={mask_logits_diff}"
    )
    assert final_masks_diff == 0.0, (
        f"r50 final soft masks max_abs_diff={final_masks_diff}"
    )
