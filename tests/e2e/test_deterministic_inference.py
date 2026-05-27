"""Nightly native inference checks for flagship detection families."""

import pytest
import torch

from libreyolo import LibreYOLO

from .conftest import (
    GENERAL_NIGHTLY_INFERENCE_PARAMS,
    cuda_cleanup,
    require_test_weights,
)

pytestmark = [pytest.mark.e2e, pytest.mark.general_nightly]


def _tensor(data):
    return (
        data.detach().cpu() if isinstance(data, torch.Tensor) else torch.as_tensor(data)
    )


def _assert_detection_output_is_stable(family, first, second):
    assert first.boxes is not None, f"{family} did not return detection boxes"
    assert second.boxes is not None, f"{family} did not return detection boxes"
    assert len(first.boxes) > 0, f"{family} returned no detections"
    assert len(first.boxes) == len(second.boxes), (
        f"{family} detection count changed: {len(first.boxes)} -> {len(second.boxes)}"
    )
    assert first.orig_shape == second.orig_shape
    assert first.names, f"{family} result has no class names"

    n = min(5, len(first.boxes))
    first_boxes = _tensor(first.boxes.xyxy[:n])
    second_boxes = _tensor(second.boxes.xyxy[:n])
    first_conf = _tensor(first.boxes.conf[:n])
    second_conf = _tensor(second.boxes.conf[:n])
    first_cls = _tensor(first.boxes.cls[:n])
    second_cls = _tensor(second.boxes.cls[:n])

    assert torch.isfinite(first_boxes).all(), f"{family} produced non-finite boxes"
    assert torch.isfinite(first_conf).all(), f"{family} produced non-finite scores"

    torch.testing.assert_close(first_boxes, second_boxes, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(first_conf, second_conf, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(first_cls, second_cls, rtol=0, atol=0)


@pytest.mark.parametrize(
    "family,size,weights",
    GENERAL_NIGHTLY_INFERENCE_PARAMS,
)
def test_native_inference_is_stable(family, size, weights, sample_image):
    """Every YOLO9/RF-DETR detection size loads and runs stable inference."""
    weights = require_test_weights(weights, expected_family=family)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LibreYOLO(weights, size=size, device=device)
    try:
        first = model(sample_image, conf=0.25)
        second = model(sample_image, conf=0.25)
        _assert_detection_output_is_stable(family, first, second)
    finally:
        del model
        cuda_cleanup()
