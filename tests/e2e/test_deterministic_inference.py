"""Nightly native inference checks for every public model family.

This is the broad tier: one smallest pretrained case per family, native model
load, two inference passes, and stable detection/gaze outputs. Keep exports and
training out of this file; those belong to flagship or backend-specific suites.
"""

import pytest
import torch
from PIL import Image

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

    # Same-process native inference should be stable. Keep a tiny tolerance so
    # GPU kernels and postprocess threshold edges do not create false nightly
    # failures while still catching meaningful drift.
    torch.testing.assert_close(first_boxes, second_boxes, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(first_conf, second_conf, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(first_cls, second_cls, rtol=0, atol=0)


def _run_l2cs(weights, size):
    from libreyolo import LibreL2CS

    weights = require_test_weights(weights)
    model = LibreL2CS(
        weights, size=size, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    image = Image.new("RGB", (96, 96), color=(128, 128, 128))
    kwargs = {"face_boxes": [(8, 8, 88, 88)]}
    first = model(image, **kwargs)
    second = model(image, **kwargs)

    assert first.gaze is not None, "l2cs did not return gaze output"
    assert second.gaze is not None, "l2cs did not return gaze output"
    assert len(first.gaze) == 1
    assert len(second.gaze) == 1
    torch.testing.assert_close(
        _tensor(first.gaze.data),
        _tensor(second.gaze.data),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize(
    "family,size,weights",
    GENERAL_NIGHTLY_INFERENCE_PARAMS,
)
def test_native_inference_is_stable(family, size, weights, sample_image):
    """Every public family loads its smallest checkpoint and runs stable inference."""
    if family == "l2cs":
        _run_l2cs(weights, size)
        cuda_cleanup()
        return

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
