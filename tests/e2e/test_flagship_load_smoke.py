"""Focused nightly load smoke checks for flagship pretrained weights."""

import pytest
import torch

from libreyolo import LibreYOLO

from .conftest import (
    FLAGSHIP_NIGHTLY_LOAD_PARAMS,
    cuda_cleanup,
    require_test_weights,
)

pytestmark = [pytest.mark.e2e, pytest.mark.flagship_nightly]


@pytest.mark.parametrize(
    "family,size,weights",
    FLAGSHIP_NIGHTLY_LOAD_PARAMS,
)
def test_flagship_weights_load(family, size, weights):
    """Load each YOLO9/RF-DETR detection size without running inference."""
    weights = require_test_weights(weights, expected_family=family)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LibreYOLO(weights, size=size, device=device)
    try:
        assert getattr(model, "nb_classes", 0) > 0
        assert getattr(model, "names", None)
    finally:
        del model
        cuda_cleanup()
