"""E2E smoke for QuickSRNet Medium 2x with published weights."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from libreyolo import LibreYOLO

from .conftest import require_test_weights

pytestmark = [pytest.mark.e2e, pytest.mark.quicksrnet]


def test_quicksrnet_m2_autodownload_and_predict():
    weights = require_test_weights(
        "weights/LibreQuickSRNetm2-restore.pt",
        expected_family="quicksrnet",
    )
    model = LibreYOLO(weights, device="cpu")
    image = Image.fromarray(
        np.random.default_rng(2026).integers(0, 256, (37, 53, 3), dtype=np.uint8),
        mode="RGB",
    )
    result = model.predict(image)
    assert result.restore_scale == 2
    assert result.restored.array.shape == (74, 106, 3)
    assert result.restored.array.dtype == np.uint8
