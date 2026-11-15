"""Task-appropriate trained-checkpoint smoke coverage for RiO-DETRv2."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from libreyolo import LibreYOLO

from .conftest import RIODETRV2_OBB_PARAMS

pytestmark = [pytest.mark.e2e, pytest.mark.network, pytest.mark.riodetrv2]


@pytest.mark.parametrize("family,size,weights", RIODETRV2_OBB_PARAMS)
def test_riodetrv2_obb_trained_checkpoint_smoke(family, size, weights):
    model = LibreYOLO(weights, device="cuda")
    assert (model.FAMILY, model.size, model.task) == (family, size, "obb")
    assert model.input_size == (800 if weights.endswith("-diorr.pt") else 1024)

    image = Image.fromarray(np.full((480, 800, 3), 127, dtype=np.uint8))
    result = model.predict(image, conf=0.0, max_det=5)[0]

    assert result.obb is not None
    assert result.obb.data.shape[1] == 7
    assert len(result.obb) == len(result.boxes) <= 5
    assert result.orig_shape == (480, 800)
