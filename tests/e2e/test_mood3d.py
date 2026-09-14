"""Real-checkpoint 3D-MOOD inference through the isolated runtime."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tests.e2e.conftest import require_test_weights

pytestmark = [pytest.mark.e2e, pytest.mark.mood3d, pytest.mark.external_data]


@pytest.mark.parametrize(
    "size,variable",
    [
        ("t", "LIBREYOLO_3DMOOD_T_CHECKPOINT"),
        ("b", "LIBREYOLO_3DMOOD_B_CHECKPOINT"),
    ],
)
def test_real_checkpoint_image_inference(size, variable, sample_image, tmp_path):
    reference = os.environ.get(variable)
    runtime_path = os.environ.get("LIBREYOLO_3DMOOD_RUNTIME")
    runtime_python = os.environ.get("LIBREYOLO_3DMOOD_PYTHON")
    if not reference or not runtime_path:
        pytest.skip(f"Set {variable} and LIBREYOLO_3DMOOD_RUNTIME.")
    checkpoint = require_test_weights(str(Path(reference).expanduser().resolve()))

    from libreyolo import Libre3DMOOD

    image = Image.open(sample_image).convert("RGB")
    width, height = image.size
    intrinsics = np.array(
        [
            [max(width, height), 0, width / 2],
            [0, max(width, height), height / 2],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )
    with Libre3DMOOD(
        checkpoint,
        size=size,
        device="auto",
        runtime_path=runtime_path,
        runtime_python=runtime_python,
    ) as model:
        result = model(image, intrinsics=intrinsics, text=["person", "car"])
        process = model._backend._process
        assert process.poll() is None
    assert process.poll() is not None
    assert len(result) == len(result.boxes3d)
    assert result.depth_map.data.shape == (height, width)
    assert torch.isfinite(result.depth_map.data).all()
    assert torch.isfinite(result.boxes3d.data).all()
    assert (result.boxes3d.dimensions > 0).all()
    assert (result.boxes3d.xyz[:, 2] > 0).all()
    assert torch.equal(result.boxes3d.conf, result.boxes3d.conf2d)
    assert torch.equal(result.boxes3d.conf3d, torch.ones_like(result.boxes3d.conf3d))
    rendered = result.plot(image)
    assert rendered.size == image.size
    rendered.save(tmp_path / f"3dmood-{size}.png")
