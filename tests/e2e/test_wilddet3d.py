"""Manual CUDA integration check against the separately installed runtime.

Set LIBREYOLO_WILDDET3D_CHECKPOINT to the local upstream full checkpoint.
This is not promoted to the nightly: its runtime and weights are user-staged.
The direct call checks adapter conversion, not an independent architecture.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tests.e2e.conftest import require_test_weights

pytestmark = [pytest.mark.e2e, pytest.mark.wilddet3d, pytest.mark.external_data]


def test_cuda_prompt_and_output_mapping(cuda_device, sample_image, tmp_path):
    reference = os.environ.get("LIBREYOLO_WILDDET3D_CHECKPOINT")
    if not reference:
        pytest.skip("Set LIBREYOLO_WILDDET3D_CHECKPOINT to a local full checkpoint.")
    checkpoint = require_test_weights(str(Path(reference).expanduser().resolve()))
    from libreyolo import LibreWildDet3D

    image = Image.open(sample_image).convert("RGB")
    w, h = image.size
    # Explicit synthetic pinhole calibration: a transport/projection check,
    # not a claim about this photograph's true calibration or metric accuracy.
    k = np.array(
        [[max(w, h), 0, w / 2], [0, max(w, h), h / 2], [0, 0, 1]], dtype=np.float32
    )
    prompt = [[w * 0.2, h * 0.2, w * 0.8, h * 0.8]]
    model = LibreWildDet3D(checkpoint, device=cuda_device)
    result = model(image, intrinsics=k, bboxes=prompt)
    assert len(result) == len(result.boxes3d) == 1
    assert torch.isfinite(result.boxes3d.data).all()
    assert (result.boxes3d.dimensions > 0).all()
    assert (result.boxes3d.xyz[:, 2] > 0).all()

    data = model._runtime.preprocess(np.asarray(image, dtype=np.float32), k)
    with torch.inference_mode():
        raw = model._predictor(
            images=data["images"].to(cuda_device),
            intrinsics=data["intrinsics"].to(cuda_device)[None],
            input_hw=[data["input_hw"]],
            original_hw=[data["original_hw"]],
            padding=[data["padding"]],
            input_boxes=prompt,
            prompt_text="geometric",
        )
    torch.testing.assert_close(result.boxes.xyxy, torch.cat(raw[0]).cpu().float())
    torch.testing.assert_close(
        result.boxes3d.data[:, :10], torch.cat(raw[1]).cpu().float()
    )
    torch.testing.assert_close(result.boxes3d.conf, torch.cat(raw[2]).cpu().float())
    rendered = result.plot(image)
    assert rendered.size == image.size
    assert not np.array_equal(np.asarray(rendered), np.asarray(image))
    rendered.save(tmp_path / "wilddet3d.png")
