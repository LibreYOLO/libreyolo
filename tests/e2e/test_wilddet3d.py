"""Manual CUDA integration check against the separately installed runtime.

Set LIBREYOLO_WILDDET3D_CHECKPOINT to the local upstream full checkpoint.
This is not promoted to the nightly: its runtime and weights are user-staged.
The direct call checks adapter conversion, not an independent architecture.
"""

import os
import platform
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
    torch.testing.assert_close(result.boxes3d.conf2d, torch.cat(raw[3]).cpu().float())
    torch.testing.assert_close(result.boxes3d.conf3d, torch.cat(raw[4]).cpu().float())
    torch.testing.assert_close(result.boxes3d.cls, torch.cat(raw[5]).cpu().float())
    torch.testing.assert_close(
        result.boxes3d.intrinsics, torch.as_tensor(k, dtype=torch.float32)
    )
    rendered = result.plot(image)
    assert rendered.size == image.size
    assert not np.array_equal(np.asarray(rendered), np.asarray(image))
    rendered.save(tmp_path / "wilddet3d.png")


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS runtime check")
def test_macos_cpu_worker(sample_image, tmp_path):
    reference = os.environ.get("LIBREYOLO_WILDDET3D_CHECKPOINT")
    runtime_path = os.environ.get("LIBREYOLO_WILDDET3D_RUNTIME")
    runtime_python = os.environ.get("LIBREYOLO_WILDDET3D_PYTHON")
    if not all((reference, runtime_path, runtime_python)):
        pytest.skip("Set LIBREYOLO_WILDDET3D_CHECKPOINT, _RUNTIME and _PYTHON for Mac.")
    checkpoint = require_test_weights(str(Path(reference).expanduser().resolve()))
    from libreyolo import LibreWildDet3D

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
    with LibreWildDet3D(
        checkpoint,
        device="auto",
        runtime_path=runtime_path,
        runtime_python=runtime_python,
    ) as model:
        assert model.device.type == "cpu"
        result = model(image, intrinsics=intrinsics, text=["person"])
        process = model._backend._process
        assert process.poll() is None
    assert process.poll() is not None
    assert len(result) == len(result.boxes3d)
    assert torch.isfinite(result.boxes3d.data).all()
    result.plot(image).save(tmp_path / "wilddet3d-macos-cpu.png")
