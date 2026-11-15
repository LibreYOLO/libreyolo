"""DPFlow clean-download and known-shift optical flow smoke."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreYOLO
from libreyolo.utils.serialization import validate_checkpoint_metadata

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.external_data,
    pytest.mark.network,
    pytest.mark.dpflow,
]

# Frame t + 1 is cropped SHIFT pixels further right/down than frame t, so the
# scene content moves by (-SHIFT_X, -SHIFT_Y) between the two frames.
SHIFT_X, SHIFT_Y = 6, 4


def _shifted_pair(sample_image) -> list[Image.Image]:
    image = Image.open(sample_image).convert("RGB")
    width = 640
    height = round(image.height * width / image.width)
    margin = 16
    base = image.resize((width + 2 * margin, height + 2 * margin), Image.BICUBIC)
    first = base.crop((margin, margin, margin + width, margin + height))
    second = base.crop(
        (margin + SHIFT_X, margin + SHIFT_Y, margin + SHIFT_X + width, margin + SHIFT_Y + height)
    )
    return [first, second]


def test_dpflow_clean_download_recovers_known_shift(tmp_path: Path, sample_image):
    checkpoint_path = tmp_path / "LibreDPFlowm-flow.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LibreYOLO(str(checkpoint_path), device=device)
    frames = _shifted_pair(sample_image)
    results = list(model.predict(frames))

    assert checkpoint_path.is_file()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert validate_checkpoint_metadata(checkpoint, strict=True) == []
    assert checkpoint["model_family"] == "dpflow"
    assert checkpoint["task"] == "flow"
    assert checkpoint["size"] == "m"

    assert len(results) == 1
    result = results[0]
    assert result.boxes is None
    flow = np.asarray(result.flow.data, dtype=np.float32)
    width, height = frames[0].size
    assert flow.shape == (height, width, 2)
    assert np.isfinite(flow).all()

    # Ignore a border where content leaves the frame.
    inner = flow[24:-24, 24:-24]
    median = np.median(inner.reshape(-1, 2), axis=0)
    assert abs(median[0] + SHIFT_X) < 0.5, median
    assert abs(median[1] + SHIFT_Y) < 0.5, median
