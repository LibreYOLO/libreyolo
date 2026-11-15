"""FlowSeek clean-download and optical flow smoke on a synthetic shift."""

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
    pytest.mark.network,
    pytest.mark.flowseek,
]

# Frame 2 shows frame 1's content moved 3 px right and 2 px down.
DX, DY = 3, 2


def _shifted_pair(sample_image) -> tuple[Image.Image, Image.Image]:
    img = Image.open(sample_image).convert("RGB")
    w, h = img.size
    img = img.resize((640, round(h * 640 / w)), Image.BILINEAR)
    arr = np.asarray(img)
    m = 10
    h, w = arr.shape[0] - 2 * m, arr.shape[1] - 2 * m
    frame1 = arr[m : m + h, m : m + w]
    frame2 = arr[m - DY : m - DY + h, m - DX : m - DX + w]
    return Image.fromarray(frame1), Image.fromarray(frame2)


def test_flowseek_t_clean_download_recovers_shift(tmp_path: Path, sample_image):
    checkpoint_path = tmp_path / "LibreFlowSeekt-flow.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LibreYOLO(str(checkpoint_path), device=device)
    frame1, frame2 = _shifted_pair(sample_image)
    results = model([frame1, frame2])

    assert checkpoint_path.is_file()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert validate_checkpoint_metadata(checkpoint, strict=True) == []
    assert (checkpoint["model_family"], checkpoint["size"], checkpoint["task"]) == (
        "flowseek",
        "t",
        "flow",
    )

    assert len(results) == 1
    result = results[0]
    assert result.boxes is None
    flow = result.flow.data
    flow = flow.cpu().numpy() if torch.is_tensor(flow) else np.asarray(flow)
    assert flow.shape == (frame1.height, frame1.width, 2)
    assert np.isfinite(flow).all()
    inner = flow[32:-32, 32:-32].reshape(-1, 2)
    assert abs(float(np.median(inner[:, 0])) - DX) < 0.5
    assert abs(float(np.median(inner[:, 1])) - DY) < 0.5
    assert result.flow.to_rgb().shape == (frame1.height, frame1.width, 3)
