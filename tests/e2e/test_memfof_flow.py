"""MEMFOF clean download and three-frame forward-flow smoke on a synthetic pan."""

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
    pytest.mark.memfof,
]

# Each frame is the previous one translated by (DX, DY) pixels, so the forward
# flow of every interior frame is (DX, DY) away from the borders.
DX, DY = 3.0, 2.0


def _pan(sample_image, count: int = 4) -> list[Image.Image]:
    base = Image.open(sample_image).convert("RGB")
    base = base.resize((640, round(640 * base.height / base.width)), Image.BICUBIC)
    return [
        base.transform(
            base.size, Image.AFFINE, (1, 0, -DX * k, 0, 1, -DY * k), resample=Image.BICUBIC
        )
        for k in range(count)
    ]


def test_memfof_clean_download_predicts_forward_flow(tmp_path: Path, sample_image):
    checkpoint_path = tmp_path / "LibreMEMFOFm-flow.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LibreYOLO(str(checkpoint_path), device=device)
    frames = _pan(sample_image)
    results = model(frames)

    assert checkpoint_path.is_file()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert validate_checkpoint_metadata(checkpoint, strict=True) == []
    assert checkpoint["model_family"] == "memfof"
    assert checkpoint["task"] == "flow"
    assert checkpoint["variant"] == "tskh"

    assert len(results) == len(frames) - 1
    h, w = frames[0].height, frames[0].width
    for result in results:
        assert result.boxes is None
        flow = result.flow.data
        flow = flow.cpu().numpy() if isinstance(flow, torch.Tensor) else np.asarray(flow)
        assert flow.shape == (h, w, 2)
        assert np.isfinite(flow).all()
        interior = flow[32:-32, 32:-32].reshape(-1, 2)
        median = np.median(interior, axis=0)
        assert abs(median[0] - DX) < 0.5, median
        assert abs(median[1] - DY) < 0.5, median
    assert result.flow.to_rgb().shape == (h, w, 3)
