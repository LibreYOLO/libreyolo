"""NeuFlow v2 clean-download and flow-result smoke on a synthetic shift."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreYOLO
from libreyolo.utils.serialization import validate_checkpoint_metadata

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.network,
    pytest.mark.neuflowv2,
]

# Frame 2 is frame 1 translated by (DX, DY) pixels, so the true flow from
# frame 1 to frame 2 is (DX, DY) everywhere the content stays in view.
DX, DY = 6.0, 3.0


def _shifted_pair(sample_image, tmp_path: Path) -> list[str]:
    img = np.asarray(Image.open(sample_image).convert("RGB"))
    h, w = img.shape[:2]
    scale = 960 / w
    img = cv2.resize(img, (960, round(h * scale)), interpolation=cv2.INTER_AREA)
    shift = np.float32([[1, 0, DX], [0, 1, DY]])
    moved = cv2.warpAffine(
        img, shift, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT
    )
    paths = []
    for name, frame in (("000.png", img), ("001.png", moved)):
        path = tmp_path / name
        Image.fromarray(frame).save(path)
        paths.append(str(path))
    return paths


def test_neuflowv2_clean_download_recovers_a_known_shift(tmp_path: Path, sample_image):
    """Flow task: not a MODEL_CATALOG row (that catalog is detect-only)."""
    checkpoint_path = tmp_path / "LibreNeuFlowv2s-flow.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LibreYOLO(str(checkpoint_path), device=device)
    frames = _shifted_pair(sample_image, tmp_path)
    results = model.predict(frames)

    assert checkpoint_path.is_file()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert validate_checkpoint_metadata(checkpoint, strict=True) == []
    assert checkpoint["model_family"] == "neuflowv2"
    assert checkpoint["task"] == "flow"

    assert len(results) == 1
    result = results[0]
    assert result.boxes is None
    flow = result.flow.data
    flow = flow if isinstance(flow, torch.Tensor) else torch.as_tensor(flow)
    assert tuple(flow.shape) == (*result.orig_shape, 2)
    assert torch.isfinite(flow).all()

    # Interior pixels only: borders are reflected, not translated.
    h, w = flow.shape[:2]
    inner = flow[h // 8 : -h // 8, w // 8 : -w // 8].reshape(-1, 2)
    median = inner.median(dim=0).values
    assert abs(float(median[0]) - DX) < 0.5, median
    assert abs(float(median[1]) - DY) < 0.5, median
    epe = (inner - inner.new_tensor([DX, DY])).norm(dim=1).mean()
    assert float(epe) < 1.0, float(epe)

    with pytest.raises(ValueError):
        model.predict(frames[0])
