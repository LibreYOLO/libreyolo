"""LibreRAFT clean-download and flow-result smoke on a synthetic shifted pair."""

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
    pytest.mark.raft,
]

_ASSET = Path(__file__).resolve().parents[2] / "libreyolo" / "assets" / "guggenheim-bilbao.jpg"
_DX, _DY = 6, 3  # content moves right and down between the two frames


def _shifted_pair(width: int = 480, height: int = 320) -> tuple[Image.Image, Image.Image]:
    img = Image.open(_ASSET).convert("RGB")
    x0 = (img.width - width) // 2
    y0 = (img.height - height) // 2
    frame1 = img.crop((x0, y0, x0 + width, y0 + height))
    # frame2(x + dx, y + dy) == frame1(x, y): ground-truth flow is (dx, dy).
    frame2 = img.crop((x0 - _DX, y0 - _DY, x0 - _DX + width, y0 - _DY + height))
    return frame1, frame2


@pytest.mark.parametrize("size", ["s", "l"])
def test_raft_clean_download_predicts_translation(tmp_path: Path, size: str):
    checkpoint_path = tmp_path / f"LibreRAFT{size}-flow.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LibreYOLO(str(checkpoint_path), device=device)
    frame1, frame2 = _shifted_pair()
    results = model([frame1, frame2])

    assert checkpoint_path.is_file()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert validate_checkpoint_metadata(checkpoint, strict=True) == []
    assert checkpoint["model_family"] == "raft"
    assert checkpoint["task"] == "flow"
    assert checkpoint["size"] == size

    assert len(results) == 1
    r = results[0]
    assert r.boxes is None
    flow = np.asarray(r.flow.data, dtype=np.float32)
    assert flow.shape == (frame1.height, frame1.width, 2)
    assert np.isfinite(flow).all()
    # Ignore a border band where the shift exposes unseen content.
    interior = flow[16:-16, 16:-16].reshape(-1, 2)
    median = np.median(interior, axis=0)
    assert abs(median[0] - _DX) < 0.5 and abs(median[1] - _DY) < 0.5, median
    epe = np.linalg.norm(interior - np.array([_DX, _DY], dtype=np.float32), axis=-1).mean()
    assert epe < 1.0, epe

    with pytest.raises(ValueError):
        model(frame1)
