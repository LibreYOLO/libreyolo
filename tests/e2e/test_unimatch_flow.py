"""UniMatch clean-download and optical flow smoke on a synthetic frame pair.

There is no video fixture, so the pair is the Guggenheim asset and a copy
shifted by a known translation: frame 2 is frame 1 moved right by ``DX`` and
down by ``DY`` pixels, so the true flow is ``(DX, DY)`` on the interior.
"""

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
    pytest.mark.unimatch,
]

ASSET = Path(__file__).resolve().parents[2] / "libreyolo" / "assets" / "guggenheim-bilbao.jpg"
DX, DY = 6, 3
W, H = 640, 480


def _shifted_pair():
    base = Image.open(ASSET).convert("RGB").resize((W + 2 * DX, H + 2 * DY))
    frame1 = base.crop((DX, DY, DX + W, DY + H))
    # Content moves right and down: frame 2 samples the base up-left of frame 1.
    frame2 = base.crop((0, 0, W, H))
    return frame1, frame2


@pytest.mark.parametrize("size", ["s", "l"])
def test_unimatch_clean_download_predicts_known_shift(tmp_path: Path, size: str):
    checkpoint_path = tmp_path / f"LibreUniMatch{size}-flow.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LibreYOLO(str(checkpoint_path), device=device)
    frame1, frame2 = _shifted_pair()
    results = model([frame1, frame2])

    assert checkpoint_path.is_file()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert validate_checkpoint_metadata(checkpoint, strict=True) == []
    assert checkpoint["model_family"] == "unimatch"
    assert checkpoint["task"] == "flow" and checkpoint["size"] == size

    assert len(results) == 1
    result = results[0]
    assert result.boxes is None
    flow = np.asarray(result.flow.data)
    assert flow.shape == (H, W, 2)
    assert np.isfinite(flow).all()
    interior = flow[32:-32, 32:-32]
    epe = np.linalg.norm(interior - np.array([DX, DY], dtype=np.float32), axis=-1)
    assert float(np.median(epe)) < 0.5, float(np.median(epe))
    assert result.flow.to_rgb().shape == (H, W, 3)
