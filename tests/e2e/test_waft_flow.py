"""WAFT clean-download and flow-result smoke on a synthetic frame pair."""

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
    pytest.mark.waft,
]

_ASSET = Path(__file__).resolve().parents[2] / "libreyolo" / "assets" / "guggenheim-bilbao.jpg"
_DX, _DY = 6.0, 3.0


def _frame_pair():
    """Frame b is frame a translated by (+6, +3) px (affine shift, PIL)."""
    img = Image.open(_ASSET).convert("RGB")
    img = img.resize((640, round(640 * img.height / img.width)))
    shifted = img.transform(
        img.size, Image.AFFINE, (1, 0, -_DX, 0, 1, -_DY), resample=Image.BILINEAR
    )
    return np.asarray(img), np.asarray(shifted)


def test_waft_m_clean_download_predicts_translation(tmp_path: Path):
    checkpoint_path = tmp_path / "LibreWAFTm-flow.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LibreYOLO(str(checkpoint_path), device=device)

    assert checkpoint_path.is_file()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert validate_checkpoint_metadata(checkpoint, strict=True) == []
    assert (checkpoint["model_family"], checkpoint["size"], checkpoint["task"]) == ("waft", "m", "flow")

    a, b = _frame_pair()
    results = model([a, b])
    assert len(results) == 1
    r = results[0]
    assert r.boxes is None
    flow = r.flow.data
    flow = flow.cpu().numpy() if torch.is_tensor(flow) else np.asarray(flow)
    assert flow.shape == (*a.shape[:2], 2)
    assert np.isfinite(flow).all()
    # Ignore the border band the shift uncovers.
    h, w = flow.shape[:2]
    inner = flow[h // 8 : -h // 8, w // 8 : -w // 8]
    median = np.median(inner.reshape(-1, 2), axis=0)
    assert abs(median[0] - _DX) < 0.25 and abs(median[1] - _DY) < 0.25, median
    rgb = r.flow.to_rgb()
    assert tuple(np.asarray(rgb).shape[:2]) == (h, w)
