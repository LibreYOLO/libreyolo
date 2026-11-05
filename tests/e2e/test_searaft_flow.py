"""SEA-RAFT clean-download and flow-result smoke on a synthetic frame pair."""

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
    pytest.mark.searaft,
]

ASSET = Path(__file__).resolve().parents[2] / "libreyolo" / "assets" / "guggenheim-bilbao.jpg"
SHIFT = (6.0, 3.0)  # (dx, dy) in pixels of frame 2 relative to frame 1


def _shifted_pair(width: int = 640):
    img = Image.open(ASSET).convert("RGB")
    img = img.resize((width, round(img.height * width / img.width)), Image.BILINEAR)
    a = np.asarray(img)
    dx, dy = int(SHIFT[0]), int(SHIFT[1])
    # frame 2 = frame 1 translated by (+dx, +dy); content moves right and down
    b = np.roll(np.roll(a, dy, axis=0), dx, axis=1)
    return Image.fromarray(a), Image.fromarray(b)


@pytest.mark.parametrize("size", ["s", "m", "l"])
def test_searaft_clean_download_predicts_translation(tmp_path: Path, size: str):
    checkpoint_path = tmp_path / f"LibreSEARAFT{size}-flow.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LibreYOLO(str(checkpoint_path), device=device)
    a, b = _shifted_pair()
    results = model([a, b])

    assert checkpoint_path.is_file()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert validate_checkpoint_metadata(checkpoint, strict=True) == []
    assert checkpoint["model_family"] == "searaft"
    assert checkpoint["size"] == size
    assert checkpoint["task"] == "flow"

    assert len(results) == 1
    r = results[0]
    assert r.boxes is None
    flow = torch.as_tensor(r.flow.data).float()
    h, w = a.height, a.width
    assert tuple(flow.shape) == (h, w, 2)
    assert torch.isfinite(flow).all()
    conf = torch.as_tensor(r.flow.confidence)
    assert conf.shape == (h, w) and 0.0 <= float(conf.min()) <= float(conf.max()) <= 1.0

    # Away from the wrapped borders the field is the global translation.
    inner = flow[32 : h - 32, 32 : w - 32]
    median = inner.reshape(-1, 2).median(dim=0).values
    assert abs(float(median[0]) - SHIFT[0]) < 0.25, median
    assert abs(float(median[1]) - SHIFT[1]) < 0.25, median
    epe = (inner - torch.tensor(SHIFT)).norm(dim=-1).mean()
    assert float(epe) < 0.5, float(epe)
    assert r.flow.to_rgb().shape[:2] == (h, w)
