"""LibreVocab1 inference parity test (env-var gated).

Runs the full SAM3 + CRADIOv4 stack on a real image with real weights and
verifies that text-prompted detection + segmentation produces sensible
outputs. Disabled by default — opt in by setting:

    LIBREVOCAB1_RUN_INFERENCE=1 \\
    LIBREVOCAB1_SIZE=s          # or 'h'
    LIBREVOCAB1_DEVICE=cuda     # or 'cpu', 'mps'
    LIBREVOCAB1_IMAGE=/path/to/test.jpg  # optional override
    LIBREVOCAB1_PROMPT=shoe     # optional override

Will download SAM3 (~3-5 GB) and CRADIOv4 weights (SO400M ~1.6 GB or H ~3 GB)
on first run.
"""

from __future__ import annotations

import os
import importlib.util
from pathlib import Path

import pytest


_REQUIRED = ("einops", "timm", "huggingface_hub", "iopath", "ftfy", "regex")
_MISSING = [m for m in _REQUIRED if importlib.util.find_spec(m) is None]

_GATE = os.environ.get("LIBREVOCAB1_RUN_INFERENCE", "0") == "1"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not _GATE,
        reason="set LIBREVOCAB1_RUN_INFERENCE=1 to enable; downloads ~5 GB of weights.",
    ),
    pytest.mark.skipif(
        bool(_MISSING),
        reason=f"libreyolo[vocab1] extras missing: {_MISSING!r}",
    ),
]


def _resolve_image_path() -> str:
    override = os.environ.get("LIBREVOCAB1_IMAGE")
    if override:
        return override
    # Default to LibreYOLO's bundled sample.
    here = Path(__file__).resolve().parents[2]
    return str(here / "libreyolo" / "assets" / "parkour.jpg")


def test_librevocab1_text_prompt_inference():
    from libreyolo.models.librevocab1 import LibreVocab1

    size = os.environ.get("LIBREVOCAB1_SIZE", "s")
    device = os.environ.get("LIBREVOCAB1_DEVICE", "cpu")
    prompt = os.environ.get("LIBREVOCAB1_PROMPT", "person")
    image_path = _resolve_image_path()

    assert Path(image_path).exists(), f"test image missing: {image_path}"

    # Construction triggers SAM3 + CRADIOv4 weight downloads on first run.
    model = LibreVocab1(size=size, device=device)

    results = model.predict(image_path, prompts=[prompt])

    assert isinstance(results, list) and len(results) == 1
    r = results[0]
    assert r["prompt"] == prompt
    # boxes/masks/scores can be empty (no detections) but must be present.
    assert "boxes" in r and "masks" in r and "scores" in r
    if r["boxes"] is not None and len(r["boxes"]) > 0:
        # boxes are xyxy in pixel coords; predict() clamps to image bounds.
        from PIL import Image
        w, h = Image.open(image_path).size
        b = r["boxes"]
        assert (b[:, 0] >= 0).all() and (b[:, 2] <= w).all()
        assert (b[:, 1] >= 0).all() and (b[:, 3] <= h).all()
        # x1 < x2, y1 < y2 (degenerate boxes would mean a postprocess bug).
        assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()
        # scores in [0, 1].
        s = r["scores"]
        assert (s >= 0).all() and (s <= 1).all()
