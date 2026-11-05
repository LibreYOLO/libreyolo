"""Falcon Perception is reachable through LibreVLM and the top-level package."""

import pytest

from libreyolo.models.falcon_perception import LibreFalconPerception

pytestmark = pytest.mark.unit


def test_factory_aliases_resolve_lazily():
    from libreyolo.models.vlm import _FALCON_ALIASES

    assert _FALCON_ALIASES["falcon-perception"] == "600m"
    assert _FALCON_ALIASES["falcon-perception-300m"] == "300m"
    import libreyolo

    assert libreyolo.LibreFalconPerception is LibreFalconPerception
