"""Unit tests for the LibreVLM open-vocabulary API (offline, no model load).

``set_classes`` only manipulates the vocabulary maps, so it can be exercised on
a bare instance without downloading or loading any model.
"""

import pytest

from libreyolo.models.vlm.base import LibreVLMModel

pytestmark = pytest.mark.unit


def _bare_model():
    # Bypass __init__ (which would load an 8GB model); we only test the
    # vocabulary-map logic of set_classes.
    return object.__new__(LibreVLMModel)


class TestSetClasses:
    def test_builds_names_and_reverse_map(self):
        m = _bare_model()
        m.set_classes(["Pink Car", "Wheel"])
        assert m.names == {0: "Pink Car", 1: "Wheel"}
        assert m.nb_classes == 2
        # reverse map is lowercased for case-insensitive label resolution
        assert m._name_to_id == {"pink car": 0, "wheel": 1}

    def test_is_sticky_and_replaces(self):
        m = _bare_model()
        m.set_classes(["boat"])
        m.set_classes(["person", "dog"])
        assert m.names == {0: "person", 1: "dog"}
        assert m.nb_classes == 2
        assert m._name_to_id == {"person": 0, "dog": 1}

    def test_returns_self_for_chaining(self):
        m = _bare_model()
        assert m.set_classes(["boat"]) is m

    def test_empty_raises(self):
        m = _bare_model()
        with pytest.raises(ValueError):
            m.set_classes([])

    def test_coerces_to_str(self):
        m = _bare_model()
        m.set_classes(["boat", 7])
        assert m.names == {0: "boat", 1: "7"}
        assert m._name_to_id["7"] == 1


class TestFactoryResolution:
    """The LibreVLM(...) name resolution (offline; no model is loaded)."""

    def test_default_resolves_to_qwen3vl_4b(self):
        from libreyolo.models.vlm import _ALIASES, _DEFAULT_MODEL
        from libreyolo.models.vlm.qwen3vl import LibreQwen3VL

        assert _ALIASES[_DEFAULT_MODEL] == (LibreQwen3VL, "4b")

    def test_known_aliases_map_to_family_and_size(self):
        from libreyolo.models.vlm import _ALIASES
        from libreyolo.models.vlm.lfm2 import LibreLFM2VL
        from libreyolo.models.vlm.qwen3vl import LibreQwen3VL
        from libreyolo.models.vlm.smolvlm import LibreSmolVLM2

        assert _ALIASES["qwen3-vl-8b"] == (LibreQwen3VL, "8b")
        assert _ALIASES["lfm2-vl-450m"] == (LibreLFM2VL, "450m")
        assert _ALIASES["smolvlm2"] == (LibreSmolVLM2, "2.2b")

        from libreyolo.models.vlm.internvl3 import LibreInternVL3

        assert _ALIASES["internvl3"] == (LibreInternVL3, "2b")

        from libreyolo.models.vlm.florence2 import LibreFlorence2
        from libreyolo.models.vlm.kosmos2 import LibreKosmos2

        assert _ALIASES["florence-2"] == (LibreFlorence2, "base")
        assert _ALIASES["kosmos-2"] == (LibreKosmos2, "224")

    def test_unknown_alias_raises_before_loading(self):
        from libreyolo.models.vlm import LibreVLM

        # Raises during resolution, before any model download/load.
        with pytest.raises(ValueError):
            LibreVLM("definitely-not-a-real-model")
