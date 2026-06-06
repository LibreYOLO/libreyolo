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

    def test_string_or_scalar_raises(self):
        # A bare string would enumerate into one-character classes; reject it.
        m = _bare_model()
        with pytest.raises(TypeError):
            m.set_classes("person")
        with pytest.raises(TypeError):
            m.set_classes(123)

    def test_coerces_to_str(self):
        m = _bare_model()
        m.set_classes(["boat", 7])
        assert m.names == {0: "boat", 1: "7"}
        assert m._name_to_id["7"] == 1

    def test_duplicate_casefolded_names_raise(self):
        m = _bare_model()
        with pytest.raises(ValueError):
            m.set_classes(["Boat", "boat"])


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


class TestSnapshotComplete:
    """The weights-completeness sentinel (offline; no download)."""

    def _base(self):
        from libreyolo.models.vlm.base import LibreVLMModel

        return LibreVLMModel

    def _mark_complete(self, path):
        (path / ".libreyolo_snapshot_complete").write_text("{}")

    def test_single_file_complete(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_text("x")
        self._mark_complete(tmp_path)
        assert self._base()._snapshot_complete(tmp_path) is True

    def test_missing_completion_marker_is_incomplete(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_text("x")
        assert self._base()._snapshot_complete(tmp_path) is False

    def test_missing_config_is_incomplete(self, tmp_path):
        (tmp_path / "model.safetensors").write_text("x")
        self._mark_complete(tmp_path)
        assert self._base()._snapshot_complete(tmp_path) is False

    def test_sharded_incomplete_when_a_shard_missing(self, tmp_path):
        import json

        (tmp_path / "config.json").write_text("{}")
        self._mark_complete(tmp_path)
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}})
        )
        (tmp_path / "s1.safetensors").write_text("x")  # only shard 1 of 2
        assert self._base()._snapshot_complete(tmp_path) is False

    def test_sharded_complete_when_all_shards_present(self, tmp_path):
        import json

        (tmp_path / "config.json").write_text("{}")
        self._mark_complete(tmp_path)
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": "s1.safetensors", "b": "s2.safetensors"}})
        )
        (tmp_path / "s1.safetensors").write_text("x")
        (tmp_path / "s2.safetensors").write_text("x")
        assert self._base()._snapshot_complete(tmp_path) is True


class TestCastInputs:
    def test_casts_float_tensors_inside_mutable_payload(self):
        import torch

        m = _bare_model()
        m._model_dtype = torch.float16
        payload = {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
            "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
            "nested": [{"image_grid_thw": torch.ones((1, 3), dtype=torch.float32)}],
        }

        out = m._cast_inputs(payload)

        assert out["input_ids"].dtype == torch.int64
        assert out["pixel_values"].dtype == torch.float16
        assert out["nested"][0]["image_grid_thw"].dtype == torch.float16

    def test_prepare_generation_inputs_drops_token_type_ids(self):
        import torch

        m = _bare_model()
        m._model_dtype = torch.float16
        payload = {
            "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
            "token_type_ids": torch.tensor([[0, 0]], dtype=torch.int64),
            "pixel_values": torch.ones((1, 3, 2, 2), dtype=torch.float32),
        }

        out = m._prepare_generation_inputs(payload)

        assert "token_type_ids" not in out
        assert out["pixel_values"].dtype == torch.float16


class _StubProc:
    def __init__(self, payload):
        self._payload = payload

    def batch_decode(self, *args, **kwargs):
        return ["<decoded>"]

    def post_process_generation(self, *args, **kwargs):
        return self._payload


class TestOverrideConfThreshold:
    """Florence-2 and Kosmos-2 honor conf= even though they build the dict directly."""

    def _florence(self):
        from libreyolo.models.vlm.florence2 import LibreFlorence2

        m = object.__new__(LibreFlorence2)
        m._name_to_id = {"boat": 0}
        m.processor = _StubProc(
            {LibreFlorence2.TASK: {"bboxes": [[0, 0, 10, 10]], "bboxes_labels": ["boat"]}}
        )
        return m

    def _kosmos(self):
        from libreyolo.models.vlm.kosmos2 import LibreKosmos2

        m = object.__new__(LibreKosmos2)
        m._name_to_id = {"boat": 0}
        m.processor = _StubProc(("a boat", [("boat", (0, 4), [[0.0, 0.0, 0.5, 0.5]])]))
        return m

    def test_florence_conf_below_score_keeps(self):
        det = self._florence()._postprocess(None, 0.5, 0.5, (100, 100))
        assert det["num_detections"] == 1

    def test_florence_conf_above_score_drops_all(self):
        det = self._florence()._postprocess(None, 1.5, 0.5, (100, 100))
        assert det["num_detections"] == 0

    def test_kosmos_conf_below_score_keeps(self):
        det = self._kosmos()._postprocess(None, 0.5, 0.5, (100, 100))
        assert det["num_detections"] == 1

    def test_kosmos_conf_above_score_drops_all(self):
        det = self._kosmos()._postprocess(None, 1.5, 0.5, (100, 100))
        assert det["num_detections"] == 0


class TestKosmosMatchLabel:
    """Kosmos-2's lenient noun-phrase to vocabulary matching (pure, offline)."""

    def _kosmos(self, names):
        from libreyolo.models.vlm.kosmos2 import LibreKosmos2

        m = object.__new__(LibreKosmos2)
        m._name_to_id = {n.lower(): i for i, n in enumerate(names)}
        return m

    def test_exact_match(self):
        assert self._kosmos(["boat", "person"])._match_label("boat") == 0

    def test_lenient_plural_phrase(self):
        # Kosmos grounds noun phrases ("the boats"); lenient substring still maps.
        assert self._kosmos(["boat"])._match_label("the boats") == 0

    def test_out_of_vocab_returns_none(self):
        assert self._kosmos(["boat"])._match_label("airplane") is None


class TestInternVL3Flatten:
    """InternVL3's nested-box flatten override (pure, offline)."""

    def _flat(self, items):
        from libreyolo.models.vlm.internvl3 import LibreInternVL3

        return LibreInternVL3._flatten_nested(items)

    def test_nested_boxes_expand_to_one_item_each(self):
        items = [{"label": "boat", "bbox": [[120, 400, 250, 550], [600, 100, 700, 200]]}]
        assert self._flat(items) == [
            {"label": "boat", "bbox": [120, 400, 250, 550]},
            {"label": "boat", "bbox": [600, 100, 700, 200]},
        ]

    def test_flat_box_passes_through(self):
        items = [{"label": "boat", "bbox": [120, 400, 250, 550]}]
        assert self._flat(items) == items

    def test_mixed_nested_and_flat(self):
        items = [
            {"label": "boat", "bbox": [[1, 2, 3, 4], [5, 6, 7, 8]]},
            {"label": "ship", "bbox": [9, 10, 11, 12]},
        ]
        assert self._flat(items) == [
            {"label": "boat", "bbox": [1, 2, 3, 4]},
            {"label": "boat", "bbox": [5, 6, 7, 8]},
            {"label": "ship", "bbox": [9, 10, 11, 12]},
        ]
