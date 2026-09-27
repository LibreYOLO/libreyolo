"""Offline unit tests for Gemma 4 and Moondream VLM adapters."""

from __future__ import annotations

import logging
import re

import pytest

from libreyolo.models.vlm.gemma4 import LibreGemma4
from libreyolo.models.vlm.moondream import (
    LibreMoondream,
    objects_to_box_items,
    objects_to_point_items,
)
from libreyolo.models.vlm.parsing import build_detection_dict

pytestmark = pytest.mark.unit

_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


class TestGemma4Adapter:
    def test_coordinate_knobs(self):
        assert LibreGemma4.BBOX_KEY == "box_2d"
        assert LibreGemma4.COORD_DIVISOR == 1000.0
        assert LibreGemma4.BOX_FORMAT == "yxyx"
        assert LibreGemma4.TRUST_REMOTE_CODE is False
        assert LibreGemma4.HF_REPOS["e2b"].startswith("LibreYOLO/")
        assert LibreGemma4.HF_REPOS["e4b"].startswith("LibreYOLO/")

    def test_official_detection_prompt(self):
        m = object.__new__(LibreGemma4)
        m._custom_prompt = None
        m.names = {0: "person", 1: "cat"}
        assert m._detection_prompt() == "detect person and cat, output only ```json"

    def test_thinking_is_disabled(self):
        m = object.__new__(LibreGemma4)
        assert m._chat_template_kwargs() == {"enable_thinking": False}

    def test_old_transformers_fails_before_download(self, monkeypatch):
        import sys
        import types

        from libreyolo.models.vlm import gemma4

        fake = types.SimpleNamespace(__version__="5.1.0")
        monkeypatch.setitem(sys.modules, "transformers", fake)
        with pytest.raises(ImportError, match="transformers>=5.10.0"):
            gemma4._require_transformers()


class TestMoondreamAdapter:
    def test_remote_code_revisions_are_commit_shas(self):
        assert LibreMoondream.TRUST_REMOTE_CODE is True
        assert set(LibreMoondream.HF_REPOS) == set(LibreMoondream.HF_REVISIONS)
        assert LibreMoondream.HF_REPOS["2"] == "LibreYOLO/LibreMoondream2"
        assert LibreMoondream.HF_REPOS["3"] == "LibreYOLO/LibreMoondream3"
        for size, revision in LibreMoondream.HF_REVISIONS.items():
            assert _COMMIT_SHA.fullmatch(revision), (size, revision)

    def test_detect_objects_become_xyxy_items(self):
        items = objects_to_box_items(
            [{"x_min": 0.1, "y_min": 0.2, "x_max": 0.3, "y_max": 0.4}],
            "car",
        )
        assert items == [{"label": "car", "bbox": [0.1, 0.2, 0.3, 0.4]}]
        det = build_detection_dict(items, {"car": 0}, (100, 50))
        assert det["num_detections"] == 1
        assert det["boxes"][0] == [10.0, 10.0, 30.0, 20.0]

    def test_point_objects_keep_normalized_xy(self):
        items = objects_to_point_items([{"x": 0.25, "y": 0.75}], "person")
        assert items == [{"label": "person", "point": [0.25, 0.75]}]

    def test_malformed_skill_rows_are_dropped(self):
        assert objects_to_box_items([{"x_min": 0.1}], "car") == []
        assert objects_to_point_items([{"x": 0.1}], "car") == []
        assert objects_to_box_items(None, "car") == []

    def test_size_3_logs_bsl_notice_once(self, caplog):
        LibreMoondream._MD3_NOTICE_SHOWN = False
        m = object.__new__(LibreMoondream)
        m.size = "3"
        caplog.set_level(logging.WARNING)
        m._notify_license_once()
        m._notify_license_once()
        assert caplog.text.count("Business Source") == 1

    def test_finalize_patch_is_idempotent(self):
        from libreyolo.models.vlm import moondream as md

        class Dummy:
            _finalize_model_loading = staticmethod(lambda *a, **k: "ok")

        md._POST_INIT_PATCHED = False
        md._ensure_remote_post_init(Dummy)
        first = Dummy._finalize_model_loading
        md._ensure_remote_post_init(Dummy)
        assert Dummy._finalize_model_loading is first
        md._POST_INIT_PATCHED = False

    def test_size_2_has_no_license_notice(self, caplog):
        m = object.__new__(LibreMoondream)
        m.size = "2"
        caplog.set_level(logging.WARNING)
        m._notify_license_once()
        assert caplog.text == ""
