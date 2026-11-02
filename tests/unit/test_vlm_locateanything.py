"""Offline tests for the LocateAnything VLM adapter."""

import re
from inspect import signature

import pytest

from libreyolo.models.vlm import locateanything as la
from libreyolo.models.vlm.locateanything import (
    LibreLocateAnything,
    _parse_locateanything_boxes,
    _parse_locateanything_points,
    build_point_dict,
)

pytestmark = pytest.mark.unit


def _bare(task="detect", names=None):
    m = object.__new__(LibreLocateAnything)
    names = names or ["person"]
    m.names = {i: name for i, name in enumerate(names)}
    m._name_to_id = {name.lower(): i for i, name in m.names.items()}
    m.task = task
    m._custom_prompt = None
    return m


class TestLocateAnythingParsing:
    def test_parses_labeled_boxes(self):
        text = (
            "result: person<box><100><200><300><400></box>"
            "car<box><10><20><30><40></box>"
        )

        assert _parse_locateanything_boxes(text) == [
            {"label": "person", "bbox": [100.0, 200.0, 300.0, 400.0]},
            {"label": "car", "bbox": [10.0, 20.0, 30.0, 40.0]},
        ]

    def test_parses_ref_labels_and_point_blocks(self):
        text = "<ref>traffic light</ref><box><500><250></box>"

        assert _parse_locateanything_points(text) == [
            {"label": "traffic light", "point": [500.0, 250.0]}
        ]

    def test_carries_ref_label_across_multiple_boxes(self):
        text = (
            "<ref>person</ref><box><100><100><200><200></box>"
            "<box><300><300><400><400></box>"
            "<ref>car</ref><box><500><500><600><600></box>"
        )

        assert _parse_locateanything_boxes(text) == [
            {"label": "person", "bbox": [100.0, 100.0, 200.0, 200.0]},
            {"label": "person", "bbox": [300.0, 300.0, 400.0, 400.0]},
            {"label": "car", "bbox": [500.0, 500.0, 600.0, 600.0]},
        ]

    def test_ignores_non_numeric_schema_echo(self):
        text = "format: <box> x1, y1, x2, y2 </box>"

        assert _parse_locateanything_boxes(text) == []
        assert _parse_locateanything_points(text) == []


class TestLocateAnythingPostprocess:
    def test_unlabeled_single_class_box_maps_to_the_only_class(self):
        det = _bare()._postprocess(
            "<box><100><200><300><400></box>",
            conf_thres=0.0,
            iou_thres=0.5,
            original_size=(1000, 500),
        )

        assert det == {
            "boxes": [[100.0, 100.0, 300.0, 200.0]],
            "scores": [1.0],
            "classes": [0],
            "num_detections": 1,
        }

    def test_multi_class_box_requires_a_matching_label(self):
        text = (
            "person<box><100><100><200><200></box>"
            "banana<box><300><300><400><400></box>"
            "<box><500><500><600><600></box>"
        )
        det = _bare(names=["person", "car"])._postprocess(
            text,
            conf_thres=0.0,
            iou_thres=0.5,
            original_size=(1000, 1000),
        )

        assert det["classes"] == [0]
        assert det["boxes"] == [[100.0, 100.0, 200.0, 200.0]]

    def test_point_task_returns_point_rows(self):
        det = _bare(task="point", names=["traffic light"])._postprocess(
            "<box><500><250></box>",
            conf_thres=0.0,
            iou_thres=0.5,
            original_size=(200, 100),
        )

        assert det == {
            "points": [[100.0, 25.0, 0.0, 1.0]],
            "num_detections": 1,
        }

    def test_point_builder_honors_conf_class_filter_and_max_det(self):
        items = [
            {"label": "person", "point": [100, 100]},
            {"label": "car", "point": [200, 200]},
            {"label": "car", "point": [300, 300]},
        ]

        det = build_point_dict(
            items,
            {"person": 0, "car": 1},
            (1000, 1000),
            conf_thres=0.5,
            max_det=1,
            classes=[1],
            default_score=1.0,
        )

        assert det == {"points": [[200.0, 200.0, 1.0, 1.0]], "num_detections": 1}

        dropped = build_point_dict(
            items,
            {"person": 0, "car": 1},
            (1000, 1000),
            conf_thres=1.5,
            default_score=1.0,
        )
        assert dropped == {"points": [], "num_detections": 0}

    def test_iou_threshold_suppresses_jitter_boxes(self):
        det = _bare()._postprocess(
            (
                "<ref>person</ref>"
                "<box><100><100><300><300></box>"
                "<box><105><105><305><305></box>"
                "<box><600><600><700><700></box>"
            ),
            conf_thres=0.0,
            iou_thres=0.5,
            original_size=(1000, 1000),
        )

        assert det["boxes"] == [
            [100.0, 100.0, 300.0, 300.0],
            [600.0, 600.0, 700.0, 700.0],
        ]


class TestLocateAnythingPrompt:
    def test_remote_code_revision_is_pinned(self):
        assert LibreLocateAnything.HF_REVISIONS["3b"]

    def test_generation_defaults_are_deterministic(self):
        params = signature(LibreLocateAnything.__init__).parameters

        assert params["do_sample"].default is False

    def test_detect_prompt_uses_category_delimiter(self):
        m = _bare(names=["person", "car"])

        assert (
            m._detection_prompt()
            == "Locate all the instances that matches the following description: person</c>car."
        )

    def test_point_prompt_uses_pointing_template(self):
        m = _bare(task="point", names=["search button"])

        assert m._detection_prompt() == "Point to: search button."

    def test_custom_prompt_wins(self):
        m = _bare()
        m._custom_prompt = "custom"

        assert m._detection_prompt() == "custom"


class TestLocateAnythingLicenseNotice:
    def test_notice_covers_weights_remote_code_and_upstream_terms(self):
        notice = LibreLocateAnything._LICENSE_NOTICE.lower()

        assert "does not redistribute" in notice
        assert "remote-code" in notice
        assert "hugging face remote code" in notice
        assert "nvidia" in notice
        assert "https://huggingface.co/nvidia/locateanything-3b" in notice


class TestDecordGuard:
    """decord ships wheels for Linux x86_64 and Windows x64 only (#989)."""

    def test_installed_decord_passes(self, monkeypatch):
        monkeypatch.setattr(la.importlib.util, "find_spec", lambda name: object())
        la._require_decord()

    @pytest.mark.parametrize(
        ("plat", "machine", "expected"),
        [
            ("linux", "x86_64", "pip install 'libreyolo[vlm]'"),
            ("win32", "AMD64", "pip install 'libreyolo[vlm]'"),
            ("darwin", "arm64", "published only for Linux x86_64 and Windows x64"),
            ("linux", "aarch64", "published only for Linux x86_64 and Windows x64"),
        ],
    )
    def test_missing_decord_message_fits_the_platform(
        self, monkeypatch, plat, machine, expected
    ):
        monkeypatch.setattr(la.importlib.util, "find_spec", lambda name: None)
        monkeypatch.setattr(la.sys, "platform", plat)
        monkeypatch.setattr(la.platform, "machine", lambda: machine)
        with pytest.raises(ImportError, match=re.escape(expected)):
            la._require_decord()

    def test_guard_runs_before_any_weight_download(self, monkeypatch):
        def no_decord():
            raise ImportError("no decord")

        def must_not_download(self):
            raise AssertionError("weights downloaded before the decord check")

        monkeypatch.setattr(la, "_require_decord", no_decord)
        monkeypatch.setattr(LibreLocateAnything, "_ensure_weights", must_not_download)
        with pytest.raises(ImportError, match="no decord"):
            _bare()._init_model()
