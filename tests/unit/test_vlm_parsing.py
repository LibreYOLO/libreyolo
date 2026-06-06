"""Unit tests for LibreVLM detection-output parsing (offline, no model)."""

import pytest

from libreyolo.models.vlm.parsing import (
    build_detection_dict,
    extract_detections,
    normalize_bbox,
    resolve_label,
)

pytestmark = pytest.mark.unit

# COCO-ish minimal vocabulary for tests.
NAME_TO_ID = {"person": 0, "ship": 8, "statue": 73}


class TestExtractDetections:
    def test_documented_single_object(self):
        # Verbatim from the LFM2.5-VL model card / Liquid docs.
        text = '[{"label": "statue", "bbox": [0.3, 0.25, 0.4, 0.65]}]'
        items = extract_detections(text)
        assert items == [{"label": "statue", "bbox": [0.3, 0.25, 0.4, 0.65]}]

    def test_multiple_objects(self):
        text = (
            '[{"label": "person", "bbox": [0.1, 0.1, 0.2, 0.5]}, '
            '{"label": "ship", "bbox": [0.37, 0.0, 0.8, 0.99]}]'
        )
        assert len(extract_detections(text)) == 2

    def test_markdown_fenced(self):
        text = '```json\n[{"label": "ship", "bbox": [0.37, 0.0, 0.8, 0.99]}]\n```'
        assert extract_detections(text) == [
            {"label": "ship", "bbox": [0.37, 0.0, 0.8, 0.99]}
        ]

    def test_prose_preamble(self):
        text = 'Here is what I found: [{"label": "person", "bbox": [0, 0, 1, 1]}] done.'
        assert extract_detections(text) == [{"label": "person", "bbox": [0, 0, 1, 1]}]

    def test_truncated_array_recovers_objects(self):
        # Generation hit the token budget mid-array, no closing bracket.
        text = (
            '[{"label": "person", "bbox": [0.1, 0.1, 0.2, 0.5]}, '
            '{"label": "ship", "bbox": [0.3, 0.3, 0.4, 0.6]}'
        )
        items = extract_detections(text)
        assert len(items) == 2

    def test_single_quotes(self):
        text = "[{'label': 'ship', 'bbox': [0.1, 0.2, 0.3, 0.4]}]"
        items = extract_detections(text)
        assert items and items[0]["label"] == "ship"

    def test_empty_and_noise(self):
        assert extract_detections("") == []
        assert extract_detections("No objects found.") == []
        assert extract_detections("[]") == []
        assert extract_detections(None) == []  # type: ignore[arg-type]

    def test_bracket_in_prose_before_json(self):
        # A prose bracket like "[0,1]" must not shadow the real detection array.
        text = 'Coordinates are normalized to [0,1]. [{"label": "person", "bbox": [0, 0, 1, 1]}]'
        assert extract_detections(text) == [{"label": "person", "bbox": [0, 0, 1, 1]}]

    def test_single_object_without_enclosing_array(self):
        # A bare object (no outer array) must still be recovered.
        text = '{"label": "person", "bbox": [0, 0, 1, 1]}'
        assert extract_detections(text) == [{"label": "person", "bbox": [0, 0, 1, 1]}]

    def test_count_bracket_before_json(self):
        text = 'I found [2] objects: [{"label": "car", "bbox": [0.1, 0.2, 0.3, 0.4]}]'
        items = extract_detections(text)
        assert items and items[0]["label"] == "car"

    def test_metadata_array_does_not_shadow_detections(self):
        # A dict-bearing but non-detection preamble array must not win.
        text = 'Classes: [{"id": 1}, {"id": 2}]. Detections: [{"label": "ship", "bbox": [0.1, 0.2, 0.3, 0.4]}]'
        assert extract_detections(text) == [{"label": "ship", "bbox": [0.1, 0.2, 0.3, 0.4]}]

    def test_schema_echo_does_not_lose_real_detection(self):
        # A restated schema example is also detection-shaped; collecting across
        # arrays keeps the real ship (its placeholder label is dropped by vocab).
        text = (
            'Format: [{"label": "object", "bbox": [0, 0, 0, 0]}]. '
            'Found: [{"label": "ship", "bbox": [0.1, 0.2, 0.3, 0.4]}]'
        )
        labels = [d["label"] for d in extract_detections(text)]
        assert "ship" in labels

    def test_truncated_real_array_behind_preamble(self):
        # A complete schema-echo array, then a real array truncated mid-content:
        # the real (complete) object is still recovered by the regex supplement.
        text = (
            'Format: [{"label": "object", "bbox": [0, 0, 0, 0]}]. '
            'Detections: [{"label": "ship", "bbox": [0.1, 0.2, 0.3, 0.4]}, '
            '{"label": "boat", "bbox": [0.5, 0.5, 0.6'
        )
        labels = [d["label"] for d in extract_detections(text)]
        assert "ship" in labels


class TestNormalizeBbox:
    def test_passthrough(self):
        assert normalize_bbox([0.1, 0.2, 0.3, 0.4]) == (0.1, 0.2, 0.3, 0.4)

    def test_clamp_out_of_range(self):
        assert normalize_bbox([-0.2, 0.0, 1.5, 1.2]) == (0.0, 0.0, 1.0, 1.0)

    def test_reorders_inverted_corners(self):
        assert normalize_bbox([0.8, 0.9, 0.2, 0.1]) == (0.2, 0.1, 0.8, 0.9)

    def test_rejects_zero_area(self):
        assert normalize_bbox([0.0, 0.0, 0.0, 0.5]) is None
        assert normalize_bbox([0.0, 0.0, 0.5, 0.0]) is None

    def test_rejects_bad_shapes(self):
        assert normalize_bbox([0.1, 0.2, 0.3]) is None
        assert normalize_bbox("nope") is None
        assert normalize_bbox([0.1, 0.2, 0.3, "x"]) is None

    def test_rejects_non_finite(self):
        assert normalize_bbox([0.1, 0.2, float("nan"), 0.4]) is None
        assert normalize_bbox([0.1, 0.2, float("inf"), 0.4]) is None


class TestResolveLabel:
    def test_case_insensitive(self):
        assert resolve_label("Person", NAME_TO_ID) == 0
        assert resolve_label("  SHIP ", NAME_TO_ID) == 8

    def test_out_of_vocab_dropped(self):
        assert resolve_label("forklift", NAME_TO_ID) is None
        assert resolve_label(123, NAME_TO_ID) is None  # type: ignore[arg-type]


class TestBuildDetectionDict:
    def test_scales_to_pixels(self):
        items = [{"label": "person", "bbox": [0.5, 0.5, 1.0, 1.0]}]
        det = build_detection_dict(items, NAME_TO_ID, (640, 480))
        assert det["num_detections"] == 1
        assert det["boxes"][0] == [320.0, 240.0, 640.0, 480.0]
        assert det["classes"][0] == 0
        assert det["scores"][0] == 1.0

    def test_drops_out_of_vocab_and_bad_boxes(self):
        items = [
            {"label": "forklift", "bbox": [0.1, 0.1, 0.2, 0.2]},  # OOV
            {"label": "ship", "bbox": [0.1, 0.1, 0.2]},  # bad box
            {"label": "ship", "bbox": [0.1, 0.1, 0.2, 0.3]},  # kept
        ]
        det = build_detection_dict(items, NAME_TO_ID, (100, 100))
        assert det["num_detections"] == 1
        assert det["classes"] == [8]

    def test_conf_threshold_filters(self):
        items = [{"label": "ship", "bbox": [0, 0, 1, 1]}]
        det = build_detection_dict(
            items, NAME_TO_ID, (100, 100), conf_thres=0.5, default_score=0.3
        )
        assert det["num_detections"] == 0

    def test_max_det_caps(self):
        # Distinct boxes so the cap (not dedup) is what limits the count.
        items = [
            {"label": "person", "bbox": [i * 0.05, i * 0.05, i * 0.05 + 0.04, i * 0.05 + 0.04]}
            for i in range(10)
        ]
        det = build_detection_dict(items, NAME_TO_ID, (100, 100), max_det=3)
        assert det["num_detections"] == 3

    def test_max_det_zero_returns_empty(self):
        items = [{"label": "person", "bbox": [0.1, 0.1, 0.2, 0.2]}]
        det = build_detection_dict(items, NAME_TO_ID, (100, 100), max_det=0)
        assert det["num_detections"] == 0

    def test_class_filter_applies_before_max_det(self):
        items = [
            {"label": "person", "bbox": [0.1, 0.1, 0.2, 0.2]},
            {"label": "ship", "bbox": [0.3, 0.3, 0.4, 0.4]},
        ]
        det = build_detection_dict(items, NAME_TO_ID, (100, 100), max_det=1, classes=[8])
        assert det["num_detections"] == 1
        assert det["classes"] == [8]

    def test_dedup_identical_boxes(self):
        # A repetition loop emits the same object many times; collapse to one.
        items = [{"label": "person", "bbox": [0.1, 0.1, 0.2, 0.2]}] * 5
        det = build_detection_dict(items, NAME_TO_ID, (100, 100))
        assert det["num_detections"] == 1

    def test_dedup_keeps_distinct_same_class(self):
        items = [
            {"label": "person", "bbox": [0.1, 0.1, 0.2, 0.2]},
            {"label": "person", "bbox": [0.5, 0.5, 0.6, 0.6]},
        ]
        det = build_detection_dict(items, NAME_TO_ID, (100, 100))
        assert det["num_detections"] == 2

    def test_empty_items(self):
        det = build_detection_dict([], NAME_TO_ID, (100, 100))
        assert det == {
            "boxes": [],
            "scores": [],
            "classes": [],
            "num_detections": 0,
        }

    def test_box_format_xywh(self):
        # x,y,w,h: [0.25,0.25,0.25,0.5] -> xyxy [0.25,0.25,0.5,0.75] -> px
        items = [{"label": "ship", "bbox": [0.25, 0.25, 0.25, 0.5]}]
        det = build_detection_dict(items, NAME_TO_ID, (100, 100), box_format="xywh")
        assert det["boxes"][0] == [25.0, 25.0, 50.0, 75.0]

    def test_box_format_cxcywh(self):
        # center 0.5,0.5 size 0.25,0.5 -> xyxy [0.375,0.25,0.625,0.75] -> px
        items = [{"label": "ship", "bbox": [0.5, 0.5, 0.25, 0.5]}]
        det = build_detection_dict(items, NAME_TO_ID, (100, 100), box_format="cxcywh")
        assert det["boxes"][0] == [37.5, 25.0, 62.5, 75.0]

    def test_schema_echo_label_dropped_keeps_real(self):
        # End to end for the preamble-shadowing fix: the echoed "object" example
        # is out of vocab and dropped; the real ship survives.
        items = extract_detections(
            'Format: [{"label": "object", "bbox": [0, 0, 0, 0]}]. '
            'Detections: [{"label": "ship", "bbox": [0.1, 0.2, 0.3, 0.4]}]'
        )
        det = build_detection_dict(items, NAME_TO_ID, (100, 100))
        assert det["num_detections"] == 1
        assert det["classes"] == [8]


class TestToXyxy:
    def test_passthrough(self):
        from libreyolo.models.vlm.parsing import to_xyxy

        assert to_xyxy([0.1, 0.2, 0.3, 0.4], "xyxy") == [0.1, 0.2, 0.3, 0.4]

    def test_xywh(self):
        from libreyolo.models.vlm.parsing import to_xyxy

        assert to_xyxy([0.25, 0.25, 0.25, 0.5], "xywh") == [0.25, 0.25, 0.5, 0.75]

    def test_cxcywh(self):
        from libreyolo.models.vlm.parsing import to_xyxy

        assert to_xyxy([0.5, 0.5, 0.25, 0.5], "cxcywh") == [0.375, 0.25, 0.625, 0.75]

    def test_unknown_format_and_bad_shape(self):
        from libreyolo.models.vlm.parsing import to_xyxy

        assert to_xyxy([0.1, 0.2, 0.3, 0.4], "weird") is None
        assert to_xyxy([0.1, 0.2, 0.3], "xyxy") is None

    def test_qwen_style_bbox_2d_on_0_1000_scale(self):
        # Qwen emits a "bbox_2d" key on a 0-1000 scale; divide by 1000 then scale.
        items = [{"label": "ship", "bbox_2d": [300, 250, 600, 750]}]
        det = build_detection_dict(
            items, NAME_TO_ID, (1000, 1000), bbox_key="bbox_2d", coord_divisor=1000.0
        )
        assert det["num_detections"] == 1
        assert det["boxes"][0] == [300.0, 250.0, 600.0, 750.0]
        assert det["classes"][0] == 8

    def test_bbox_key_mismatch_drops_item(self):
        # With bbox_key="bbox_2d", a plain "bbox" item has no usable box.
        items = [{"label": "ship", "bbox": [0.1, 0.1, 0.2, 0.2]}]
        det = build_detection_dict(
            items, NAME_TO_ID, (100, 100), bbox_key="bbox_2d", coord_divisor=1000.0
        )
        assert det["num_detections"] == 0
