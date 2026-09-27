"""Offline tests for LibreGround click-text parsing."""

import pytest

from libreyolo.models.ground.parsing import (
    build_point_dict,
    coerce_queries,
    extract_clicks,
    scale_point,
)

pytestmark = pytest.mark.unit


class TestExtractClicks:
    def test_click_function(self):
        assert extract_clicks("Click(352, 348)") == [{"point": [352.0, 348.0]}]

    def test_click_kwargs(self):
        assert extract_clicks("pyautogui.click(x=120, y=80)") == [
            {"point": [120.0, 80.0]}
        ]

    def test_click_point_tag(self):
        text = "Action: click(point='<point>450 320</point>')"
        assert extract_clicks(text) == [{"point": [450.0, 320.0]}]

    def test_bare_point_tag(self):
        assert extract_clicks("<point>0.5, 0.25</point>") == [{"point": [0.5, 0.25]}]

    def test_showui_pair(self):
        assert extract_clicks("[0.73, 0.21]") == [{"point": [0.73, 0.21]}]

    def test_json_point(self):
        text = '{"point": [120, 80], "label": "Save"}'
        assert extract_clicks(text) == [{"label": "Save", "point": [120.0, 80.0]}]

    def test_json_xy(self):
        assert extract_clicks('{"x": 10, "y": 20}') == [
            {"label": None, "point": [10.0, 20.0]}
        ]

    def test_box_becomes_center(self):
        items = extract_clicks('{"bbox": [0, 0, 10, 10], "label": "icon"}')
        assert items == [{"label": "icon", "point": [5.0, 5.0]}]

    def test_florence_loc_point(self):
        assert extract_clicks("<loc_250><loc_500>") == [{"point": [250.0, 500.0]}]

    def test_florence_loc_box_center(self):
        assert extract_clicks("<loc_0><loc_0><loc_100><loc_200>") == [
            {"point": [50.0, 100.0]}
        ]

    def test_empty_and_noise(self):
        assert extract_clicks("") == []
        assert extract_clicks(None) == []
        assert extract_clicks("I cannot see that control.") == []

    def test_scale_aside_does_not_win_over_named_click(self):
        text = "Coordinates are [0, 1]. Click(12, 34)"
        assert extract_clicks(text) == [{"point": [12.0, 34.0]}]


class TestScalePoint:
    def test_unit(self):
        assert scale_point(0.5, 0.25, (200, 100), "unit") == (100.0, 25.0)

    def test_milli(self):
        assert scale_point(500, 250, (200, 100), "milli") == (100.0, 25.0)

    def test_pixel_is_not_clamped(self):
        assert scale_point(-5, 999, (200, 100), "pixel") == (-5.0, 999.0)

    def test_pixel_view(self):
        assert scale_point(50, 25, (200, 100), "pixel_view", (100, 50)) == (100.0, 50.0)

    def test_unknown_space_raises(self):
        with pytest.raises(ValueError, match="coord space"):
            scale_point(0, 0, (10, 10), "nope")


class TestBuildPointDict:
    def test_singleton_label_fill(self):
        out = build_point_dict(
            [{"point": [0.5, 0.5]}],
            {"bluetooth": 0},
            (100, 50),
            coord_space="unit",
        )
        assert out["num_detections"] == 1
        assert out["points"][0][:2] == [50.0, 25.0]
        assert out["points"][0][2] == 0.0

    def test_conf_filter(self):
        out = build_point_dict(
            [{"point": [0.5, 0.5]}],
            {"bluetooth": 0},
            (100, 50),
            conf_thres=1.1,
            default_score=1.0,
        )
        assert out["num_detections"] == 0

    def test_far_out_of_range_is_dropped(self):
        out = build_point_dict(
            [{"point": [-5, 999]}],
            {"a": 0},
            (200, 100),
            coord_space="pixel",
        )
        assert out["num_detections"] == 0

    def test_tiny_overshoot_is_snapped(self):
        out = build_point_dict(
            [{"point": [199.2, -0.2]}],
            {"a": 0},
            (200, 100),
            coord_space="pixel",
        )
        assert out["num_detections"] == 1
        assert out["points"][0][:2] == [199.0, 0.0]

    @pytest.mark.parametrize(
        ("point", "coord_space"),
        [
            ([1.0, 1.0], "unit"),
            ([0.999, 0.999], "unit"),
            ([1000, 1000], "milli"),
        ],
    )
    def test_normalized_right_bottom_edge_is_kept(self, point, coord_space):
        out = build_point_dict(
            [{"point": point}],
            {"a": 0},
            (200, 100),
            coord_space=coord_space,
        )
        assert out["num_detections"] == 1
        assert out["points"][0][:2] == [199.0, 99.0]

    @pytest.mark.parametrize(
        ("point", "coord_space"),
        [
            ([1.01, 0.5], "unit"),
            ([1010, 500], "milli"),
        ],
    )
    def test_normalized_far_overshoot_is_dropped(self, point, coord_space):
        out = build_point_dict(
            [{"point": point}],
            {"a": 0},
            (200, 100),
            coord_space=coord_space,
        )
        assert out["num_detections"] == 0


class TestCoerceQueries:
    def test_string_and_list(self):
        assert coerce_queries("  Bluetooth ") == ["Bluetooth"]
        assert coerce_queries(["Wi-Fi", "Bluetooth"]) == ["Wi-Fi", "Bluetooth"]

    def test_rejects_empty_and_dupes(self):
        with pytest.raises(ValueError):
            coerce_queries("  ")
        with pytest.raises(ValueError):
            coerce_queries(["A", "a"])
        with pytest.raises(TypeError):
            coerce_queries(12)
