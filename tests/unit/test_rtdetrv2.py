"""Unit tests for LibreRTDETRv2.detect_size_from_filename."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


class TestRTDETRv2SizeDetectionFromFilename:
    """Test LibreRTDETRv2.detect_size_from_filename."""

    @pytest.mark.parametrize("size", ["r18", "r34", "r50", "r50m", "r101"])
    def test_all_sizes_via_libre_prefix(self, size):
        from libreyolo.models.rtdetrv2.model import LibreRTDETRv2

        assert LibreRTDETRv2.detect_size_from_filename(f"LibreRTDETRv2-{size}.pt") == size

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("rtdetrv2_r18vd_120e_coco_rerun.pth", "r18"),
            ("rtdetrv2_r34vd_120e_coco_rerun.pth", "r34"),
            ("rtdetrv2_r50vd_6x_coco_ema.pth", "r50"),
            ("rtdetrv2_r50vd_m_7x_coco_ema.pth", "r50m"),
            ("rtdetrv2_r101vd_6x_coco_ema.pth", "r101"),
        ],
    )
    def test_upstream_v2_vd_checkpoints(self, filename, expected):
        from libreyolo.models.rtdetrv2.model import LibreRTDETRv2

        assert LibreRTDETRv2.detect_size_from_filename(filename) == expected

    def test_r50vd_m_returns_r50m_not_r50(self):
        # Regression: super()'s substring fallback would match "_r50" inside
        # "rtdetrv2_r50vd_m_..." and return "r50" before the v2 regex ran.
        from libreyolo.models.rtdetrv2.model import LibreRTDETRv2

        assert LibreRTDETRv2.detect_size_from_filename("rtdetrv2_r50vd_m_7x_coco_ema.pth") == "r50m"

    def test_unknown_filename_returns_none(self):
        from libreyolo.models.rtdetrv2.model import LibreRTDETRv2

        assert LibreRTDETRv2.detect_size_from_filename("yolov8n.pt") is None
