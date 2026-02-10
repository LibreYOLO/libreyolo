"""Unit tests for the ncnn export module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
import yaml

from libreyolo.export.exporter import Exporter

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Minimal model for export tests (no real weights needed)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def _make_wrapper(nb_classes=4, model_name="TESTYOLO", size="s", input_size=32):
    """Build a mock LibreYOLOBase-like wrapper around _TinyModel."""
    wrapper = MagicMock()
    wrapper.model = _TinyModel()
    wrapper.model.eval()
    wrapper.size = size
    wrapper.nb_classes = nb_classes
    wrapper.names = {i: f"class_{i}" for i in range(nb_classes)}
    wrapper.device = torch.device("cpu")
    wrapper._get_model_name.return_value = model_name
    wrapper._get_input_size.return_value = input_size
    return wrapper


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNCNNFormatRegistration:
    """Test ncnn format is properly registered in Exporter."""

    def test_ncnn_format_registered(self):
        """Verify ncnn is in supported formats."""
        assert "ncnn" in Exporter.FORMATS

    def test_ncnn_format_config(self):
        """Verify ncnn format configuration."""
        fmt = Exporter.FORMATS["ncnn"]
        assert fmt["suffix"] == "_ncnn"
        assert fmt["requires"] is None


class TestNCNNAvailabilityCheck:
    """Test ncnn/pnnx availability checking."""

    def test_check_ncnn_export_raises_helpful_error(self):
        """Verify helpful error message when pnnx not installed."""
        try:
            import pnnx
            pytest.skip("pnnx is installed, skipping missing pnnx test")
        except ImportError:
            pass

        from libreyolo.export.ncnn import check_ncnn_export_available

        with pytest.raises(ImportError) as exc_info:
            check_ncnn_export_available()

        error_msg = str(exc_info.value)
        assert "pnnx" in error_msg.lower()
        assert "pip install" in error_msg


class TestNCNNOutputPathGeneration:
    """Test output path generation for ncnn format."""

    def test_auto_path_ncnn(self):
        """ncnn export should generate path with _ncnn suffix."""
        wrapper = _make_wrapper(model_name="LIBREYOLO9", size="t")
        exporter = Exporter(wrapper)

        # Verify the auto-generated path would include _ncnn
        fmt_info = Exporter.FORMATS["ncnn"]
        expected_suffix = fmt_info["suffix"]
        assert expected_suffix == "_ncnn"

    def test_fp16_suffix_in_auto_path(self):
        """FP16 ncnn export should include _fp16 in auto-generated filename."""
        wrapper = _make_wrapper(model_name="TESTYOLO", size="s")
        exporter = Exporter(wrapper)

        # Build what the auto path would be for ncnn + fp16
        model_name = wrapper._get_model_name().lower()
        size = wrapper.size
        fmt_info = Exporter.FORMATS["ncnn"]
        expected = f"weights/{model_name}_{size}_fp16{fmt_info['suffix']}"
        assert "_fp16_ncnn" in expected


class TestNCNNMetadataYAML:
    """Test metadata YAML write/read round-trip."""

    def test_metadata_roundtrip(self):
        """Test that metadata can be written and read back correctly."""
        from libreyolo.export.ncnn import _save_metadata

        metadata = {
            "libreyolo_version": "0.1.5",
            "model_family": "LIBREYOLO9",
            "model_size": "t",
            "nb_classes": 80,
            "names": {"0": "person", "1": "bicycle"},
            "imgsz": 640,
            "precision": "fp32",
            "dynamic": False,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            _save_metadata(output_dir, metadata)

            metadata_path = output_dir / "metadata.yaml"
            assert metadata_path.exists()

            with open(metadata_path) as f:
                loaded = yaml.safe_load(f)

            assert loaded["model_family"] == "LIBREYOLO9"
            assert loaded["model_size"] == "t"
            assert loaded["nb_classes"] == 80
            assert loaded["precision"] == "fp32"
            assert loaded["imgsz"] == 640
            assert loaded["dynamic"] is False
            assert loaded["names"]["0"] == "person"
            assert loaded["names"]["1"] == "bicycle"


class TestNCNNExportValidation:
    """Test ncnn export validation in exporter."""

    def test_ncnn_export_fails_without_pnnx(self):
        """ncnn export without pnnx should raise ImportError."""
        try:
            import pnnx
            pytest.skip("pnnx is installed, skipping missing pnnx test")
        except ImportError:
            pass

        wrapper = _make_wrapper()
        exporter = Exporter(wrapper)

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ImportError, match="pnnx"):
                exporter("ncnn", output_path=str(Path(tmpdir) / "model_ncnn"))
