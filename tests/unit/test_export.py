"""Unit tests for the unified Exporter module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

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


class TestExporterFormats:
    def test_expected_keys(self):
        assert "onnx" in Exporter.FORMATS
        assert "torchscript" in Exporter.FORMATS

    def test_suffix_present(self):
        for fmt_info in Exporter.FORMATS.values():
            assert "suffix" in fmt_info
            assert fmt_info["suffix"].startswith(".")

    def test_method_present(self):
        for fmt_info in Exporter.FORMATS.values():
            assert "method" in fmt_info
            assert hasattr(Exporter, fmt_info["method"])


class TestExporterValidation:
    def test_invalid_format_raises(self):
        wrapper = _make_wrapper()
        exporter = Exporter(wrapper)
        with pytest.raises(ValueError, match="Unsupported export format"):
            exporter("badformat")

    def test_invalid_format_case_insensitive(self):
        wrapper = _make_wrapper()
        exporter = Exporter(wrapper)
        # Should NOT raise — format names are lowered
        with tempfile.TemporaryDirectory() as tmpdir:
            path = exporter(
                "ONNX",
                output_path=str(Path(tmpdir) / "model.onnx"),
                simplify=False,
            )
            assert Path(path).exists()


class TestOutputPathGeneration:
    def test_auto_path_onnx(self):
        wrapper = _make_wrapper(model_name="LIBREYOLOX", size="m")
        exporter = Exporter(wrapper)
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            orig = os.getcwd()
            try:
                os.chdir(tmpdir)
                path = exporter("onnx", simplify=False)
                assert path == "libreyolox_m.onnx"
                assert Path(path).exists()
            finally:
                os.chdir(orig)

    def test_auto_path_torchscript(self):
        wrapper = _make_wrapper(model_name="LIBREYOLO9", size="t")
        exporter = Exporter(wrapper)
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            orig = os.getcwd()
            try:
                os.chdir(tmpdir)
                path = exporter("torchscript")
                assert path == "libreyolo9_t.torchscript"
                assert Path(path).exists()
            finally:
                os.chdir(orig)

    def test_explicit_path(self):
        wrapper = _make_wrapper()
        exporter = Exporter(wrapper)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "custom.onnx")
            path = exporter("onnx", output_path=out, simplify=False)
            assert path == out
            assert Path(out).exists()


class TestOnnxMetadata:
    def test_metadata_written(self):
        import onnx

        wrapper = _make_wrapper(nb_classes=4, model_name="LIBREYOLOX", size="s")
        exporter = Exporter(wrapper)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "meta.onnx")
            exporter("onnx", output_path=out, simplify=False)

            model_proto = onnx.load(out)
            meta = {p.key: p.value for p in model_proto.metadata_props}

            assert meta["model_family"] == "LIBREYOLOX"
            assert meta["model_size"] == "s"
            assert meta["nb_classes"] == "4"
            assert meta["dynamic"] == "True"
            assert meta["half"] == "False"

            names = json.loads(meta["names"])
            assert names["0"] == "class_0"
            assert len(names) == 4


class TestOnnxMetadataReading:
    def test_round_trip(self):
        """Export with metadata, then load via LIBREYOLOOnnx and verify auto-read."""
        import onnx

        wrapper = _make_wrapper(nb_classes=3, model_name="TESTYOLO", size="s")
        wrapper.names = {0: "cat", 1: "dog", 2: "bird"}
        exporter = Exporter(wrapper)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "rt.onnx")
            exporter("onnx", output_path=out, simplify=False)

            from libreyolo.common.onnx import LIBREYOLOOnnx

            onnx_model = LIBREYOLOOnnx(out, nb_classes=80)  # deliberately wrong
            # After metadata reading, names should be overwritten
            assert onnx_model.names[0] == "cat"
            assert onnx_model.names[1] == "dog"
            assert onnx_model.names[2] == "bird"
            assert onnx_model.nb_classes == 3


class TestDynamicAxes:
    def test_dynamic_true(self):
        import onnx

        wrapper = _make_wrapper()
        exporter = Exporter(wrapper)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "dyn.onnx")
            exporter("onnx", output_path=out, dynamic=True, simplify=False)

            model_proto = onnx.load(out)
            graph = model_proto.graph

            # Input batch dim should be symbolic (dim_param is a non-empty string)
            input_shape = graph.input[0].type.tensor_type.shape
            dim0 = input_shape.dim[0]
            assert dim0.dim_param != "", "Batch dim should be dynamic (symbolic)"
            assert dim0.dim_value == 0, "Dynamic dim should not have a fixed value"

    def test_dynamic_false(self):
        import onnx

        wrapper = _make_wrapper()
        exporter = Exporter(wrapper)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "static.onnx")
            exporter("onnx", output_path=out, dynamic=False, simplify=False)

            model_proto = onnx.load(out)
            graph = model_proto.graph

            # Input batch dim should be a fixed value (1)
            input_shape = graph.input[0].type.tensor_type.shape
            dim0 = input_shape.dim[0]
            assert dim0.dim_value == 1


class TestHalfExport:
    def test_half_produces_float16_input(self):
        import onnx
        from onnx import TensorProto

        wrapper = _make_wrapper()
        exporter = Exporter(wrapper)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "fp16.onnx")
            exporter("onnx", output_path=out, half=True, simplify=False)

            model_proto = onnx.load(out)
            input_type = model_proto.graph.input[0].type.tensor_type.elem_type
            assert input_type == TensorProto.FLOAT16


class TestSimplify:
    def test_simplify_runs_without_error(self):
        wrapper = _make_wrapper()
        exporter = Exporter(wrapper)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "simp.onnx")
            # Should not raise even if onnxsim is missing (warns instead)
            path = exporter("onnx", output_path=out, simplify=True)
            assert Path(path).exists()


class TestTorchScriptExport:
    def test_basic_torchscript(self):
        wrapper = _make_wrapper()
        exporter = Exporter(wrapper)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "model.torchscript")
            path = exporter("torchscript", output_path=out)
            assert Path(path).exists()

            # Verify the file is loadable
            loaded = torch.jit.load(out)
            dummy = torch.randn(1, 3, 32, 32)
            result = loaded(dummy)
            assert result.shape == (1, 4)


class TestModelStateRestored:
    def test_model_stays_on_original_device(self):
        wrapper = _make_wrapper()
        original_device = next(wrapper.model.parameters()).device

        exporter = Exporter(wrapper)
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter(
                "onnx",
                output_path=str(Path(tmpdir) / "test.onnx"),
                simplify=False,
            )

        current_device = next(wrapper.model.parameters()).device
        assert current_device == original_device

    def test_half_restored_to_float32(self):
        wrapper = _make_wrapper()
        exporter = Exporter(wrapper)
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter(
                "onnx",
                output_path=str(Path(tmpdir) / "test.onnx"),
                half=True,
                simplify=False,
            )

        param = next(wrapper.model.parameters())
        assert param.dtype == torch.float32
