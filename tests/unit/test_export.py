"""Unit tests for the unified Exporter module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from libreyolo.export.exporter import (
    BaseExporter,
    NcnnExporter,
    OnnxExporter,
    OpenVINOExporter,
    TensorRTExporter,
    TorchScriptExporter,
)

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
    """Build a mock BaseModel-like wrapper around _TinyModel."""
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
        assert "onnx" in BaseExporter._registry
        assert "torchscript" in BaseExporter._registry
        assert "tensorrt" in BaseExporter._registry
        assert "openvino" in BaseExporter._registry
        assert "ncnn" in BaseExporter._registry

    def test_suffix_present(self):
        for cls in BaseExporter._registry.values():
            assert cls.suffix.startswith(".") or cls.suffix.startswith("_")

    def test_subclass_attributes(self):
        assert OnnxExporter.suffix == ".onnx"
        assert TensorRTExporter.requires_onnx is True
        assert TorchScriptExporter.apply_model_half is True
        assert NcnnExporter.supports_int8 is False


class TestExporterValidation:
    def test_invalid_format_raises(self):
        wrapper = _make_wrapper()
        with pytest.raises(ValueError, match="Unsupported export format"):
            BaseExporter.create("badformat", wrapper)

    def test_invalid_format_case_insensitive(self):
        wrapper = _make_wrapper()
        # Should NOT raise — format names are lowered via create()
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = BaseExporter.create("TORCHSCRIPT", wrapper)
            path = exporter(
                output_path=str(Path(tmpdir) / "model.torchscript"),
            )
            assert Path(path).exists()


class TestOutputPathGeneration:
    def test_auto_path_torchscript(self):
        wrapper = _make_wrapper(model_name="yolo9", size="t")
        exporter = TorchScriptExporter(wrapper)
        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            orig = os.getcwd()
            try:
                os.chdir(tmpdir)
                path = exporter()
                assert path == str(Path("weights") / "yolo9_t.torchscript")
                assert Path(path).exists()
            finally:
                os.chdir(orig)

    def test_explicit_path(self):
        wrapper = _make_wrapper()
        exporter = TorchScriptExporter(wrapper)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "custom.torchscript")
            path = exporter(output_path=out)
            assert path == out
            assert Path(out).exists()


class TestTorchScriptExport:
    def test_basic_torchscript(self):
        wrapper = _make_wrapper()
        exporter = TorchScriptExporter(wrapper)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = str(Path(tmpdir) / "model.torchscript")
            path = exporter(output_path=out)
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

        exporter = TorchScriptExporter(wrapper)
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter(output_path=str(Path(tmpdir) / "test.torchscript"))

        current_device = next(wrapper.model.parameters()).device
        assert current_device == original_device

    def test_half_restored_to_float32(self):
        wrapper = _make_wrapper()
        exporter = TorchScriptExporter(wrapper)
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter(
                output_path=str(Path(tmpdir) / "test.torchscript"),
                half=True,
            )

        param = next(wrapper.model.parameters())
        assert param.dtype == torch.float32


# ---------------------------------------------------------------------------
# TensorRT Export Tests
# ---------------------------------------------------------------------------


class TestTensorRTFormat:
    """Test TensorRT format registration and validation."""

    def test_tensorrt_format_registered(self):
        """Verify TensorRT is in registry."""
        assert "tensorrt" in BaseExporter._registry

    def test_tensorrt_format_config(self):
        """Verify TensorRT format configuration."""
        assert TensorRTExporter.suffix == ".engine"
        assert TensorRTExporter.requires_onnx is True


class TestTensorRTValidation:
    """Test TensorRT export parameter validation."""

    def test_int8_requires_data(self):
        """INT8 export without data should raise ValueError."""
        wrapper = _make_wrapper()
        exporter = TensorRTExporter(wrapper)

        with pytest.raises(ValueError, match="calibration data"):
            exporter(int8=True)

    def test_int8_with_data_no_immediate_error(self):
        """INT8 with data parameter should not raise validation error.

        Note: Will fail later due to missing TensorRT (or ONNX), but validation should pass.
        """
        try:
            import tensorrt
            pytest.skip("TensorRT is installed, skipping missing TensorRT test")
        except ImportError:
            pass

        wrapper = _make_wrapper()
        exporter = TensorRTExporter(wrapper)

        # Should fail with ImportError (missing onnx or tensorrt), not ValueError
        with pytest.raises(ImportError):
            exporter(int8=True, data="coco8.yaml")


class TestTensorRTImportCheck:
    """Test TensorRT availability checking."""

    def test_check_tensorrt_raises_helpful_error(self):
        """Verify helpful error message when TensorRT not installed."""
        # Skip if TensorRT is actually installed
        try:
            import tensorrt
            pytest.skip("TensorRT is installed, skipping missing TensorRT test")
        except ImportError:
            pass

        from libreyolo.export.tensorrt import check_tensorrt_available

        with pytest.raises(ImportError) as exc_info:
            check_tensorrt_available()

        error_msg = str(exc_info.value)
        assert "tensorrt" in error_msg.lower()
        assert "pip install" in error_msg


class TestCalibrationDataLoader:
    """Test calibration data loader for INT8 quantization."""

    def test_calibration_loader_import(self):
        """Verify calibration module can be imported."""
        from libreyolo.export.calibration import CalibrationDataLoader, get_calibration_dataloader
        assert CalibrationDataLoader is not None
        assert get_calibration_dataloader is not None

    def test_calibration_loader_properties(self):
        """Test calibration loader with mock data would have correct properties."""
        from libreyolo.export.calibration import CalibrationDataLoader
        import numpy as np

        # Check that dtype and shape properties are defined
        assert hasattr(CalibrationDataLoader, "shape")
        assert hasattr(CalibrationDataLoader, "dtype")


# ---------------------------------------------------------------------------
# OpenVINO Export Tests
# ---------------------------------------------------------------------------


class TestOpenVINOFormat:
    """Test OpenVINO format registration and validation."""

    def test_openvino_format_registered(self):
        """Verify OpenVINO is in registry."""
        assert "openvino" in BaseExporter._registry

    def test_openvino_format_config(self):
        """Verify OpenVINO format configuration."""
        assert OpenVINOExporter.suffix == "_openvino"
        assert OpenVINOExporter.requires_onnx is True


class TestOpenVINOValidation:
    """Test OpenVINO export parameter validation."""

    def test_int8_requires_data(self):
        """INT8 export without data should raise ValueError."""
        wrapper = _make_wrapper()
        exporter = OpenVINOExporter(wrapper)

        with pytest.raises(ValueError, match="calibration data"):
            exporter(int8=True)

    def test_int8_with_data_no_immediate_error(self):
        """INT8 with data parameter should not raise validation error.

        Note: Will fail later due to missing OpenVINO (or ONNX), but validation should pass.
        """
        try:
            import openvino
            pytest.skip("OpenVINO is installed, skipping missing OpenVINO test")
        except ImportError:
            pass

        wrapper = _make_wrapper()
        exporter = OpenVINOExporter(wrapper)

        # Should fail with ImportError (missing onnx or openvino), not ValueError
        with pytest.raises(ImportError):
            exporter(int8=True, data="coco8.yaml")


class TestOpenVINOImportCheck:
    """Test OpenVINO availability checking."""

    def test_check_openvino_raises_helpful_error(self):
        """Verify helpful error message when OpenVINO not installed."""
        try:
            import openvino
            pytest.skip("OpenVINO is installed, skipping missing OpenVINO test")
        except ImportError:
            pass

        from libreyolo.export.openvino import check_openvino_available

        with pytest.raises(ImportError) as exc_info:
            check_openvino_available()

        error_msg = str(exc_info.value)
        assert "openvino" in error_msg.lower()
        assert "pip install" in error_msg


class TestExportPrecisionSuffix:
    """Test output filename generation with precision suffixes."""

    def test_fp16_suffix_in_auto_path(self):
        """FP16 export should include _fp16 in auto-generated filename."""
        wrapper = _make_wrapper(model_name="TESTYOLO", size="s")
        exporter = TorchScriptExporter(wrapper)

        with tempfile.TemporaryDirectory() as tmpdir:
            import os
            orig = os.getcwd()
            try:
                os.chdir(tmpdir)
                path = exporter(half=True)
                assert "_fp16" in path, f"Expected _fp16 in path, got: {path}"
                assert path == str(Path("weights") / "testyolo_s_fp16.torchscript")
            finally:
                os.chdir(orig)

    def test_half_and_int8_uses_int8(self):
        """When both half and int8 are True, int8 takes precedence."""
        import warnings

        wrapper = _make_wrapper()
        exporter = TensorRTExporter(wrapper)

        # Should warn about using INT8 when both specified
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                exporter(half=True, int8=True, data="coco8.yaml")
            except ImportError:
                # Expected - TensorRT not installed
                pass
            except Exception:
                # May fail for other reasons if TensorRT is installed but
                # calibration data can't be loaded, etc. That's OK for this test.
                pass

            # Check that a warning was issued about INT8 precedence
            warning_msgs = [str(warning.message) for warning in w]
            assert any("INT8" in msg for msg in warning_msgs)


# ---------------------------------------------------------------------------
# TensorRT Export Config Tests
# ---------------------------------------------------------------------------


class TestTensorRTExportConfig:
    """Test TensorRT export configuration system."""

    def test_default_config(self):
        """Test default configuration values."""
        from libreyolo.export.config import TensorRTExportConfig

        config = TensorRTExportConfig()
        assert config.precision == "fp16"
        assert config.workspace == 4.0
        assert config.verbose is False
        assert config.hardware_compatibility == "none"
        assert config.device == 0
        assert config.dynamic.enabled is False
        assert config.int8_calibration.fraction == 0.1

    def test_config_half_property(self):
        """Test half property for different precisions."""
        from libreyolo.export.config import TensorRTExportConfig

        fp32_config = TensorRTExportConfig(precision="fp32")
        fp16_config = TensorRTExportConfig(precision="fp16")
        int8_config = TensorRTExportConfig(precision="int8")

        assert fp32_config.half is False
        assert fp16_config.half is True
        assert int8_config.half is True  # INT8 includes FP16 fallback

    def test_config_int8_property(self):
        """Test int8 property for different precisions."""
        from libreyolo.export.config import TensorRTExportConfig

        fp32_config = TensorRTExportConfig(precision="fp32")
        fp16_config = TensorRTExportConfig(precision="fp16")
        int8_config = TensorRTExportConfig(precision="int8")

        assert fp32_config.int8 is False
        assert fp16_config.int8 is False
        assert int8_config.int8 is True

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        from libreyolo.export.config import TensorRTExportConfig

        config = TensorRTExportConfig.from_dict({
            "precision": "int8",
            "workspace": 8.0,
            "hardware_compatibility": "ampere_plus",
            "dynamic": {"enabled": True, "max_batch": 16},
        })

        assert config.precision == "int8"
        assert config.workspace == 8.0
        assert config.hardware_compatibility == "ampere_plus"
        assert config.dynamic.enabled is True
        assert config.dynamic.max_batch == 16

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        from libreyolo.export.config import TensorRTExportConfig

        config = TensorRTExportConfig(precision="fp32", workspace=2.0)
        data = config.to_dict()

        assert data["precision"] == "fp32"
        assert data["workspace"] == 2.0
        assert "dynamic" in data
        assert "int8_calibration" in data

    def test_config_validation_invalid_precision(self):
        """Test validation rejects invalid precision."""
        from libreyolo.export.config import TensorRTExportConfig

        with pytest.raises(ValueError, match="Invalid precision"):
            TensorRTExportConfig(precision="fp8")

    def test_config_validation_invalid_workspace(self):
        """Test validation rejects invalid workspace."""
        from libreyolo.export.config import TensorRTExportConfig

        with pytest.raises(ValueError, match="workspace must be positive"):
            TensorRTExportConfig(workspace=-1.0)

    def test_config_validation_invalid_hardware_compat(self):
        """Test validation rejects invalid hardware compatibility."""
        from libreyolo.export.config import TensorRTExportConfig

        with pytest.raises(ValueError, match="Invalid hardware_compatibility"):
            TensorRTExportConfig(hardware_compatibility="invalid")

    def test_load_export_config_none(self):
        """Test load_export_config with None returns default."""
        from libreyolo.export.config import load_export_config, TensorRTExportConfig

        config = load_export_config(None)
        assert isinstance(config, TensorRTExportConfig)
        assert config.precision == "fp16"

    def test_load_export_config_dict(self):
        """Test load_export_config with dict."""
        from libreyolo.export.config import load_export_config

        config = load_export_config({"precision": "fp32"})
        assert config.precision == "fp32"

    def test_load_export_config_passthrough(self):
        """Test load_export_config passes through existing config."""
        from libreyolo.export.config import load_export_config, TensorRTExportConfig

        original = TensorRTExportConfig(precision="int8")
        config = load_export_config(original)
        assert config is original

    def test_load_export_config_yaml(self):
        """Test load_export_config from YAML file."""
        from libreyolo.export.config import load_export_config

        config = load_export_config("tensorrt_default.yaml")
        assert config.precision == "fp16"
        assert config.workspace == 4.0
