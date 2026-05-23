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
from libreyolo.export.onnx import export_onnx

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


class _TinyRFDETRExport(nn.Module):
    """Small RF-DETR-shaped export module for ONNX schema tests."""

    def __init__(self, *, segmentation=False):
        super().__init__()
        self.segmentation = segmentation
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        batch = x.shape[0]
        signal = x.mean(dim=(1, 2, 3), keepdim=True) + self.anchor
        boxes = signal.reshape(batch, 1, 1).expand(batch, 3, 4)
        logits = signal.reshape(batch, 1, 1).expand(batch, 3, 2)
        if self.segmentation:
            masks = signal.expand(batch, 3, 8, 8)
            return boxes, logits, masks
        return boxes, logits


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

    def test_metadata_includes_task_contract(self):
        wrapper = _make_wrapper()
        wrapper.task = "segment"
        wrapper.SUPPORTED_TASKS = ("detect", "segment")
        wrapper.DEFAULT_TASK = "detect"

        metadata = TensorRTExporter(wrapper)._build_metadata(
            precision="fp32",
            dynamic=False,
            onnx_path=None,
        )

        assert metadata["task"] == "segment"
        assert metadata["supported_tasks"] == ["detect", "segment"]
        assert metadata["default_task"] == "detect"

    def test_rfdetr_export_metadata_is_single_task(self):
        wrapper = _make_wrapper(model_name="rfdetr")
        wrapper.task = "segment"
        wrapper.SUPPORTED_TASKS = ("detect", "segment")
        wrapper.DEFAULT_TASK = "detect"

        metadata = TensorRTExporter(wrapper)._build_metadata(
            precision="fp32",
            dynamic=False,
            onnx_path=None,
        )

        assert metadata["task"] == "segment"
        assert metadata["supported_tasks"] == ["segment"]
        assert metadata["default_task"] == "segment"

    def test_tensorrt_export_forwards_dynamic_batch_profile(
        self, monkeypatch, tmp_path
    ):
        wrapper = _make_wrapper(model_name="rfdetr")
        captured = {}

        def fake_export_tensorrt(**kwargs):
            captured.update(kwargs)
            return str(tmp_path / "model.engine")

        monkeypatch.setattr(
            "libreyolo.export.tensorrt.export_tensorrt",
            fake_export_tensorrt,
        )

        metadata = {"model_family": "rfdetr"}
        TensorRTExporter(wrapper)._export(
            wrapper.model,
            torch.zeros(1, 3, 32, 32),
            output_path=str(tmp_path / "model.engine"),
            precision="fp16",
            metadata=metadata,
            calibration_data=None,
            onnx_path=str(tmp_path / "model.onnx"),
            half=True,
            int8=False,
            dynamic=True,
            verbose=False,
            min_batch=2,
            opt_batch=4,
            max_batch=16,
        )

        assert captured["min_batch"] == 2
        assert captured["opt_batch"] == 4
        assert captured["max_batch"] == 16
        assert captured["metadata"]["trt_min_batch"] == 2
        assert captured["metadata"]["trt_opt_batch"] == 4
        assert captured["metadata"]["trt_max_batch"] == 16
        assert "trt_max_batch" not in metadata

    def test_rfdetr_export_defaults_to_cpu(self):
        wrapper = _make_wrapper(model_name="rfdetr")
        wrapper.device = torch.device("cuda")

        imgsz, device, output_path = OnnxExporter(wrapper)._resolve_params(
            output_path=None,
            imgsz=None,
            device=None,
            half=False,
            int8=False,
        )

        assert imgsz == 32
        assert device == torch.device("cpu")
        assert output_path.endswith(".onnx")

    def test_rfdetr_export_auto_device_defaults_to_cpu(self):
        wrapper = _make_wrapper(model_name="rfdetr")
        wrapper.device = torch.device("cuda")

        _imgsz, device, _output_path = OnnxExporter(wrapper)._resolve_params(
            output_path=None,
            imgsz=None,
            device="auto",
            half=False,
            int8=False,
        )

        assert device == torch.device("cpu")

    def test_rfdetr_export_auto_opset_is_17(self, monkeypatch, tmp_path):
        captured = {}
        wrapper = _make_wrapper(model_name="rfdetr")
        wrapper.model = _TinyRFDETRExport(segmentation=False)
        wrapper.task = "detect"
        wrapper.SUPPORTED_TASKS = ("detect",)
        wrapper.DEFAULT_TASK = "detect"

        def fake_export_onnx(_nn_model, _dummy, **kwargs):
            captured.update(kwargs)
            Path(kwargs["output_path"]).write_bytes(b"onnx")
            return kwargs["output_path"]

        monkeypatch.setattr("libreyolo.export.exporter.export_onnx", fake_export_onnx)
        output_path = tmp_path / "rfdetr.onnx"

        exported = OnnxExporter(wrapper)(
            output_path=str(output_path),
            simplify=False,
            dynamic=False,
            device="cpu",
        )

        assert exported == str(output_path)
        assert captured["opset"] == 17

    @pytest.mark.parametrize(
        ("segmentation", "expected_outputs"),
        [
            (False, ["dets", "labels"]),
            (True, ["dets", "labels", "masks"]),
        ],
    )
    def test_rfdetr_onnx_uses_upstream_io_names(
        self, tmp_path, segmentation, expected_outputs
    ):
        onnx = pytest.importorskip("onnx")
        output_path = tmp_path / "rfdetr.onnx"

        export_onnx(
            _TinyRFDETRExport(segmentation=segmentation),
            torch.zeros(1, 3, 32, 32),
            output_path=str(output_path),
            opset=17,
            simplify=False,
            dynamic=False,
            half=False,
            metadata={
                "model_family": "rfdetr",
                "task": "segment" if segmentation else "detect",
                "segmentation": "true" if segmentation else "false",
            },
        )

        proto = onnx.load(output_path)
        assert [i.name for i in proto.graph.input] == ["input"]
        assert [o.name for o in proto.graph.output] == expected_outputs

    def test_onnx_metadata_uses_export_imgsz_override(self, tmp_path):
        onnx = pytest.importorskip("onnx")
        wrapper = _make_wrapper(model_name="TESTYOLO", input_size=32)
        output_path = tmp_path / "custom_imgsz.onnx"

        OnnxExporter(wrapper)(
            output_path=str(output_path),
            imgsz=48,
            simplify=False,
            dynamic=False,
        )

        proto = onnx.load(output_path)
        meta = {p.key: p.value for p in proto.metadata_props}
        assert meta["imgsz"] == "48"

        from libreyolo.backends.onnx import OnnxBackend

        assert OnnxBackend._read_onnx_metadata(str(output_path), 4)[-1] == 48


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

    def test_auto_path_includes_segmentation_task(self):
        wrapper = _make_wrapper(model_name="rfdetr", size="n")
        wrapper.task = "segment"
        exporter = OnnxExporter(wrapper)
        assert exporter._auto_output_path(half=False, int8=False) == str(
            Path("weights") / "rfdetr_n_seg.onnx"
        )
        assert exporter._auto_output_path(half=True, int8=False) == str(
            Path("weights") / "rfdetr_n_seg_fp16.onnx"
        )

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

    def test_rfdetr_position_embedding_dim_buffer_not_checkpointed(self):
        from libreyolo.models.rfdetr.backbone import PositionEmbeddingSine

        module = PositionEmbeddingSine(num_pos_feats=8, normalize=True)

        assert "dim_t" not in module.state_dict()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_rfdetr_position_embedding_torchscript_loads_on_cuda(self, tmp_path):
        from libreyolo.models.rfdetr.backbone import PositionEmbeddingSine

        module = PositionEmbeddingSine(num_pos_feats=8, normalize=True)
        module.export()
        mask = torch.zeros(1, 4, 4, dtype=torch.bool)
        traced = torch.jit.trace(module, mask)
        path = tmp_path / "position_embedding.pt"
        torch.jit.save(traced, str(path))

        loaded = torch.jit.load(str(path), map_location="cuda")
        out = loaded(mask.to("cuda"))

        assert out.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_rfdetr_proposal_grid_torchscript_loads_on_cuda(self, tmp_path):
        from libreyolo.models.rfdetr.transformer import gen_encoder_output_proposals

        class ProposalModule(nn.Module):
            def forward(self, memory):
                _, proposals = gen_encoder_output_proposals(
                    memory,
                    spatial_shapes=[(2, 2)],
                    unsigmoid=False,
                )
                return proposals

        module = ProposalModule()
        memory = torch.zeros(1, 4, 8)
        traced = torch.jit.trace(module, memory)
        path = tmp_path / "proposal_grid.pt"
        torch.jit.save(traced, str(path))

        loaded = torch.jit.load(str(path), map_location="cuda")
        out = loaded(memory.to("cuda"))

        assert out.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_rfdetr_sine_embedding_torchscript_loads_on_cuda(self, tmp_path):
        from libreyolo.models.rfdetr.transformer import gen_sineembed_for_position

        class SineModule(nn.Module):
            def forward(self, pos):
                return gen_sineembed_for_position(pos, 128.0)

        module = SineModule()
        pos = torch.rand(2, 3, 4)
        traced = torch.jit.trace(module, pos)
        path = tmp_path / "sine_embedding.pt"
        torch.jit.save(traced, str(path))

        loaded = torch.jit.load(str(path), map_location="cuda")
        out = loaded(pos.to("cuda"))

        assert out.device.type == "cuda"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_rfdetr_seg_depthwise_block_torchscript_loads_on_cuda(self, tmp_path):
        from libreyolo.models.rfdetr.segmentation import DepthwiseConvBlock

        module = DepthwiseConvBlock(4)
        module.export()
        x = torch.randn(1, 4, 8, 8)
        traced = torch.jit.trace(module, x)
        path = tmp_path / "seg_depthwise_block.pt"
        torch.jit.save(traced, str(path))

        loaded = torch.jit.load(str(path), map_location="cuda")
        out = loaded(x.to("cuda"))

        assert out.device.type == "cuda"


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
            import tensorrt  # noqa: F401

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
            import tensorrt  # noqa: F401

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
        from libreyolo.export.calibration import (
            CalibrationDataLoader,
            get_calibration_dataloader,
        )

        assert CalibrationDataLoader is not None
        assert get_calibration_dataloader is not None

    def test_calibration_loader_properties(self):
        """Test calibration loader with mock data would have correct properties."""
        from libreyolo.export.calibration import CalibrationDataLoader

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
            import openvino  # noqa: F401

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
            import openvino  # noqa: F401

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

        config = TensorRTExportConfig.from_dict(
            {
                "precision": "int8",
                "workspace": 8.0,
                "hardware_compatibility": "ampere_plus",
                "dynamic": {"enabled": True, "max_batch": 16},
            }
        )

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
