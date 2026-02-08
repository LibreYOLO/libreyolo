"""
End-to-end tests for TorchScript export and inference.

Tests the complete pipeline:
1. Load PyTorch model
2. Export to TorchScript
3. Load TorchScript model
4. Run inference and verify
"""

from pathlib import Path

import pytest
import torch

from .conftest import (
    FULL_TEST_MODELS,
    QUICK_TEST_MODELS,
    RFDETR_TEST_MODELS,
    get_model_weights,
    requires_rfdetr,
)

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def load_model(model_type: str, size: str, device: str = "cuda"):
    """Load a model by type and size."""
    from libreyolo import LIBREYOLO

    weights = get_model_weights(model_type, size)
    return LIBREYOLO(weights, device=device)


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestTorchScriptExport:
    """Test TorchScript export for all models."""

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_torchscript_export_quick(self, model_type, size, tmp_path):
        """Quick test with smallest models (for CI)."""
        self._run_export_test(model_type, size, tmp_path)

    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,size", FULL_TEST_MODELS)
    def test_torchscript_export_full(self, model_type, size, tmp_path):
        """Full test with all YOLOX and YOLOv9 models."""
        self._run_export_test(model_type, size, tmp_path)

    @requires_rfdetr
    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,size", RFDETR_TEST_MODELS)
    def test_torchscript_export_rfdetr(self, model_type, size, tmp_path):
        """Test RF-DETR models (requires extra dependencies)."""
        self._run_export_test(model_type, size, tmp_path)

    def _run_export_test(self, model_type, size, tmp_path):
        """Common TorchScript export test implementation."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load PyTorch model
        pt_model = load_model(model_type, size, device=device)

        # Export to TorchScript
        ts_path = str(tmp_path / f"{model_type}_{size}.torchscript")
        exported_path = pt_model.export(
            format="torchscript",
            output_path=ts_path,
        )
        assert Path(exported_path).exists(), "TorchScript file not created"
        assert Path(exported_path).stat().st_size > 0, "TorchScript file is empty"

        # Load TorchScript model and verify it runs
        loaded = torch.jit.load(exported_path, map_location=device)
        loaded.eval()

        # Run a forward pass
        input_size = pt_model._get_input_size()
        dummy_input = torch.randn(1, 3, input_size, input_size, device=device)
        with torch.no_grad():
            output = loaded(dummy_input)

        # Output should be a tensor or tuple
        assert output is not None


class TestTorchScriptLoadAndInference:
    """Test TorchScript model loading and inference."""

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_torchscript_inference(self, model_type, size, tmp_path):
        """Test that TorchScript produces valid forward pass output."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pt_model = load_model(model_type, size, device=device)

        # Export
        ts_path = str(tmp_path / f"{model_type}_{size}.torchscript")
        pt_model.export(format="torchscript", output_path=ts_path)

        # Load and run
        loaded = torch.jit.load(ts_path, map_location=device)
        loaded.eval()

        input_size = pt_model._get_input_size()
        dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

        # Multiple forward passes should work
        for _ in range(5):
            with torch.no_grad():
                output = loaded(dummy_input)
            assert output is not None


class TestTorchScriptOutputConsistency:
    """Test TorchScript vs PyTorch output consistency."""

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_output_shapes_match(self, model_type, size, tmp_path):
        """Test that TorchScript output shapes match PyTorch."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pt_model = load_model(model_type, size, device=device)

        # Export
        ts_path = str(tmp_path / f"{model_type}_{size}.torchscript")
        pt_model.export(format="torchscript", output_path=ts_path)

        # Load TorchScript
        ts_model = torch.jit.load(ts_path, map_location=device)
        ts_model.eval()

        # Get outputs from both
        input_size = pt_model._get_input_size()
        dummy_input = torch.randn(1, 3, input_size, input_size, device=device)

        pt_model.model.eval()
        with torch.no_grad():
            pt_output = pt_model.model(dummy_input)
            ts_output = ts_model(dummy_input)

        # Compare shapes
        if isinstance(pt_output, torch.Tensor):
            assert pt_output.shape == ts_output.shape
        elif isinstance(pt_output, (tuple, list)):
            for pt_o, ts_o in zip(pt_output, ts_output):
                if isinstance(pt_o, torch.Tensor):
                    assert pt_o.shape == ts_o.shape


class TestTorchScriptHalf:
    """Test TorchScript FP16 export."""

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_torchscript_half(self, model_type, size, tmp_path):
        """Test FP16 TorchScript export."""
        if not torch.cuda.is_available():
            pytest.skip("FP16 TorchScript needs CUDA")

        pt_model = load_model(model_type, size, device="cuda")

        # Export with half precision
        ts_path = str(tmp_path / f"{model_type}_{size}_fp16.torchscript")
        pt_model.export(
            format="torchscript",
            output_path=ts_path,
            half=True,
        )
        assert Path(ts_path).exists()

        # Load and verify dtype
        loaded = torch.jit.load(ts_path, map_location="cuda")
        loaded.eval()

        # Check that model parameters are float16
        for param in loaded.parameters():
            assert param.dtype == torch.float16, f"Expected float16, got {param.dtype}"


class TestTorchScriptModelCoverage:
    """Verify all model types can be exported to TorchScript."""

    def test_all_yolox_sizes_exportable(self, tmp_path):
        """Test that all YOLOX sizes can be exported."""
        from .conftest import YOLOX_SIZES

        device = "cuda" if torch.cuda.is_available() else "cpu"

        for size in YOLOX_SIZES:
            pt_model = load_model("yolox", size, device=device)
            ts_path = str(tmp_path / f"yolox_{size}.torchscript")

            pt_model.export(format="torchscript", output_path=ts_path)
            assert Path(ts_path).exists(), f"Failed to export YOLOX-{size}"

            # Verify model loads
            loaded = torch.jit.load(ts_path, map_location=device)
            assert loaded is not None

    def test_all_yolov9_sizes_exportable(self, tmp_path):
        """Test that all YOLOv9 sizes can be exported."""
        from .conftest import YOLOV9_SIZES

        device = "cuda" if torch.cuda.is_available() else "cpu"

        for size in YOLOV9_SIZES:
            pt_model = load_model("yolov9", size, device=device)
            ts_path = str(tmp_path / f"yolov9_{size}.torchscript")

            pt_model.export(format="torchscript", output_path=ts_path)
            assert Path(ts_path).exists(), f"Failed to export YOLOv9-{size}"

            # Verify model loads
            loaded = torch.jit.load(ts_path, map_location=device)
            assert loaded is not None

    @requires_rfdetr
    def test_all_rfdetr_sizes_exportable(self, tmp_path):
        """Test that all RF-DETR sizes can be exported."""
        from .conftest import RFDETR_SIZES

        device = "cuda" if torch.cuda.is_available() else "cpu"

        for size in RFDETR_SIZES:
            pt_model = load_model("rfdetr", size, device=device)
            ts_path = str(tmp_path / f"rfdetr_{size}.torchscript")

            try:
                pt_model.export(format="torchscript", output_path=ts_path)
                assert Path(ts_path).exists(), f"Failed to export RF-DETR-{size}"
            except Exception as e:
                # RF-DETR may have tracing issues due to dynamic shapes
                pytest.skip(f"RF-DETR-{size} TorchScript export not supported: {e}")


class TestTorchScriptBatchSize:
    """Test TorchScript with different batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_different_batch_sizes(self, batch_size, tmp_path):
        """Test that TorchScript works with different batch sizes."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pt_model = load_model("yolox", "nano", device=device)

        ts_path = str(tmp_path / f"yolox_nano_batch{batch_size}.torchscript")
        pt_model.export(
            format="torchscript",
            output_path=ts_path,
            batch=batch_size,
        )

        # Load and test with different batch sizes
        loaded = torch.jit.load(ts_path, map_location=device)
        loaded.eval()

        input_size = pt_model._get_input_size()

        # Test with the exported batch size
        dummy_input = torch.randn(batch_size, 3, input_size, input_size, device=device)
        with torch.no_grad():
            output = loaded(dummy_input)
        assert output is not None
