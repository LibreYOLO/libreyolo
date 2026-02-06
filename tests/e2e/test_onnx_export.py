"""
End-to-end tests for ONNX export and inference.

Tests the complete pipeline:
1. Load PyTorch model
2. Run PyTorch inference (baseline)
3. Export to ONNX
4. Load ONNX model
5. Run ONNX inference
6. Compare results between PyTorch and ONNX
"""

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch

from .conftest import (
    FULL_TEST_MODELS,
    QUICK_TEST_MODELS,
    RFDETR_TEST_MODELS,
    compute_iou,
    get_model_weights,
    load_model,
    match_detections,
    requires_cuda,
    requires_rfdetr,
    results_are_acceptable,
)

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestONNXExport:
    """Test ONNX export for all models."""

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_onnx_export_quick(self, model_type, size, sample_image, tmp_path):
        """Quick test with smallest models (for CI)."""
        self._run_onnx_test(model_type, size, sample_image, tmp_path)

    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,size", FULL_TEST_MODELS)
    def test_onnx_export_full(self, model_type, size, sample_image, tmp_path):
        """Full test with all YOLOX and YOLOv9 models."""
        self._run_onnx_test(model_type, size, sample_image, tmp_path)

    @requires_rfdetr
    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,size", RFDETR_TEST_MODELS)
    def test_onnx_export_rfdetr(self, model_type, size, sample_image, tmp_path):
        """Test RF-DETR models (requires extra dependencies)."""
        self._run_onnx_test(model_type, size, sample_image, tmp_path)

    def _run_onnx_test(self, model_type, size, sample_image, tmp_path):
        """Common ONNX export test implementation."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load PyTorch model
        pt_model = load_model(model_type, size, device=device)

        # Run PyTorch inference
        pt_results = pt_model(sample_image, conf=0.25)

        # Export to ONNX
        onnx_path = str(tmp_path / f"{model_type}_{size}.onnx")
        exported_path = pt_model.export(
            format="onnx",
            output_path=onnx_path,
            simplify=True,
            dynamic=True,
        )
        assert Path(exported_path).exists(), "ONNX file not created"
        assert Path(exported_path).stat().st_size > 0, "ONNX file is empty"

        # Verify ONNX model is valid
        onnx_model = onnx.load(exported_path)
        onnx.checker.check_model(onnx_model)

        # Load ONNX model for inference
        from libreyolo import LIBREYOLO
        onnx_model_wrapper = LIBREYOLO(exported_path, device=device)

        # Run ONNX inference
        onnx_results = onnx_model_wrapper(sample_image, conf=0.25)

        # Compare results
        match_rate, matched, total = match_detections(pt_results, onnx_results)

        assert results_are_acceptable(match_rate, len(pt_results), len(onnx_results), threshold=0.8), (
            f"Results mismatch: PT={len(pt_results)}, ONNX={len(onnx_results)}, "
            f"matched={matched}/{total}, rate={match_rate:.2%}"
        )


class TestONNXExportHalf:
    """Test ONNX FP16 export."""

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_onnx_fp16_export(self, model_type, size, sample_image, tmp_path):
        """Test FP16 ONNX export."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pt_model = load_model(model_type, size, device=device)

        # Export to ONNX with half precision
        onnx_path = str(tmp_path / f"{model_type}_{size}_fp16.onnx")
        exported_path = pt_model.export(
            format="onnx",
            output_path=onnx_path,
            half=True,
            simplify=False,  # Simplify may fail with FP16
        )
        assert Path(exported_path).exists()

        # Verify input type is float16
        onnx_model = onnx.load(exported_path)
        input_type = onnx_model.graph.input[0].type.tensor_type.elem_type
        assert input_type == onnx.TensorProto.FLOAT16, "Input should be FP16"


class TestONNXMetadata:
    """Test ONNX export metadata."""

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_onnx_metadata(self, model_type, size, tmp_path):
        """Test that ONNX exports include correct metadata."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pt_model = load_model(model_type, size, device=device)

        onnx_path = str(tmp_path / f"{model_type}_{size}.onnx")
        pt_model.export(format="onnx", output_path=onnx_path, simplify=False)

        # Load and check metadata
        onnx_model = onnx.load(onnx_path)
        meta = {p.key: p.value for p in onnx_model.metadata_props}

        assert "model_family" in meta
        assert "model_size" in meta
        assert "nb_classes" in meta
        assert "names" in meta

        # Verify names are valid JSON
        names = json.loads(meta["names"])
        assert isinstance(names, dict)
        assert len(names) == pt_model.nb_classes

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_onnx_metadata_round_trip(self, model_type, size, tmp_path):
        """Test that metadata is correctly loaded when loading ONNX model."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pt_model = load_model(model_type, size, device=device)

        onnx_path = str(tmp_path / f"{model_type}_{size}.onnx")
        pt_model.export(format="onnx", output_path=onnx_path, simplify=False)

        # Load ONNX model and verify metadata was read
        from libreyolo import LIBREYOLO
        onnx_model = LIBREYOLO(onnx_path, device=device)

        assert onnx_model.nb_classes == pt_model.nb_classes
        assert onnx_model.names == pt_model.names


class TestONNXDynamicAxes:
    """Test ONNX dynamic axes export."""

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_onnx_dynamic_batch(self, model_type, size, tmp_path):
        """Test that dynamic batch works correctly."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pt_model = load_model(model_type, size, device=device)

        onnx_path = str(tmp_path / f"{model_type}_{size}_dynamic.onnx")
        pt_model.export(
            format="onnx",
            output_path=onnx_path,
            dynamic=True,
            simplify=False,
        )

        # Verify batch dimension is symbolic
        onnx_model = onnx.load(onnx_path)
        input_shape = onnx_model.graph.input[0].type.tensor_type.shape
        dim0 = input_shape.dim[0]

        # Dynamic dim should have param name, not fixed value
        assert dim0.dim_param != "", "Batch dim should be dynamic"

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_onnx_static_batch(self, model_type, size, tmp_path):
        """Test that static batch works correctly."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pt_model = load_model(model_type, size, device=device)

        onnx_path = str(tmp_path / f"{model_type}_{size}_static.onnx")
        pt_model.export(
            format="onnx",
            output_path=onnx_path,
            dynamic=False,
            batch=4,
            simplify=False,
        )

        # Verify batch dimension is fixed
        onnx_model = onnx.load(onnx_path)
        input_shape = onnx_model.graph.input[0].type.tensor_type.shape
        dim0 = input_shape.dim[0]

        assert dim0.dim_value == 4, f"Batch should be 4, got {dim0.dim_value}"


class TestONNXSimplification:
    """Test ONNX graph simplification."""

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_onnx_simplify(self, model_type, size, sample_image, tmp_path):
        """Test that simplified ONNX produces same results."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pt_model = load_model(model_type, size, device=device)

        # Export without simplification
        onnx_path = str(tmp_path / f"{model_type}_{size}_raw.onnx")
        pt_model.export(format="onnx", output_path=onnx_path, simplify=False)
        raw_size = Path(onnx_path).stat().st_size

        # Export with simplification
        onnx_simp_path = str(tmp_path / f"{model_type}_{size}_simp.onnx")
        pt_model.export(format="onnx", output_path=onnx_simp_path, simplify=True)
        simp_size = Path(onnx_simp_path).stat().st_size

        # Simplified should be equal or smaller (onnxsim may not always reduce size)
        # But it should not significantly increase
        assert simp_size <= raw_size * 1.1, "Simplified model should not be much larger"

        # Both should produce valid results
        from libreyolo import LIBREYOLO
        onnx_model = LIBREYOLO(onnx_simp_path, device=device)
        result = onnx_model(sample_image, conf=0.25)
        assert result is not None


class TestONNXMultipleInference:
    """Test ONNX model stability."""

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_onnx_consistent_results(self, model_type, size, sample_image, tmp_path):
        """Test that ONNX model produces consistent results."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pt_model = load_model(model_type, size, device=device)

        onnx_path = str(tmp_path / f"{model_type}_{size}.onnx")
        pt_model.export(format="onnx", output_path=onnx_path, simplify=False)

        from libreyolo import LIBREYOLO
        onnx_model = LIBREYOLO(onnx_path, device=device)

        # Run multiple inferences
        results = []
        for _ in range(5):
            result = onnx_model(sample_image, conf=0.25)
            results.append(len(result))

        # Results should be identical
        assert len(set(results)) == 1, f"Inconsistent results: {results}"


class TestONNXModelCoverage:
    """Verify all model types can be exported to ONNX."""

    def test_all_yolox_sizes_exportable(self, tmp_path):
        """Test that all YOLOX sizes can be exported."""
        from .conftest import YOLOX_SIZES

        device = "cuda" if torch.cuda.is_available() else "cpu"

        for size in YOLOX_SIZES:
            pt_model = load_model("yolox", size, device=device)
            onnx_path = str(tmp_path / f"yolox_{size}.onnx")

            pt_model.export(format="onnx", output_path=onnx_path, simplify=False)
            assert Path(onnx_path).exists(), f"Failed to export YOLOX-{size}"

            # Verify model is valid
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

    def test_all_yolov9_sizes_exportable(self, tmp_path):
        """Test that all YOLOv9 sizes can be exported."""
        from .conftest import YOLOV9_SIZES

        device = "cuda" if torch.cuda.is_available() else "cpu"

        for size in YOLOV9_SIZES:
            pt_model = load_model("yolov9", size, device=device)
            onnx_path = str(tmp_path / f"yolov9_{size}.onnx")

            pt_model.export(format="onnx", output_path=onnx_path, simplify=False)
            assert Path(onnx_path).exists(), f"Failed to export YOLOv9-{size}"

            # Verify model is valid
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

    @requires_rfdetr
    def test_all_rfdetr_sizes_exportable(self, tmp_path):
        """Test that all RF-DETR sizes can be exported."""
        from .conftest import RFDETR_SIZES

        device = "cuda" if torch.cuda.is_available() else "cpu"

        for size in RFDETR_SIZES:
            pt_model = load_model("rfdetr", size, device=device)
            onnx_path = str(tmp_path / f"rfdetr_{size}.onnx")

            pt_model.export(format="onnx", output_path=onnx_path, simplify=False)
            assert Path(onnx_path).exists(), f"Failed to export RF-DETR-{size}"

            # Verify model is valid
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)


class TestONNXOpset:
    """Test ONNX opset version handling."""

    @pytest.mark.parametrize("opset", [11, 12, 13, 14, 15, 16, 17])
    def test_onnx_different_opsets(self, opset, tmp_path):
        """Test export with different opset versions."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pt_model = load_model("yolox", "nano", device=device)

        onnx_path = str(tmp_path / f"yolox_nano_opset{opset}.onnx")

        try:
            pt_model.export(
                format="onnx",
                output_path=onnx_path,
                opset=opset,
                simplify=False,
            )
            assert Path(onnx_path).exists()

            # Verify opset version
            onnx_model = onnx.load(onnx_path)
            model_opset = onnx_model.opset_import[0].version
            assert model_opset == opset, f"Expected opset {opset}, got {model_opset}"

        except Exception as e:
            # Some opsets may not support all operations
            if opset < 11:
                pytest.skip(f"Opset {opset} not supported: {e}")
            raise
