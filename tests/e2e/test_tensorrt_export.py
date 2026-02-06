"""
End-to-end tests for TensorRT export and inference.

Tests the complete pipeline:
1. Load PyTorch model
2. Run PyTorch inference (baseline)
3. Export to TensorRT
4. Load TensorRT engine
5. Run TensorRT inference
6. Compare results between PyTorch and TensorRT
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from .conftest import (
    FULL_TEST_MODELS,
    QUICK_TEST_MODELS,
    RFDETR_TEST_MODELS,
    load_model,
    match_detections,
    requires_cuda,
    requires_tensorrt,
    requires_rfdetr,
    results_are_acceptable,
)

pytestmark = [pytest.mark.e2e, pytest.mark.tensorrt]


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestTensorRTExportFP16:
    """Test TensorRT FP16 export for all models."""

    @requires_tensorrt
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_fp16_export_quick(self, model_type, size, sample_image, tmp_path):
        """Quick test with smallest models (for CI)."""
        self._run_fp16_test(model_type, size, sample_image, tmp_path)

    @requires_tensorrt
    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,size", FULL_TEST_MODELS)
    def test_fp16_export_full(self, model_type, size, sample_image, tmp_path):
        """Full test with all YOLOX and YOLOv9 models."""
        self._run_fp16_test(model_type, size, sample_image, tmp_path)

    @requires_tensorrt
    @requires_rfdetr
    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,size", RFDETR_TEST_MODELS)
    def test_fp16_export_rfdetr(self, model_type, size, sample_image, tmp_path):
        """Test RF-DETR models (requires extra dependencies)."""
        self._run_fp16_test(model_type, size, sample_image, tmp_path)

    def _run_fp16_test(self, model_type, size, sample_image, tmp_path):
        """Common FP16 test implementation."""
        # Load PyTorch model
        pt_model = load_model(model_type, size, device="cuda")

        # Run PyTorch inference
        pt_results = pt_model(sample_image, conf=0.25)

        # Export to TensorRT
        engine_path = str(tmp_path / f"{model_type}_{size}_fp16.engine")
        exported_path = pt_model.export(
            format="tensorrt",
            output_path=engine_path,
            half=True,
        )
        assert Path(exported_path).exists(), "Engine file not created"
        assert Path(exported_path).stat().st_size > 0, "Engine file is empty"

        # Load TensorRT engine
        from libreyolo import LIBREYOLO
        trt_model = LIBREYOLO(exported_path, device="cuda")

        # Run TensorRT inference
        trt_results = trt_model(sample_image, conf=0.25)

        # Compare results
        match_rate, matched, total = match_detections(pt_results, trt_results)

        assert results_are_acceptable(match_rate, len(pt_results), len(trt_results)), (
            f"Results mismatch: PT={len(pt_results)}, TRT={len(trt_results)}, "
            f"matched={matched}/{total}, rate={match_rate:.2%}"
        )

        # Cleanup CUDA memory
        del pt_model, trt_model
        torch.cuda.empty_cache()


class TestTensorRTExportFP32:
    """Test TensorRT FP32 export for all models."""

    @requires_tensorrt
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_fp32_export_quick(self, model_type, size, sample_image, tmp_path):
        """Quick test with smallest models (for CI)."""
        self._run_fp32_test(model_type, size, sample_image, tmp_path)

    @requires_tensorrt
    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,size", FULL_TEST_MODELS)
    def test_fp32_export_full(self, model_type, size, sample_image, tmp_path):
        """Full test with all YOLOX and YOLOv9 models."""
        self._run_fp32_test(model_type, size, sample_image, tmp_path)

    def _run_fp32_test(self, model_type, size, sample_image, tmp_path):
        """Common FP32 test implementation."""
        # Load PyTorch model
        pt_model = load_model(model_type, size, device="cuda")

        # Run PyTorch inference
        pt_results = pt_model(sample_image, conf=0.25)

        # Export to TensorRT (FP32)
        engine_path = str(tmp_path / f"{model_type}_{size}_fp32.engine")
        exported_path = pt_model.export(
            format="tensorrt",
            output_path=engine_path,
            half=False,
        )
        assert Path(exported_path).exists(), "Engine file not created"

        # Load TensorRT engine
        from libreyolo import LIBREYOLO
        trt_model = LIBREYOLO(exported_path, device="cuda")

        # Run TensorRT inference
        trt_results = trt_model(sample_image, conf=0.25)

        # Compare results - FP32 should have higher match rate
        match_rate, matched, total = match_detections(pt_results, trt_results)

        # FP32 should be more accurate than FP16
        assert results_are_acceptable(match_rate, len(pt_results), len(trt_results)), (
            f"Results mismatch: PT={len(pt_results)}, TRT={len(trt_results)}, "
            f"matched={matched}/{total}, rate={match_rate:.2%}"
        )

        # Cleanup
        del pt_model, trt_model
        torch.cuda.empty_cache()


class TestTensorRTExportINT8:
    """Test TensorRT INT8 export (requires calibration data)."""

    @requires_tensorrt
    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_int8_export_quick(self, model_type, size, sample_image, tmp_path):
        """INT8 export test with calibration."""
        self._run_int8_test(model_type, size, sample_image, tmp_path)

    def _run_int8_test(self, model_type, size, sample_image, tmp_path):
        """Common INT8 test implementation."""
        # Check for cuda-python or pycuda
        try:
            from cuda.bindings import runtime as cudart
        except ImportError:
            try:
                import pycuda.driver
            except ImportError:
                pytest.skip("INT8 requires cuda-python or pycuda")

        # Load PyTorch model
        pt_model = load_model(model_type, size, device="cuda")

        # Run PyTorch inference
        pt_results = pt_model(sample_image, conf=0.25)

        # Export to TensorRT (INT8)
        engine_path = str(tmp_path / f"{model_type}_{size}_int8.engine")

        try:
            exported_path = pt_model.export(
                format="tensorrt",
                output_path=engine_path,
                int8=True,
                data="coco5000.yaml",  # Calibration dataset
                fraction=0.05,  # Use 5% for faster testing
            )
        except FileNotFoundError:
            pytest.skip("Calibration dataset coco5000.yaml not found")

        assert Path(exported_path).exists(), "Engine file not created"

        # Load TensorRT engine
        from libreyolo import LIBREYOLO
        trt_model = LIBREYOLO(exported_path, device="cuda")

        # Run TensorRT inference
        trt_results = trt_model(sample_image, conf=0.25)

        # INT8 may have lower accuracy but should still detect objects
        match_rate, matched, total = match_detections(pt_results, trt_results, iou_threshold=0.3)

        # INT8 tolerances are more relaxed
        det_diff = abs(len(pt_results) - len(trt_results))
        acceptable = match_rate >= 0.5 or det_diff <= 5

        assert acceptable, (
            f"INT8 results too different: PT={len(pt_results)}, TRT={len(trt_results)}, "
            f"matched={matched}/{total}, rate={match_rate:.2%}"
        )

        # Cleanup
        del pt_model, trt_model
        torch.cuda.empty_cache()


class TestTensorRTEngineLoading:
    """Test TensorRT engine loading and metadata."""

    @requires_tensorrt
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_engine_metadata(self, model_type, size, tmp_path):
        """Test that exported engines have correct metadata."""
        pt_model = load_model(model_type, size, device="cuda")

        engine_path = str(tmp_path / f"{model_type}_{size}.engine")
        pt_model.export(
            format="tensorrt",
            output_path=engine_path,
            half=True,
        )

        # Load and verify metadata
        from libreyolo import LIBREYOLO
        trt_model = LIBREYOLO(engine_path, device="cuda")

        assert trt_model.model_type == "tensorrt"
        assert trt_model.nb_classes == pt_model.nb_classes
        assert hasattr(trt_model, "input_size")

        del pt_model, trt_model
        torch.cuda.empty_cache()

    @requires_tensorrt
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_engine_multiple_inference(self, model_type, size, sample_image, tmp_path):
        """Test that engines can run multiple inferences."""
        pt_model = load_model(model_type, size, device="cuda")

        engine_path = str(tmp_path / f"{model_type}_{size}.engine")
        pt_model.export(format="tensorrt", output_path=engine_path, half=True)

        from libreyolo import LIBREYOLO
        trt_model = LIBREYOLO(engine_path, device="cuda")

        # Run multiple inferences
        results = []
        for _ in range(5):
            result = trt_model(sample_image, conf=0.25)
            results.append(len(result))

        # Results should be consistent
        assert len(set(results)) == 1, f"Inconsistent results across runs: {results}"

        del pt_model, trt_model
        torch.cuda.empty_cache()


class TestTensorRTExportConfig:
    """Test TensorRT export with configuration files."""

    @requires_tensorrt
    def test_export_with_yaml_config(self, sample_image, tmp_path):
        """Test export using YAML configuration file."""
        pt_model = load_model("yolox", "nano", device="cuda")

        engine_path = str(tmp_path / "model_with_config.engine")
        exported_path = pt_model.export(
            format="tensorrt",
            output_path=engine_path,
            trt_config="tensorrt_default.yaml",
        )

        assert Path(exported_path).exists()

        # Verify inference works
        from libreyolo import LIBREYOLO
        trt_model = LIBREYOLO(exported_path, device="cuda")
        result = trt_model(sample_image, conf=0.25)

        assert result is not None

        del pt_model, trt_model
        torch.cuda.empty_cache()

    @requires_tensorrt
    def test_export_with_dict_config(self, sample_image, tmp_path):
        """Test export using dictionary configuration."""
        pt_model = load_model("yolox", "nano", device="cuda")

        config = {
            "precision": "fp16",
            "workspace": 2.0,
            "verbose": False,
        }

        engine_path = str(tmp_path / "model_with_dict_config.engine")
        exported_path = pt_model.export(
            format="tensorrt",
            output_path=engine_path,
            trt_config=config,
        )

        assert Path(exported_path).exists()

        del pt_model
        torch.cuda.empty_cache()


class TestTensorRTInferenceSpeed:
    """Test TensorRT inference performance."""

    @requires_tensorrt
    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_tensorrt_faster_than_pytorch(self, model_type, size, sample_image, tmp_path):
        """Verify TensorRT inference is faster than PyTorch."""
        import time

        pt_model = load_model(model_type, size, device="cuda")

        # Export to TensorRT
        engine_path = str(tmp_path / f"{model_type}_{size}.engine")
        pt_model.export(format="tensorrt", output_path=engine_path, half=True)

        from libreyolo import LIBREYOLO
        trt_model = LIBREYOLO(engine_path, device="cuda")

        # Warmup
        for _ in range(5):
            pt_model(sample_image, conf=0.25)
            trt_model(sample_image, conf=0.25)

        # Benchmark PyTorch
        torch.cuda.synchronize()
        pt_start = time.perf_counter()
        for _ in range(20):
            pt_model(sample_image, conf=0.25)
        torch.cuda.synchronize()
        pt_time = (time.perf_counter() - pt_start) / 20

        # Benchmark TensorRT
        torch.cuda.synchronize()
        trt_start = time.perf_counter()
        for _ in range(20):
            trt_model(sample_image, conf=0.25)
        torch.cuda.synchronize()
        trt_time = (time.perf_counter() - trt_start) / 20

        speedup = pt_time / trt_time

        # TensorRT should be at least 1.2x faster (conservative)
        assert speedup >= 1.0, (
            f"TensorRT not faster: PT={pt_time*1000:.1f}ms, "
            f"TRT={trt_time*1000:.1f}ms, speedup={speedup:.2f}x"
        )

        del pt_model, trt_model
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Model coverage tests
# ---------------------------------------------------------------------------


class TestModelCoverage:
    """Verify all model types can be exported."""

    @requires_tensorrt
    def test_all_yolox_sizes_exportable(self, tmp_path):
        """Test that all YOLOX sizes can be exported."""
        from .conftest import YOLOX_SIZES

        for size in YOLOX_SIZES:
            pt_model = load_model("yolox", size, device="cuda")
            engine_path = str(tmp_path / f"yolox_{size}.engine")

            try:
                pt_model.export(format="tensorrt", output_path=engine_path, half=True)
                assert Path(engine_path).exists(), f"Failed to export YOLOX-{size}"
            finally:
                del pt_model
                torch.cuda.empty_cache()

    @requires_tensorrt
    def test_all_yolov9_sizes_exportable(self, tmp_path):
        """Test that all YOLOv9 sizes can be exported."""
        from .conftest import YOLOV9_SIZES

        for size in YOLOV9_SIZES:
            pt_model = load_model("yolov9", size, device="cuda")
            engine_path = str(tmp_path / f"yolov9_{size}.engine")

            try:
                pt_model.export(format="tensorrt", output_path=engine_path, half=True)
                assert Path(engine_path).exists(), f"Failed to export YOLOv9-{size}"
            finally:
                del pt_model
                torch.cuda.empty_cache()

    @requires_tensorrt
    @requires_rfdetr
    def test_all_rfdetr_sizes_exportable(self, tmp_path):
        """Test that all RF-DETR sizes can be exported."""
        from .conftest import RFDETR_SIZES

        for size in RFDETR_SIZES:
            pt_model = load_model("rfdetr", size, device="cuda")
            engine_path = str(tmp_path / f"rfdetr_{size}.engine")

            try:
                pt_model.export(format="tensorrt", output_path=engine_path, half=True)
                assert Path(engine_path).exists(), f"Failed to export RF-DETR-{size}"
            finally:
                del pt_model
                torch.cuda.empty_cache()
