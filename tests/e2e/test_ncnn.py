"""
End-to-end tests for ncnn export and inference.

Tests the complete pipeline:
1. Load PyTorch model
2. Run PyTorch inference (baseline)
3. Export to ncnn via PNNX
4. Load ncnn model via LIBREYOLO() factory / LIBREYOLONCNN backend
5. Run ncnn inference
6. Compare results between PyTorch and ncnn

Known ncnn/PNNX limitations:
- YOLOX: Focus layer uses slice-with-step-2, unsupported by ncnn.
  Accuracy tests are xfail; export/metadata/consistency tests still pass.
- RF-DETR: torch.tile is unsupported by ncnn. All inference tests are xfail.
"""

from pathlib import Path

import pytest
import yaml

from .conftest import (
    FULL_TEST_MODELS,
    QUICK_TEST_MODELS,
    RFDETR_TEST_MODELS,
    RFDETR_SIZES,
    YOLOX_SIZES,
    YOLOV9_SIZES,
    load_model,
    match_detections,
    requires_ncnn,
    requires_rfdetr,
    results_are_acceptable,
)

pytestmark = [pytest.mark.e2e, pytest.mark.ncnn]

# ---------------------------------------------------------------------------
# xfail markers for ncnn op limitations
# ---------------------------------------------------------------------------

_yolox_xfail = pytest.mark.xfail(
    reason="PNNX: Focus layer slice-with-step-2 unsupported by ncnn",
    strict=True,
)
_rfdetr_xfail = pytest.mark.xfail(
    reason="ncnn does not support torch.tile",
    strict=True,
)

# Accuracy test parameters — YOLOX and RF-DETR are expected to fail
QUICK_ACCURACY_PARAMS = [
    pytest.param("yolox", "nano", marks=_yolox_xfail),
    ("yolov9", "t"),
]

FULL_ACCURACY_PARAMS = [
    *[pytest.param("yolox", size, marks=_yolox_xfail) for size in YOLOX_SIZES],
    *[("yolov9", size) for size in YOLOV9_SIZES],
]

RFDETR_ACCURACY_PARAMS = [
    pytest.param("rfdetr", size, marks=_rfdetr_xfail) for size in RFDETR_SIZES
]


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestNCNNExportFP32:
    """Test ncnn FP32 export + inference for all models."""

    @requires_ncnn
    @pytest.mark.parametrize("model_type,size", QUICK_ACCURACY_PARAMS)
    def test_fp32_export_quick(self, model_type, size, sample_image, tmp_path):
        """Quick test with smallest models (for CI)."""
        self._run_fp32_test(model_type, size, sample_image, tmp_path)

    @requires_ncnn
    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,size", FULL_ACCURACY_PARAMS)
    def test_fp32_export_full(self, model_type, size, sample_image, tmp_path):
        """Full test with all YOLOX and YOLOv9 models."""
        self._run_fp32_test(model_type, size, sample_image, tmp_path)

    @requires_ncnn
    @requires_rfdetr
    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,size", RFDETR_ACCURACY_PARAMS)
    def test_fp32_export_rfdetr(self, model_type, size, sample_image, tmp_path):
        """Test RF-DETR models (requires extra dependencies)."""
        self._run_fp32_test(model_type, size, sample_image, tmp_path)

    def _run_fp32_test(self, model_type, size, sample_image, tmp_path):
        """Common FP32 test implementation."""
        from libreyolo import LIBREYOLO

        pt_model = load_model(model_type, size, device="cpu")
        pt_results = pt_model(sample_image, conf=0.25)

        ncnn_path = str(tmp_path / f"{model_type}_{size}_fp32_ncnn")
        exported_path = pt_model.export(
            format="ncnn",
            output_path=ncnn_path,
            half=False,
        )
        exported_dir = Path(exported_path)
        assert exported_dir.is_dir(), "ncnn output directory not created"
        assert (exported_dir / "model.ncnn.param").exists(), "model.ncnn.param not found"
        assert (exported_dir / "model.ncnn.bin").exists(), "model.ncnn.bin not found"

        # Load via factory and run inference
        ncnn_model = LIBREYOLO(exported_path)
        ncnn_results = ncnn_model(sample_image, conf=0.25)

        # Compare results
        match_rate, matched, total = match_detections(pt_results, ncnn_results)
        assert results_are_acceptable(match_rate, len(pt_results), len(ncnn_results)), (
            f"Results mismatch: PT={len(pt_results)}, ncnn={len(ncnn_results)}, "
            f"matched={matched}/{total}, rate={match_rate:.2%}"
        )

        del pt_model


class TestNCNNExportFP16:
    """Test ncnn FP16 export + inference for all models."""

    @requires_ncnn
    @pytest.mark.parametrize("model_type,size", QUICK_ACCURACY_PARAMS)
    def test_fp16_export_quick(self, model_type, size, sample_image, tmp_path):
        """Quick test with smallest models (for CI)."""
        self._run_fp16_test(model_type, size, sample_image, tmp_path)

    @requires_ncnn
    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,size", FULL_ACCURACY_PARAMS)
    def test_fp16_export_full(self, model_type, size, sample_image, tmp_path):
        """Full test with all YOLOX and YOLOv9 models."""
        self._run_fp16_test(model_type, size, sample_image, tmp_path)

    @requires_ncnn
    @requires_rfdetr
    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,size", RFDETR_ACCURACY_PARAMS)
    def test_fp16_export_rfdetr(self, model_type, size, sample_image, tmp_path):
        """Test RF-DETR models (requires extra dependencies)."""
        self._run_fp16_test(model_type, size, sample_image, tmp_path)

    def _run_fp16_test(self, model_type, size, sample_image, tmp_path):
        """Common FP16 test implementation."""
        from libreyolo import LIBREYOLO

        pt_model = load_model(model_type, size, device="cpu")
        pt_results = pt_model(sample_image, conf=0.25)

        ncnn_path = str(tmp_path / f"{model_type}_{size}_fp16_ncnn")
        exported_path = pt_model.export(
            format="ncnn",
            output_path=ncnn_path,
            half=True,
        )
        exported_dir = Path(exported_path)
        assert exported_dir.is_dir(), "ncnn output directory not created"
        assert (exported_dir / "model.ncnn.param").exists(), "model.ncnn.param not found"
        assert (exported_dir / "model.ncnn.bin").exists(), "model.ncnn.bin not found"

        # Load via factory and run inference
        ncnn_model = LIBREYOLO(exported_path)
        ncnn_results = ncnn_model(sample_image, conf=0.25)

        # Compare results — FP16 may lose a bit of precision
        match_rate, matched, total = match_detections(pt_results, ncnn_results)
        assert results_are_acceptable(match_rate, len(pt_results), len(ncnn_results)), (
            f"Results mismatch: PT={len(pt_results)}, ncnn={len(ncnn_results)}, "
            f"matched={matched}/{total}, rate={match_rate:.2%}"
        )

        del pt_model


class TestNCNNMetadata:
    """Test ncnn metadata export."""

    @requires_ncnn
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_metadata_saved(self, model_type, size, tmp_path):
        """Test that exported models have correct metadata."""
        pt_model = load_model(model_type, size, device="cpu")

        ncnn_path = str(tmp_path / f"{model_type}_{size}_ncnn")
        exported_path = pt_model.export(
            format="ncnn",
            output_path=ncnn_path,
            half=False,
        )

        # Verify metadata file exists and has correct content
        metadata_path = Path(exported_path) / "metadata.yaml"
        assert metadata_path.exists(), "metadata.yaml not found"

        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)

        assert metadata["model_family"] == pt_model._get_model_name()
        assert metadata["model_size"] == pt_model.size
        assert metadata["nb_classes"] == pt_model.nb_classes
        assert metadata["precision"] == "fp32"
        assert "names" in metadata

        names = metadata["names"]
        assert isinstance(names, dict)
        assert len(names) == pt_model.nb_classes

        del pt_model

    @requires_ncnn
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_metadata_round_trip(self, model_type, size, tmp_path):
        """Test that metadata is correctly loaded when loading ncnn model."""
        from libreyolo import LIBREYOLO

        pt_model = load_model(model_type, size, device="cpu")

        ncnn_path = str(tmp_path / f"{model_type}_{size}_ncnn")
        exported_path = pt_model.export(
            format="ncnn",
            output_path=ncnn_path,
            half=False,
        )

        # Load via factory and verify metadata was read
        ncnn_model = LIBREYOLO(exported_path)
        assert ncnn_model.nb_classes == pt_model.nb_classes
        assert ncnn_model.names == pt_model.names

        del pt_model


class TestNCNNFactory:
    """Test loading ncnn models through the LIBREYOLO() factory."""

    @requires_ncnn
    @pytest.mark.parametrize("model_type,size", QUICK_ACCURACY_PARAMS)
    def test_factory_dispatch(self, model_type, size, sample_image, tmp_path):
        """Export model, load via LIBREYOLO(dir), verify type and inference."""
        from libreyolo import LIBREYOLO
        from libreyolo.common.ncnn import LIBREYOLONCNN

        pt_model = load_model(model_type, size, device="cpu")
        pt_results = pt_model(sample_image, conf=0.25)

        ncnn_path = str(tmp_path / f"{model_type}_{size}_ncnn")
        exported_path = pt_model.export(
            format="ncnn", output_path=ncnn_path, half=False,
        )

        # Load through factory
        factory_model = LIBREYOLO(exported_path)
        assert isinstance(factory_model, LIBREYOLONCNN), (
            f"Expected LIBREYOLONCNN, got {type(factory_model).__name__}"
        )

        # Run inference and compare
        factory_results = factory_model(sample_image, conf=0.25)
        match_rate, matched, total = match_detections(pt_results, factory_results)
        assert results_are_acceptable(match_rate, len(pt_results), len(factory_results)), (
            f"Results mismatch: PT={len(pt_results)}, Factory={len(factory_results)}, "
            f"matched={matched}/{total}, rate={match_rate:.2%}"
        )

        del pt_model


class TestNCNNBackend:
    """Test the LIBREYOLONCNN inference backend class directly."""

    @requires_ncnn
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_predict_alias(self, model_type, size, sample_image, tmp_path):
        """Test that predict() is an alias for __call__."""
        from libreyolo.common.ncnn import LIBREYOLONCNN

        pt_model = load_model(model_type, size, device="cpu")
        ncnn_path = str(tmp_path / f"{model_type}_{size}_ncnn")
        exported_path = pt_model.export(
            format="ncnn", output_path=ncnn_path, half=False,
        )

        ncnn_model = LIBREYOLONCNN(exported_path)
        result_call = ncnn_model(sample_image, conf=0.25)
        result_predict = ncnn_model.predict(sample_image, conf=0.25)

        assert len(result_call) == len(result_predict)

        del pt_model

    @requires_ncnn
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_save_output(self, model_type, size, sample_image, tmp_path):
        """Test that save=True produces an annotated image."""
        from libreyolo.common.ncnn import LIBREYOLONCNN

        pt_model = load_model(model_type, size, device="cpu")
        ncnn_path = str(tmp_path / f"{model_type}_{size}_ncnn")
        exported_path = pt_model.export(
            format="ncnn", output_path=ncnn_path, half=False,
        )

        save_path = str(tmp_path / "annotated.jpg")
        ncnn_model = LIBREYOLONCNN(exported_path)
        result = ncnn_model(sample_image, conf=0.25, save=True, output_path=save_path)

        assert Path(save_path).exists(), "Annotated image was not saved"
        assert hasattr(result, "saved_path")

        del pt_model

    @requires_ncnn
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_classes_filter(self, model_type, size, sample_image, tmp_path):
        """Test that classes filter limits detections to specified class IDs."""
        from libreyolo.common.ncnn import LIBREYOLONCNN

        pt_model = load_model(model_type, size, device="cpu")
        ncnn_path = str(tmp_path / f"{model_type}_{size}_ncnn")
        exported_path = pt_model.export(
            format="ncnn", output_path=ncnn_path, half=False,
        )

        ncnn_model = LIBREYOLONCNN(exported_path)

        # Run with class filter (class 0 = person in COCO)
        result = ncnn_model(sample_image, conf=0.25, classes=[0])

        if len(result) > 0:
            unique_classes = result.boxes.cls.unique().tolist()
            assert all(c == 0.0 for c in unique_classes), (
                f"Expected only class 0, got classes: {unique_classes}"
            )

        del pt_model


class TestNCNNMultipleInference:
    """Test ncnn inference consistency."""

    @requires_ncnn
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_consistent_results(self, model_type, size, sample_image, tmp_path):
        """Test that ncnn model produces consistent results across runs."""
        from libreyolo import LIBREYOLO

        pt_model = load_model(model_type, size, device="cpu")

        ncnn_path = str(tmp_path / f"{model_type}_{size}_ncnn")
        exported_path = pt_model.export(
            format="ncnn",
            output_path=ncnn_path,
            half=False,
        )

        ncnn_model = LIBREYOLO(exported_path)

        # Run multiple inferences
        results = []
        for _ in range(5):
            result = ncnn_model(sample_image, conf=0.25)
            results.append(len(result))

        # Results should be identical
        assert len(set(results)) == 1, f"Inconsistent results across runs: {results}"

        del pt_model


# ---------------------------------------------------------------------------
# Model coverage tests
# ---------------------------------------------------------------------------


class TestNCNNModelCoverage:
    """Verify all model types can be exported to ncnn."""

    @requires_ncnn
    @pytest.mark.slow
    def test_all_yolox_sizes_exportable(self, sample_image, tmp_path):
        """Test that all YOLOX sizes can be exported and run."""
        from libreyolo import LIBREYOLO

        for size in YOLOX_SIZES:
            pt_model = load_model("yolox", size, device="cpu")
            ncnn_path = str(tmp_path / f"yolox_{size}_ncnn")

            exported_path = pt_model.export(
                format="ncnn", output_path=ncnn_path, half=False
            )
            assert Path(exported_path).is_dir(), f"Failed to export YOLOX-{size}"

            # Verify inference works via backend
            ncnn_model = LIBREYOLO(exported_path)
            result = ncnn_model(sample_image, conf=0.25)
            assert result is not None

            del pt_model

    @requires_ncnn
    @pytest.mark.slow
    def test_all_yolov9_sizes_exportable(self, sample_image, tmp_path):
        """Test that all YOLOv9 sizes can be exported and run."""
        from libreyolo import LIBREYOLO

        for size in YOLOV9_SIZES:
            pt_model = load_model("yolov9", size, device="cpu")
            ncnn_path = str(tmp_path / f"yolov9_{size}_ncnn")

            exported_path = pt_model.export(
                format="ncnn", output_path=ncnn_path, half=False
            )
            assert Path(exported_path).is_dir(), f"Failed to export YOLOv9-{size}"

            # Verify inference works via backend
            ncnn_model = LIBREYOLO(exported_path)
            result = ncnn_model(sample_image, conf=0.25)
            assert result is not None

            del pt_model

    @requires_ncnn
    @requires_rfdetr
    @pytest.mark.slow
    @_rfdetr_xfail
    def test_all_rfdetr_sizes_exportable(self, sample_image, tmp_path):
        """Test that all RF-DETR sizes can be exported and run."""
        from libreyolo import LIBREYOLO

        for size in RFDETR_SIZES:
            pt_model = load_model("rfdetr", size, device="cpu")
            ncnn_path = str(tmp_path / f"rfdetr_{size}_ncnn")

            exported_path = pt_model.export(
                format="ncnn", output_path=ncnn_path, half=False
            )
            assert Path(exported_path).is_dir(), f"Failed to export RF-DETR-{size}"

            # Verify inference works via backend
            ncnn_model = LIBREYOLO(exported_path)
            result = ncnn_model(sample_image, conf=0.25)
            assert result is not None

            del pt_model
