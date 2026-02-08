"""
End-to-end tests for OpenVINO export and inference.

Tests the complete pipeline:
1. Load PyTorch model
2. Run PyTorch inference (baseline)
3. Export to OpenVINO
4. Load OpenVINO model via LIBREYOLO() factory / LIBREYOLOOpenVINO backend
5. Run OpenVINO inference
6. Compare results between PyTorch and OpenVINO
"""

from pathlib import Path

import pytest
import yaml

from .conftest import (
    FULL_TEST_MODELS,
    QUICK_TEST_MODELS,
    RFDETR_TEST_MODELS,
    load_model,
    match_detections,
    requires_openvino,
    requires_rfdetr,
    results_are_acceptable,
)

pytestmark = [pytest.mark.e2e, pytest.mark.openvino]


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestOpenVINOExportFP16:
    """Test OpenVINO FP16 export + inference for all models."""

    @requires_openvino
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_fp16_export_quick(self, model_type, size, sample_image, tmp_path):
        """Quick test with smallest models (for CI)."""
        self._run_fp16_test(model_type, size, sample_image, tmp_path)

    @requires_openvino
    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,size", FULL_TEST_MODELS)
    def test_fp16_export_full(self, model_type, size, sample_image, tmp_path):
        """Full test with all YOLOX and YOLOv9 models."""
        self._run_fp16_test(model_type, size, sample_image, tmp_path)

    @requires_openvino
    @requires_rfdetr
    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,size", RFDETR_TEST_MODELS)
    def test_fp16_export_rfdetr(self, model_type, size, sample_image, tmp_path):
        """Test RF-DETR models (requires extra dependencies)."""
        self._run_fp16_test(model_type, size, sample_image, tmp_path)

    def _run_fp16_test(self, model_type, size, sample_image, tmp_path):
        """Common FP16 test implementation."""
        from libreyolo import LIBREYOLO

        pt_model = load_model(model_type, size, device="cpu")
        pt_results = pt_model(sample_image, conf=0.25)

        ov_path = str(tmp_path / f"{model_type}_{size}_fp16_openvino")
        exported_path = pt_model.export(
            format="openvino",
            output_path=ov_path,
            half=True,
        )
        exported_dir = Path(exported_path)
        assert exported_dir.is_dir(), "OpenVINO output directory not created"
        assert (exported_dir / "model.xml").exists(), "model.xml not found"
        assert (exported_dir / "model.bin").exists(), "model.bin not found"

        # Load via factory and run inference
        ov_model = LIBREYOLO(exported_path)
        ov_results = ov_model(sample_image, conf=0.25)

        # Compare results — FP16 may lose a bit of precision
        match_rate, matched, total = match_detections(pt_results, ov_results)
        assert results_are_acceptable(match_rate, len(pt_results), len(ov_results)), (
            f"Results mismatch: PT={len(pt_results)}, OV={len(ov_results)}, "
            f"matched={matched}/{total}, rate={match_rate:.2%}"
        )

        del pt_model


class TestOpenVINOExportFP32:
    """Test OpenVINO FP32 export + inference for all models."""

    @requires_openvino
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_fp32_export_quick(self, model_type, size, sample_image, tmp_path):
        """Quick FP32 export test."""
        self._run_fp32_test(model_type, size, sample_image, tmp_path)

    @requires_openvino
    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,size", FULL_TEST_MODELS)
    def test_fp32_export_full(self, model_type, size, sample_image, tmp_path):
        """Full test with all YOLOX and YOLOv9 models."""
        self._run_fp32_test(model_type, size, sample_image, tmp_path)

    def _run_fp32_test(self, model_type, size, sample_image, tmp_path):
        """Common FP32 test implementation."""
        from libreyolo import LIBREYOLO

        pt_model = load_model(model_type, size, device="cpu")
        pt_results = pt_model(sample_image, conf=0.25)

        ov_path = str(tmp_path / f"{model_type}_{size}_fp32_openvino")
        exported_path = pt_model.export(
            format="openvino",
            output_path=ov_path,
            half=False,
        )
        assert Path(exported_path).is_dir(), "OpenVINO output directory not created"

        # Load via factory and run inference
        ov_model = LIBREYOLO(exported_path)
        ov_results = ov_model(sample_image, conf=0.25)

        # Compare results — FP32 should have higher match rate
        match_rate, matched, total = match_detections(pt_results, ov_results)
        assert results_are_acceptable(match_rate, len(pt_results), len(ov_results)), (
            f"Results mismatch: PT={len(pt_results)}, OV={len(ov_results)}, "
            f"matched={matched}/{total}, rate={match_rate:.2%}"
        )

        del pt_model


class TestOpenVINOMetadata:
    """Test OpenVINO metadata export."""

    @requires_openvino
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_metadata_saved(self, model_type, size, tmp_path):
        """Test that exported models have correct metadata."""
        pt_model = load_model(model_type, size, device="cpu")

        ov_path = str(tmp_path / f"{model_type}_{size}_openvino")
        exported_path = pt_model.export(
            format="openvino",
            output_path=ov_path,
            half=True,
        )

        # Verify metadata file exists and has correct content
        metadata_path = Path(exported_path) / "metadata.yaml"
        assert metadata_path.exists(), "metadata.yaml not found"

        with open(metadata_path) as f:
            metadata = yaml.safe_load(f)

        assert metadata["model_family"] == pt_model._get_model_name()
        assert metadata["model_size"] == pt_model.size
        assert metadata["nb_classes"] == pt_model.nb_classes
        assert metadata["precision"] == "fp16"
        assert "names" in metadata

        names = metadata["names"]
        assert isinstance(names, dict)
        assert len(names) == pt_model.nb_classes

        del pt_model

    @requires_openvino
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_metadata_round_trip(self, model_type, size, tmp_path):
        """Test that metadata is correctly loaded when loading OpenVINO model."""
        from libreyolo import LIBREYOLO

        pt_model = load_model(model_type, size, device="cpu")

        ov_path = str(tmp_path / f"{model_type}_{size}_openvino")
        exported_path = pt_model.export(
            format="openvino",
            output_path=ov_path,
            half=True,
        )

        # Load via factory and verify metadata was read
        ov_model = LIBREYOLO(exported_path)
        assert ov_model.nb_classes == pt_model.nb_classes
        assert ov_model.names == pt_model.names

        del pt_model


class TestOpenVINOMultipleInference:
    """Test OpenVINO inference consistency."""

    @requires_openvino
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_consistent_results(self, model_type, size, sample_image, tmp_path):
        """Test that OpenVINO model produces consistent results across runs."""
        from libreyolo import LIBREYOLO

        pt_model = load_model(model_type, size, device="cpu")

        ov_path = str(tmp_path / f"{model_type}_{size}_openvino")
        exported_path = pt_model.export(
            format="openvino",
            output_path=ov_path,
            half=True,
        )

        ov_model = LIBREYOLO(exported_path)

        # Run multiple inferences
        results = []
        for _ in range(5):
            result = ov_model(sample_image, conf=0.25)
            results.append(len(result))

        # Results should be identical
        assert len(set(results)) == 1, f"Inconsistent results across runs: {results}"

        del pt_model


class TestOpenVINOModelLoading:
    """Test OpenVINO model loading and structure."""

    @requires_openvino
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_model_loadable(self, model_type, size, tmp_path):
        """Test that exported OpenVINO model can be loaded by the runtime."""
        import openvino as ov

        pt_model = load_model(model_type, size, device="cpu")

        ov_path = str(tmp_path / f"{model_type}_{size}_openvino")
        exported_path = pt_model.export(
            format="openvino",
            output_path=ov_path,
            half=True,
        )

        # Load and compile with OpenVINO
        core = ov.Core()
        model = core.read_model(str(Path(exported_path) / "model.xml"))
        compiled = core.compile_model(model, "CPU")

        # Verify model has expected input/output structure
        assert len(compiled.inputs) >= 1
        assert len(compiled.outputs) >= 1

        # Verify input shape is correct
        input_shape = model.inputs[0].shape
        assert len(input_shape) == 4  # (B, C, H, W)
        assert input_shape[1] == 3    # RGB channels

        del pt_model

    @requires_openvino
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_fp32_and_fp16_both_work(self, model_type, size, sample_image, tmp_path):
        """Test that both FP32 and FP16 exports produce valid results."""
        from libreyolo import LIBREYOLO

        pt_model = load_model(model_type, size, device="cpu")

        # Export FP32
        fp32_path = str(tmp_path / f"{model_type}_{size}_fp32_openvino")
        fp32_exported = pt_model.export(
            format="openvino", output_path=fp32_path, half=False
        )

        # Export FP16
        fp16_path = str(tmp_path / f"{model_type}_{size}_fp16_openvino")
        fp16_exported = pt_model.export(
            format="openvino", output_path=fp16_path, half=True
        )

        # Both should produce inference results via the backend
        fp32_model = LIBREYOLO(fp32_exported)
        fp16_model = LIBREYOLO(fp16_exported)

        fp32_results = fp32_model(sample_image, conf=0.25)
        fp16_results = fp16_model(sample_image, conf=0.25)

        assert fp32_results is not None
        assert fp16_results is not None

        # FP16 file should be smaller than FP32
        fp32_size = (Path(fp32_exported) / "model.bin").stat().st_size
        fp16_size = (Path(fp16_exported) / "model.bin").stat().st_size
        assert fp16_size <= fp32_size, (
            f"FP16 ({fp16_size}) should be <= FP32 ({fp32_size})"
        )

        del pt_model


# ---------------------------------------------------------------------------
# Backend-specific tests
# ---------------------------------------------------------------------------


class TestOpenVINOBackend:
    """Test the LIBREYOLOOpenVINO inference backend class directly."""

    @requires_openvino
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_predict_alias(self, model_type, size, sample_image, tmp_path):
        """Test that predict() is an alias for __call__."""
        from libreyolo.common.openvino import LIBREYOLOOpenVINO

        pt_model = load_model(model_type, size, device="cpu")
        ov_path = str(tmp_path / f"{model_type}_{size}_openvino")
        exported_path = pt_model.export(
            format="openvino", output_path=ov_path, half=True,
        )

        ov_model = LIBREYOLOOpenVINO(exported_path)
        result_call = ov_model(sample_image, conf=0.25)
        result_predict = ov_model.predict(sample_image, conf=0.25)

        assert len(result_call) == len(result_predict)

        del pt_model

    @requires_openvino
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_save_output(self, model_type, size, sample_image, tmp_path):
        """Test that save=True produces an annotated image."""
        from libreyolo.common.openvino import LIBREYOLOOpenVINO

        pt_model = load_model(model_type, size, device="cpu")
        ov_path = str(tmp_path / f"{model_type}_{size}_openvino")
        exported_path = pt_model.export(
            format="openvino", output_path=ov_path, half=True,
        )

        save_path = str(tmp_path / "annotated.jpg")
        ov_model = LIBREYOLOOpenVINO(exported_path)
        result = ov_model(sample_image, conf=0.25, save=True, output_path=save_path)

        assert Path(save_path).exists(), "Annotated image was not saved"
        assert hasattr(result, "saved_path")

        del pt_model

    @requires_openvino
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_classes_filter(self, model_type, size, sample_image, tmp_path):
        """Test that classes filter limits detections to specified class IDs."""
        from libreyolo.common.openvino import LIBREYOLOOpenVINO

        pt_model = load_model(model_type, size, device="cpu")
        ov_path = str(tmp_path / f"{model_type}_{size}_openvino")
        exported_path = pt_model.export(
            format="openvino", output_path=ov_path, half=True,
        )

        ov_model = LIBREYOLOOpenVINO(exported_path)

        # Run with class filter (class 0 = person in COCO)
        result = ov_model(sample_image, conf=0.25, classes=[0])

        if len(result) > 0:
            unique_classes = result.boxes.cls.unique().tolist()
            assert all(c == 0.0 for c in unique_classes), (
                f"Expected only class 0, got classes: {unique_classes}"
            )

        del pt_model


class TestOpenVINOFactory:
    """Test loading OpenVINO models through the LIBREYOLO() factory."""

    @requires_openvino
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_factory_dispatch(self, model_type, size, sample_image, tmp_path):
        """Export model, load via LIBREYOLO(dir), verify type and inference."""
        from libreyolo import LIBREYOLO
        from libreyolo.common.openvino import LIBREYOLOOpenVINO

        pt_model = load_model(model_type, size, device="cpu")
        pt_results = pt_model(sample_image, conf=0.25)

        ov_path = str(tmp_path / f"{model_type}_{size}_openvino")
        exported_path = pt_model.export(
            format="openvino", output_path=ov_path, half=True,
        )

        # Load through factory
        factory_model = LIBREYOLO(exported_path)
        assert isinstance(factory_model, LIBREYOLOOpenVINO), (
            f"Expected LIBREYOLOOpenVINO, got {type(factory_model).__name__}"
        )

        # Run inference and compare
        factory_results = factory_model(sample_image, conf=0.25)
        match_rate, matched, total = match_detections(pt_results, factory_results)
        assert results_are_acceptable(match_rate, len(pt_results), len(factory_results)), (
            f"Results mismatch: PT={len(pt_results)}, Factory={len(factory_results)}, "
            f"matched={matched}/{total}, rate={match_rate:.2%}"
        )

        del pt_model


# ---------------------------------------------------------------------------
# Model coverage tests
# ---------------------------------------------------------------------------


class TestOpenVINOModelCoverage:
    """Verify all model types can be exported to OpenVINO."""

    @requires_openvino
    @pytest.mark.slow
    def test_all_yolox_sizes_exportable(self, sample_image, tmp_path):
        """Test that all YOLOX sizes can be exported and run."""
        from libreyolo import LIBREYOLO
        from .conftest import YOLOX_SIZES

        for size in YOLOX_SIZES:
            pt_model = load_model("yolox", size, device="cpu")
            ov_path = str(tmp_path / f"yolox_{size}_openvino")

            exported_path = pt_model.export(
                format="openvino", output_path=ov_path, half=True
            )
            assert Path(exported_path).is_dir(), f"Failed to export YOLOX-{size}"

            # Verify inference works via backend
            ov_model = LIBREYOLO(exported_path)
            result = ov_model(sample_image, conf=0.25)
            assert result is not None

            del pt_model

    @requires_openvino
    @pytest.mark.slow
    def test_all_yolov9_sizes_exportable(self, sample_image, tmp_path):
        """Test that all YOLOv9 sizes can be exported and run."""
        from libreyolo import LIBREYOLO
        from .conftest import YOLOV9_SIZES

        for size in YOLOV9_SIZES:
            pt_model = load_model("yolov9", size, device="cpu")
            ov_path = str(tmp_path / f"yolov9_{size}_openvino")

            exported_path = pt_model.export(
                format="openvino", output_path=ov_path, half=True
            )
            assert Path(exported_path).is_dir(), f"Failed to export YOLOv9-{size}"

            # Verify inference works via backend
            ov_model = LIBREYOLO(exported_path)
            result = ov_model(sample_image, conf=0.25)
            assert result is not None

            del pt_model

    @requires_openvino
    @requires_rfdetr
    @pytest.mark.slow
    def test_all_rfdetr_sizes_exportable(self, sample_image, tmp_path):
        """Test that all RF-DETR sizes can be exported and run."""
        from libreyolo import LIBREYOLO
        from .conftest import RFDETR_SIZES

        for size in RFDETR_SIZES:
            pt_model = load_model("rfdetr", size, device="cpu")
            ov_path = str(tmp_path / f"rfdetr_{size}_openvino")

            exported_path = pt_model.export(
                format="openvino", output_path=ov_path, half=True
            )
            assert Path(exported_path).is_dir(), f"Failed to export RF-DETR-{size}"

            # Verify inference works via backend
            ov_model = LIBREYOLO(exported_path)
            result = ov_model(sample_image, conf=0.25)
            assert result is not None

            del pt_model
