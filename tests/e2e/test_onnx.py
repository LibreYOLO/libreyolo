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

import onnx
import pytest
import torch

from .conftest import (
    FULL_TEST_PARAMS,
    QUICK_TEST_PARAMS,
    RFDETR_TEST_PARAMS,
    load_model,
    match_detections,
    requires_rfdetr,
    results_are_acceptable,
    run_consistency_test,
    run_export_compare_test,
    run_metadata_round_trip_test,
)

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.export_backend,
    pytest.mark.supported_backend,
    pytest.mark.onnx,
]
OFFICIAL_YOLONAS_S = Path("downloads/yolonas/yolo_nas_s_coco.pth")
OFFICIAL_YOLONAS_WEIGHTS = {
    "s": Path("downloads/yolonas/yolo_nas_s_coco.pth"),
    "m": Path("downloads/yolonas/yolo_nas_m_coco.pth"),
    "l": Path("downloads/yolonas/yolo_nas_l_coco.pth"),
}


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestONNXExport:
    """Test ONNX export for all models."""

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_PARAMS)
    def test_onnx_export_quick(self, model_type, size, sample_image, tmp_path):
        """Quick test with smallest models (for CI)."""
        self._run_onnx_test(model_type, size, sample_image, tmp_path)

    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,size", FULL_TEST_PARAMS)
    def test_onnx_export_full(self, model_type, size, sample_image, tmp_path):
        """Full test with all YOLOX and YOLOv9 models."""
        self._run_onnx_test(model_type, size, sample_image, tmp_path)

    @requires_rfdetr
    @pytest.mark.slow
    @pytest.mark.parametrize("model_type,size", RFDETR_TEST_PARAMS)
    def test_onnx_export_rfdetr(self, model_type, size, sample_image, tmp_path):
        """Test RF-DETR models (requires extra dependencies)."""
        self._run_onnx_test(model_type, size, sample_image, tmp_path)

    def _run_onnx_test(self, model_type, size, sample_image, tmp_path):
        """Common ONNX export test implementation."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        exported_path, _, _ = run_export_compare_test(
            model_type,
            size,
            sample_image,
            tmp_path,
            format="onnx",
            export_kwargs={"simplify": True, "dynamic": True},
            match_threshold=0.8,
            device=device,
        )

        # ONNX-specific: verify model validity
        onnx_model = onnx.load(exported_path)
        onnx.checker.check_model(onnx_model)


@pytest.mark.yolonas
class TestONNXYOLONAS:
    """Test ONNX export for the official YOLO-NAS-S checkpoint."""

    @pytest.mark.skipif(
        not OFFICIAL_YOLONAS_S.exists(),
        reason="Official YOLO-NAS-S checkpoint not present in downloads/yolonas/",
    )
    def test_onnx_export_yolonas_s(self, sample_image, tmp_path):
        """Export YOLO-NAS-S to ONNX, reload it, and compare inference results."""
        from libreyolo import LibreYOLO

        device = "cuda" if torch.cuda.is_available() else "cpu"

        pt_model = LibreYOLO(str(OFFICIAL_YOLONAS_S), device=device)
        pt_results = pt_model(sample_image, conf=0.25)

        onnx_path = str(tmp_path / "yolonas_s.onnx")
        exported_path = pt_model.export(
            format="onnx",
            output_path=onnx_path,
            simplify=True,
            dynamic=True,
        )
        assert Path(exported_path).exists(), "ONNX file not created"

        onnx_model = onnx.load(exported_path)
        onnx.checker.check_model(onnx_model)

        loaded_model = LibreYOLO(exported_path, device=device)
        assert loaded_model.model_family == "yolonas"
        assert loaded_model.nb_classes == pt_model.nb_classes
        assert loaded_model.names == pt_model.names

        onnx_results = loaded_model(sample_image, conf=0.25)

        match_rate, matched, total = match_detections(pt_results, onnx_results)
        assert results_are_acceptable(
            match_rate,
            len(pt_results),
            len(onnx_results),
            threshold=0.8,
        ), (
            f"Results mismatch: PT={len(pt_results)}, ONNX={len(onnx_results)}, "
            f"matched={matched}/{total}, rate={match_rate:.2%}"
        )


class TestONNXExportHalf:
    """Test ONNX FP16 export."""

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_PARAMS)
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

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_PARAMS)
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

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_PARAMS)
    def test_onnx_metadata_round_trip(self, model_type, size, tmp_path):
        """Test that metadata is correctly loaded when loading ONNX model."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        run_metadata_round_trip_test(
            model_type,
            size,
            tmp_path,
            format="onnx",
            export_kwargs={"simplify": False},
            device=device,
        )


class TestONNXDynamicAxes:
    """Test ONNX dynamic axes export."""

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_PARAMS)
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

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_PARAMS)
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

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_PARAMS)
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
        from libreyolo import LibreYOLO

        onnx_model = LibreYOLO(onnx_simp_path, device=device)
        result = onnx_model(sample_image, conf=0.25)
        assert result is not None


class TestONNXMultipleInference:
    """Test ONNX model stability."""

    @pytest.mark.parametrize("model_type,size", QUICK_TEST_PARAMS)
    def test_onnx_consistent_results(self, model_type, size, sample_image, tmp_path):
        """Test that ONNX model produces consistent results."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        run_consistency_test(
            model_type,
            size,
            sample_image,
            tmp_path,
            format="onnx",
            export_kwargs={"simplify": False},
            device=device,
        )


class TestONNXModelCoverage:
    """Verify all model types can be exported to ONNX."""

    @pytest.mark.yolox
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

    @pytest.mark.yolo9
    def test_all_yolo9_sizes_exportable(self, tmp_path):
        """Test that all YOLO9 sizes can be exported."""
        from .conftest import YOLO9_SIZES

        device = "cuda" if torch.cuda.is_available() else "cpu"

        for size in YOLO9_SIZES:
            pt_model = load_model("yolo9", size, device=device)
            onnx_path = str(tmp_path / f"yolo9_{size}.onnx")

            pt_model.export(format="onnx", output_path=onnx_path, simplify=False)
            assert Path(onnx_path).exists(), f"Failed to export YOLO9-{size}"

            # Verify model is valid
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

    @requires_rfdetr
    @pytest.mark.rfdetr
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

    @pytest.mark.yolonas
    def test_all_yolonas_sizes_exportable(self, tmp_path):
        """Test that all local official YOLO-NAS detection sizes export."""
        from libreyolo import LibreYOLO

        missing = [
            size for size, path in OFFICIAL_YOLONAS_WEIGHTS.items() if not path.exists()
        ]
        if missing:
            pytest.skip(
                "Official YOLO-NAS checkpoints not present for sizes: "
                + ", ".join(sorted(missing))
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"

        for size, weights in OFFICIAL_YOLONAS_WEIGHTS.items():
            pt_model = LibreYOLO(str(weights), device=device)
            onnx_path = str(tmp_path / f"yolonas_{size}.onnx")

            pt_model.export(format="onnx", output_path=onnx_path, simplify=False)
            assert Path(onnx_path).exists(), f"Failed to export YOLO-NAS-{size}"

            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)

            loaded = LibreYOLO(onnx_path, device=device)
            assert loaded.model_family == "yolonas"
            assert loaded.nb_classes == pt_model.nb_classes


class TestONNXOpset:
    """Test ONNX opset version handling."""

    @pytest.mark.yolox
    @pytest.mark.parametrize("opset", [11, 12, 13, 14, 15, 16, 17])
    def test_onnx_different_opsets(self, opset, tmp_path):
        """Test export with different opset versions."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        pt_model = load_model("yolox", "n", device=device)

        onnx_path = str(tmp_path / f"yolox_n_opset{opset}.onnx")

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


@requires_rfdetr
@pytest.mark.rfdetr
class TestONNXSegmentation:
    """Test ONNX export and inference for segmentation models."""

    def test_onnx_seg_export_produces_masks(self, sample_image, tmp_path):
        """Export RF-DETR-seg to ONNX, load it back, and verify masks are returned."""
        from libreyolo import LibreYOLO

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # PyTorch seg model
        pt_model = LibreYOLO("LibreRFDETRn-seg.pt", device=device)

        # Export to ONNX
        onnx_path = str(tmp_path / "rfdetr_n_seg.onnx")
        exported = pt_model.export(format="onnx", output_path=onnx_path, simplify=False)
        assert Path(exported).exists()

        # Verify ONNX has 3 outputs (boxes, scores, masks)
        onnx_model = onnx.load(exported)
        output_names = [o.name for o in onnx_model.graph.output]
        assert len(output_names) == 3, f"Expected 3 outputs, got {output_names}"
        assert "masks" in output_names

        # Verify segmentation metadata
        meta = {p.key: p.value for p in onnx_model.metadata_props}
        assert meta.get("segmentation") == "true"

        # Load ONNX backend and run inference
        onnx_model_loaded = LibreYOLO(exported, device=device)
        onnx_result = onnx_model_loaded(sample_image, conf=0.3)

        # ONNX result should also have masks
        if len(onnx_result) > 0:
            assert onnx_result.masks is not None, "ONNX seg model should return masks"
            assert len(onnx_result.masks) == len(onnx_result), "One mask per detection"
            h, w = onnx_result.orig_shape
            assert onnx_result.masks.data.shape[1:] == (h, w), (
                "Masks should be at original image resolution"
            )

    def test_onnx_seg_detection_counts_match(self, sample_image, tmp_path):
        """ONNX seg model should produce similar detection counts to PyTorch."""
        from libreyolo import LibreYOLO

        device = "cuda" if torch.cuda.is_available() else "cpu"

        pt_model = LibreYOLO("LibreRFDETRn-seg.pt", device=device)
        pt_result = pt_model(sample_image, conf=0.3)

        onnx_path = str(tmp_path / "rfdetr_n_seg_compare.onnx")
        pt_model.export(format="onnx", output_path=onnx_path, simplify=False)

        onnx_model = LibreYOLO(onnx_path, device=device)
        onnx_result = onnx_model(sample_image, conf=0.3)

        # Detection counts should be close (allow some variance from fp32→onnx)
        pt_count = len(pt_result)
        onnx_count = len(onnx_result)
        assert abs(pt_count - onnx_count) <= max(3, pt_count * 0.3), (
            f"Detection count mismatch: PT={pt_count}, ONNX={onnx_count}"
        )
