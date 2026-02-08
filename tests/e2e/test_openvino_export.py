"""
End-to-end tests for OpenVINO export and inference.

Tests the complete pipeline:
1. Load PyTorch model
2. Run PyTorch inference (baseline)
3. Export to OpenVINO
4. Load OpenVINO model via OpenVINO runtime
5. Run OpenVINO inference
6. Compare results between PyTorch and OpenVINO
"""

from pathlib import Path

import numpy as np
import pytest
import torch
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
# OpenVINO inference helper
# ---------------------------------------------------------------------------


def _nms(boxes, scores, iou_threshold=0.45):
    """Numpy NMS (mirrors libreyolo.common.onnx._nms)."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= iou_threshold)[0] + 1]
    return keep


def _openvino_predict(model_dir, image, conf=0.25, iou=0.45):
    """Run full inference pipeline using OpenVINO runtime.

    Mirrors LIBREYOLOOnnx preprocessing and postprocessing to produce
    comparable Results objects for detection matching.
    """
    import openvino as ov

    from libreyolo.common.utils import preprocess_image
    from libreyolo.common.results import Boxes, Results
    from libreyolo.yolox.utils import preprocess_image as yolox_preprocess

    model_dir = Path(model_dir)

    # Read metadata
    with open(model_dir / "metadata.yaml") as f:
        meta = yaml.safe_load(f)

    model_family = meta["model_family"]
    nb_classes = meta["nb_classes"]
    names = {int(k): v for k, v in meta["names"].items()}
    imgsz = meta["imgsz"]

    # Load and compile model
    core = ov.Core()
    ov_model = core.read_model(str(model_dir / "model.xml"))
    compiled = core.compile_model(ov_model, "CPU")

    # Preprocess based on model family
    ratio = None
    if model_family == "LIBREYOLOX":
        input_tensor, original_img, original_size, ratio = yolox_preprocess(
            image, input_size=imgsz, color_format="auto"
        )
    elif model_family == "LIBREYOLORFDETR":
        input_tensor, original_img, original_size = _preprocess_rfdetr(image, imgsz)
    else:
        input_tensor, original_img, original_size = preprocess_image(
            image, input_size=imgsz, color_format="auto"
        )

    blob = input_tensor.numpy()

    # Run inference
    result = compiled(blob)
    all_outputs = [result[output] for output in compiled.outputs]

    # Parse outputs based on model family
    orig_w, orig_h = original_size

    if model_family == "LIBREYOLOX":
        boxes, max_scores, class_ids = _parse_yolox(
            all_outputs, imgsz, orig_w, orig_h, conf, ratio
        )
    elif model_family == "LIBREYOLORFDETR":
        boxes, max_scores, class_ids = _parse_rfdetr(
            all_outputs, orig_w, orig_h, conf
        )
    else:
        boxes, max_scores, class_ids = _parse_yolov9(
            all_outputs, imgsz, orig_w, orig_h, conf
        )

    orig_shape = (orig_h, orig_w)

    if len(boxes) == 0:
        return Results(
            boxes=Boxes(
                torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.float32),
            ),
            orig_shape=orig_shape,
            path=str(image) if isinstance(image, (str, Path)) else None,
            names=names,
        )

    # Apply NMS
    keep = _nms(boxes, max_scores, iou)
    boxes, max_scores, class_ids = boxes[keep], max_scores[keep], class_ids[keep]

    return Results(
        boxes=Boxes(
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(max_scores, dtype=torch.float32),
            torch.tensor(class_ids, dtype=torch.float32),
        ),
        orig_shape=orig_shape,
        path=str(image) if isinstance(image, (str, Path)) else None,
        names=names,
    )


# ---------------------------------------------------------------------------
# Output parsers (mirror LIBREYOLOOnnx._parse_* methods)
# ---------------------------------------------------------------------------


def _parse_yolox(all_outputs, imgsz, orig_w, orig_h, conf, ratio):
    """Parse YOLOX output: (B, N, 5+nc) with cxcywh + objectness + class_scores."""
    outputs = all_outputs[0][0]  # (N, 5+nc)

    cx, cy, w, h = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]
    objectness = outputs[:, 4]
    class_scores = outputs[:, 5:]

    max_class_scores = np.max(class_scores, axis=1)
    max_scores = objectness * max_class_scores
    class_ids = np.argmax(class_scores, axis=1)

    mask = max_scores > conf
    cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
    max_scores, class_ids = max_scores[mask], class_ids[mask]

    if len(max_scores) == 0:
        return np.empty((0, 4)), max_scores, class_ids

    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    r = ratio if ratio is not None else 1.0
    boxes /= r
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    return boxes, max_scores, class_ids


def _parse_yolov9(all_outputs, imgsz, orig_w, orig_h, conf):
    """Parse YOLOv9 output: (B, 4+nc, N) with xyxy + class_scores."""
    outputs = all_outputs[0][0].T  # (N, 4+nc)

    boxes = outputs[:, :4]
    scores = outputs[:, 4:]

    max_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)

    mask = max_scores > conf
    boxes, max_scores, class_ids = boxes[mask], max_scores[mask], class_ids[mask]

    if len(boxes) == 0:
        return boxes, max_scores, class_ids

    scale_x = orig_w / imgsz
    scale_y = orig_h / imgsz
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    return boxes, max_scores, class_ids


def _parse_rfdetr(all_outputs, orig_w, orig_h, conf):
    """Parse RF-DETR output: boxes (B,300,4) cxcywh [0,1] + logits (B,300,nc)."""
    boxes_raw = all_outputs[0][0]  # (300, 4) normalized cxcywh
    logits = all_outputs[1][0]     # (300, nc) raw logits

    scores = 1.0 / (1.0 + np.exp(-logits.astype(np.float64))).astype(np.float32)

    max_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)

    mask = max_scores > conf
    boxes_raw, max_scores, class_ids = boxes_raw[mask], max_scores[mask], class_ids[mask]

    if len(boxes_raw) == 0:
        return boxes_raw, max_scores, class_ids

    cx, cy, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    x1 = (cx - w / 2) * orig_w
    y1 = (cy - h / 2) * orig_h
    x2 = (cx + w / 2) * orig_w
    y2 = (cy + h / 2) * orig_h
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    return boxes, max_scores, class_ids


def _preprocess_rfdetr(image, input_size):
    """RF-DETR preprocessing: direct resize + ImageNet normalization."""
    from libreyolo.common.image_loader import ImageLoader
    from PIL import Image

    img = ImageLoader.load(image, color_format="auto")
    original_size = img.size  # (W, H)
    original_img = img.copy()

    img_resized = img.resize((input_size, input_size), Image.Resampling.BILINEAR)
    img_array = np.array(img_resized, dtype=np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std

    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
    return img_tensor, original_img, original_size


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
        # Load PyTorch model
        pt_model = load_model(model_type, size, device="cpu")

        # Run PyTorch inference
        pt_results = pt_model(sample_image, conf=0.25)

        # Export to OpenVINO
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

        # Run OpenVINO inference
        ov_results = _openvino_predict(exported_path, sample_image, conf=0.25)

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
        # Load PyTorch model
        pt_model = load_model(model_type, size, device="cpu")

        # Run PyTorch inference
        pt_results = pt_model(sample_image, conf=0.25)

        # Export to OpenVINO (FP32)
        ov_path = str(tmp_path / f"{model_type}_{size}_fp32_openvino")
        exported_path = pt_model.export(
            format="openvino",
            output_path=ov_path,
            half=False,
        )
        assert Path(exported_path).is_dir(), "OpenVINO output directory not created"

        # Run OpenVINO inference
        ov_results = _openvino_predict(exported_path, sample_image, conf=0.25)

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
        """Test that metadata values match original model attributes."""
        pt_model = load_model(model_type, size, device="cpu")

        ov_path = str(tmp_path / f"{model_type}_{size}_openvino")
        exported_path = pt_model.export(
            format="openvino",
            output_path=ov_path,
            half=True,
        )

        with open(Path(exported_path) / "metadata.yaml") as f:
            metadata = yaml.safe_load(f)

        # Verify names round-trip correctly
        for k, v in pt_model.names.items():
            assert metadata["names"][str(k)] == v

        assert metadata["imgsz"] == pt_model._get_input_size()

        del pt_model


class TestOpenVINOMultipleInference:
    """Test OpenVINO inference consistency."""

    @requires_openvino
    @pytest.mark.parametrize("model_type,size", QUICK_TEST_MODELS)
    def test_consistent_results(self, model_type, size, sample_image, tmp_path):
        """Test that OpenVINO model produces consistent results across runs."""
        pt_model = load_model(model_type, size, device="cpu")

        ov_path = str(tmp_path / f"{model_type}_{size}_openvino")
        exported_path = pt_model.export(
            format="openvino",
            output_path=ov_path,
            half=True,
        )

        # Run multiple inferences
        results = []
        for _ in range(5):
            result = _openvino_predict(exported_path, sample_image, conf=0.25)
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

        # Both should produce inference results
        fp32_results = _openvino_predict(fp32_exported, sample_image, conf=0.25)
        fp16_results = _openvino_predict(fp16_exported, sample_image, conf=0.25)

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
# Model coverage tests
# ---------------------------------------------------------------------------


class TestOpenVINOModelCoverage:
    """Verify all model types can be exported to OpenVINO."""

    @requires_openvino
    @pytest.mark.slow
    def test_all_yolox_sizes_exportable(self, sample_image, tmp_path):
        """Test that all YOLOX sizes can be exported and run."""
        from .conftest import YOLOX_SIZES

        for size in YOLOX_SIZES:
            pt_model = load_model("yolox", size, device="cpu")
            ov_path = str(tmp_path / f"yolox_{size}_openvino")

            exported_path = pt_model.export(
                format="openvino", output_path=ov_path, half=True
            )
            assert Path(exported_path).is_dir(), f"Failed to export YOLOX-{size}"

            # Verify inference works
            result = _openvino_predict(exported_path, sample_image, conf=0.25)
            assert result is not None

            del pt_model

    @requires_openvino
    @pytest.mark.slow
    def test_all_yolov9_sizes_exportable(self, sample_image, tmp_path):
        """Test that all YOLOv9 sizes can be exported and run."""
        from .conftest import YOLOV9_SIZES

        for size in YOLOV9_SIZES:
            pt_model = load_model("yolov9", size, device="cpu")
            ov_path = str(tmp_path / f"yolov9_{size}_openvino")

            exported_path = pt_model.export(
                format="openvino", output_path=ov_path, half=True
            )
            assert Path(exported_path).is_dir(), f"Failed to export YOLOv9-{size}"

            # Verify inference works
            result = _openvino_predict(exported_path, sample_image, conf=0.25)
            assert result is not None

            del pt_model

    @requires_openvino
    @requires_rfdetr
    @pytest.mark.slow
    def test_all_rfdetr_sizes_exportable(self, sample_image, tmp_path):
        """Test that all RF-DETR sizes can be exported and run."""
        from .conftest import RFDETR_SIZES

        for size in RFDETR_SIZES:
            pt_model = load_model("rfdetr", size, device="cpu")
            ov_path = str(tmp_path / f"rfdetr_{size}_openvino")

            exported_path = pt_model.export(
                format="openvino", output_path=ov_path, half=True
            )
            assert Path(exported_path).is_dir(), f"Failed to export RF-DETR-{size}"

            # Verify inference works
            result = _openvino_predict(exported_path, sample_image, conf=0.25)
            assert result is not None

            del pt_model
