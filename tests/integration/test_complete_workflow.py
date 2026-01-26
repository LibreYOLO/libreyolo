"""
Comprehensive integration tests for LibreYOLO covering:
- Downloading all 10 models from HuggingFace
- Inference with test image
- Export to ONNX (EXAMPLE CODE - NOT YET IMPLEMENTED)
- ONNX inference (EXAMPLE CODE - NOT YET IMPLEMENTED)
- Feature map saving with custom output paths
"""

import shutil
from pathlib import Path
import pytest
import torch
import numpy as np

from libreyolo import LIBREYOLO

pytestmark = pytest.mark.integration

# ==============================================================================
# NOTE: ONNX tests are currently SKIPPED as ONNX export is not yet fully
# implemented in the library. The test code below serves as example/template
# code for when ONNX support is added in the future.
# ==============================================================================
ONNX_AVAILABLE = False
try:
    import onnxruntime as ort  # type: ignore  # Optional dependency for future ONNX support
    ONNX_AVAILABLE = True
except ImportError:
    pass


# All model combinations (4 YOLOv9)
ALL_MODELS = [
    # YOLOv9 variants (anchor-free with DFL)
    ("9", "t"),
    ("9", "s"),
    ("9", "m"),
    ("9", "c"),
]

# Subset of quick models for faster testing
QUICK_MODELS = [
    ("9", "t"),
]

# Model filename patterns for different versions
MODEL_FILENAME_PATTERNS = {
    "9": "yolov9{size}.pt",
}

def get_model_filename(version: str, size: str) -> str:
    """Get the correct filename for a model version and size."""
    if version == "9":
        return f"yolov9{size}.pt"
    else:
        return f"libreyolo{version}{size}.pt"


@pytest.fixture(scope="module")
def test_output_dir(project_root):
    """Create a clean output directory for this test module."""
    output_dir = project_root / "tests" / "output" / "complete_workflow"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


class TestModelDownload:
    """Test downloading all models from HuggingFace."""

    @pytest.mark.parametrize("version, size", ALL_MODELS)
    def test_download_model(self, version, size, weights_dir):
        """Test that each model can be downloaded from HuggingFace."""
        filename = get_model_filename(version, size)
        weight_file = weights_dir / filename

        # If it already exists, we're good
        if weight_file.exists():
            pytest.skip(f"Weights {weight_file.name} already exist")
            return

        # Try to download using the LIBREYOLO factory with auto-download
        try:
            model = LIBREYOLO(str(weight_file), size=size)
            assert weight_file.exists(), f"Download failed, {weight_file} not found"

            # Verify it's a valid torch file
            state_dict = torch.load(str(weight_file), map_location='cpu', weights_only=False)
            assert len(state_dict) > 0, "Downloaded weights file is empty"
            print(f"✓ Successfully downloaded {weight_file.name}")

        except Exception as e:
            pytest.fail(f"Failed to download model {version}{size}: {e}")


class TestInference:
    """Test inference with all models."""

    @pytest.mark.parametrize("version, size", ALL_MODELS)
    def test_inference_all_models(self, version, size, weights_dir, test_image, test_output_dir):
        """Test inference on a test image for each model."""
        filename = get_model_filename(version, size)
        weight_file = weights_dir / filename

        if not weight_file.exists():
            pytest.skip(f"Weights {weight_file.name} not found. Run download tests first.")

        # Load model
        model = LIBREYOLO(str(weight_file), size=size)

        # Run inference
        results = model.predict(test_image, save=False, conf_thres=0.25, iou_thres=0.45)

        # Validate results structure
        assert results is not None, "Results should not be None"
        assert "boxes" in results, "Results should contain 'boxes'"
        assert "scores" in results, "Results should contain 'scores'"
        assert "classes" in results, "Results should contain 'classes'"
        assert "num_detections" in results, "Results should contain 'num_detections'"

        # Validate data types
        assert isinstance(results["num_detections"], int), "num_detections should be int"
        num_dets = results["num_detections"]

        if num_dets > 0:
            assert len(results["boxes"]) == num_dets, "boxes length should match num_detections"
            assert len(results["scores"]) == num_dets, "scores length should match num_detections"
            assert len(results["classes"]) == num_dets, "classes length should match num_detections"

            # Validate ranges
            assert all(0 <= score <= 1 for score in results["scores"]), "Scores should be in [0, 1]"
            assert all(0 <= cls < 80 for cls in results["classes"]), "Classes should be in [0, 79] for COCO"

        print(f"✓ Inference successful for {version}{size}: {num_dets} detections")


class TestONNXExport:
    """
    ⚠️  WARNING: ONNX EXPORT IS NOT YET IMPLEMENTED IN LIBREYOLO ⚠️
    
    These tests are SKIPPED and serve as example/template code for when
    ONNX export functionality is added to the library in the future.
    
    The test code below shows the expected API and behavior once ONNX
    support is properly implemented.
    """
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="onnxruntime not installed")
    @pytest.mark.skip(reason="⚠️  ONNX export not yet implemented in LibreYOLO - this is example code for future implementation")
    @pytest.mark.parametrize("version, size", ALL_MODELS)
    def test_export_to_onnx(self, version, size, weights_dir, test_output_dir):
        """Test exporting each model to ONNX format."""
        filename = get_model_filename(version, size)
        weight_file = weights_dir / filename

        if not weight_file.exists():
            pytest.skip(f"Weights {weight_file.name} not found. Run download tests first.")
        
        # Define output path in test output directory
        onnx_output = test_output_dir / f"libreyolo{version}{size}.onnx"
        
        # Load model and export
        model = LIBREYOLO(str(weight_file), size=size)
        exported_path = model.export(output_path=str(onnx_output), input_size=640, opset=12)
        
        # Verify export
        assert Path(exported_path).exists(), f"ONNX file not created at {exported_path}"
        assert onnx_output.exists(), "ONNX file not found at expected location"
        
        # Verify it's a valid ONNX file by loading it
        if ONNX_AVAILABLE:
            try:
                session = ort.InferenceSession(str(onnx_output), providers=['CPUExecutionProvider'])
                
                # Check input/output shapes
                assert len(session.get_inputs()) == 1, "Should have 1 input"
                assert len(session.get_outputs()) == 1, "Should have 1 output"
                
                input_shape = session.get_inputs()[0].shape
                assert input_shape[1] == 3, "Input should have 3 channels"
                
                print(f"✓ Successfully exported {version}{size} to ONNX")
                
            except Exception as e:
                pytest.fail(f"Failed to load exported ONNX file: {e}")


class TestONNXInference:
    """
    ⚠️  WARNING: ONNX INFERENCE IS NOT YET IMPLEMENTED IN LIBREYOLO ⚠️
    
    These tests are SKIPPED and serve as example/template code for when
    ONNX inference functionality is added to the library in the future.
    
    The test code below shows the expected API and behavior once ONNX
    support is properly implemented.
    """
    
    @staticmethod
    def preprocess_image_for_onnx(image_path, input_size=640):
        """Preprocess image for ONNX inference (matching LibreYOLO preprocessing)."""
        import cv2  # type: ignore  # Optional dependency for ONNX preprocessing
        
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        h0, w0 = img.shape[:2]
        r = min(input_size / h0, input_size / w0)
        img_resized = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=cv2.INTER_LINEAR)
        
        h, w = img_resized.shape[:2]
        dw, dh = input_size - w, input_size - h
        dw /= 2
        dh /= 2
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        blob = cv2.dnn.blobFromImage(
            img_padded, 1/255.0, (input_size, input_size), 
            swapRB=True, crop=False
        )
        
        return blob
    
    @pytest.mark.skipif(not ONNX_AVAILABLE, reason="onnxruntime not installed")
    @pytest.mark.skip(reason="⚠️  ONNX inference not yet implemented in LibreYOLO - this is example code for future implementation")
    @pytest.mark.parametrize("version, size", ALL_MODELS)
    def test_onnx_inference(self, version, size, test_output_dir, test_image):
        """Test ONNX inference for each exported model."""
        onnx_file = test_output_dir / f"libreyolo{version}{size}.onnx"
        
        if not onnx_file.exists():
            pytest.skip(f"ONNX file {onnx_file.name} not found. Run export tests first.")
        
        if not ONNX_AVAILABLE:
            pytest.skip("onnxruntime not installed")
        
        # Load ONNX model
        session = ort.InferenceSession(str(onnx_file), providers=['CPUExecutionProvider'])
        
        # Preprocess image
        input_blob = self.preprocess_image_for_onnx(test_image, input_size=640)
        
        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        outputs = session.run([output_name], {input_name: input_blob})
        
        # Validate output
        assert outputs is not None, "ONNX inference returned None"
        assert len(outputs) == 1, "Should have 1 output tensor"
        
        output_tensor = outputs[0]
        assert output_tensor.shape[0] == 1, "Batch size should be 1"
        assert output_tensor.shape[2] == 84, "Output should have 84 channels (4 bbox + 80 classes)"
        
        # Extract boxes and scores
        batch_output = output_tensor[0]  # (N, 84)
        boxes = batch_output[:, :4]
        scores = batch_output[:, 4:]
        
        # Apply confidence threshold
        max_scores = np.max(scores, axis=1)
        confident_detections = np.sum(max_scores > 0.25)
        
        print(f"✓ ONNX inference successful for {version}{size}: {confident_detections} confident detections")


class TestCustomOutputPath:
    """Test saving detection results with custom output paths."""
    
    @pytest.mark.parametrize("version, size", [("9", "t")])
    def test_inference_with_custom_output_path(self, version, size, weights_dir, test_image, test_output_dir):
        """Test saving detection images to custom output paths."""
        filename = get_model_filename(version, size)
        weight_file = weights_dir / filename

        if not weight_file.exists():
            pytest.skip(f"Weights {weight_file.name} not found. Run download tests first.")
        
        # Define custom output path
        custom_output = test_output_dir / f"detections_{version}{size}.jpg"
        
        # Load model and run inference with save=True and custom output_path
        model = LIBREYOLO(str(weight_file), size=size)
        results = model.predict(test_image, save=True, output_path=str(custom_output))
        
        # Verify results
        assert results is not None, "Inference failed"
        
        # Verify output file was created at custom path
        assert custom_output.exists(), f"Output image not saved at custom path: {custom_output}"
        
        # Verify it's a valid image file
        assert custom_output.stat().st_size > 0, "Output image file is empty"
        
        print(f"✓ Detection image saved to custom path: {custom_output}")


class TestEndToEndWorkflow:
    """
    End-to-end workflow test combining multiple operations.
    
    NOTE: ONNX steps (4-5) are currently skipped as ONNX support is not yet
    fully implemented. They will be enabled once ONNX export/inference is added.
    """
    
    def test_full_workflow_single_model(self, weights_dir, test_image, test_output_dir, tmp_path, monkeypatch):
        """
        Complete workflow test for a single model (YOLOv9t):
        1. Load model (with auto-download if needed)
        2. Run inference
        3. Save detection image to custom path
        """
        version, size = "9", "t"
        weight_file = weights_dir / get_model_filename(version, size)

        if not weight_file.exists():
            pytest.skip(f"Weights {weight_file.name} not found.")

        # Use a dedicated subdirectory for this test
        workflow_dir = test_output_dir / "end_to_end"
        workflow_dir.mkdir(exist_ok=True)

        # Step 1: Load model (will auto-download if needed)
        print("\n[1/3] Loading model...")
        model = LIBREYOLO(str(weight_file), size=size)
        assert weight_file.exists(), "Model weights not found/downloaded"

        # Step 2: Run inference
        print("[2/3] Running inference...")
        results = model.predict(test_image, save=False)
        assert results is not None
        assert results["num_detections"] >= 0
        print(f"      Found {results['num_detections']} detections")

        # Step 3: Save detection image to custom path
        print("[3/3] Saving detection image...")
        detection_output = workflow_dir / "detection_result.jpg"
        results = model.predict(test_image, save=True, output_path=str(detection_output))
        assert detection_output.exists(), "Detection image not saved"

        print(f"\n End-to-end workflow completed successfully!")
        print(f"  - Detections: {results['num_detections']}")
        print(f"  - Detection image: {detection_output}")


class TestNewModelVersions:
    """Test specific functionality for new model versions (v9, v7, rd)."""

    def test_v9_anchor_free_detection(self, weights_dir, test_image):
        """Test YOLOv9 anchor-free detection with DFL."""
        filename = get_model_filename("9", "t")
        weight_file = weights_dir / filename

        if not weight_file.exists():
            pytest.skip(f"Weights {weight_file.name} not found.")

        model = LIBREYOLO(str(weight_file), size="t")
        results = model.predict(test_image, save=False, conf_thres=0.25)

        assert results is not None
        assert "boxes" in results
        assert model.version == "9"
        print(f"✓ YOLOv9 anchor-free detection: {results['num_detections']} detections")

    @pytest.mark.parametrize("version, size", QUICK_MODELS)
    def test_auto_size_detection(self, version, size, weights_dir, test_image):
        """Test automatic size detection (new feature - no size parameter needed)."""
        filename = get_model_filename(version, size)
        weight_file = weights_dir / filename

        if not weight_file.exists():
            pytest.skip(f"Weights {weight_file.name} not found.")

        # Load model WITHOUT specifying size - should auto-detect
        model = LIBREYOLO(str(weight_file))

        # Verify size was correctly detected
        assert model.size == size, f"Auto-detected size {model.size} != expected {size}"
        assert model.version == version, f"Version {model.version} != expected {version}"

        # Verify model works correctly with auto-detected size
        results = model.predict(test_image, save=False, conf_thres=0.25)
        assert results is not None
        assert "boxes" in results

        print(f"✓ Auto-detected size={size} for {version}{size}: {results['num_detections']} detections")

    @pytest.mark.parametrize("version, size", QUICK_MODELS)
    def test_version_detection(self, version, size, weights_dir, test_image):
        """Test that version is correctly detected from weights."""
        filename = get_model_filename(version, size)
        weight_file = weights_dir / filename

        if not weight_file.exists():
            pytest.skip(f"Weights {weight_file.name} not found.")

        model = LIBREYOLO(str(weight_file), size=size)
        assert model.version == version, f"Expected version {version}, got {model.version}"
        print(f"✓ Version detection correct for {version}{size}")


class TestModelComparison:
    """Compare behavior across different model versions."""

    def test_all_models_return_same_schema(self, weights_dir, test_image):
        """Test that all model versions return the same detection schema."""
        expected_keys = {"boxes", "scores", "classes", "num_detections"}

        for version, size in QUICK_MODELS:
            filename = get_model_filename(version, size)
            weight_file = weights_dir / filename

            if not weight_file.exists():
                continue

            model = LIBREYOLO(str(weight_file), size=size)
            results = model.predict(test_image, save=False, conf_thres=0.25)

            assert results is not None, f"{version}{size} returned None"
            assert set(results.keys()) >= expected_keys, f"{version}{size} missing required keys"

        print("✓ All models return consistent detection schema")

