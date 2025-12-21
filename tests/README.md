# LibreYOLO Test Suite

This directory contains comprehensive tests for LibreYOLO models.

## Test Structure

```
tests/
├── unit/                          # Fast unit tests (no weights needed)
│   ├── test_predict_contract.py  # API contract tests
│   └── test_utils.py              # Utility function tests
└── integration/                   # Integration tests (require models/weights)
    ├── test_conversion.py         # Weight conversion tests
    ├── test_inference.py          # Basic inference tests
    └── test_complete_workflow.py  # NEW: Comprehensive workflow tests
```

## Quick Start

### Run Unit Tests (Fast, No Weights Needed)
```bash
make test
```

### Run All Integration Tests (Requires Weights)
```bash
make test_integration
```

### Run Specific Test Class
```bash
uv run pytest tests/integration/test_complete_workflow.py::TestInference -v
```

### Run Specific Test
```bash
uv run pytest tests/integration/test_complete_workflow.py::TestInference::test_inference_all_models -v
```

## New Comprehensive Test Suite

The `test_complete_workflow.py` file contains extensive tests covering:

### 1. **Model Download Tests** (`TestModelDownload`)
- Downloads all 10 models from HuggingFace:
  - YOLOv8: n, s, m, l, x
  - YOLOv11: n, s, m, l, x
- Verifies each downloaded model is valid

```bash
uv run pytest tests/integration/test_complete_workflow.py::TestModelDownload -v
```

### 2. **Inference Tests** (`TestInference`)
- Tests inference on a test image for all 10 models
- Validates detection output structure
- Checks detection quality (boxes, scores, classes)

```bash
uv run pytest tests/integration/test_complete_workflow.py::TestInference -v
```

### 3. **ONNX Export Tests** (`TestONNXExport`)
- Exports all 10 models to ONNX format
- Validates ONNX model structure
- Verifies input/output shapes

```bash
uv run pytest tests/integration/test_complete_workflow.py::TestONNXExport -v
```

### 4. **ONNX Inference Tests** (`TestONNXInference`)
- Tests ONNX inference for all 10 exported models
- Validates ONNX output format
- Checks detection results

```bash
uv run pytest tests/integration/test_complete_workflow.py::TestONNXInference -v
```

### 5. **Feature Map Tests** (`TestFeatureMaps`)
- Tests feature map visualization saving
- Tests selective layer saving
- Validates feature map output structure

```bash
uv run pytest tests/integration/test_complete_workflow.py::TestFeatureMaps -v
```

### 6. **Custom Output Path Tests** (`TestCustomOutputPath`)
- Tests saving detection images to custom paths
- Validates output file creation

```bash
uv run pytest tests/integration/test_complete_workflow.py::TestCustomOutputPath -v
```

### 7. **End-to-End Workflow Test** (`TestEndToEndWorkflow`)
- Complete workflow for YOLO11n:
  1. Load model (auto-download if needed)
  2. Run inference
  3. Save detection image to custom path
  4. Export to ONNX
  5. Run ONNX inference
  6. Save feature maps

```bash
uv run pytest tests/integration/test_complete_workflow.py::TestEndToEndWorkflow -v
```

## Test Output

All test outputs are saved to:
```
tests/output/complete_workflow/
├── libreyolo8n.onnx
├── libreyolo8s.onnx
├── ...
├── libreyolo11x.onnx
├── detections_11n.jpg
├── detections_8n.jpg
└── end_to_end/
    ├── detection_result.jpg
    └── model_11n.onnx
```

## Running Tests Sequentially

For a complete test run with all models:

```bash
# 1. Run download tests first (downloads all 10 models)
uv run pytest tests/integration/test_complete_workflow.py::TestModelDownload -v

# 2. Run inference tests
uv run pytest tests/integration/test_complete_workflow.py::TestInference -v

# 3. Run ONNX export tests
uv run pytest tests/integration/test_complete_workflow.py::TestONNXExport -v

# 4. Run ONNX inference tests
uv run pytest tests/integration/test_complete_workflow.py::TestONNXInference -v

# 5. Run feature map tests
uv run pytest tests/integration/test_complete_workflow.py::TestFeatureMaps -v

# 6. Run custom output tests
uv run pytest tests/integration/test_complete_workflow.py::TestCustomOutputPath -v

# 7. Run end-to-end workflow test
uv run pytest tests/integration/test_complete_workflow.py::TestEndToEndWorkflow -v
```

Or run everything at once:
```bash
uv run pytest tests/integration/test_complete_workflow.py -v
```

## Test Duration

- **Unit tests**: ~2 seconds
- **Download tests**: ~5-30 minutes (depending on model sizes and network speed)
- **Inference tests**: ~1-2 minutes (all 10 models)
- **ONNX export tests**: ~2-3 minutes (all 10 models)
- **ONNX inference tests**: ~1-2 minutes (all 10 models)
- **Feature map tests**: ~10-15 seconds
- **End-to-end workflow**: ~30-60 seconds

## Notes

- Tests automatically skip if required weights are missing
- Downloads happen automatically via HuggingFace integration
- Each test is isolated and can run independently
- Tests use `tmp_path` fixture to avoid polluting the workspace
- All test artifacts are saved in `tests/output/`

## Parallel Execution

For faster test execution (requires `pytest-xdist`):

```bash
uv run pytest tests/integration/test_complete_workflow.py -v -n auto
```

## Troubleshooting

### Models Not Downloading
If auto-download fails, manually download models:
```bash
python examples/auto_download.py
```

### ONNX Tests Failing
Ensure ONNX Runtime is installed:
```bash
uv pip install onnxruntime
```

### Memory Issues
For large models (l, x), you may need to run tests one at a time:
```bash
uv run pytest tests/integration/test_complete_workflow.py::TestInference::test_inference_all_models[11-n] -v
```

