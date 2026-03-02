# LibreYOLO Tests

## Structure

```
tests/
├── conftest.py                          # Markers (e2e, unit)
├── unit/                                # Fast, no GPU needed
│   ├── test_coco_validation.py          # COCO evaluation with mock data
│   ├── test_export.py                   # Export logic unit tests
│   ├── test_export_ncnn.py              # ncnn export unit tests
│   ├── test_factory.py                  # LibreYOLO() factory tests
│   ├── test_image_loader.py             # Image format handling
│   ├── test_results.py                  # Results class tests
│   ├── test_v9_layers.py                # YOLOv9 layer forward passes
│   ├── test_validation_metrics.py       # IoU, mAP, precision/recall
│   └── test_yolo_coco_api.py            # COCO format conversion
└── e2e/                                 # End-to-end tests (GPU recommended)
    ├── conftest.py                      # Fixtures, helpers, RF5 infra, export helpers
    ├── configs/                         # Training config YAMLs (yolox, yolo9, rfdetr)
    ├── test_ncnn.py                     # ncnn export + inference
    ├── test_onnx.py                     # ONNX export + inference
    ├── test_openvino.py                 # OpenVINO export + inference
    ├── test_rf1_training.py             # All-models training smoke test (marbles dataset)
    ├── test_rf5_training.py             # RF5 training validation (CLI tool)
    ├── test_tensorrt.py                 # TensorRT export + inference
    ├── test_torchscript.py              # TorchScript export
    └── test_val_coco128.py              # Validation sanity check (all 15 pretrained models)
```

## Running Tests

### Unit Tests (Fast, CPU)

```bash
pytest tests/unit/ -v

# Run specific test
pytest tests/unit/test_v9_layers.py -v
```

### E2E Export Tests

Export tests validate the full export + inference pipeline per backend.
Each backend requires its runtime to be installed.

```bash
# ONNX (always available)
pytest tests/e2e/test_onnx.py -v

# TensorRT (requires CUDA + TensorRT)
pytest tests/e2e/test_tensorrt.py -v

# OpenVINO
pytest tests/e2e/test_openvino.py -v

# ncnn
pytest tests/e2e/test_ncnn.py -v

# Quick tests only (smallest models)
pytest tests/e2e/ -v -k "quick" --ignore=tests/e2e/test_rf1_training.py --ignore=tests/e2e/test_rf5_training.py

# Skip slow tests
pytest tests/e2e/ -v -m "not slow"
```

### E2E Training Tests

```bash
# RF1: smoke test on marbles dataset (all 15 models, 10 epochs each)
pytest tests/e2e/test_rf1_training.py -v -s

# RF5: config-driven benchmark (standalone CLI)
python -m tests.e2e.test_rf5_training --config yolox.yaml --size nano
python -m tests.e2e.test_rf5_training --config yolox.yaml --size nano --dataset aquarium-qlnqy
python -m tests.e2e.test_rf5_training --list-configs
```

## RF5 - Training Validation Suite

RF5 is a minimal subset of Roboflow100 designed to quickly verify training code works:

| Dataset | Classes | Train | Purpose |
|---------|---------|-------|---------|
| bacteria-ptywi | 1 | 30 | Tiny dataset, tiny objects, dense |
| circuit-elements | 45 | 672 | Many classes, extremely dense |
| aquarium-qlnqy | 7 | 448 | Balanced baseline |
| aerial-cows | 1 | 1084 | Small objects, aerial imagery |
| road-signs-6ih4y | 21 | 1376 | Large objects, sparse |

**Total: ~3,600 training images** (vs ~180k for full RF100)

### When to Run

- **Unit tests**: Always, before any commit
- **Export tests**: When modifying export code or inference backends
- **RF1 training**: When changing training code, loss functions, or model architectures
- **RF5 training**: Before merging significant training changes
