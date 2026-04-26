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
│   ├── test_yolo9_layers.py              # YOLOv9 layer forward passes
│   ├── test_validation_metrics.py       # IoU, mAP, precision/recall
│   └── test_yolo_coco_api.py            # COCO format conversion
└── e2e/                                 # End-to-end tests (GPU recommended)
    ├── conftest.py                      # Fixtures, helpers, RF5 infra, export helpers
    ├── configs/                         # Training config YAMLs (yolox, yolo9, rfdetr)
    ├── test_ncnn.py                     # ncnn export + inference
    ├── test_onnx.py                     # ONNX export + inference
    ├── test_openvino.py                 # OpenVINO export + inference
    ├── test_rf1_training.py             # All-models training test (marbles dataset)
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
pytest tests/unit/test_yolo9_layers.py -v
```

### E2E Tests (Export + Training)

`make test_e2e` runs all e2e tests (export, training, validation) with each test
file in its own process to avoid CUDA driver state corruption between files.

```bash
# All e2e tests (recommended)
make test_e2e

# Marker-driven subsets
pytest tests/e2e/ -v -m "e2e and onnx"
pytest tests/e2e/ -v -m "e2e and torchscript and yolonas"
pytest tests/e2e/ -v -m "e2e and yolo9 and not ncnn"

# Same marker filtering through the Makefile runner
make test_e2e MARKERS='e2e and onnx'
make test_e2e MARKERS='e2e and (onnx or torchscript) and not ncnn'

# Individual test files
pytest tests/e2e/test_onnx.py -v        # ONNX export + inference
pytest tests/e2e/test_tensorrt.py -v     # TensorRT (requires CUDA + TensorRT)
pytest tests/e2e/test_openvino.py -v     # OpenVINO
pytest tests/e2e/test_ncnn.py -v         # ncnn
pytest tests/e2e/test_rf1_training.py -v # RF1: all 15 models, 10 epochs (marbles)

# Quick tests only (smallest models)
pytest tests/e2e/ -v -k "quick" --ignore=tests/e2e/test_rf5_training.py

# RF5: config-driven benchmark (standalone CLI)
python -m tests.e2e.test_rf5_training --config yolox.yaml --size nano
python -m tests.e2e.test_rf5_training --list-configs
```

### Useful Markers

- Backends: `onnx`, `torchscript`, `tensorrt`, `trt`, `openvino`, `ncnn`
- Model families: `yolox`, `yolo9`, `yolonas`, `rfdetr`, `dfine`, `rtdetr`
- Suites: `rf1`, `rf5`, `slow`

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
- **E2E tests** (`make test_e2e`): When modifying export, training, or model code
- **RF5 training**: Before merging significant training changes
