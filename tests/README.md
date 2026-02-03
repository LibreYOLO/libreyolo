# LibreYOLO Tests

## Structure

```
tests/
├── unit/                              # Fast, no weights needed
│   ├── test_v9_layers.py              # YOLOv9 layer forward passes
│   ├── test_validation_metrics.py     # IoU, mAP, precision/recall
│   ├── test_image_loader.py           # Image format handling
│   └── test_yolo_coco_api.py          # COCO format conversion
├── integration/                       # Requires datasets
│   └── test_coco_validation.py        # COCO evaluation pipeline
└── e2e/                               # End-to-end tests (GPU required)
    ├── rf5_datasets.py                # RF5 dataset definitions
    └── test_rf5_training.py           # RF5 training validation
```

## Running Tests

### Unit Tests (Fast, CPU)

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific test
pytest tests/unit/test_v9_layers.py -v
```

### Integration Tests

```bash
# Run integration tests
pytest tests/integration/ -v
```

### E2E Tests (GPU Required)

E2E tests validate complete training pipelines. They require:
- GPU with CUDA support
- RF5 datasets (auto-downloaded to ~/.cache/rf100/)

```bash
# Quick smoke test (1 model, 1 dataset)
pytest tests/e2e/test_rf5_training.py::test_rf5_quick -v -s

# Test specific model family on all RF5 datasets
pytest tests/e2e/test_rf5_training.py -v -s -k "yolox"

# Run as standalone script with more control
python -m tests.e2e.test_rf5_training --model yolox-nano --epochs 5

# Test all models (full matrix - takes longer)
python -m tests.e2e.test_rf5_training --all-models --epochs 3
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

### When to Run RF5

Run RF5 E2E tests when:
- Modifying training code (`libreyolo/training/`)
- Adding new model architectures
- Changing loss functions or optimizers
- Before merging significant changes

### CLI Options

```bash
python -m tests.e2e.test_rf5_training --help

# List available models
python -m tests.e2e.test_rf5_training --list-models

# List RF5 datasets
python -m tests.e2e.test_rf5_training --list-datasets

# Run specific combination
python -m tests.e2e.test_rf5_training --model yolox-nano --dataset aquarium-qlnqy --epochs 10
```
