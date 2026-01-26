# LibreYOLO Tests

## Structure

```
tests/
├── unit/                              # Fast, no weights needed
│   ├── test_v9_layers.py              # YOLOv9 layer forward passes
│   ├── test_validation_metrics.py     # IoU, mAP, precision/recall
│   ├── test_image_loader.py           # Image format handling
│   └── test_yolo_coco_api.py          # COCO format conversion
└── integration/                       # Requires datasets
    └── test_coco_validation.py        # COCO evaluation pipeline
```

## Running

```bash
# Unit tests (default)
uv run pytest -m unit

# Integration tests
uv run pytest -m integration

# Everything
uv run pytest -m "unit or integration"
```
