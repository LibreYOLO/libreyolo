# LibreYOLO

LibreYOLO is an MIT-licensed object detection library supporting YOLOv8, YOLOv9, YOLOv11, YOLOX, and RF-DETR models.

## Installation

```bash
git clone https://github.com/Libre-YOLO/libreyolo.git
cd libreyolo
pip install -e .
```

## Quick Example

```python
from libreyolo import LIBREYOLO

# Load model (auto-detects version and size)
model = LIBREYOLO("weights/libreyolo8n.pt")

# Run inference
results = model(image="image.jpg", save=True)
print(f"Found {results['num_detections']} objects")

# Validate on dataset
val_results = model.val(data="coco128.yaml")
print(f"mAP50-95: {val_results['metrics/mAP50-95']:.3f}")
```

## Supported Models

| Model | Sizes | Training |
|-------|-------|----------|
| YOLOv8 | n, s, m, l, x | No |
| YOLOv9 | t, s, m, c | No |
| YOLOv11 | n, s, m, l, x | No |
| YOLOX | nano, tiny, s, m, l, x | Yes |
| RF-DETR | n, s, b, m, l | Yes |

```{toctree}
:maxdepth: 2
:caption: Guide

getting-started
inference
validation
training
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```
