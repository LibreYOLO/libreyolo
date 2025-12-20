# LibreYOLO User Guide

## Installation

### Prerequisites
- Python 3.10+
- PyTorch

### Install from Source

```bash
git clone https://github.com/yourusername/libreyolo.git
cd libreyolo
pip install -e .
```

## Basic Usage

LibreYOLO supports both YOLOv8 and YOLOv11 architectures.

### Python API

You can use the model directly in your Python scripts.

#### YOLOv8 Example

```python
from libreyolo import LIBREYOLO8

# Initialize model
# size can be "n", "s", "m", "l", "x"
model = LIBREYOLO8(model_path="weights/libreyolo8n.pt", size="n")

# Run inference
# image can be a file path or numpy array
results = model(image="path/to/image.jpg", save=True)

print(f"Found {results['num_detections']} detections")
print(f"Saved result to: {results.get('saved_path')}")
```

#### YOLOv11 Example

```python
from libreyolo import LIBREYOLO11

# Initialize model
model = LIBREYOLO11(model_path="weights/libreyolo11n.pt", size="n")

# Run inference
results = model(image="path/to/image.jpg", save=True)
```

### Inference Options

The `model()` call (or `model.predict()`) accepts the following arguments:

- `image`: Path to image file or numpy array (required).
- `save`: Boolean, whether to save the annotated image (default: `False`).
- `conf_threshold`: Confidence threshold for detections (default: `0.25`).
- `iou_threshold`: NMS IoU threshold (default: `0.45`).

The returned `results` dictionary contains:
- `boxes`: List of detected bounding boxes `[x1, y1, x2, y2]`.
- `scores`: List of confidence scores.
- `class_ids`: List of class indices.
- `num_detections`: Total number of detections.
- `saved_path`: Path to the saved image (if `save=True`).

