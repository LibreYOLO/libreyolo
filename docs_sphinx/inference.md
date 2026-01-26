# Inference

## Basic Usage

```python
from libreyolo import LIBREYOLO

model = LIBREYOLO("weights/libreyoloXs.pt")
results = model(image="image.jpg")
```

## Input Types

```python
# File path
results = model(image="path/to/image.jpg")

# URL
results = model(image="https://example.com/image.jpg")

# PIL Image
from PIL import Image
results = model(image=Image.open("image.jpg"))

# NumPy array
import numpy as np
results = model(image=np.array(...), color_format="rgb")

# OpenCV (BGR)
import cv2
results = model(image=cv2.imread("image.jpg"), color_format="bgr")

# PyTorch tensor
import torch
results = model(image=torch.randn(3, 640, 640))

# Directory (batch)
results = model(image="path/to/images/")  # Returns list
```

## Parameters

```python
results = model(
    image="image.jpg",
    save=True,              # Save annotated image
    output_path="results/", # Save location
    conf_thres=0.25,        # Confidence threshold
    iou_thres=0.45,         # NMS IoU threshold
)
```

## Result Format

```python
{
    "boxes": [[x1, y1, x2, y2], ...],  # Bounding boxes (xyxy format)
    "scores": [0.95, 0.87, ...],        # Confidence scores
    "classes": [0, 2, ...],             # Class IDs (0-indexed)
    "num_detections": 5,
    "source": "image.jpg",              # Input source path
    "saved_path": "runs/..."            # Output path (if save=True)
}
```

For tiled inference, additional fields are included:

```python
{
    # ... standard fields ...
    "tiled": True,
    "num_tiles": 9,
    "tiles_path": "runs/tiled_detections/.../tiles",
    "grid_path": "runs/tiled_detections/.../grid_visualization.jpg"
}
```

## Tiled Inference

For high-resolution images:

```python
results = model(
    image="large_image.jpg",
    tiling=True,
    overlap_ratio=0.2
)
```

## Batch Processing

```python
results = model(
    image="path/to/images/",
    save=True,
    batch_size=4
)

for r in results:
    print(f"{r['source']}: {r['num_detections']} detections")
```

## ONNX Export & Inference

```python
# Export to ONNX
model.export(output_path="model.onnx")

# Load and run ONNX model
from libreyolo import LIBREYOLOOnnx
onnx_model = LIBREYOLOOnnx("model.onnx")
results = onnx_model(image="image.jpg")
```

YOLOX models support additional export options:

```python
from libreyolo import LIBREYOLOX

model = LIBREYOLOX("weights/libreyoloXs.pt", size="s")

# ONNX with options
model.export(
    format="onnx",
    output_path="model.onnx",
    opset=11,
    simplify=True,   # Simplify ONNX graph
    dynamic=False    # Dynamic input shapes
)

# TorchScript
model.export(format="torchscript", output_path="model.pt")
```
