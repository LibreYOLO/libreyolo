# Inference Guide

LibreYOLO provides a flexible API for running object detection inference.

## Basic Usage

```python
from libreyolo import LIBREYOLO

# Auto-detect model version and size
model = LIBREYOLO(model_path="weights/libreyolo8n.pt")
results = model(image="image.jpg")
```

## Input Types

LibreYOLO accepts multiple input types:

```python
# File path
results = model(image="path/to/image.jpg")

# URL (http, https, s3, gs)
results = model(image="https://example.com/image.jpg")

# PIL Image
from PIL import Image
img = Image.open("image.jpg")
results = model(image=img)

# NumPy array (RGB or BGR)
import numpy as np
img_array = np.array(img)
results = model(image=img_array, color_format="rgb")

# OpenCV (BGR by default)
import cv2
img_cv = cv2.imread("image.jpg")
results = model(image=img_cv, color_format="bgr")

# PyTorch Tensor (CHW or NCHW)
import torch
tensor = torch.randn(3, 640, 640)
results = model(image=tensor)

# Raw bytes
with open("image.jpg", "rb") as f:
    img_bytes = f.read()
results = model(image=img_bytes)

# BytesIO object
import io
buffer = io.BytesIO(img_bytes)
results = model(image=buffer)

# pathlib.Path
from pathlib import Path
results = model(image=Path("path/to/image.jpg"))

# Directory of images
results = model(image="path/to/images/")  # Returns list of results
```

## Inference Parameters

```python
results = model(
    image="image.jpg",
    save=True,              # Save annotated image
    output_path="result/",  # Custom save location
    conf_thres=0.25,        # Confidence threshold
    iou_thres=0.45,         # NMS IoU threshold
)
```

## Result Format

The inference returns a dictionary:

```python
{
    "boxes": [[x1, y1, x2, y2], ...],  # Bounding boxes in xyxy format
    "scores": [0.95, 0.87, ...],        # Confidence scores
    "classes": [0, 2, ...],             # Class IDs (COCO classes)
    "num_detections": 5,                # Total detections
    "source": "image.jpg",              # Input source
    "saved_path": "runs/detections/..." # If save=True
}
```

## Tiled Inference

For high-resolution images, use tiled inference:

```python
model = LIBREYOLO("weights/libreyolo8n.pt")

# Enable tiling for large images - splits into overlapping tiles
results = model(
    image="high_res_image.jpg",
    tiling=True,         # Enable tiled inference
    overlap_ratio=0.2    # Tile overlap (optional, default=0.2)
)
```

## Batch Processing

Process a directory of images:

```python
# Process all images in a directory
results = model(
    image="path/to/images/",
    save=True,
    batch_size=4  # Process 4 images at a time
)

# results is a list of detection dictionaries
for r in results:
    print(f"{r['source']}: {r['num_detections']} detections")
```

## ONNX Inference

Export and use ONNX models:

```python
# Export to ONNX
model.export(output_path="model.onnx")

# Load ONNX model
from libreyolo import LIBREYOLOOnnx
onnx_model = LIBREYOLOOnnx("model.onnx")
results = onnx_model(image="image.jpg")
```

## Device Selection

```python
# Auto-detect best device (CUDA > MPS > CPU)
model = LIBREYOLO("weights/libreyolo8n.pt", device="auto")

# Force specific device
model = LIBREYOLO("weights/libreyolo8n.pt", device="cuda:0")
model = LIBREYOLO("weights/libreyolo8n.pt", device="cpu")
```

