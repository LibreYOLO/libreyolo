# Getting Started

This guide will help you get up and running with LibreYOLO.

## Installation

### Prerequisites

- Python 3.10+
- PyTorch (with CUDA support for GPU inference)

### Install from Source

```bash
git clone https://github.com/Libre-YOLO/libreyolo.git
cd libreyolo

# Recommended: install with all extras
uv sync --all-extras --group dev

# Or with pip
pip install -e .[convert,onnx,dev]
```

### Minimal Installation

```bash
pip install -e .
```

## Download Weights

LibreYOLO can automatically download weights from Hugging Face:

```python
from libreyolo import LIBREYOLO

# Weights are auto-downloaded if not found locally
model = LIBREYOLO(model_path="weights/libreyolo8n.pt", size="n")
```

Or manually download from the [Hugging Face repository](https://huggingface.co/Libre-YOLO).

## Your First Detection

```python
from libreyolo import LIBREYOLO

# Initialize model
model = LIBREYOLO(model_path="weights/libreyolo8n.pt", size="n")

# Run inference on an image
results = model(image="path/to/image.jpg", save=True)

# Print results
print(f"Found {results['num_detections']} objects")
for i, (box, score, cls) in enumerate(zip(
    results['boxes'], results['scores'], results['classes']
)):
    print(f"  {i+1}. Class {cls}: {score:.2f} at {box}")
```

## Model Sizes

LibreYOLO supports multiple model sizes for each architecture:

### YOLOv8 / YOLOv11

| Size | Parameter | Speed | Accuracy |
|------|-----------|-------|----------|
| `n` (nano) | Smallest | Fastest | Lower |
| `s` (small) | Small | Fast | Good |
| `m` (medium) | Medium | Balanced | Better |
| `l` (large) | Large | Slower | High |
| `x` (xlarge) | Largest | Slowest | Highest |

```python
# Use different sizes
model_nano = LIBREYOLO("weights/libreyolo8n.pt", size="n")
model_large = LIBREYOLO("weights/libreyolo8l.pt", size="l")
```

### YOLOv7

| Size | Description |
|------|-------------|
| `base` | Standard YOLOv7 |
| `tiny` | Lightweight variant |

```python
from libreyolo import LIBREYOLO7
model = LIBREYOLO7("weights/libreyolo7.pt", size="base")
```

### YOLOv9

| Size | Description |
|------|-------------|
| `t` | Tiny |
| `s` | Small |
| `m` | Medium |
| `c` | Compact (largest) |

```python
from libreyolo import LIBREYOLO9
model = LIBREYOLO9("weights/libreyolo9s.pt", size="s")
```

### YOLOX

| Size | Input Size | Description |
|------|------------|-------------|
| `nano` | 416 | Smallest |
| `tiny` | 416 | Very small |
| `s` | 640 | Small |
| `m` | 640 | Medium |
| `l` | 640 | Large |
| `x` | 640 | Extra large |

```python
# YOLOX uses different size names
model = LIBREYOLO("weights/libreyoloXs.pt", size="s")
```

### YOLO-RD

| Size | Description |
|------|-------------|
| `c` | Based on YOLOv9-c with DConv |

```python
from libreyolo import LIBREYOLORD
model = LIBREYOLORD("weights/libreyolord.pt", size="c")
```

### RT-DETR

| Size | Backbone | Description |
|------|----------|-------------|
| `s` | ResNet18-vd | Small |
| `ms` | ResNet34-vd | Medium* |
| `m` | ResNet50-vd_m | Medium |
| `l` | ResNet50-vd | Large |
| `x` | ResNet101-vd | Extra large |

```python
from libreyolo import LIBREYOLORTDETR
model = LIBREYOLORTDETR("weights/rtdetr_l.pt", size="l")
```

## Next Steps

- {doc}`inference` - Learn about inference options
- {doc}`training` - Train on custom datasets
- {doc}`yolox` - YOLOX-specific features and training
- {doc}`explainability` - Visualize model attention with CAM methods

