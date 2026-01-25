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
# Model version and size are auto-detected from the weights
model = LIBREYOLO(model_path="weights/libreyolo8n.pt")
```

Or manually download from the [Hugging Face repository](https://huggingface.co/Libre-YOLO).

## Your First Detection

```python
from libreyolo import LIBREYOLO

# Initialize model (auto-detects version and size)
model = LIBREYOLO(model_path="weights/libreyolo8n.pt")

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
# Auto-detection (recommended)
model_nano = LIBREYOLO("weights/libreyolo8n.pt")
model_large = LIBREYOLO("weights/libreyolo8l.pt")

# Or specify size explicitly (backward compatible)
model_nano = LIBREYOLO("weights/libreyolo8n.pt", size="n")
model_large = LIBREYOLO("weights/libreyolo8l.pt", size="l")
```

### YOLOv9

| Size | Description |
|------|-------------|
| `t` | Tiny |
| `s` | Small |
| `m` | Medium |
| `c` | Compact (largest) |

```python
# Auto-detection (recommended)
model = LIBREYOLO("weights/libreyolo9s.pt")

# Or use version-specific class
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
# Auto-detection (recommended)
model = LIBREYOLO("weights/libreyoloXs.pt")

# Or specify size explicitly
model = LIBREYOLO("weights/libreyoloXs.pt", size="s")
```

### RF-DETR (Detection Transformer)

| Size | Description |
|------|-------------|
| `n` | Nano |
| `s` | Small |
| `b` | Base |
| `m` | Medium |
| `l` | Large |

```python
# Auto-detection (recommended)
model = LIBREYOLO("weights/libreyolorfdetrn.pt")

# Or use version-specific class
from libreyolo import LIBREYOLORFDETR
model = LIBREYOLORFDETR("weights/libreyolorfdetrn.pt", size="n")
```

RF-DETR uses a Detection Transformer architecture with DINOv2 backbone and deformable attention. It requires ImageNet normalization preprocessing (handled automatically).

## Next Steps

- {doc}`inference` - Learn about inference options
- {doc}`validation` - Validate models on COCO datasets
- {doc}`training` - Train on custom datasets
- {doc}`yolox` - YOLOX-specific features and training

