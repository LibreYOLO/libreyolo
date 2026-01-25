# YOLOX Guide

LibreYOLO includes full support for YOLOX models, providing both inference and training capabilities.

## Overview

YOLOX is an anchor-free object detection model known for its strong performance and simplicity. LibreYOLO supports all standard YOLOX sizes.

## Model Sizes

| Size | Input Size | Description |
|------|------------|-------------|
| `nano` | 416 | Smallest, fastest |
| `tiny` | 416 | Very small |
| `s` | 640 | Small |
| `m` | 640 | Medium |
| `l` | 640 | Large |
| `x` | 640 | Extra large, highest accuracy |

```{note}
YOLOX uses different size names than YOLOv8/v11. Use `nano`, `tiny`, `s`, `m`, `l`, `x` instead of `n`, `s`, `m`, `l`, `x`.
```

## Inference

### Loading a Model

```python
from libreyolo import LIBREYOLO

# Using the unified factory (auto-detects YOLOX version and size)
model = LIBREYOLO("weights/libreyoloXs.pt")

# Or use the YOLOX-specific class directly
from libreyolo import LIBREYOLOX
model = LIBREYOLOX("weights/libreyoloXs.pt", size="s")
```

### Running Inference

```python
# Single image
results = model(image="image.jpg", save=True)

# With custom thresholds
results = model(
    image="image.jpg",
    conf_thres=0.25,
    iou_thres=0.45,
    save=True
)

# Process a directory
results = model(image="path/to/images/", save=True)
```

### Auto-Download Weights

Weights are automatically downloaded from Hugging Face if not found locally:

```python
# Weights will be downloaded automatically (auto-detects size)
model = LIBREYOLO("weights/libreyoloXs.pt")
```

## Training

YOLOX models support training directly via the model's `.train()` method.

### Training from Scratch

```python
from libreyolo import LIBREYOLOX

# Create a new untrained model
model = LIBREYOLOX.new(size="s", num_classes=80)

# Train on your dataset
results = model.train(
    data="path/to/data.yaml",
    epochs=300,
    batch_size=16
)
```

### Fine-tuning a Pretrained Model

```python
from libreyolo import LIBREYOLOX

# Load pretrained weights
model = LIBREYOLOX("libreyoloXs.pt", size="s")

# Fine-tune on custom dataset
results = model.train(
    data="custom_data.yaml",
    epochs=100,
    batch_size=16
)
```

### Training Configuration

The training supports many options:

```python
results = model.train(
    data="data.yaml",           # Dataset config
    epochs=300,                 # Training epochs
    batch_size=16,              # Batch size
    imgsz=640,                  # Image size
    # Additional options via kwargs:
    # lr=0.01,                  # Learning rate
    # weight_decay=0.0005,      # Weight decay
    # warmup_epochs=5,          # Warmup epochs
    # mosaic_prob=1.0,          # Mosaic probability
    # mixup_prob=1.0,           # Mixup probability
)
```

### Training Features

The YOLOX trainer includes:

- **Mixed Precision (AMP)**: Faster training with reduced memory usage
- **EMA (Exponential Moving Average)**: Better model generalization
- **Learning Rate Scheduling**: Warmup and cosine annealing
- **Mosaic & Mixup**: Advanced data augmentation
- **TensorBoard Logging**: Training visualization
- **Checkpoint Saving**: Automatic best/last model saving

### Resume Training

```python
results = model.train(
    data="data.yaml",
    epochs=300,
    resume="runs/exp_20240101/last.pt"  # Resume from checkpoint
)
```

## Model Export

### ONNX Export

```python
model = LIBREYOLOX("libreyoloXs.pt", size="s")

# Export to ONNX
output_path = model.export(
    format="onnx",
    output_path="libreyoloXs.onnx",
    opset=11,
    simplify=True,      # Simplify with onnx-simplifier
    dynamic=False       # Dynamic input shapes
)
```

### TorchScript Export

```python
output_path = model.export(
    format="torchscript",
    output_path="libreyoloXs.pt"
)
```

## Tiled Inference

For high-resolution images, enable tiling:

```python
model = LIBREYOLOX("libreyoloXs.pt")

# Enable tiling for large images - splits into overlapping tiles
results = model(
    image="high_res_image.jpg",
    tiling=True,
    overlap_ratio=0.2
)
```

## Device Selection

```python
# Auto-detect best device (CUDA > MPS > CPU)
model = LIBREYOLOX("libreyoloXs.pt", device="auto")

# Force specific device
model = LIBREYOLOX("libreyoloXs.pt", device="cuda:0")
model = LIBREYOLOX("libreyoloXs.pt", device="cpu")
```

## Comparison with YOLOv8/v11

| Feature | YOLOX | YOLOv8/v11 |
|---------|-------|------------|
| Architecture | Anchor-free | Anchor-free |
| Backbone | CSPDarknet | C2F/C3k2 |
| Head | Decoupled | Coupled |
| Sizes | nano, tiny, s, m, l, x | n, s, m, l, x |
| Training API | `.train()` method (stable) | Under development |
| Validation | Supported | Supported |
| Export formats | ONNX, TorchScript | ONNX |

