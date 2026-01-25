# LibreYOLO Documentation

**LibreYOLO** is an open-source, MIT-licensed implementation of YOLO object detection models. It provides a clean, independent codebase for training and inference.

```{note}
While this codebase is MIT licensed, pre-trained weights converted from other repositories may inherit their original licenses (often AGPL-3.0).
```

## Features

- **Supported Models:** YOLOv8, YOLOv9, YOLOv11, YOLOX, and RF-DETR (Detection Transformer)
- **Auto-Detection:** Automatic model version and size detection from weights
- **Unified API:** Simple, consistent interface for all model architectures
- **Validation:** COCO-style evaluation with mAP metrics
- **Training:** YOLOX training support (other models under development)
- **ONNX Export:** Export models for deployment with ONNX Runtime
- **MIT License:** Permissive licensing for the codebase
- **Weight Conversion:** Tools to convert weights from Ultralytics format

## Quick Start

```python
from libreyolo import LIBREYOLO

# Load a model (auto-detects version and size)
model = LIBREYOLO(model_path="weights/libreyolo8n.pt")

# Run inference
results = model(image="path/to/image.jpg", save=True)

# Access results
print(f"Found {results['num_detections']} objects")
for box, score, cls in zip(results['boxes'], results['scores'], results['classes']):
    print(f"Class {cls}: {score:.2f} at {box}")

# Validate on COCO dataset
val_results = model.val(data="coco128.yaml")
print(f"mAP50-95: {val_results['metrics/mAP50-95']:.3f}")
```

## Installation

```bash
git clone https://github.com/Libre-YOLO/libreyolo.git
cd libreyolo
pip install -e .
```

```{toctree}
:maxdepth: 2
:caption: User Guide

getting-started
inference
validation
training
yolox
finetuning
```

```{toctree}
:maxdepth: 2
:caption: Technical Reference

model-architecture
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`

