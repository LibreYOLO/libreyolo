# LibreYOLO Documentation

**LibreYOLO** is an open-source, MIT-licensed implementation of YOLO object detection models. It provides a clean, independent codebase for training and inference.

```{note}
While this codebase is MIT licensed, pre-trained weights converted from other repositories may inherit their original licenses (often AGPL-3.0).
```

## Features

- **Supported Models:** Full support for YOLOv8, YOLOv11, and YOLOX architectures
- **Unified API:** Simple, consistent interface for loading and using different YOLO versions
- **Training Engine:** Built-in support for training models on custom datasets with Mosaic/Mixup augmentation
- **MIT License:** Permissive licensing for the codebase
- **Weight Conversion:** Tools to convert weights from Ultralytics format
- **Explainability:** Built-in CAM methods (GradCAM, EigenCAM, etc.)
- **ONNX Export:** Export models for deployment with ONNX Runtime

## Quick Start

```python
from libreyolo import LIBREYOLO

# Load a model (auto-detects v8 vs v11)
model = LIBREYOLO(model_path="weights/libreyolo8n.pt", size="n")

# Run inference
results = model(image="path/to/image.jpg", save=True)

# Access results
print(f"Found {results['num_detections']} objects")
for box, score, cls in zip(results['boxes'], results['scores'], results['classes']):
    print(f"Class {cls}: {score:.2f} at {box}")
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
training
finetuning
yolox
explainability
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

