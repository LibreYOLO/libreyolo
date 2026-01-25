# Getting Started

## Installation

```bash
git clone https://github.com/Libre-YOLO/libreyolo.git
cd libreyolo
pip install -e .
```

With optional extras:

```bash
# ONNX export support
pip install -e .[onnx]

# Weight conversion from Ultralytics
pip install -e .[convert]

# RF-DETR model support
pip install -e .[rfdetr]

# Multiple extras
pip install -e .[onnx,rfdetr]
```

For development:

```bash
# Using pip
pip install -e .[onnx,convert] && pip install pytest ruff jupyterlab

# Using uv (recommended)
uv sync --extra onnx --extra convert --group dev
```

## Loading Models

```python
from libreyolo import LIBREYOLO

# Auto-detects model version and size from weights
model = LIBREYOLO("weights/libreyolo8n.pt")
```

Weights are auto-downloaded from [Hugging Face](https://huggingface.co/Libre-YOLO) if not found locally.

## Model Classes

Use the unified factory (recommended):

```python
from libreyolo import LIBREYOLO
model = LIBREYOLO("weights/libreyolo8n.pt")  # Works for any model
```

Or use specific classes:

```python
from libreyolo import LIBREYOLO8, LIBREYOLO9, LIBREYOLO11, LIBREYOLOX, LIBREYOLORFDETR

model = LIBREYOLO8("weights/libreyolo8n.pt", size="n")
model = LIBREYOLO9("weights/libreyolo9s.pt", size="s")
model = LIBREYOLO11("weights/libreyolo11n.pt", size="n")
model = LIBREYOLOX("weights/libreyoloXs.pt", size="s")
model = LIBREYOLORFDETR("weights/libreyolorfdetrn.pt", size="n")
```

## Model Sizes

### YOLOv8 / YOLOv11

`n` (nano), `s` (small), `m` (medium), `l` (large), `x` (xlarge)

### YOLOv9

`t` (tiny), `s` (small), `m` (medium), `c` (compact)

### YOLOX

`nano`, `tiny`, `s`, `m`, `l`, `x`

Note: `nano` and `tiny` use 416x416 input, others use 640x640.

### RF-DETR

`n` (nano), `s` (small), `b` (base), `m` (medium), `l` (large)

## Device Selection

```python
# Auto-detect (CUDA > MPS > CPU)
model = LIBREYOLO("weights/libreyolo8n.pt", device="auto")

# Force specific device
model = LIBREYOLO("weights/libreyolo8n.pt", device="cuda:0")
model = LIBREYOLO("weights/libreyolo8n.pt", device="cpu")
model = LIBREYOLO("weights/libreyolo8n.pt", device="mps")  # Apple Silicon
```
