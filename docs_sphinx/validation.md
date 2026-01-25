# Validation

## Basic Usage

```python
from libreyolo import LIBREYOLO

model = LIBREYOLO("weights/libreyolo8n.pt")

results = model.val(data="coco128.yaml")
print(f"mAP50-95: {results['metrics/mAP50-95']:.3f}")
```

## Parameters

```python
results = model.val(
    data="coco.yaml",       # Dataset config (required)
    batch=16,               # Batch size
    imgsz=640,              # Image size (default: model's native size)
    conf=0.001,             # Confidence threshold (low for mAP)
    iou=0.6,                # NMS IoU threshold
    device="auto",          # Device ("auto", "cuda:0", "cpu", "mps")
    split="val",            # "val", "test", or "train"
    save_json=False,        # Export COCO JSON predictions
    plots=True,             # Generate confusion matrix and plots
    verbose=True            # Print detailed metrics
)
```

## Result Metrics

```python
{
    # Primary metrics
    "metrics/mAP50-95": 0.451,  # Mean AP @ IoU 0.50:0.95
    "metrics/mAP50": 0.632,     # Mean AP @ IoU 0.50
    "metrics/mAP75": 0.489,     # Mean AP @ IoU 0.75
    "metrics/precision": 0.721,
    "metrics/recall": 0.654,

    # Per-size metrics (COCO evaluation)
    "metrics/mAP_small": 0.234,   # Objects < 32x32 px
    "metrics/mAP_medium": 0.498,  # Objects 32x32 to 96x96 px
    "metrics/mAP_large": 0.612    # Objects > 96x96 px
}
```

Per-class AP scores are also computed when `verbose=True`.

## Dataset Configuration

Create a YAML file:

```yaml
# custom.yaml
path: /path/to/dataset
train: images/train
val: images/val

names:
  0: cat
  1: dog
  2: bird
```

Expected directory structure:

```
dataset/
├── images/
│   ├── train/
│   │   └── *.jpg
│   └── val/
│       └── *.jpg
└── labels/
    ├── train/
    │   └── *.txt
    └── val/
        └── *.txt
```

Label format (YOLO): `class_id x_center y_center width height` (normalized 0-1)

## Built-in Datasets

| Config | Images | Use Case |
|--------|--------|----------|
| `coco8.yaml` | 8 | Quick testing |
| `coco128.yaml` | 128 | Development |
| `coco.yaml` | 5000 | Benchmarking |

Datasets auto-download when first used.

## Comparing Models

```python
models = ["weights/libreyolo8n.pt", "weights/libreyolo8s.pt"]

for weights in models:
    model = LIBREYOLO(weights)
    results = model.val(data="coco128.yaml", verbose=False)
    print(f"{weights}: mAP50-95 = {results['metrics/mAP50-95']:.3f}")
```
