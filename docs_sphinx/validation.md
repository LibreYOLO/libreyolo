# Validation Guide

This guide explains how to validate LibreYOLO models on COCO datasets and interpret evaluation metrics.

## Quick Start

```python
from libreyolo import LIBREYOLO

# Load model
model = LIBREYOLO("weights/libreyolo8n.pt")

# Validate on COCO dataset
results = model.val(
    data="coco128.yaml",  # Path to dataset config
    batch=16,             # Batch size
    imgsz=640,            # Image size
    conf=0.001,           # Confidence threshold
    iou=0.6               # IoU threshold for NMS
)

# Print metrics
print(f"mAP50-95: {results['metrics/mAP50-95']:.3f}")
print(f"mAP50: {results['metrics/mAP50']:.3f}")
print(f"Precision: {results['metrics/precision']:.3f}")
print(f"Recall: {results['metrics/recall']:.3f}")
```

## Validation Parameters

### `model.val()` Method

```python
results = model.val(
    data="coco.yaml",       # Dataset config file (required)
    batch=16,               # Batch size (default: 16)
    imgsz=640,              # Image size (default: 640)
    conf=0.001,             # Confidence threshold (default: 0.001)
    iou=0.6,                # IoU threshold for NMS (default: 0.6)
    device="auto",          # Device: "auto", "cuda:0", "cpu"
    split="val",            # Dataset split: "val" or "test"
    save_json=False,        # Save results to JSON (COCO format)
    plots=False,            # Generate plots (confusion matrix, etc.)
    verbose=True            # Print detailed output
)
```

### Parameters Explained

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `data` | Path to dataset YAML config | Required | Must contain val/test image paths |
| `batch` | Validation batch size | 16 | Higher = faster but more memory |
| `imgsz` | Input image size | 640 | Model-dependent (YOLOX nano/tiny use 416) |
| `conf` | Confidence threshold | 0.001 | Lower = more detections (better recall) |
| `iou` | IoU threshold for NMS | 0.6 | Higher = fewer duplicate boxes |
| `device` | Compute device | "auto" | Auto-detects CUDA > MPS > CPU |
| `split` | Dataset split to use | "val" | "val" or "test" |
| `save_json` | Export COCO JSON | False | Useful for custom evaluation tools |
| `plots` | Generate visualization | False | Creates plots in output directory |
| `verbose` | Print progress | True | Disable for cleaner logs |

## Dataset Configuration

### YAML Format

Create a YAML file describing your dataset:

```yaml
# coco.yaml
path: /path/to/coco  # Dataset root directory
train: images/train  # Train images (relative to 'path')
val: images/val      # Validation images (relative to 'path')
test: images/test    # Test images (optional)

# Class names (COCO has 80 classes)
names:
  0: person
  1: bicycle
  2: car
  # ... (80 total)
```

### Built-in Datasets

LibreYOLO includes pre-configured datasets:

| Config | Description | Size | Use Case |
|--------|-------------|------|----------|
| `coco8.yaml` | 8 COCO images | ~6MB | Quick testing |
| `coco128.yaml` | 128 COCO images | ~30MB | Development |
| `coco.yaml` | Full COCO val2017 | ~1GB | Official benchmarking |
| `coco-val-only.yaml` | COCO val (no train) | ~1GB | Validation-only setup |

```python
# Use built-in datasets (auto-downloaded)
results = model.val(data="coco8.yaml")
```

## Understanding Metrics

Validation returns a dictionary with COCO-style metrics:

```python
{
    "metrics/mAP50-95": 0.451,  # Mean Average Precision at IoU 0.50-0.95
    "metrics/mAP50": 0.632,     # Mean Average Precision at IoU 0.50
    "metrics/mAP75": 0.489,     # Mean Average Precision at IoU 0.75
    "metrics/precision": 0.721, # Precision (TP / (TP + FP))
    "metrics/recall": 0.654,    # Recall (TP / (TP + FN))
    "metrics/mAP_small": 0.234, # mAP for small objects (area < 32²)
    "metrics/mAP_medium": 0.498,# mAP for medium objects (32² < area < 96²)
    "metrics/mAP_large": 0.612  # mAP for large objects (area > 96²)
}
```

### Key Metrics

#### mAP50-95 (Primary Metric)

Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95 (step 0.05):
- **Best overall metric** for model quality
- Used in official COCO leaderboard
- Ranges from 0.0 to 1.0 (higher is better)
- Penalizes poor localization (requires tight boxes)

#### mAP50

Mean Average Precision at IoU threshold 0.50:
- More lenient than mAP50-95
- Commonly used in older YOLO papers
- Good for rough object localization
- Typically higher than mAP50-95

#### Precision

Proportion of correct detections among all predictions:
```
Precision = True Positives / (True Positives + False Positives)
```
- High precision = few false alarms
- Trade-off with recall (adjust `conf` threshold)

#### Recall

Proportion of ground truth objects detected:
```
Recall = True Positives / (True Positives + False Negatives)
```
- High recall = few missed objects
- Trade-off with precision (adjust `conf` threshold)

### Object Size Categories

COCO defines three object size categories based on pixel area:

| Category | Area Range | Characteristics |
|----------|------------|-----------------|
| Small | < 32² (1024 px²) | Difficult to detect (distant objects) |
| Medium | 32² - 96² (1024-9216 px²) | Moderate difficulty |
| Large | > 96² (9216 px²) | Easier to detect (close objects) |

## Model-Specific Preprocessing

LibreYOLO automatically applies correct preprocessing for each model:

| Model | Preprocessing | Notes |
|-------|---------------|-------|
| YOLOv8 | Normalize to 0-1 | `img / 255.0` |
| YOLOv9 | Normalize to 0-1 | `img / 255.0` |
| YOLOv11 | Normalize to 0-1 | `img / 255.0` |
| YOLOX | Keep 0-255 range | No normalization |
| RF-DETR | ImageNet normalization | `mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]` |

**You don't need to handle this manually** - LibreYOLO detects the model type and applies the correct preprocessing automatically.

## Complete Example

```python
from libreyolo import LIBREYOLO

# Load model (auto-detects version and size)
model = LIBREYOLO("weights/libreyolo8n.pt")

# Validate on COCO val2017
results = model.val(
    data="coco.yaml",
    batch=32,          # Large batch for faster validation
    imgsz=640,
    conf=0.001,        # Low threshold for maximum recall
    iou=0.6,
    device="cuda:0",   # Use GPU
    save_json=True,    # Save predictions in COCO format
    verbose=True
)

# Print all metrics
print("\n=== Validation Results ===")
for metric, value in results.items():
    if metric.startswith("metrics/"):
        print(f"{metric:30s}: {value:.4f}")

# Check performance
if results["metrics/mAP50-95"] > 0.45:
    print("\n✓ Model performance: Good")
elif results["metrics/mAP50-95"] > 0.35:
    print("\n⚠ Model performance: Moderate")
else:
    print("\n✗ Model performance: Poor")
```

## Validation on Custom Datasets

### 1. Organize Your Data

```
custom_dataset/
├── images/
│   ├── val/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── train/  # Optional
└── labels/
    ├── val/
    │   ├── img001.txt
    │   ├── img002.txt
    │   └── ...
    └── train/  # Optional
```

### 2. Create Dataset Config

```yaml
# custom.yaml
path: /path/to/custom_dataset
train: images/train  # Optional
val: images/val

# Define your classes
names:
  0: cat
  1: dog
  2: bird
```

### 3. Run Validation

```python
results = model.val(
    data="custom.yaml",
    batch=16,
    imgsz=640
)
```

## Comparing Models

```python
models = {
    "YOLOv8n": "weights/libreyolo8n.pt",
    "YOLOv8s": "weights/libreyolo8s.pt",
    "YOLOv8m": "weights/libreyolo8m.pt",
}

results_comparison = {}

for name, weights in models.items():
    model = LIBREYOLO(weights)
    results = model.val(
        data="coco128.yaml",
        batch=16,
        verbose=False
    )
    results_comparison[name] = results["metrics/mAP50-95"]
    print(f"{name:12s}: mAP50-95 = {results['metrics/mAP50-95']:.3f}")

# Find best model
best_model = max(results_comparison, key=results_comparison.get)
print(f"\nBest model: {best_model} ({results_comparison[best_model]:.3f})")
```

## Troubleshooting

### Low mAP Scores

**Possible causes:**
- Incorrect `conf` threshold (try lowering to 0.001)
- Wrong dataset configuration (check paths in YAML)
- Model-dataset mismatch (e.g., trained on COCO, tested on custom data)
- Incorrect class names/IDs in YAML config

### Out of Memory

**Solutions:**
- Reduce batch size: `batch=8` or `batch=4`
- Reduce image size: `imgsz=416` (if model supports it)
- Use CPU: `device="cpu"` (slower but no memory limits)

### Slow Validation

**Optimizations:**
- Increase batch size: `batch=32` or higher
- Use GPU: `device="cuda:0"`
- Disable plots: `plots=False`
- Disable verbose output: `verbose=False`

### Dataset Not Found

**Fixes:**
- Check absolute paths in YAML config
- Ensure images exist: `ls /path/to/dataset/images/val`
- Verify YAML syntax (no tabs, proper indentation)

## Advanced: COCO Evaluator Integration

For research or official benchmarking, use COCO's official evaluation:

```python
from libreyolo import LIBREYOLO
from libreyolo.validation.coco_evaluator import COCOEvaluator

model = LIBREYOLO("weights/libreyolo8n.pt")

# Run validation with JSON export
results = model.val(
    data="coco.yaml",
    save_json=True,  # Exports predictions.json
    batch=32
)

# Use COCO evaluator for detailed analysis
evaluator = COCOEvaluator(
    gt_json="path/to/instances_val2017.json",
    pred_json="runs/val/predictions.json"
)
coco_results = evaluator.evaluate()

# Per-class AP scores
for class_id, ap in enumerate(coco_results["per_class_AP"]):
    print(f"Class {class_id}: AP = {ap:.3f}")
```

## Next Steps

- {doc}`inference` - Run inference on new images
- {doc}`training` - Train models on custom data
- {doc}`model-architecture` - Understand model internals
