# CLI Test Checklist

Test commands for the LibreYOLO CLI before merging.
Uses `coco8.yaml` (8 images) for fast training/validation runs.

## Special Commands

```bash
# Help (no args)
libreyolo

# Version
libreyolo version
libreyolo version --json

# System checks
libreyolo checks
libreyolo checks --json

# List models
libreyolo models
libreyolo models --json

# List export formats
libreyolo formats
libreyolo formats --json

# Default config
libreyolo cfg
libreyolo cfg --json

# Model info (downloads yolox-s if not present)
libreyolo info model=yolox-s
libreyolo info model=yolox-s --json
```

## Predict

```bash
# Basic inference
libreyolo predict source=libreyolo/assets/parkour.jpg model=yolox-s

# With conf threshold
libreyolo predict source=libreyolo/assets/parkour.jpg model=yolox-s conf=0.5

# JSON output
libreyolo predict source=libreyolo/assets/parkour.jpg model=yolox-s --json --quiet

# Save annotated image
libreyolo predict source=libreyolo/assets/parkour.jpg model=yolox-s save

# Bare bool flags
libreyolo predict source=libreyolo/assets/parkour.jpg model=yolox-s half save

# --key value syntax (standard CLI)
libreyolo predict --source libreyolo/assets/parkour.jpg --model yolox-s --conf 0.5

# Mixed syntax
libreyolo predict source=libreyolo/assets/parkour.jpg --model yolox-s conf=0.5

# Task prefix (detect is stripped)
libreyolo detect predict source=libreyolo/assets/parkour.jpg model=yolox-s

# YOLOv9 model (auto-downloads)
libreyolo predict source=libreyolo/assets/parkour.jpg model=yolo9-t

# Local weights path
libreyolo predict source=libreyolo/assets/parkour.jpg model=weights/LibreYOLOXs.pt

# Schema discovery
libreyolo predict --help-json
libreyolo predict --help
```

## Train

```bash
# YOLOX — short real training on coco8
libreyolo train data=coco8.yaml model=yolox-s epochs=3 batch=4 imgsz=416

# YOLOX — JSON output
libreyolo train data=coco8.yaml model=yolox-s epochs=3 batch=4 imgsz=416 --json

# YOLOv9 — short real training
libreyolo train data=coco8.yaml model=yolo9-t epochs=3 batch=4

# RF-DETR — short real training
libreyolo train data=coco8.yaml model=rfdetr-s epochs=3 batch=2

# Dry run — verify family defaults without training
libreyolo train data=coco8.yaml model=yolox-s --dry-run --json
libreyolo train data=coco8.yaml model=yolo9-t --dry-run --json
libreyolo train data=coco8.yaml model=rfdetr-s --dry-run --json

# User override wins over family default
libreyolo train data=coco8.yaml model=yolox-s momentum=0.5 --dry-run --json

# Schema discovery
libreyolo train --help-json
libreyolo train --help
```

## Val

```bash
# Validate with pretrained weights
libreyolo val model=yolox-s data=coco8.yaml

# JSON output
libreyolo val model=yolox-s data=coco8.yaml --json

# With overrides
libreyolo val model=yolox-s data=coco8.yaml batch=8 conf=0.01

# Local weights
libreyolo val model=weights/LibreYOLOXs.pt data=coco8.yaml

# YOLOv9
libreyolo val model=yolo9-t data=coco8.yaml
```

## Export

```bash
# ONNX export
libreyolo export model=weights/LibreYOLOXs.pt format=onnx

# ONNX with options
libreyolo export model=weights/LibreYOLOXs.pt format=onnx half dynamic

# JSON output
libreyolo export model=weights/LibreYOLOXs.pt format=onnx --json

# TensorRT (engine alias)
libreyolo export model=weights/LibreYOLOXs.pt format=engine half

# OpenVINO
libreyolo export model=weights/LibreYOLOXs.pt format=openvino
```

## Error Cases

```bash
# Missing source file
libreyolo predict source=nonexistent.jpg model=yolox-s

# Missing source — JSON error
libreyolo predict source=nonexistent.jpg model=yolox-s --json

# Missing required arg
libreyolo train model=yolox-s
libreyolo predict model=yolox-s

# Half + int8 conflict
libreyolo export model=weights/LibreYOLOXs.pt format=onnx half int8
```
