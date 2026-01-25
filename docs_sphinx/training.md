# Training

Training is available for **YOLOX** and **RF-DETR** models.

## YOLOX Training API

### Train from Scratch

```python
from libreyolo import LIBREYOLOX

# Create new model
model = LIBREYOLOX.new(size="s", num_classes=80)

# Train
results = model.train(
    data="path/to/data.yaml",
    epochs=300,
    batch=16,
    imgsz=640
)

print(f"Best mAP: {results['best_mAP50_95']:.3f}")
```

### Fine-tune Pretrained Model

```python
from libreyolo import LIBREYOLOX

# Load pretrained
model = LIBREYOLOX("weights/libreyoloXs.pt", size="s")

# Fine-tune
results = model.train(
    data="custom_data.yaml",
    epochs=100,
    batch=16
)
```

## Training Parameters

```python
results = model.train(
    # Required
    data="data.yaml",           # Dataset config path

    # Training
    epochs=100,                 # Number of epochs
    batch=16,                   # Batch size
    imgsz=640,                  # Image size

    # Optimizer
    lr0=0.01,                   # Initial learning rate
    optimizer="SGD",            # "SGD", "Adam", "AdamW"

    # System
    device="",                  # "" = auto, "cuda:0", "cpu"
    workers=8,                  # Dataloader workers
    seed=0,                     # Random seed

    # Output
    project="runs/train",       # Output directory
    name="exp",                 # Experiment name
    exist_ok=False,             # Overwrite existing

    # Features
    amp=True,                   # Mixed precision
    patience=50,                # Early stopping patience
    resume=None,                # Resume from checkpoint
)
```

## Return Value

```python
{
    'final_loss': 0.045,
    'best_mAP50': 0.65,
    'best_mAP50_95': 0.48,
    'best_epoch': 85,
    'save_dir': 'runs/train/exp',
    'best_checkpoint': 'runs/train/exp/weights/best.pt',
    'last_checkpoint': 'runs/train/exp/weights/last.pt'
}
```

## Resume Training

```python
results = model.train(
    data="data.yaml",
    epochs=300,
    resume="runs/train/exp/weights/last.pt"
)
```

## Dataset Format

Create `data.yaml`:

```yaml
path: /path/to/dataset
train: images/train
val: images/val

nc: 3
names:
  0: cat
  1: dog
  2: bird
```

Directory structure:

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

Label format: `class_id x_center y_center width height` (normalized 0-1)

## Training Output

```
runs/train/exp/
├── train_config.yaml
├── weights/
│   ├── best.pt
│   ├── last.pt
│   └── epoch_*.pt
└── tensorboard/
```

## Using Trained Model

```python
from libreyolo import LIBREYOLOX

# Load best checkpoint
model = LIBREYOLOX("runs/train/exp/weights/best.pt", size="s")

# Run inference
results = model(image="test.jpg", save=True)
```

## Configuration File

Optionally load config from YAML:

```python
results = model.train(
    data="data.yaml",
    cfg="train_config.yaml",  # Load config from file
    epochs=200                # Override config value
)
```

## Model Sizes

| Size | Input | Parameters |
|------|-------|------------|
| `nano` | 416 | 0.9M |
| `tiny` | 416 | 5.1M |
| `s` | 640 | 9.0M |
| `m` | 640 | 25.3M |
| `l` | 640 | 54.2M |
| `x` | 640 | 99.1M |

## RF-DETR Training API

RF-DETR uses the original rfdetr training implementation with EMA, warmup scheduling, and Hungarian matching loss.

### Basic Training

```python
from libreyolo import LIBREYOLORFDETR

model = LIBREYOLORFDETR(size="b")

results = model.train(
    data="path/to/dataset",
    epochs=100,
    batch_size=4,
    lr=1e-4
)
```

### Training Parameters

```python
results = model.train(
    # Required
    data="path/to/dataset",      # Dataset path (COCO format)

    # Training
    epochs=100,                  # Number of epochs
    batch_size=4,                # Batch size
    lr=1e-4,                     # Learning rate

    # Output
    output_dir="runs/train",     # Output directory

    # Resume
    resume=None,                 # Checkpoint path to resume from
)
```

### RF-DETR Dataset Format

RF-DETR requires **COCO format** annotations (different from YOLO format):

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   └── image1.jpg, image2.jpg, ...
├── valid/
│   ├── _annotations.coco.json
│   └── image1.jpg, image2.jpg, ...
└── test/  (optional)
    ├── _annotations.coco.json
    └── image1.jpg, image2.jpg, ...
```

The `_annotations.coco.json` file follows standard COCO format with `images`, `annotations`, and `categories` fields.

### RF-DETR Model Sizes

| Size | Resolution | Description |
|------|------------|-------------|
| `n` | 384 | Nano |
| `s` | 512 | Small |
| `b` | 560 | Base |
| `m` | 576 | Medium |
| `l` | 704 | Large |
