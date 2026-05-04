---
name: libreyolo-add-native-segmentation-model
description: >-
  Add a native segmentation model to LibreYOLO after the explicit task-paradigm
  refactor. Use this for simple guidance on task metadata, model registration,
  filename suffixes, validation, results, and segment data plumbing.
---

# Add a Native Segmentation Model to LibreYOLO

This is a short starter checklist. Prefer existing family patterns over new
framework abstractions.

## 1. Register The Task

In the family model class, declare segmentation support explicitly:

```python
class LibreMyFamily(BaseModel):
    FAMILY = "myfamily"
    SUPPORTED_TASKS = ("detect", "segment")
    DEFAULT_TASK = "detect"
    TASK_INPUT_SIZES = {
        "detect": {"n": 640, "s": 640},
        "segment": {"n": 640, "s": 640},
    }
```

Accept `task: str | None = None` in `__init__` and pass it to `BaseModel`:

```python
super().__init__(..., task=task, ...)
```

Use `self.task == "segment"` as the source of truth. Do not add a second
long-lived `is_segmentation` flag.

## 2. Make The Factory See It

Keep all code under the existing family package, for example:

```text
libreyolo/models/yolo9/
libreyolo/models/edgecrafter/
```

Do not create long-term task packages like `ecseg/` or `ecpose/`.

Ensure the family package is imported from `libreyolo/models/__init__.py` so
`BaseModel.__init_subclass__` can register it. Implement or update `can_load()`
so checkpoints from the family match only that family.

Task resolution already follows:

```text
explicit task= -> checkpoint["task"] -> filename suffix -> family default
```

Use LibreYOLO-branded weight names with Ultralytics-style task suffixes:

```text
LibreMyFamilyn-seg.pt
LibreYOLO9s-seg.pt
```

## 3. Persist Metadata

When saving checkpoints or export metadata, preserve existing metadata and add:

```python
"family": self.family
"task": self.task
"supported_tasks": self.SUPPORTED_TASKS
"default_task": self.DEFAULT_TASK
```

Backends should read the same metadata and initialize with the resolved task.

## 4. Return Flat Results

Public results must stay Ultralytics-style and flat:

```python
Results(..., boxes=boxes, masks=masks)
```

For segmentation, populate:

```python
result.boxes
result.masks
```

Leave unsupported slots present but `None`:

```python
result.keypoints is None
result.probs is None
result.obb is None
```

Use `Results._select(indices)` for filtering/tracking so boxes and masks stay
aligned.

## 5. Train Data Contract

Enable segment labels only when the trainer/model needs them:

```python
YOLODataset(..., load_segments=True)
COCODataset(..., load_segments=True)
```

The trainer receives:

```python
on_forward(imgs, targets, polygons=polygons)
```

`polygons` uses this contract:

```text
batch -> image -> instance -> polygon ring
```

Each ring is an `Nx2` array in original image pixel coordinates. Detection rows
without polygon labels use an empty ring list for that instance.

Native YOLO-style segmentation still needs the family-specific pieces: proto
head, mask coefficients, target assignment, polygon/mask rasterization, and mask
loss. Do not pretend the shared data plumbing implements those by itself.

## 6. Validate As Segment

`BaseModel.val()` dispatches by task:

```text
detect  -> DetectionValidator
segment -> SegmentationValidator
```

Segment validation should return both box and mask metrics:

```text
metrics/mAP50-95(B)
metrics/mAP50-95(M)
```

For segment training, set the trainer best metric to the mask key:

```python
best_metric_key = "metrics/mAP50-95(M)"
```

`model.val()` still returns a dict in this refactor. Typed Ultralytics metric
objects are not part of the current contract.

## 7. Minimum Tests

Add focused tests before broad e2e runs:

- task suffix: `LibreMyFamilyn-seg.pt -> task == "segment"`
- explicit unsupported task raises clearly
- checkpoint saves and loads `"task": "segment"`
- `model.val()` for segment emits `(B)` and `(M)` keys
- `Results(boxes=..., masks=...)` survives filtering/tracking
- `load_segments=True` preserves polygons; default detection loading stays
  `(B, max_labels, 5)`

Then run:

```bash
pytest tests/unit
pytest tests/e2e/test_tracking.py
pytest tests/e2e/test_val_coco128.py
```
