---
name: libreyolo-add-native-pose-model
description: >-
  Add a native pose/keypoint model to LibreYOLO after the explicit
  task-paradigm refactor. Use this for simple guidance on task metadata, model
  registration, filename suffixes, flat Results keypoints, and the minimum
  trainer/validator/data surfaces.
---

# Add a Native Pose Model to LibreYOLO

This is a small starter checklist. Pose support is less mature than detection
and segmentation, so keep the first implementation narrow and explicit.

## 1. Register The Task

In the family model class, declare pose support:

```python
class LibreMyFamily(BaseModel):
    FAMILY = "myfamily"
    SUPPORTED_TASKS = ("detect", "pose")
    DEFAULT_TASK = "detect"
    TASK_INPUT_SIZES = {
        "detect": {"n": 640, "s": 640},
        "pose": {"n": 640, "s": 640},
    }
```

Accept `task: str | None = None` in `__init__` and pass it to `BaseModel`:

```python
super().__init__(..., task=task, ...)
```

Use `self.task == "pose"` as the source of truth. Avoid extra stored flags like
`is_pose`.

## 2. File Names And Factory

Keep pose code inside the existing family package:

```text
libreyolo/models/edgecrafter/
libreyolo/models/yolo9/
```

Task resolution already follows:

```text
explicit task= -> checkpoint["task"] -> filename suffix -> family default
```

Use LibreYOLO-branded weights with the Ultralytics-style pose suffix:

```text
LibreMyFamilyn-pose.pt
LibreEdgeCrafterS-pose.pt
```

Ensure the family is imported from `libreyolo/models/__init__.py`, and make
`can_load()` identify the family without relying only on the filename.

## 3. Persist Metadata

Checkpoints and export metadata should preserve existing metadata and include:

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
Results(..., boxes=boxes, keypoints=keypoints)
```

For pose, populate:

```python
result.boxes
result.keypoints
```

Leave unsupported slots present but `None`:

```python
result.masks is None
result.probs is None
result.obb is None
```

Use `Results._select(indices)` for filtering/tracking so boxes and keypoints
stay aligned.

## 5. Keypoint Data Contract

Pose labels should extend detection labels with keypoints:

```text
class cx cy w h kx ky [v] ...
```

Keep the initial parser conservative:

- preserve the default detection path as `(B, max_labels, 5)`
- add pose loading only behind an explicit flag, for example `load_keypoints=True`
- pass keypoints through a single optional trainer channel, not by growing tuple
  arity repeatedly
- document whether keypoints are normalized or pixel-space at the trainer
  boundary

Do not silently drop visibility values if the dataset provides them.

## 6. Trainer And Validator

Add a pose trainer path only when there is a native pose head and loss. The
family-specific implementation still needs:

- pose head output shape
- target assignment
- keypoint loss
- keypoint decoding/postprocess
- NMS/filtering that preserves keypoints through `Results._select`

Add `PoseValidator` before claiming pose validation support. It should be
selected by task:

```text
detect -> DetectionValidator
pose   -> PoseValidator
```

Until `PoseValidator` exists, keep `SUPPORTED_TASKS` limited to tasks that the
model can actually validate.

## 7. Minimum Tests

Add focused tests before broad e2e runs:

- task suffix: `LibreMyFamilyn-pose.pt -> task == "pose"`
- explicit unsupported task raises clearly
- checkpoint saves and loads `"task": "pose"`
- `Results(boxes=..., keypoints=...)` exposes `result.keypoints`
- filtering/tracking preserves keypoints with boxes
- default detection data loading still emits `(B, max_labels, 5)`
- pose label loading preserves keypoints and visibility values when enabled

Then run:

```bash
pytest tests/unit
pytest tests/e2e/test_tracking.py
pytest tests/e2e/test_val_coco128.py
```
