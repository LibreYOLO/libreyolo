# ADR 0001: Multi-Task Architecture For Detection, Segmentation, and Keypoints

- Status: Proposed
- Date: 2026-04-27
- Scope: Core library architecture

## Context

LibreYOLO is clean for object detection, but its shared architecture is still
detection-centric.

Current pressure points:

- Task and family are mixed together.
  - `-seg` is effectively a RF-DETR special case in
    [libreyolo/models/__init__.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/models/__init__.py:285).
- Shared results only model boxes plus optional masks.
  - [libreyolo/utils/results.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/utils/results.py:183)
- Shared datasets collapse polygons to boxes and drop COCO segmentation.
  - [libreyolo/data/dataset.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/data/dataset.py:107)
  - [libreyolo/data/dataset.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/data/dataset.py:279)
- Validation is hardwired to bbox detection.
  - [libreyolo/models/base/model.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/models/base/model.py:521)
  - [libreyolo/training/trainer.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/training/trainer.py:482)
- Export and backend support infer task through heuristics instead of a stable
  contract.
  - [libreyolo/export/exporter.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/export/exporter.py:410)
  - [libreyolo/export/onnx.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/export/onnx.py:105)
  - [libreyolo/backends/base.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/backends/base.py:190)

This is manageable for wrapped RF-DETR segmentation, but it will not scale to:

- YOLO9 segmentation
- native EdgeCrafter detection + segmentation + keypoints
- future keypoint families beyond EdgeCrafter

## Decision

LibreYOLO should adopt a two-axis architecture:

- `family`: architecture and checkpoint layout
- `task`: `detect | segment | pose`

The library should standardize around one shared instance-level internal
contract:

- boxes are always present
- masks are optional
- keypoints are optional

Everything shared in the library should depend on that contract instead of on
family-specific heuristics or on the current `(N, 5)` box-only dataset shape.

## Goals

- Add multiple segmentation families without more one-off branches.
- Add multiple keypoint families later without redesigning detection again.
- Preserve the current public UX for detection users.
- Keep wrapped families viable.
- Keep native ports clean and family-local.

## Non-Goals

- This ADR does not require a full rewrite in one PR.
- This ADR does not require wrapper families to move onto the native training
  loop.
- This ADR does not require every family to support every task.

## Target Architecture

### 1. Task Model

Add a first-class task axis to shared code.

Proposed shared types:

- `libreyolo/tasks.py`
  - `TaskType = Literal["detect", "segment", "pose"]`
  - `DEFAULT_TASK = "detect"`

Shared model metadata:

- `BaseModel.family: str`
- `BaseModel.task: TaskType`
- `BaseModel.supported_tasks: tuple[TaskType, ...]`

Task resolution precedence:

1. explicit user argument
2. checkpoint metadata
3. filename suffix or task-specific naming
4. family default

### 2. Results Contract

Keep one image-level result object, but make the per-instance payload explicit.

Proposed updates in
[libreyolo/utils/results.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/utils/results.py:1):

- keep `Boxes`
- keep `Masks`
- add `Keypoints`
- add `Instances`
- evolve `Results`

Proposed classes:

```python
class Keypoints:
    data: torch.Tensor | np.ndarray      # (N, K, 3) => x, y, conf_or_vis
    orig_shape: tuple[int, int]

class Instances:
    boxes: Boxes
    masks: Masks | None = None
    keypoints: Keypoints | None = None
    track_id: torch.Tensor | None = None

    def select(self, indices) -> "Instances": ...
    def filter_classes(self, classes) -> "Instances": ...
    def with_track_ids(self, ids) -> "Instances": ...
    def cpu(self) -> "Instances": ...
    def numpy(self) -> "Instances": ...

class Results:
    task: TaskType
    instances: Instances
    orig_shape: tuple[int, int]
    path: str | None
    names: dict[int, str]
    frame_idx: int | None = None
```

Compatibility rules:

- keep `result.boxes`
- keep `result.masks`
- add `result.keypoints`
- keep `result.track_id`
- make them aliases over `result.instances`

### 3. Drawing and Tracking

Shared drawing should become task-aware, but tracking should remain box-based.

Proposed shared additions:

- `libreyolo/utils/drawing.py`
  - add `draw_keypoints(...)`
  - add `draw_results(...)`
- `Results.plot(...)` delegates to `draw_results(...)`

Tracking boundary:

- association stays box-based
- payload preservation is generic

Instead of manually slicing masks in
[libreyolo/tracking/tracker.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/tracking/tracker.py:265),
the tracker should call:

```python
tracked_instances = results.instances.select(indices).with_track_ids(track_ids)
```

This also makes pose tracking inherit the same behavior automatically.

### 4. Annotation and Dataset Contract

The shared sample contract should be instance-centric and original-canvas-based,
not model-centric and pre-scaled.

Proposed new shared types:

- `libreyolo/data/types.py`
  - `ImageMeta`
  - `SegmentSet`
  - `Instances`
  - `Sample`

Proposed shape:

```python
@dataclass
class ImageMeta:
    image_id: int | str
    path: Path
    orig_shape: tuple[int, int]   # h, w

@dataclass
class SegmentSet:
    kind: Literal["polygon", "rle", "bitmap"]
    data: list[Any]
    canvas_size: tuple[int, int]

@dataclass
class Instances:
    classes: NDArray[np.int64]
    boxes_xyxy: NDArray[np.float32]
    segments: SegmentSet | None = None
    keypoints: NDArray[np.float32] | None = None   # (N, K, 3)
    area: NDArray[np.float32] | None = None
    iscrowd: NDArray[np.uint8] | None = None
    ignore: NDArray[np.uint8] | None = None

@dataclass
class Sample:
    image: np.ndarray
    instances: Instances
    meta: ImageMeta
```

Rules:

- canonical geometry is absolute pixels
- dataset loaders do not scale to `imgsz`
- dataset loaders do not normalize
- polygons and RLE are preserved
- pose keypoints are preserved

Required new shared files:

- `libreyolo/data/types.py`
- `libreyolo/data/parsers/yolo.py`
- `libreyolo/data/parsers/coco.py`
- `libreyolo/data/collate.py`
- `libreyolo/data/transforms/geometry.py`

The current training-oriented datasets can survive during migration, but they
should stop being the canonical source of truth for annotations.

### 5. Validation and Metrics

Validation should be task-level, not detection-only.

Proposed new files:

- `libreyolo/validation/factory.py`
- `libreyolo/validation/types.py`
- `libreyolo/validation/base_coco.py`
- `libreyolo/validation/segmentation_validator.py`
- `libreyolo/validation/keypoints_validator.py`
- `libreyolo/validation/metric_suite.py`
- `libreyolo/validation/serializers.py`
- `libreyolo/validation/ground_truth.py`

Proposed internal metric prediction type:

```python
@dataclass
class ValidationPrediction:
    image_id: int | str
    orig_shape: tuple[int, int]
    boxes_xyxy: torch.Tensor | np.ndarray
    scores: torch.Tensor | np.ndarray
    class_ids: torch.Tensor | np.ndarray
    masks: torch.Tensor | np.ndarray | None = None
    keypoints: torch.Tensor | np.ndarray | None = None
```

Validator dispatch:

- `DetectionValidator`
- `SegmentationValidator`
- `KeypointsValidator`

All task validators should share one COCO metric suite:

- bbox metrics
- segm metrics
- keypoints metrics

This removes the current hard dependency on box-only postprocessing in
[libreyolo/validation/detection_validator.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/validation/detection_validator.py:310).

### 6. Runtime and Export Contract

Exported models should declare an explicit runtime contract.

Proposed new files:

- `libreyolo/export/contract.py`
- `libreyolo/backends/contracts.py`
- `libreyolo/backends/parsers.py`
- `libreyolo/backends/types.py`

Contract name:

- `libreyolo_contract`

Suggested fields:

```json
{
  "schema_version": 1,
  "model": {"family": "rfdetr", "size": "n"},
  "task": "segment",
  "label_space": {"num_classes": 80, "names": {"0": "person"}},
  "preprocess": {
    "layout": "nchw",
    "color_space": "rgb",
    "resize_mode": "stretch",
    "imgsz": [560, 560]
  },
  "runtime": {
    "parser": "detr_instance_seg_v1",
    "nms_mode": "none",
    "outputs": [
      {"name": "pred_boxes", "semantic": "boxes"},
      {"name": "pred_logits", "semantic": "class_logits"},
      {"name": "pred_masks", "semantic": "mask_logits", "encoding": "dense_logits"}
    ]
  }
}
```

Shared runtime prediction type:

```python
@dataclass
class TaskPredictions:
    task: TaskType
    boxes_xyxy: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    masks: np.ndarray | None = None
    keypoints: np.ndarray | None = None
```

Backends should stop inferring task from:

- output count
- family name
- segmentation booleans

They should instead load the declared contract and dispatch to a parser.

### 7. Training Layering

`BaseTrainer` should become orchestration-only.

Current overload:

- data setup
- validation selection
- trainer loop assumptions tied to box-only batches

Proposed training layout:

- `libreyolo/training/trainer.py`
  - keep `BaseTrainer`, slim it down
- `libreyolo/training/architecture/grid.py`
  - `GridTrainerBase`
- `libreyolo/training/architecture/detr.py`
  - `DETRTrainerBase`
- `libreyolo/training/task.py`
  - `TaskTrainingSpec`

Proposed shared task-facing methods:

- `build_train_pipeline()`
- `prepare_batch()`
- `encode_targets_for_model()`
- `build_validator()`

Rules:

- architecture bases own repeated mechanics
- family trainers own optimizer groups, losses, recipe details
- wrapper families can bypass the native trainer stack

### 8. Wrapper Family Boundary

Wrapped families remain valid.

RF-DETR should not be forced onto the native training stack if upstream still
provides the best recipe.

Proposed wrapper adapter:

- `libreyolo/training/external.py`
  - `ExternalTrainerAdapter`

Responsibilities:

- translate `TrainConfig`
- call upstream train API
- normalize checkpoints and metadata

### 9. EdgeCrafter Family Naming

Do not keep `models/ec/` as the long-term family package.

Future EdgeCrafter support is one family with multiple tasks:

- ECDet
- ECSeg
- ECKey

Recommended rename:

- `libreyolo/models/ec/` -> `libreyolo/models/edgecrafter/`

Proposed class:

- `LibreYOLOEdgeCrafter(BaseModel)`

Proposed family-local structure:

- `libreyolo/models/edgecrafter/model.py`
- `libreyolo/models/edgecrafter/nn.py`
- `libreyolo/models/edgecrafter/backbone.py`
- `libreyolo/models/edgecrafter/encoder.py`
- `libreyolo/models/edgecrafter/decoder.py`
- `libreyolo/models/edgecrafter/postprocess.py`
- `libreyolo/models/edgecrafter/transforms.py`
- `libreyolo/models/edgecrafter/trainer.py`
- `libreyolo/models/edgecrafter/config.py`

Task selection should be a constructor argument or checkpoint property, not a
new top-level family folder per task.

## Proposed Public API

No breaking UX changes should be required for current users.

Continue supporting:

```python
model = LibreYOLO("LibreYOLO9s.pt")
result = model.predict("image.jpg")
tracked = model.track("video.mp4")
```

Add task override when needed:

```python
model = LibreYOLO("weights.pt", task="segment")
model = LibreYOLO("weights.pt", task="pose")
```

Result access:

```python
result.boxes
result.masks
result.keypoints
result.instances
result.plot()
```

## Phased Refactor Plan

### Phase 0: Documentation and Metadata

Goal:

- establish vocabulary before code moves

Changes:

- add this ADR
- add `TaskType`
- add `task` to checkpoint metadata schema going forward

Files:

- new `docs/adr/0001-multi-task-architecture.md`
- new `libreyolo/tasks.py`

### Phase 1: Results And Task Metadata

Goal:

- separate family vs task without breaking users

Changes:

- add `BaseModel.task`
- add `BaseModel.supported_tasks`
- add `Keypoints`
- add `Instances`
- evolve `Results`
- keep legacy aliases

Primary files:

- [libreyolo/models/base/model.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/models/base/model.py)
- [libreyolo/utils/results.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/utils/results.py)
- [libreyolo/models/__init__.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/models/__init__.py)
- [libreyolo/utils/drawing.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/utils/drawing.py)
- [libreyolo/tracking/tracker.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/tracking/tracker.py)

Compatibility:

- `InferenceRunner` accepts both old detection dicts and new `Instances`

### Phase 2: Shared Annotation Schema And Compatibility Adapters

Goal:

- stop destroying segmentation and keypoint annotations

Changes:

- add canonical data types
- add parsers
- preserve polygons, RLE, keypoints
- add `legacy_box_collate`
- add `instance_list_collate`

Primary files:

- new `libreyolo/data/types.py`
- new `libreyolo/data/parsers/yolo.py`
- new `libreyolo/data/parsers/coco.py`
- new `libreyolo/data/collate.py`
- update [libreyolo/data/dataset.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/data/dataset.py)
- update [libreyolo/data/yolo_coco_api.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/data/yolo_coco_api.py)

Key rule:

- datasets load originals, transforms resize later

### Phase 3: Task-Aware Validation

Goal:

- make seg and pose metrics first-class

Changes:

- add `ValidatorFactory`
- add `ValidationPrediction`
- split COCO metrics by `bbox`, `segm`, `keypoints`
- implement `SegmentationValidator`
- later implement `KeypointsValidator`

Primary files:

- [libreyolo/validation/base.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/validation/base.py)
- [libreyolo/validation/detection_validator.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/validation/detection_validator.py)
- [libreyolo/validation/coco_evaluator.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/validation/coco_evaluator.py)
- new `libreyolo/validation/factory.py`
- new `libreyolo/validation/types.py`
- new `libreyolo/validation/segmentation_validator.py`
- new `libreyolo/validation/keypoints_validator.py`
- new `libreyolo/validation/metric_suite.py`
- new `libreyolo/validation/serializers.py`
- new `libreyolo/validation/ground_truth.py`

Migration order:

1. refactor detection onto shared metric suite
2. add segmentation validator
3. add keypoints validator

### Phase 4: Export And Backend Contract

Goal:

- stop using heuristic runtime parsing

Changes:

- add contract builder
- add parser registry
- dual-write legacy metadata plus `libreyolo_contract`
- dual-read contract first, heuristics second

Primary files:

- [libreyolo/export/exporter.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/export/exporter.py)
- [libreyolo/export/onnx.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/export/onnx.py)
- [libreyolo/backends/base.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/backends/base.py)
- [libreyolo/backends/onnx.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/backends/onnx.py)
- new `libreyolo/export/contract.py`
- new `libreyolo/backends/contracts.py`
- new `libreyolo/backends/parsers.py`
- new `libreyolo/backends/types.py`

### Phase 5: Training Refactor

Goal:

- make native multi-task families clean to integrate

Changes:

- slim `BaseTrainer`
- add `GridTrainerBase`
- add `DETRTrainerBase`
- move validator selection behind trainer hooks
- add task training spec

Primary files:

- [libreyolo/training/trainer.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/training/trainer.py)
- [libreyolo/training/config.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/training/config.py)
- new `libreyolo/training/architecture/grid.py`
- new `libreyolo/training/architecture/detr.py`
- new `libreyolo/training/task.py`
- optional new `libreyolo/training/external.py`

Immediate beneficiaries:

- D-FINE
- RT-DETR
- EdgeCrafter native family

### Phase 6: EdgeCrafter Family Consolidation

Goal:

- avoid a second architecture split when ECSeg and ECKey land

Changes:

- rename `models/ec` to `models/edgecrafter`
- replace task-named family folder with one family wrapper
- task selects head/decoder/postprocess behavior

Primary files:

- rename `libreyolo/models/ec/` to `libreyolo/models/edgecrafter/`
- update [libreyolo/models/__init__.py](/Users/xuban.ceccon/Documents/GitHub/libreyolo/libreyolo/models/__init__.py)
- add `LibreYOLOEdgeCrafter`

## Exact Class And File Targets

### Shared Core

- `libreyolo/tasks.py`
  - `TaskType`
- `libreyolo/utils/results.py`
  - `Boxes`
  - `Masks`
  - `Keypoints`
  - `Instances`
  - `Results`
- `libreyolo/utils/drawing.py`
  - `draw_boxes`
  - `draw_masks`
  - `draw_keypoints`
  - `draw_results`

### Data

- `libreyolo/data/types.py`
  - `ImageMeta`
  - `SegmentSet`
  - `Instances`
  - `Sample`
- `libreyolo/data/parsers/yolo.py`
  - `YOLOLabelParser`
- `libreyolo/data/parsers/coco.py`
  - `COCOAnnotationParser`
- `libreyolo/data/collate.py`
  - `legacy_box_collate`
  - `instance_list_collate`
  - `multiscale_collate`

### Validation

- `libreyolo/validation/types.py`
  - `ValidationPrediction`
- `libreyolo/validation/factory.py`
  - `ValidatorFactory`
- `libreyolo/validation/base_coco.py`
  - `BaseCOCOValidator`
- `libreyolo/validation/metric_suite.py`
  - `COCOMetricSuite`
  - `COCOMetricEvaluator`
- `libreyolo/validation/serializers.py`
  - `BBoxPredictionSerializer`
  - `SegmentationPredictionSerializer`
  - `KeypointsPredictionSerializer`
- `libreyolo/validation/segmentation_validator.py`
  - `SegmentationValidator`
- `libreyolo/validation/keypoints_validator.py`
  - `KeypointsValidator`

### Runtime / Export

- `libreyolo/export/contract.py`
  - `build_runtime_contract`
- `libreyolo/backends/contracts.py`
  - `RuntimeContract`
  - `load_runtime_contract`
- `libreyolo/backends/types.py`
  - `TaskPredictions`
- `libreyolo/backends/parsers.py`
  - `BasePredictionParser`
  - `DETRInstanceDetectParser`
  - `DETRInstanceSegParser`
  - `YOLOProtoSegParser`
  - `InstanceKeypointsParser`

### Training

- `libreyolo/training/task.py`
  - `TaskTrainingSpec`
- `libreyolo/training/architecture/grid.py`
  - `GridTrainerBase`
- `libreyolo/training/architecture/detr.py`
  - `DETRTrainerBase`
- `libreyolo/training/external.py`
  - `ExternalTrainerAdapter`

### Families

- `libreyolo/models/rfdetr/model.py`
  - replace `segmentation: bool` with `task: TaskType`
- `libreyolo/models/yolo9/model.py`
  - future `supported_tasks = ("detect", "segment")`
- `libreyolo/models/edgecrafter/model.py`
  - `LibreYOLOEdgeCrafter`

## Consequences

### Positive

- multiple segmentation families become normal
- keypoint families fit the same shared contract
- detection UX stays stable
- shared validation becomes trustworthy
- exported runtimes stop depending on fragile heuristics

### Negative

- migration will touch a wide surface area
- compatibility code will exist for at least one release
- dataset parsing and geometry transforms become more complex
- pose evaluation requires explicit skeleton/OKS metadata

## Main Risks

- keeping `(N, 5)` as the true shared batch contract for too long
- not making dataset schema explicit for YOLO txt segmentation vs keypoints
- adding too many parser variants instead of standardizing by paradigm
- renaming EdgeCrafter too late
- breaking soft ABI expectations around `Results`

## Recommended First Slice

The highest-leverage first implementation slice is:

1. add `TaskType`
2. extend `Results` with `Keypoints` and `Instances`
3. make `InferenceRunner` accept `Instances`
4. add `ValidatorFactory`
5. preserve segmentation ground truth in data adapters

That sequence gives immediate architectural value without requiring a full
training rewrite first.
