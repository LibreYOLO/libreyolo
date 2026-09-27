# FiftyOne integration

[FiftyOne](https://github.com/voxel51/fiftyone) (Apache-2.0) is a dataset
curation and prediction-analysis tool. `libreyolo.integrations.fiftyone` sends
LibreYOLO predictions into a FiftyOne dataset and moves datasets in both
directions between a LibreYOLO dataset yaml and a FiftyOne dataset.

Nothing is vendored and nothing is imported at `import libreyolo` time. The
module imports `fiftyone` lazily and raises an install hint when it is absent.

## Install

```bash
pip install "libreyolo[fiftyone]"
```

Two things to know before installing into an existing environment:

- FiftyOne depends on `opencv-python-headless` while LibreYOLO depends on
  `opencv-python`. Both ship the same `cv2` module, so pip leaves whichever
  landed last in place. Installing FiftyOne second gives a headless `cv2`
  (`cv2.getBuildInformation()` reports `GUI: NONE`), which breaks the window
  paths: `predict(show=True)`, video display, and the labeller preview.
  Reinstall `opencv-python` afterwards if you need those, or keep FiftyOne in
  its own environment. This is why `libreyolo[fiftyone]` is not part of
  `libreyolo[all]`.
- FiftyOne runs a local MongoDB (`fiftyone-db`), started on first use. It needs
  write access to `~/.fiftyone`.

## Predictions into FiftyOne

```python
import fiftyone as fo
from libreyolo import LibreYOLO
from libreyolo.integrations.fiftyone import apply_model

dataset = fo.load_dataset("my-dataset")
model = LibreYOLO("LibreYOLO9s.pt")
apply_model(dataset, model, label_field="predictions", conf=0.25, batch_size=8)

session = fo.launch_app(dataset)
```

`apply_model` accepts a loaded model or a checkpoint name, forwards `conf`,
`iou`, `imgsz`, `device`, `classes`, and `max_det` to `model.predict`, and runs
through FiftyOne's own `apply_model`, so the progress bar, `skip_failures`, and
batching behave the way FiftyOne users expect. `batch_size` maps onto
LibreYOLO's batched list inference: one stacked forward per chunk.

To use the model with any other FiftyOne API that takes a model, wrap it
directly:

```python
from libreyolo.integrations.fiftyone import to_fiftyone_model

fo_model = to_fiftyone_model("LibreYOLO9s.pt", conf=0.25)
dataset.apply_model(fo_model, label_field="predictions")
```

### What each task writes

| Task | FiftyOne label | Field |
|---|---|---|
| detect | `fo.Detections` | `label_field` |
| segment | `fo.Detections` with per-box masks | `label_field` |
| segment, `mask_format="polyline"` | `fo.Polylines` | `label_field` |
| obb | `fo.Polylines`, closed and filled | `label_field` |
| pose | `fo.Keypoints` plus `fo.Detections` | `label_field_keypoints`, `label_field_detections` |
| classify | `fo.Classification` (top-1) | `label_field` |

Coordinates follow FiftyOne's convention: boxes are normalized `[x, y, w, h]`
against the original image, keypoints are normalized `[x, y]` with
`[nan, nan]` for points the model did not see, and instance masks are boolean
arrays cropped to their box. Boxes are clipped to the image before
normalization, which is what the COCO evaluators do too. Track ids, when the
source was tracked, land in `Detection.index`.

## Datasets in and out

```python
from libreyolo.integrations.fiftyone import from_fiftyone, to_fiftyone

dataset = to_fiftyone("coco128.yaml", split="val")     # ground truth attached
yaml_path = from_fiftyone(curated_view, "data/curated", split="train")
```

`to_fiftyone` reads both layouts in
[`docs/dataset_schema.md`](dataset_schema.md): the YOLO layout (`images/` plus
`labels/`, including `.txt` image lists) and native COCO JSON through the
yaml's `annotations` mapping. Ground truth lands in `ground_truth` by default.
Labels in both layouts are already normalized, so no image is decoded.

`from_fiftyone` exports a dataset or a filtered view as a LibreYOLO-trainable
dataset and returns the yaml path. Pass `classes=` to fix the class ids:
without it, the exported yaml only contains the classes present in the view,
so ids shift. Calling it twice against the same directory with `split="train"`
and `split="val"` writes one yaml with both splits.

## Workflow: find your model's worst predictions

```python
import fiftyone as fo
from libreyolo import LibreYOLO
from libreyolo.integrations.fiftyone import apply_model, to_fiftyone

dataset = to_fiftyone("coco128.yaml", split="val")
apply_model(dataset, LibreYOLO("LibreYOLO9s.pt"), label_field="predictions")

results = dataset.evaluate_detections(
    "predictions", gt_field="ground_truth", eval_key="eval", compute_mAP=True
)
print(results.mAP())
results.print_report()

# Images where the model is most wrong, worst first.
worst = dataset.sort_by("eval_fp", reverse=True)
session = fo.launch_app(worst)
```

`eval_key` writes per-sample `eval_tp` / `eval_fp` / `eval_fn` counts and tags
every box as a true positive, false positive, or false negative, so the App can
filter down to the failures themselves.

## Workflow: find label errors before training

```python
import fiftyone.brain as fob
from libreyolo import LibreYOLO
from libreyolo.integrations.fiftyone import apply_model, from_fiftyone, to_fiftyone

dataset = to_fiftyone("my-dataset.yaml", split="train")
apply_model(dataset, LibreYOLO("LibreYOLO9s.pt"), label_field="predictions")

fob.compute_mistakenness(dataset, "predictions", label_field="ground_truth")

suspect = dataset.sort_by("mistakenness", reverse=True).limit(50)
# Fix or drop the bad labels in the App, then export what survives.
clean = dataset.match_tags("bad_label", bool=False)
yaml_path = from_fiftyone(clean, "data/clean", split="train", classes=classes)
```

Then train on `yaml_path` as usual. Mistakenness needs predictions from a model
that did not train on these exact labels for the ranking to mean anything.

## Validated

Local CPU run against coco128 (128 images) with `LibreYOLO9s.pt`:

- `to_fiftyone` loads 128 samples with ground truth; `apply_model` writes 704
  boxes; `evaluate_detections` reports mAP 0.5238.
- Box coordinates round-trip against `model.predict` to 3e-5 px, so the
  letterbox geometry the App draws is the geometry the model produced.
- `from_fiftyone` output loads back through `load_data_config`.
- Tasks exercised: YOLO9 detect (single and batched), RF-DETR detect,
  RF-DETR instance segmentation in both mask and polyline form, RTDETRv2 obb,
  HRNet pose, DeiT classify.

The `fiftyone`-marked tests in `tests/unit/test_integrations_fiftyone.py` cover
the same round trip on synthetic data, with no weight downloads.
