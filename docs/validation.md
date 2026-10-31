# Validation

What `val()` returns besides its metric keys, and the class-agnostic NMS
option it shares with `predict()`. Code: `libreyolo/validation/base.py`,
`libreyolo/validation/confusion_matrix.py`, `libreyolo/ops/agnostic_nms.py`.

## Results object

`val()` returns a flat `dict` of metric keys to numbers. Detect and segment
return a `ValidationMetrics` and classify a `ClassifyMetrics`, both `dict`
subclasses with the same keys plus the attributes below. Other tasks return a
plain `dict`.

| Attribute | Tasks | Value |
| --- | --- | --- |
| `confusion_matrix` | detect, segment, classify | The run's `ConfusionMatrix` |
| `box.px` | detect, segment | 1000 confidences evenly spaced over [0, 1] |
| `box.ap_class_index` | detect, segment | Class indices with at least one ground truth, ascending |
| `box.p_curve`, `box.r_curve`, `box.f1_curve` | detect, segment | `(len(ap_class_index), len(px))` precision, recall and F1, one row per class in `ap_class_index` |
| `box.p`, `box.r`, `box.f1` | detect, segment | Per-class values at the confidence that maximizes the mean F1 |
| `top1`, `top5` | classify | `metrics/accuracy_top1` and `metrics/accuracy_top5` |

`box` also carries `image_metrics` and `best_conf_per_class`.

The curves use IoU 0.5 matching over the same matched detections as
`metrics/best_conf`, and count boxes for segmentation too. At confidence `x` a
class keeps its detections scoring at least `x`; where none survive, precision
is 1.0 and recall and F1 are 0.0. The arrays are empty when the evaluator
produced no curves.

## Confusion matrix

`results.confusion_matrix` is computed on every `val()`, not only with
`plots=True`. Counts are integers indexed `matrix[predicted, true]`.

- Detect and segment: `(nc + 1, nc + 1)` with a `background` row and column.
  Predictions at confidence 0.25, or the run's `conf` if higher, pair one to
  one with ground-truth boxes at IoU 0.5 or more, highest IoU first, whatever
  their classes. An unpaired ground truth counts in the `background` row, an
  unpaired prediction in the `background` column. Segmentation pairs boxes.
  With `classes=`, ground truth of the other classes is not counted.
- Classify: `(nc, nc)`, one count per image at `[top-1 class, label]`.

Methods: `summary(normalize=False)` (one row per predicted class, keyed
`Predicted` plus the true-class labels), `to_json()`, `to_csv()`, `to_df()`
(Polars, needs `polars`), `normalized()` (each true-class column sums to 1),
`tp_fp()`, `nonzero()` (aligned `predicted`, `true`, `count` arrays),
`class_accuracy()` and `plot(normalize=True, save_dir=...)`. `labels` holds
the class names, plus `background` for detection. `class_accuracy()` is, per
class, the diagonal count over its true-class column; for detection the column
leaves out missed boxes, so it is the class accuracy among found objects.

Classification heads with more than 2,000 classes (`MAX_DENSE_CLASSES`) do not
build the dense matrix: `matrix`, `normalized()`, `summary()`, `to_*()` and
`plot()` raise `ValueError`, and `plots=True` skips the confusion matrix
images. `nonzero()`, `tp_fp()` and `class_accuracy()` still work.

## Plot files

With `plots=True`, under the run's `plots/` directory:

- `confusion_matrix.png`: raw counts (detect, segment, classify).
- `confusion_matrix_normalized.png`: per-true-class shares (detect, segment,
  classify).
- `f1_conf_box.png`: the `box.f1_curve` rows against `box.px` (detect,
  segment).

## Class-agnostic NMS

`agnostic_nms=True` keeps only the highest-scoring box among boxes that
overlap above `iou`, whatever their classes. Default `False`.

- Python: `predict(agnostic_nms=True)`, `val(agnostic_nms=True)`.
- CLI: `libreyolo predict agnostic_nms=True` or `--agnostic-nms`, and the
  same for `libreyolo val`.

It runs on a family's finished detections, after the `classes` filter, so it
behaves the same for NMS-free families, exported backends, `augment=True` and
tiling. `max_det` is applied after suppression: the postprocess is asked for
at least 300 candidates, so suppressed boxes are replaced by the next ones.
Masks and keypoints follow their box; oriented boxes use rotated IoU.

- `predict()`: detect, segment, pose and obb. On the standard predict path
  other tasks raise `ValueError`, and predict paths that do not implement it
  raise `NotImplementedError`.
- `val()`: detect and segment. Other tasks raise `ValueError`. Every metric of
  the run (mAP, `confusion_matrix`, the `box` curves, `box.image_metrics`)
  sees the suppressed boxes.
