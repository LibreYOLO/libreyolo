# LibreYOLO Checkpoint Metadata Schema

LibreYOLO `.pt` files are checkpoint wrapper dictionaries saved with
`torch.save()`. The top-level `model` key stores the PyTorch `state_dict`; the
other required top-level keys are metadata used to identify and load the
checkpoint without filename parsing or state-dict sniffing.

## Schema v1.0

Every official LibreYOLO `.pt` checkpoint must contain:

```python
{
    "model": state_dict,
    "schema_version": "1.0",
    "libreyolo_version": "0.x.y",
    "model_family": "yolo9",
    "size": "t",
    "task": "detect",
    "nc": 80,
    "names": {0: "cat", 1: "dog"},
    "imgsz": 640,
}
```

Required field meanings:

- `model`: PyTorch state dict for the model weights.
- `schema_version`: metadata contract version. v1.0 uses the string `"1.0"`.
- `libreyolo_version`: LibreYOLO version that produced the checkpoint.
- `model_family`: registered LibreYOLO family, such as `yolo9`, `rfdetr`,
  `dfine`, or `ec`.
- `size`: model variant within the family, such as `t`, `s`, `r18`, or `atto`.
- `task`: canonical task, one of `detect`, `segment`, `pose`, `classify`,
  `gaze`, `obb`, or `point`.
- `nc`: positive integer class count.
- `names`: `dict[int, str]` with keys in `0..nc-1`. Official checkpoints
  should write every key. Readers may pad missing keys with `class_i` labels for
  legacy sparse mappings, but out-of-range keys are invalid.
- `imgsz`: positive integer square input resolution.

Pose checkpoints additionally include:

- `num_keypoints`: positive integer keypoint count used by the pose head.
- `keypoint_dim`: pose label dimension from the dataset contract, either `2`
  for `x,y` labels or `3` for `x,y,visibility` labels. Model outputs always
  expose keypoints as `x,y,visibility`.
- `oks_sigmas`: optional list of per-keypoint OKS sigmas. When omitted, loaders
  and validators use the task default for `num_keypoints`.

The schema is intentionally flat. Existing LibreYOLO checkpoints and loaders
already use top-level keys such as `model_family`, `size`, `nc`, `names`, and
`task`; nesting the metadata would increase migration risk before release.
The top-level `model` value is deliberately a `state_dict`, matching existing
LibreYOLO behavior. Other checkpoint formats may differ.

## Export Runtime Metadata

The checkpoint schema above remains square-only. Exported runtime artifacts may
also carry metadata for graph tracing and backend loading. For rectangular
graph exports, exporters may dual-write `imgsz_h` and `imgsz_w` next to the
legacy scalar `imgsz`; readers that do not understand the rectangular fields
must not silently treat the scalar as a square runtime contract.

Backend support for rectangular runtime metadata is family- and format-scoped.
YOLO9-family exports may use non-square `imgsz_h/imgsz_w` in supported runtime
formats; families or formats without explicit rectangular support must reject
the metadata instead of preprocessing those artifacts as square inputs.

Embedded-NMS runtime exports may also write these flat metadata keys:

- `nms`: string boolean. `"true"` means the exported graph includes an
  embedded post-processing output.
- `nms_conf`: confidence threshold baked into the embedded NMS graph output.
- `nms_iou`: IoU threshold baked into the embedded NMS graph output.
- `max_det`: maximum number of post-NMS detection rows emitted by the embedded
  graph output.
- `nms_raw_output`: string boolean. `"true"` means the exported graph also
  exposes an auxiliary raw detector output for LibreYOLO backend parsing.

For ONNX YOLO9 detection exports with `nms=true`, output `0` / `output` is the
standalone post-NMS tensor using the export-time `nms_conf`, `nms_iou`, and
`max_det` values. When `nms_raw_output=true`, output `1` / `raw` is reserved for
LibreYOLO backends so they can apply native original-canvas clipping and runtime
`predict(conf=..., iou=..., max_det=...)` semantics. Third-party consumers that
want graph-embedded NMS should use the first output.

## Training Checkpoints

Trainer checkpoints use the same required metadata core and may also contain
flat training/resume fields:

```python
{
    "model": state_dict,
    "...": "all required v1.0 metadata",
    "epoch": 42,
    "optimizer": optimizer_state_dict,
    "config": {...},
    "loss": 1.23,
    "best_metric_key": "metrics/mAP50-95",
    "best_metric_value": 0.51,
    "best_epoch": 39,
    "is_ema_weights": True,
    "train_model": raw_state_dict,
    "ema": ema_state_dict,
    "ema_updates": 12345,
}
```

`is_ema_weights` declares whether the top-level `model` is EMA-smoothed. When
EMA is enabled, `train_model`, `ema`, and `ema_updates` preserve resume state.
Published inference weights should be lean checkpoints and should not include
optimizer, epoch, config, loss, or EMA resume state unless intentionally
distributed as training checkpoints.

For release compatibility, readers accept legacy best-metric aliases such as
`best_mAP50_95`, `best_mAP50`, `best_metric`, and `best_metric_name`.

## Legacy And Foreign Weights

New LibreYOLO writers validate strictly and must emit v1.0 metadata.

When metadata is missing or incomplete:

- Legacy LibreYOLO-looking checkpoints load through the compatibility path with
  a warning and conversion instructions.
- Foreign upstream checkpoints are not loaded by `LibreYOLO(...)` as LibreYOLO
  checkpoints. Convert them with the appropriate `weights/convert_*.py` script
  before loading.

Schema helpers live in `libreyolo/utils/serialization.py`:

```python
wrap_libreyolo_checkpoint(...)
unwrap_libreyolo_checkpoint(...)
validate_checkpoint_metadata(...)
```
