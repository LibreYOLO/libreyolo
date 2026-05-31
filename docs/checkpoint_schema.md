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
- `task`: canonical task, one of `detect`, `segment`, `pose`, `classify`, or
  `gaze`.
- `nc`: positive integer class count.
- `names`: `dict[int, str]` with keys in `0..nc-1`. Official checkpoints
  should write every key. Readers may pad missing keys with `class_i` labels for
  legacy sparse mappings, but out-of-range keys are invalid.
- `imgsz`: positive integer square input resolution.

The schema is intentionally flat. Existing LibreYOLO checkpoints and loaders
already use top-level keys such as `model_family`, `size`, `nc`, `names`, and
`task`; nesting the metadata would increase migration risk before release.
The top-level `model` value is deliberately a `state_dict`, matching existing
LibreYOLO behavior. This differs from Ultralytics checkpoints, where `model`
may hold a model object.

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
