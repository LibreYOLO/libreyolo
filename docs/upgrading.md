# Upgrading

What changes when you move between LibreYOLO versions, and what you have to do
about it. Only versions with user-visible migration work are listed.

The full list of changes for any release is in
[CHANGELOG.md](../CHANGELOG.md); this page carries only the parts that require
you to edit code or re-check numbers.

## v1.6.0 to the next release

YOLOv9 checkpoints from 1.6.0 and earlier keep loading and keep the same
float predict/val boxes. What changes is training behaviour and the names
seen by code that reaches into the network.

### What does not break

- Float, quantized and training checkpoints written by earlier releases
  load: legacy tensor names, and the module names stored in quantization
  manifests (`keep_high_precision`, `fp8_tensorwise_weights`), are renamed
  on load and on resume.
- Model inputs, outputs and export output names are unchanged.

### What does change

- **PGI default.** A fine-tune from weights without `aux.*` tensors, which
  includes the published `LibreYOLO9{t,s,m,c}.pt`, trains the main head
  only, as in 1.5. 1.6.0 attached a randomly initialised PGI branch at
  `aux_weight=0.25` to those runs. Weights that carry `aux.*` tensors and
  from-scratch runs still train with PGI. Pass `aux_weight=0.25` to attach a
  new branch anyway, or `aux_weight=0` to force it off.
- **Gradient clipping.** YOLO9, YOLO9-E2E and YOLO9-P2 clip the gradient
  L2 norm at 10. `clip_max_norm=0` restores the 1.6.0 behaviour.
- **`perspective`.** A non-zero value now warps by corner displacement, so
  the same number gives a different warp for YOLO9, YOLOX and YOLO-NAS,
  including runs resumed from a 1.6.0 checkpoint that set it; values of
  `0.002` and above saturate at the same maximum. There is no setting that
  reproduces the 1.6.0 warp. `perspective=0`, the default, is unchanged.
- **YOLO9 mixup.** When mixup is enabled, the blend ratio comes from
  Beta(1, 1) instead of Beta(32, 32). There is no setting for the old value.
- **Internal names.** `DDetect` is `YOLO9Head` (`anchor_convs` /
  `class_convs`), `RepConvN` is `RepConv`, `RepNBottleneck` is `Bottleneck`,
  and block sublayers are `conv1`/`conv2`/`bottleneck`.
  `LibreYOLO9Model.fuse()` and `RepConvN.fuse_convs()` are removed. ONNX
  node names follow the new module names; update Hailo end-node configs.

## v1.5.x to v1.6.0

Existing YOLOv9 checkpoints keep the same predict/val boxes after the
upgrade. Do not re-export or re-evaluate old weights expecting a silent
letterbox flip: there is none.

### What does not break

- Unmarked YOLOv9 `.pt` files (every LibreYOLO ≤1.5 fine-tune, and the
  already-published `LibreYOLO9{t,s,m,c}.pt` mirrors) keep **top-left**
  letterbox. Inference of those files is bit-identical to 1.5.
- The PGI auxiliary head is training-only. Old checkpoints load and infer
  as single-head models. Export graphs stay on the main head.
- Validation NMS defaults (`0.001` / `0.6` / `300`) are unchanged. Val
  numbers from 1.5 remain comparable.

### What does change (new training / new converts only)

- Newly converted official MultimediaTechLab weights stamp
  `letterbox_pad: center`; v9-t and v9-s conversions also keep the auxiliary
  PGI tensors (the v9-m and v9-c auxiliary branch has a different topology
  and is not converted). Fine-tunes that start from those files train with
  center-pad, and with PGI for t and s.
- New YOLOv9 training defaults: `max_labels=300`, SGD momentum warmup
  `0.8 → 0.937` over the existing 3-epoch LR warmup, and `aux_weight=0.25`
  on stock detect (not P2/E2E). Resume of a 1.5 checkpoint without `aux.*`
  keys stays single-head.
- To force center-pad on an unmarked checkpoint: `model.train(...,
  letterbox_pad="center")`. To disable PGI: `aux_weight=0`.

### What you should re-check

- If you compare val mAP of a **new** official convert (center-stamped)
  against a 1.5 number that used the same weights under top-left pad, the
  scores will move. That is the intended geometry match with upstream, not
  a silent default flip of your old files.

### Inputs, resume and validation

- NumPy image arrays are read as BGR, the OpenCV order; 1.5 read them as
  RGB unless `color_format="bgr"` was passed. Add `color_format="rgb"` where
  you pass an RGB array such as `np.asarray(pil_image)`. `cv2.imread()`
  output and video frames need no change.
- A 4-D NumPy array or tensor is a batch: `predict()` returns a list with one
  `Results` per image instead of using only the first image.
- `train(resume=True)` and `train(resume="<run>/weights/last.pt")` restore
  the run's saved training arguments and keep writing into that run's
  directory; arguments you pass explicitly win. Resuming released weights,
  or a run that already reached its `epochs`, raises a `ValueError` that
  says so.
- With validation on, the final epoch always validates, so runs shorter than
  `eval_interval` now report metrics and write `best.pt`. `train(val=False)`
  turns validation off entirely, final plots and precise-BN refresh
  included; such runs write no `best.pt`, only `last.pt` and the periodic
  `epoch_<n>.pt` files `save_period` asks for.
- Validation during training writes into `<run>/val` instead of `runs/val/`
  in the working directory.
- CLI `train` failures are classified by exception type (`config_type_error`,
  `config_unknown_key`, `config_unsupported`, `cuda_oom`, `io_error`)
  instead of always `io_error`, so configuration errors now exit with 2.

## v1.4.0 to v1.5.0

Nothing was removed from the public API surface: every class and function that
worked in v1.4.0 still imports and still works (`__all__` grew from 101 names
to 142 with zero removals). Four things need a code change, and three change
numbers you may be comparing against.

### Code changes you must make

#### `allow_experimental=True` no longer exists

The acknowledgement gate is gone, along with the
`ddp_aware(experimental_key=...)` mechanism behind it. EC, RTMDet, PicoDet and
FOMO training and export previously required the argument, so any script that
trains one of those families is affected.

```python
# v1.4.0
model.train(data="data.yaml", epochs=100, allow_experimental=True)

# v1.5.0: delete the argument
model.train(data="data.yaml", epochs=100)
```

The argument no longer has any effect. `train()` warns
`Unknown training config keys (ignored): ['allow_experimental']` and trains
normally; `export()` ignores it without a warning.

`BaseModel.EXPERIMENTAL_WEIGHT_FILENAMES` was removed with it. The
`get_download_notice()` hook survives and is still overridden by midas,
segformer and yolo9_p2; only the base implementation returns `None`.

#### The export support tier `"experimental"` no longer exists

```python
from libreyolo.export.support import Tier
# v1.4.0: Literal["validated", "experimental", "blocked"]
# v1.5.0: Literal["validated", "available", "blocked"]
```

If you branch on the tier string, replace `"experimental"` with `"available"`.
`BaseExporter` no longer emits a `RuntimeWarning` for those formats.

#### `pretrained=False` with `resume` is now rejected

The combination previously proceeded incoherently. It now raises:

```
ValueError: pretrained=False cannot be combined with resume.
```

Pick one. `pretrained=False` starts from a fresh seeded initialization;
`resume` continues an interrupted run from its checkpoint.

#### CLI `--imgsz` is a string, not an int

This one is narrower than it sounds. Both of these are unaffected:

```bash
libreyolo predict --model yolo9-t --source img.jpg --imgsz 640   # still fine
```

```python
model.predict("img.jpg", imgsz=640)   # still fine
```

Only code that calls the CLI *command functions* directly from Python needs to
change, because `predict`, `train` and `val` widened `--imgsz` from `int` to
`str` so it can accept rectangular sizes:

```python
from libreyolo.cli.commands.predict import predict_cmd

predict_cmd(..., imgsz=640)     # v1.4.0
predict_cmd(..., imgsz="640")   # v1.5.0, and "480x640" now works too
```

`train`'s default is now the string `"640"`. `export --imgsz` was already a
string, and `profile` is unchanged.

### Numbers that change

If you track metrics across versions, three changes move them at default
settings.

#### faster-coco-eval is the default COCO metrics backend

`val()` and per-epoch training validation now compute COCO metrics with the
faster-coco-eval C++ backend instead of pycocotools.

The switch was measured across all 100 RF100-VL test splits: 1381 of 1400
metric values bit-identical, maximum deviation 2.22e-16, headline deltas
exactly 0, at 15.6x faster overall and 56x on detection-dense datasets. In
practice your numbers should not move, but they are produced by a different
implementation, so this is worth knowing before you compare a v1.5.0 run
against a v1.4.0 one.

pycocotools remains the automatic fallback when faster-coco-eval is not
installed. To force it:

```bash
libreyolo val --model yolo9-t --data coco.yaml --no-faster-coco-eval
```

```python
model.val(data="coco.yaml", faster_coco_eval=False)
```

or set `LIBREYOLO_FASTER_COCO_EVAL=0`. The backend actually used is logged at
INFO, exposed as `model.last_eval_backend` after `val()`, and included as
`eval_backend` in the CLI JSON payload. Install the fast path with
`pip install "libreyolo[fast-eval]"`.

#### YOLOX checkpoints trained before v1.5.0 need an eps override to score faithfully

This one is a real trap, so read it if you have fine-tuned YOLOX.

YOLOX specifies BatchNorm `eps=1e-3` and `momentum=0.03`. Until v1.5.0 those
values were applied as a post-hoc fixup that did **not** survive the
class-count rebuild `train()` performs when your dataset's `nc` differs from
the checkpoint's. So such a fine-tune trained and reported in-training
validation at torch's default `eps=1e-5`, then reloaded for inference at
`1e-3`: the same tensors under different normalization.

Regular-conv sizes barely move. Depthwise `n` moves a lot, because its
per-channel `running_var` is small enough for eps to dominate. On RF100-VL
`ball`, the same nano checkpoint scores **0.566** mAP50-95 evaluated at its
trained eps and **0.151** after a stock reload.

A checkpoint trained before v1.5.0 carries eps=1e-5 semantics. To report
faithful numbers for it, either evaluate with BN eps overridden to 1e-5, or
fold `sqrt((var + 1e-3) / (var + 1e-5))` into the BN weights. Checkpoints
trained on v1.5.0 and later need neither.

#### D-FINE multi-scale training uses the upstream per-size recipe

`base_size_repeat` was hardcoded to 3 for every size; it now resolves per size
as upstream specifies: **n** trains at fixed size (multi-scale off), **s** 20,
**m** 6, **l** 4, **x** 3. Only x matched before, so n/s/m/l now see a
different scale distribution and converge to different metrics.

To restore the old behavior, set it explicitly:

```python
from libreyolo.training.config import DFINEConfig

config = DFINEConfig(base_size_repeat=3)
```

DEIM still uses the hardcoded 3.

### Worth knowing, but no action needed

- **Rectangular `imgsz` results changed because they were wrong before.** Box
  coordinates, RTMDet mask resizing, YOLO-NAS rescaling and validator
  ground-truth scaling now use per-axis height and width instead of one
  scalar. Square `imgsz` is bit-unchanged. If you ran rectangular inference or
  validation on v1.4.0, those numbers were mis-scaled. YOLO-NAS now rejects
  rectangular `imgsz` outright rather than silently producing wrong output.
- **Metrics dictionaries gained keys.** `max_det`, `ar_max_det` and
  `AR_max_det` from the COCO evaluator, and `metrics/loss` plus
  `metrics/loss/ce` from FOMO. Values at defaults are unchanged, but anything
  iterating metric keys (custom loggers, CSV headers) sees new columns.
- **Seeded YOLO9 runs that trigger a head rebuild** start from a different
  initialization, because the seed is now applied before the rebuild rather
  than after. A seeded v1.4.0 fine-tune onto a different class count is not
  reproducible bit-for-bit on v1.5.0.
- **`libreyolo[hub-kernels]` on CUDA now actually engages the native
  MS-deform-attn kernel.** v1.4.0 gated it behind a condition RF-DETR never
  took, so the kernel never ran. Predictions can now shift at float tolerance
  for RF-DETR and the other deformable-attention families. Stock installs are
  unaffected; `LIBREYOLO_HUB_KERNELS=0` disables it.
- **`libreyolo predict` drops unsupported options instead of raising.** The
  CLI filters kwargs against the model's `__call__` signature, so an option a
  family does not accept is now ignored rather than raising `TypeError`. A
  typo in a flag name will be silently ignored.
- **Live sources change the JSON output shape.** Webcams, RTSP streams and
  screen capture implicitly enable streaming, which emits one record per frame
  rather than one for the call. These sources are new in v1.5.0, so no v1.4.0
  script is affected.
- **Re-exporting `rfdetr-pose` or `yolonas-pose` to ONNX yields different
  output names.** v1.4.0 misread their multi-tensor pose heads as segmentation
  via an output-count heuristic. Existing `.onnx` files on disk are untouched.
- **On a torch-free install**, results hold numpy arrays rather than
  `torch.Tensor`, so `.boxes.data` returns a different type and NMS
  tie-breaking may differ. With torch installed, behavior is byte-for-byte
  unchanged.
- **Config objects validate more at construction.** `TrainConfig` gained a
  `__post_init__` where it had none, so a config that was already invalid now
  raises immediately instead of failing deep into a run.
- **Weight filenames for task-suffixed families resolve differently.**
  `segformer-b0` now resolves to `LibreSegformerb0-sem.pt`. This fixes
  auto-download 404s, but breaks any script that hardcoded the old
  unsuffixed filename.
- **The pytest marker `experimental_backend` is now `extended_backend`.**
  Only relevant if you run the test suite with `-m`.

### Checkpoints and datasets

Checkpoints written by v1.4.0 load unchanged. The schema gained
`imgsz_h`/`imgsz_w` for rectangular models, and still writes the scalar
`imgsz = max(h, w)` for older readers. ExecuTorch and MNN exports now require
a sidecar (`<program>.pte.json`, `<model>.mnn.json`), and HRNet exports carry
`pose_input: "person_crop"`. Dataset formats are unchanged.
