# Training hooks and experiment loggers

## Training hooks

Every trainable LibreYOLO model family accepts `callbacks=` and emits four
events through its trainer. This includes YOLOv9, YOLOv9-E2E, YOLOX,
YOLOv7, YOLO-NAS, RT-DETR, RT-DETRv2, RT-DETRv4, RF-DETR, D-FINE, DEIM,
DEIMv2, PicoDet, RTMDet, EC, and FOMO. Inference-only families still raise
`NotImplementedError` from `train()`; this includes SAM, L2CS, Depth
Anything V2, and the VLM tier.

Pass handlers via `callbacks=` on `model.train(...)`:

| Event | When | Key fields |
|---|---|---|
| `TrainStartEvent` | After setup, before the first epoch | `start_epoch`, `total_epochs`, `model_family`, `model_size`, `task`, `save_dir`, `config` |
| `TrainEpochEvent` | After each epoch (train + val) | `epoch`, `train_loss`, `train_loss_items`, `lr`, `val_metrics`, `validated`, `is_best`, `best_metric`, `best_epoch`, `epoch_seconds` |
| `TrainEndEvent` | After training completes | `completed_epochs`, `final_loss`, `best_metric`, `best_epoch`, `total_seconds`, `results` |
| `TrainExceptionEvent` | If training raises | `epoch`, `exception`, `exception_type`, `exception_message`, `elapsed_seconds` |

`TrainStartEvent.config` is the fully resolved training configuration
(user kwargs merged with model-family defaults) as a read-only mapping.

A plain callable receives `TrainEpochEvent` only. An object may implement
any subset of `on_train_start`, `on_train_epoch_end`, `on_train_end`,
`on_train_exception`:

```python
from libreyolo import LibreYOLO9
from libreyolo.training import TrainEpochEvent

def on_epoch(e: TrainEpochEvent):
    print(f"epoch {e.epoch}/{e.total_epochs} loss={e.train_loss:.4f}")

model = LibreYOLO9("yolo9-s.pt")
model.train(data="coco8.yaml", epochs=10, callbacks=on_epoch)
```

Callbacks fire on rank 0 only under DDP. For multi-GPU spawn
(`device="0,1"`), callbacks must be picklable: define them as a
module-level function or class, not a closure or lambda.
Automatic spawn works from both guarded and ordinary unguarded top-level
Python scripts. Guarded-script model classes and callbacks are transported to
the ranks, but arbitrary top-level runtime side effects are not replayed there.
Put code needed by every rank in importable model or trainer execution paths
and key it from the rank environment variables instead of relying on a guarded
script to run again in each rank. A rare callback that standard pickle can
reference but cannot be transported by value (for example, one that captures a
write-only file handle) uses the guarded-script compatibility launcher and
emits a warning; keep that call under the `__main__` guard.

## Built-in loggers

Built-in loggers are callback objects layered on the same universal hooks.
Enable TensorBoard, MLflow, Weights & Biases, Comet, ClearML, Neptune or
DVCLive by name, or pass configured instances:

```python
model.train(data="coco8.yaml", loggers="tensorboard")
model.train(data="coco8.yaml", loggers="mlflow")

from libreyolo.training import MLflowLogger
model.train(
    data="coco8.yaml",
    loggers=[MLflowLogger(experiment_name="my-exp"), "tensorboard"],
)
```

All seven log the same canonical metric names per epoch: `train/loss`,
`train/loss/<component>`, `lr/<group>`, `val/<metric>`,
`time/epoch_seconds`. They also log the resolved training config at
start. A backend failure mid-run (server down, auth expired) disables
the logger with a warning; training is never interrupted. A missing
backend package raises at construction with the install command.

### Validation loss

Every trainable family (`g0`, `g1` and `g2` in the model registry) can opt in
to a validation loss reported alongside its accuracy metric:

```python
model.train(data="coco8.yaml", val_loss=True)
```

Supported families and the components each reports, all prefixed `val/loss/`:

| Task | Family | Components |
| --- | --- | --- |
| detect | `yolo9`, `yolo9_p2` | `box`, `cls`, `dfl` |
| detect | `yolo9_e2e` | `box`, `cls`, `dfl` (one-to-many plus one-to-one) |
| detect | `yolonas` | `cls`, `iou`, `dfl` |
| detect | `rfdetr` | `ce`, `bbox`, `giou` |
| detect | `rtdetr`, `rtdetrv2` | `vfl`, `bbox`, `giou` |
| detect | `dfine` | `vfl`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `deim`, `deimv2`, `rtdetrv4`, `ec` | `mal`, `bbox`, `giou`, `fgl`, `ddf` |
| detect | `rtmdet` | `cls`, `bbox` |
| detect | `picodet` | `cls`, `bbox`, `dfl` |
| detect | `yolox` | `iou`, `obj`, `cls`, `l1` |
| detect | `yolo7` | `iou`, `obj`, `cls` |
| point | `fomo` | `ce` |
| classify | `resnet`, `convnext`, `mobilenetv4`, `efficientnetv2` | `ce` |
| semantic | `segformer`, `lingbotvision`, `dinov2` | `sem` |
| restore | `nafnet` | `restore` |

Components are weighted exactly as training weights them, so they sum to the
reported `val/loss`. The always-on artifact names are the corresponding
`metrics/loss...` keys, and `libreyolo monitor` overlays `metrics/loss` with
`train/loss`.

The validator reuses the model output already produced for the accuracy
metric; it does not run a second network forward. Most families need nothing
beyond that output. Three need a little more, still without re-running the
backbone:

- The DETR-line decoders score only their evaluation layer and return a
  two-key dict in eval. For the duration of the validation pass they also
  emit the auxiliary-decoder, encoder and pre-decoder outputs their criterion
  consumes. The predictions the metrics use are unchanged, and the extra work
  is per-layer prediction heads, not extra decoder layers.
- YOLO9-E2E infers through its one-to-one branch only, so the one-to-many
  branch is rebuilt from the neck features the eval forward already published.
- YOLOX's eval branch sigmoids obj/cls and skips the grid bookkeeping its
  criterion needs. Inside the validation pass its head assembles a second,
  training-shaped set of tensors from the same convolution outputs. The
  returned inference tensor is untouched, so predictions and mAP are
  unchanged.

The reported total covers the same terms as training, with two documented
exceptions:

- Contrastive-denoising groups need the ground truth at forward time, and
  validation forwards without it, so `dn_*` terms are never included. RF-DETR
  is the analogous case with `group_detr` collapsing to 1 in eval.
- The number is computed on the evaluation/EMA model, exactly like the
  accuracy metric. Where a family's train and eval forwards genuinely differ
  (BatchNorm running statistics, stochastic depth), the validation loss
  reflects the eval-mode model. That is the intended comparison, not a
  discrepancy.

FOMO is the one family where `val_loss=True` changes nothing: its validator
has always computed this loss unconditionally. It is now published under the
shared `metrics/loss` keys as well as the older `metrics/val_loss`, so the
monitor overlays it like every other family.

This option is off by default because target assignment adds work and memory
to validation. It runs under `torch.no_grad()` with the evaluation/EMA model,
and distributed training computes it locally on rank 0 without collectives.
Best-checkpoint selection remains based on the configured accuracy metric.
Augmented validation, a task a family has not implemented it for, and
inference-only (`g3`/`g4`) families all raise a clear configuration error.

### TensorBoard

```
pip install libreyolo[tensorboard]
```

`TensorBoardLogger(log_dir=None)` — event files default to
`<save_dir>/tensorboard`. View with `tensorboard --logdir runs/train`.

### MLflow

```
pip install libreyolo[mlflow]
```

`MLflowLogger(tracking_uri=None, experiment_name=None, run_name=None,
log_artifacts=True, log_checkpoints=False)` — the tracking URI falls
back to `MLFLOW_TRACKING_URI`, then MLflow's default local store. At
train end it uploads `results.csv`, `train_config.yaml` and
`summary.json` (plus `weights/best.pt` with `log_checkpoints=True`) and
closes the run as FINISHED, or FAILED if training raised.

Note: MLflow 3.x deprecated the local `./mlruns` file store and raises
unless `MLFLOW_ALLOW_FILE_STORE=true`. For server-less local tracking
pass a database URI instead, e.g.
`MLflowLogger(tracking_uri="sqlite:///mlflow.db")`, and view it with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`.

### Weights & Biases

```
pip install libreyolo[wandb]
```

`WandbLogger(project=None, name=None, entity=None,
log_checkpoints=False)` — project falls back to `WANDB_PROJECT`, then
`"libreyolo"`. The resolved config becomes the run config;
`log_checkpoints=True` uploads `weights/best.pt` as a model artifact.

Run names default to `<family><size>-<task>` (e.g. `yolo9s-detect`).

### Comet

```
pip install libreyolo[comet]
```

`CometLogger(project_name=None, workspace=None, name=None, api_key=None,
online=None, log_artifacts=True, log_checkpoints=False)` uses the current
`comet_ml.start()` API. The project falls back to `COMET_PROJECT_NAME`, then
`"libreyolo"`; credentials fall back to Comet's normal `COMET_API_KEY` or
`comet login` configuration. Set `online=False` for an offline experiment.

### ClearML

```
pip install libreyolo[clearml]
```

`ClearMLLogger(project_name="LibreYOLO", task_name=None, tags=None,
output_uri=None, log_artifacts=True, log_checkpoints=False)` creates a fresh
ClearML training task, reports the resolved config under `TrainConfig`, and
marks the task failed when training raises. Automatic framework connection is
disabled to prevent duplicate metric reporting; ClearML's configured server
and credentials are otherwise used normally.

### Neptune

```
pip install libreyolo[neptune]
```

`NeptuneLogger(project=None, api_token=None, name=None, run_id=None,
tags=None, mode=None, capture_console=False, log_artifacts=True,
log_checkpoints=False)` uses Neptune's current `neptune-scale` client, not the
legacy `neptune` package. Project and credentials fall back to
`NEPTUNE_PROJECT` and `NEPTUNE_API_TOKEN`. Use `mode="offline"` for local
logging or `run_id=` to attach to an existing run.

The stable Neptune client currently requires `protobuf<7`, while the TFLite
extra requires protobuf 7 through `onnx2tf`. For that reason Neptune is not
included in `libreyolo[all]`; install `libreyolo[neptune]` in an environment
without the TFLite extra.

### DVCLive / DVC

```
pip install libreyolo[dvclive]  # or libreyolo[dvc]
```

`DVCLiveLogger(log_dir=None, resume=None, report=None, save_dvc_exp=False,
dvcyaml=None, monitor_system=False, log_checkpoints=False)` writes to
`<save_dir>/dvclive`; both `loggers="dvclive"` and `loggers="dvc"` activate
it. On resumed LibreYOLO training it resumes the DVCLive history and uses the
trainer's 1-based epoch as the DVCLive step.

DVCLive uses `/` to build its summary tree and cannot store a float at a path
that is also a parent (for example, both `train/loss` and
`train/loss/box`). LibreYOLO keeps the parent name and dot-encodes only the
conflicting child separator: `train/loss.box`. This preserves every metric
without disabling the logger; non-conflicting names stay unchanged.

DVCLive itself normally defaults to saving a DVC experiment and writing a
root `dvc.yaml`. LibreYOLO deliberately defaults both behaviours off so an
opt-in logger does not create Git/DVC state outside the training run. Pass
`save_dvc_exp=True` and/or an explicit `dvcyaml=".../dvc.yaml"` when that is
the desired workflow. When `log_checkpoints=True` is explicitly enabled,
checkpoint registration uses `cache=False`, so it never silently adds the
checkpoint to the DVC cache. As with DVCLive directly, artifact registration
requires a DVC repository.

## Always-on run status (`status.json`, `metrics.jsonl`, `train.log`)

Separate from the opt-in loggers above, every training run (all
families, no configuration) writes a small set of monitoring artifacts into
its `save_dir`. They exist so an agent-launched run can be watched cheaply,
without a third-party account or tailing the full log.

| File | Written | Contents |
|---|---|---|
| `status.json` | rewritten atomically every epoch (+ on start/end/failure) | live snapshot: `state` (`running`/`completed`/`failed`), `current_epoch` (one-based, the epoch that last finished), `completed_epochs`, `total_epochs`, `progress`, `eta_seconds`, latest `metrics`, `best_metric`/`best_epoch`, and on failure an `error` `{type, message}` |
| `metrics.jsonl` | appended once per epoch | one JSON row per epoch (same schema as the family `results.csv`), the full history for charts |
| `train.log` | tee'd live | the run's `libreyolo` console output |

These are produced by `TrainingStatusCallback`, attached automatically
alongside the family artifact writer. `status.json` is the cheap read for a
polling agent (a few tokens vs. re-parsing a log); the atomic write means a
reader never observes a half-written file.

### Live web dashboard

```bash
libreyolo monitor                     # watch the most recent run under runs/
libreyolo monitor runs/train/exp      # watch a specific run
```

`libreyolo monitor` serves a zero-dependency (stdlib HTTP server) browser
dashboard over the files above: live metric charts, the log tail, and any
validation/plot images, auto-refreshing while the run is active. It is
read-only and never touches the training process, so it attaches to a live
run, re-opens a finished one, or inspects a crashed one, and keeps working
even if the trainer dies.
