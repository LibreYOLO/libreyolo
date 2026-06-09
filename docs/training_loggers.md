# Training hooks and experiment loggers

## Training hooks

Every trainer emits four events. Pass handlers via `callbacks=` on
`model.train(...)`:

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
module-level class, not a closure or lambda.

## Built-in loggers

Built-in loggers are callback objects layered on the hooks. Enable by
name or pass configured instances:

```python
model.train(data="coco8.yaml", loggers="tensorboard")

from libreyolo.training import MLflowLogger
model.train(
    data="coco8.yaml",
    loggers=[MLflowLogger(experiment_name="my-exp"), "tensorboard"],
)
```

All three log the same canonical metric names per epoch: `train/loss`,
`train/loss/<component>`, `lr/<group>`, `val/<metric>`,
`time/epoch_seconds`. They also log the resolved training config at
start. A backend failure mid-run (server down, auth expired) disables
the logger with a warning; training is never interrupted. A missing
backend package raises at construction with the install command.

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
