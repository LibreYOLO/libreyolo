# LibreVLA

`LibreVLA` runs vision-language-action (VLA) policies: camera frames plus the
robot's state plus an instruction in, an action chunk out. It is the
perception-to-control layer of a robot stack, behind the same three lines as
every other LibreYOLO task. The contract lives in
`docs/adr/0028-librevla-contract.md`.

The library returns actions. It does not drive motors, own cameras or run a
simulator. Those sit on either side of the call.

## Install

```bash
pip install "libreyolo[vla]"
```

The extra pulls the Apache-2.0 `lerobot` package, which needs Python 3.12 or
newer. On an older interpreter the extra installs nothing and the tier
raises an `ImportError` that says so. Importing `LibreVLA` never imports
`lerobot`; it loads on first use.

## Predict

```python
from libreyolo import LibreVLA

model = LibreVLA()                                   # smolvla-base, autodownloads
model.set_instruction("pick up the red cube")        # sticky, like set_classes
result = model.predict(frame, state=q)               # one observation

result.actions.data                                  # (50, 6) float32
result.actions.first                                 # (6,) the action to send now
result.actions.names                                 # per-dimension names, if known
result.actions.fps                                   # control rate, if known
result.plot(frame).save("chunk.png")                 # frame + trajectory strip
```

- `frame` is any LibreYOLO image input (path, PIL, NumPy BGR, tensor). It is
  the original camera pixels; the policy's processor owns resize and
  normalization.
- `state` is the proprioceptive vector the policy was trained with (six
  joint positions for the SmolVLA base). Pass a callable and it is read
  once per frame. `None` runs with zeros and logs one warning: the call
  works, the actions are not meaningful.
- `instruction` can also be passed per call: `predict(frame, state=q,
  instruction="open the drawer")`.
  Language-conditioned families require text. Action-policy families that
  declare `REQUIRES_INSTRUCTION = False` accept an omitted instruction and
  return an empty `Actions.instruction` string.

Several cameras:

```python
result = model.predict({"front": a, "wrist": b}, state=q)
model = LibreVLA(cameras=["front", "wrist"])          # fix the slot order once
```

A dict key equal to a policy slot (`camera1`, `camera2`, `camera3` on the
base) takes that slot; other keys fill the free slots in order. `cameras=`
pins the mapping by position. Fewer frames than slots is fine on SmolVLA.

Folders, videos and streams replay open loop, one chunk per frame:

```python
results = model.predict("episode_frames/", state=q)             # list
for r in model.predict("episode.mp4", state=q, stream=True):    # generator
    ...
for r in model.predict(0, state=robot.read, stream=True):       # webcam, closed loop
    robot.send(r.actions.first)
```

`save=True` writes the rendered chunk under `runs/act/predict`. Call
`model.reset()` between episodes to clear the policy's action queue.

## Train

```python
results = model.train(data="lerobot/svla_so101_pickplace", epochs=5, batch=8)
model = LibreVLA(results["best"])
```

- `data` is a LeRobot dataset: a Hub repo id or a local directory in the
  LeRobot v3 layout. There is no LibreYOLO YAML for `act`; the LeRobot
  format already is the interchange format of the field.
- The last ten percent of episodes are held out for validation
  (`val_split`), or pass `val_episodes=[...]` / `train_episodes=[...]`.
- Dataset cameras map onto the policy's slots in order (`up` becomes
  `camera1`, and so on). The checkpoint remembers the dataset's camera
  names, so `predict({"up": frame})` works on the fine-tune.
- `lr0` overrides the family's optimizer preset; `accumulate` scales the
  effective batch; `max_steps` caps steps per epoch for smoke runs;
  `callbacks=` and `loggers=` are the standard training layers.
- Best and last checkpoints are directories under `runs/act/train/weights`,
selected on validation loss.

The SmolVLA recipe trains the action expert with the vision backbone
frozen, as upstream does. LoRA and full backbone training are not exposed
yet.

Families declaring `PRETRAINED_BASE = False` construct without downloading
a base policy. Their default config and feature shapes come from the family
and the training dataset. Prediction before training raises an untrained-model
error. After `train()`, that instance uses the last checkpoint; load
`results["best"]` to use the best checkpoint instead. Training a loaded
checkpoint starts from its saved policy and processors.

## Action policies without language

```python
model = LibreVLA("act")                 # or "diffusion"; initially untrained
run = model.train(data="lerobot/svla_so101_pickplace", epochs=5)
model = LibreVLA(run["best"])
result = model.predict({"up": front, "wrist": wrist}, state=q)
```

Both policies learn the dataset's action representation without a language
instruction. Supply all camera slots used during training. Training starts
with a randomly initialized action policy and the upstream ResNet18 backbone
weights; there is no pretrained robot-policy download.

Diffusion keeps the last `n_obs_steps` observations across calls, padding the
first observation by repetition. Call `reset()` between episodes or before
predicting an unrelated frame. The returned chunk starts at the current
step; validation aligns it with the corresponding part of the dataset's
longer training horizon, including padding masks.

The offline CPU smoke uses tiny configs and a synthetic in-memory dataset:
`PYTHONPATH=. python tests/smoke/vla_action_policies.py` (Python 3.12+, VLA
and Diffusion extras installed). It exercises the real policies, processors,
training, checkpoint reload, prediction and validation without downloads.
It verifies the runtime path, not robot task success.

## Validate

```python
model.val(data="lerobot/svla_so101_pickplace")
# {'val/action_l1': ..., 'val/action_mse': ..., 'val/action_l1_first': ...,
#  'val/action_l1_dims': {'shoulder_pan.pos': ..., ...}, 'val/steps': ..., 'episodes': [...]}
```

Offline action error on held-out episodes, in the dataset's units: mean L1
and MSE between predicted and recorded chunks, the first-step L1, and a
per-dimension table. `split="train"` or `split="all"` change the episodes;
`max_batches=` bounds the run. A fine-tune defaults to its own dataset.

This is the proxy metric VLA papers report offline. It is not a success
rate. Simulator and real-robot evaluation are the next rungs and will land
behind the same `val()` surface.

## Checkpoints

A VLA checkpoint is a directory, not a `.pt`:

```
runs/act/train/weights/best/
  config.json                       # upstream policy config
  model.safetensors                 # policy weights
  policy_preprocessor.json          # saved processor pipelines (+ stats)
  policy_postprocessor.json
  libreyolo_vla.json                # the contract file
```

`libreyolo_vla.json` records the family, size, base repo and pinned
revision, the dataset, its control rate, the camera names in slot order,
the action and state names, and the chunk size. `LibreVLA(path)` reads it
and rebuilds the right family.

For policies trained without a base checkpoint, `base_repo` and
`base_revision` are both null. Reloading uses the saved directory only.

## Families

| Alias | Upstream | Weights | Notes |
| --- | --- | --- | --- |
| `smolvla-base` (default) | SmolVLA 450M (`lerobot/smolvla_base`) | Apache-2.0 | 3 camera slots, state 6, action 6, 50-step chunk |
| `act`, `act-policy` | LeRobot ACT | No pretrained policy; torchvision ResNet18 backbone (BSD-3-Clause code) | No instruction; dataset camera/state/action shapes; 100-step chunk |
| `diffusion`, `diffusion-policy` | LeRobot Diffusion Policy | No pretrained policy; torchvision ResNet18 backbone (BSD-3-Clause code) | No instruction; two observations of history; 64-step training horizon, 32 predicted steps with LeRobot 0.6.1 defaults |

`pi0`, `pi05`, `molmoact2`, `xvla`, `groot` and `openvla` are reserved names
that raise with a message until an adapter is load-tested. Adding a family
is a small class over its lerobot policy: see `libreyolo/models/vla/smolvla.py`.

## Limits

- CPU inference is seconds per chunk; live control needs a GPU. The base
  checkpoint takes a couple of minutes to load the first time.
- `export()`, `track()`, tiling and TTA raise.
- No CLI verb in this version; the tier is a Python-API surface.
- The trainer is imitation learning only.
