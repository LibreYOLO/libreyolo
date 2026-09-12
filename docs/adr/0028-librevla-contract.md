# ADR 0028: LibreVLA Contract For Vision-Language-Action Policies

- Status: Accepted
- Date: 2026-09-12
- Scope: New sibling tier (`LibreVLA`), new canonical task (`act`), new
  result payload (`Results.actions`)

## Context

A vision-language-action (VLA) model takes camera frames, the robot's
proprioceptive state and a language instruction, and returns the actions the
robot should execute next. It is the perception-to-control layer of a robot
stack, and every current open family (SmolVLA, pi0, pi0.5, MolmoAct2, X-VLA,
GR00T) is published as a Hugging Face policy through the `lerobot` package
(Apache-2.0).

None of LibreYOLO's existing contracts fit it:

- `LibreYOLO(...)` sniffs a `.pt` state dict and runs one forward pass to
  calibrated boxes. A VLA is a multi-file policy directory with a sampling
  loop and no boxes.
- `LibreVLM(...)` is image plus text in, text out, parsed into boxes. A VLA
  emits a continuous action chunk, not text.
- `LibreLLM(...)` is a hosted chat client with no local weights.

The output primitive is also new. Every task in `libreyolo/tasks.py` names an
image-space or per-object primitive. An action chunk is a `(T, D)` sequence
of robot commands. Following ADR 0003 and ADR 0021, the task names the output
primitive, not the domain, so `act` is a canonical task and not a `robot` or
`vla` task.

## Decision

Add `LibreVLA` as a sibling factory, the task `act`, and `Results.actions`.
The library owns the observation contract, the result payload, the offline
training loop, the offline evaluation metric, the checkpoint directory
contract and the user-facing API. The policy network, its processors and
the dataset reader come from `lerobot` through its public API. No upstream
source is ported, adapted or vendored.

The line is the same as ADR 0002 drew: the contract, not the architecture.

- `LibreYOLO(...)`: closed-set detector, real scores, `.pt` checkpoints.
- `LibreVLM(...)`: generative detector, sticky vocabulary, `Results.boxes`.
- `LibreGround(...)`: instruction, click, `Results.points`.
- `LibreVLA(...)`: frames plus state plus instruction, `Results.actions`.

The default model is SmolVLA base (`lerobot/smolvla_base`, Apache-2.0,
about 450M parameters), pinned to a commit and downloaded on first use.

## Task contract: `act`

`act` is the canonical task for models whose public prediction primitive is
an action chunk. Filename suffix `-act`. Aliases `action`, `actions`, `vla`,
`policy`, `robot-policy` resolve to `act` at the API boundary.

`Results.actions` holds an `Actions` payload:

- `data` is `(T, D)` float32: `T` timesteps of a `D`-dimensional action.
  Row 0 is the action to execute now, later rows are the model's plan.
- Values are in the units the model was trained on. LibreYOLO does not
  reinterpret them: for the shipped SmolVLA checkpoints they are absolute
  joint targets in the dataset's own unit. `Actions.names` carries the
  per-dimension names when the checkpoint knows them, `Actions.fps` the
  control rate the chunk was planned at, `Actions.instruction` the text
  that produced it. All three may be `None`.
- `Results.boxes` is `None`. `Results.orig_shape` is the primary camera
  frame's `(H, W)`. `len(result)` is `T`. Slicing a result slices
  timesteps. `summary()` returns one row with the chunk and its metadata.
- `result.plot(image)` renders the frame with a per-dimension trajectory
  strip under it. `save=True` writes that under `runs/act/predict`.

Box-specific paths (tracking, tiling, export backends, mAP validation)
reject `act` results.

## Observation contract

`predict(source, *, state=None, instruction=None, cameras=None, ...)`.

- `source` is one frame in any `ImageInput` form for a single camera, a
  `dict[str, ImageInput]` naming several cameras, a list or directory of
  frames, a video, a webcam index or a stream URL. Frames are original
  camera pixels; the family's processor owns resize and normalization.
- Camera naming: a dict key that matches one of the family's camera slots
  (`camera1`, `camera2`, ... for SmolVLA) is used as such; other keys are
  assigned to free slots in order. Set `cameras=["front", "wrist"]` on the
  model or the call to fix the mapping once. A single frame goes to the
  first slot. Missing slots are legal for families that tolerate them.
- `state` is the proprioceptive vector `(D_state,)`. A callable is called
  once per frame and must return that vector; this is how a live source
  becomes a closed loop without LibreYOLO owning a robot driver. `None`
  substitutes zeros and logs one warning: the call runs, the actions are
  not meaningful.
- `instruction` is the task text. `set_instruction(text)` makes it sticky,
  the way `set_classes` and `set_query` work on the other tiers. A call
  with neither raises.
- `reset()` clears any family-side action queue between episodes.

Depth, tactile, audio and multi-step observation histories are not part of
this version. They enter as additional named observation entries without
changing the call shape.

## Training, validation, checkpoints

`train(data=..., epochs=..., batch=..., lr0=..., ...)` fine-tunes on a
LeRobot dataset (a Hub repo id or a local directory in the LeRobot v3
layout). There is no LibreYOLO dataset YAML for `act`: the LeRobot format
already is the interchange format of the field, and inventing a second one
would only lose users. The trainer:

- splits held-out episodes off the end for validation (`val_split`, default
  10 percent of episodes, or explicit `val_episodes`),
- builds the policy from the pinned base with the dataset's feature shapes
  and statistics through the upstream factory,
- runs a compact epoch loop with the family's optimizer and schedule
  presets, gradient clipping and best/last checkpoints,
- emits the standard training callbacks and loggers,
- returns the standard results dict with `save_dir`, `best`, `last`.

`val(data=..., split=...)` reports offline action error on held-out
episodes: mean L1 and MSE between predicted and recorded action chunks,
overall and per dimension, in the dataset's units. This is the metric the
VLA papers report for offline evaluation, it needs no robot, and it is what
CI can run. It is not a success rate. Simulator success (LIBERO through
`lerobot[libero]`) and real-robot evaluation are the next rungs and land
behind the same `val()` surface with a `sim=` argument; they are out of
scope here.

A VLA checkpoint is a directory, not a `.pt`: the upstream policy files,
the saved pre and post processors, and `libreyolo_vla.json` recording
family, size, base repo and revision, the dataset, action names and the
control rate. `LibreVLA(path)` loads it and restores the instruction-free
default. Schema-v1 `.pt` metadata does not apply.

`export()` raises. The flow-matching action expert has no exportable
graph contract yet; adding one is a later ADR.

## Public API

Amendment (2026-09-12): action policies may opt out of language conditioning
with `REQUIRES_INSTRUCTION = False`; omitted text then resolves to an empty
string. Families with `PRETRAINED_BASE = False` construct an untrained
wrapper and build their default config through `_scratch_config(meta)`.
Training supplies dataset features and statistics without a base-policy
download. Their checkpoints record null `base_repo` and `base_revision`,
and reload exclusively from the saved directory. Prediction requires either
a saved checkpoint or a completed training run; the trained wrapper uses
the last checkpoint. Language-conditioned and pretrained families retain
their existing defaults. `LeRobotPolicyFamily` shares the public-runtime
adapter hooks while allowing family-specific overrides.

```python
from libreyolo import LibreVLA

model = LibreVLA()                                   # smolvla-base, autodownloads
model.set_instruction("pick up the red cube")
result = model.predict(frame, state=q)               # one chunk
result.actions.data                                  # (50, 6)
result.actions.first                                 # (6,) the next action
result.actions.names                                 # per-dimension names

result = model.predict({"front": a, "wrist": b}, state=q)   # several cameras

for result in model.predict(0, state=robot.read, stream=True):   # closed loop
    robot.send(result.actions.first)

results = model.train(data="lerobot/svla_so101_pickplace", epochs=5)
model = LibreVLA(results["best"])
model.val(data="lerobot/svla_so101_pickplace")       # offline action error
```

Install: `pip install "libreyolo[vla]"`. The extra requires Python 3.12 or
newer because `lerobot` does; on older interpreters the extra installs
nothing and the tier raises an `ImportError` that says so. The extra stays
out of `all` for the same reason.

## Families

| Alias | Upstream | License | Status |
| --- | --- | --- | --- |
| `smolvla-base` (default) | `lerobot/smolvla_base` | Apache-2.0 | shipped |
| `pi0`, `pi05` | `lerobot/pi0_base`, `lerobot/pi05_base` | Gemma terms on the PaliGemma base | reserved, not an alias yet |
| `molmoact2` | `allenai/MolmoAct2-*-LeRobot` | Apache-2.0 | reserved, not an alias yet |
| `xvla`, `groot` | lerobot policies | see upstream | reserved, not an alias yet |

Reserved names raise with a message rather than resolving, the way
LibreGround handles unverified families. A family becomes an alias when its
adapter is load-tested against a real checkpoint.

## Licensing

LibreYOLO ships only its own adapter, trainer, metric and payload code. The
policy loads through the Apache-2.0 `lerobot` API. Weights are downloaded
from the upstream Hub repo at a pinned revision; LibreYOLO does not
redistribute them. The SmolVLA base backbone (SmolVLM2-500M) is Apache-2.0.

## Out of scope

- Robot drivers, motor buses, cameras: the library returns actions.
- Simulators and real-robot rollouts in CI.
- Export, tracking, tiling, TTA.
- A CLI verb. The tier is a Python-API surface in this version, as the
  VLM tier was in its first version.
- A reinforcement-learning loop. The trainer is imitation only.

## Consequences

Positive: robot policies get the same three-line experience as every other
task, with training and offline evaluation included; the tier is fully
isolated from the detector factory; a new family is a small adapter over a
lerobot policy class; the observation and action contracts are fixed before
the first simulator or robot integration lands, so those can be added
without changing what users already wrote.

Negative: a second optional heavyweight dependency with its own Python
floor; offline action error is a proxy metric; CPU inference is seconds per
chunk, so live control needs a GPU.
