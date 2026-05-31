# LibreYOLO Model Nomenclature

This document catalogs the model-naming conventions **currently in use** in
the LibreYOLO repository. It is descriptive — it records what is there today,
not a proposal. Sources of truth are the `FAMILY` and `FILENAME_PREFIX`
class constants in `libreyolo/models/<family>/model.py` and the
task-resolution rules in [`libreyolo/tasks.py`](../libreyolo/tasks.py).

## Filename schema

Every weight file follows:

```
Libre<FAMILY><size>[-<task>].pt
```

- `FAMILY` — family-specific prefix (see table below).
- `<size>` — single-letter or backbone-named size code. Always **lowercase**,
  attached directly to the family prefix with no separator.
- `<task>` — optional task suffix, hyphen-prefixed.
  Detect is **implicit** (no suffix), following the common YOLO naming convention.

## Family prefixes

The 12 families currently registered:

| Family id (`FAMILY`) | Filename prefix | Casing rule applied |
|---|---|---|
| `yolox`     | `LibreYOLOX`    | All-caps acronym |
| `yolo9`     | `LibreYOLO9`    | All-caps acronym + version digit |
| `yolo9_e2e` | `LibreYOLO9E2E` | All-caps acronym + version + variant |
| `yolonas`   | `LibreYOLONAS`  | All-caps acronym (hyphen dropped from `YOLO-NAS`) |
| `dfine`     | `LibreDFINE`    | All-caps acronym (hyphen dropped from `D-FINE`) |
| `deim`      | `LibreDEIM`     | All-caps acronym |
| `deimv2`    | `LibreDEIMv2`   | All-caps acronym + lowercase version |
| `rtdetr`    | `LibreRTDETR`   | All-caps acronym (hyphen dropped from `RT-DETR`) |
| `rfdetr`    | `LibreRFDETR`   | All-caps acronym (hyphen dropped from `RF-DETR`) |
| `picodet`   | `LibrePICODET`  | All-caps (`PicoDet` rendered uppercase) |
| `ec`     | `LibreEC`    | Short form of EdgeCrafter — used as the family alias for the three sibling upstream models `ECDet`, `ECPose`, `ECSeg` |
| `l2cs`      | `LibreL2CS`     | All-caps acronym (`L2CS` gaze estimation) — inference-only |

Casing rules observed in the table:

1. **Acronyms remain all-caps** (`YOLOX`, `YOLO9`, `YOLONAS`, `DFINE`, `DEIM`,
   `RTDETR`, `RFDETR`).
2. **Hyphens and dots from upstream branding are dropped**
   (`D-FINE` → `DFINE`, `RT-DETR` → `RTDETR`, `RF-DETR` → `RFDETR`,
   `YOLO-NAS` → `YOLONAS`).
3. **Version suffixes are lowercase** (`DEIMv2`, not `DEIMV2`).
4. **`ec` is a family alias, not a single model name.** The EdgeCrafter
   project ships three sibling upstream models — `ECDet`, `ECPose`, `ECSeg`
   — that share a backbone+encoder and differ only in the head. LibreYOLO
   collapses all three into one family (`FAMILY = "ec"`) with three task
   variants (`SUPPORTED_TASKS = ("detect", "pose", "segment")`); the
   filename prefix `LibreEC` is the short form of EdgeCrafter, with the
   task carried in the `-pose` / `-seg` suffix.

The casing rule is now uniform: **every family prefix is all-caps after
`Libre`**, with the only mixed-case fragment being the lowercase version
suffix `DEIMv2`. There are no CamelCase-preserving exceptions.

## Size codes

Sizes are family-specific. The table below records what each family currently
ships:

| Family | Size codes (detect) |
|---|---|
| `yolox`     | `n`, `t`, `s`, `m`, `l`, `x` |
| `yolo9`     | `t`, `s`, `m`, `c` |
| `yolo9_e2e` | `t`, `s`, `m`, `c` (inherited from yolo9) |
| `yolonas`   | `s`, `m`, `l` |
| `dfine`     | `n`, `s`, `m`, `l`, `x` |
| `deim`      | `n`, `s`, `m`, `l`, `x` |
| `deimv2`    | per-cfg (see `SIZE_CONFIGS`) |
| `rtdetr`    | `r18`, `r34`, `r50`, `r50m`, `r101`, `l`, `x` |
| `rfdetr`    | `n`, `s`, `m`, `l` |
| `picodet`   | `s`, `m`, `l` (320 / 416 / 640 input) |
| `ec`     | `s`, `m`, `l`, `x` |
| `l2cs`      | `r18`, `r34`, `r50`, `r101`, `r152` (ResNet backbone depth) |

Notes:

- Standard codes are `n` (nano), `t` (tiny), `s` (small), `m` (medium),
  `l` (large), `x` (xlarge).
- `yolo9` uses `c` for "compact" instead of `l`.
- `rtdetr` mixes backbone-named codes (`r18`, `r50`, …) with letter codes
  (`l`, `x`).

## Task suffixes

From `libreyolo/tasks.py`:

| Task          | Filename suffix |
|---|---|
| `detect`      | *(none — implicit)* |
| `segment`     | `-seg` |
| `pose`        | `-pose` |
| `classify`    | `-cls` |
| `gaze`        | `-gaze` |

The factory accepts upstream-style aliases (`detection`, `det`, `segmentation`,
`keypoints`, `cls`, …) at the API boundary; only the canonical names above
appear in filenames.

## Per-family task support

| Family    | `SUPPORTED_TASKS`                   | Default | Notes |
|---|---|---|---|
| `yolox`     | `("detect",)` (default)             | detect | detect-only |
| `yolo9`     | `("detect",)` (default)             | detect | detect-only |
| `yolo9_e2e` | `("detect",)` (default)             | detect | detect-only |
| `dfine`     | `("detect",)` (default)             | detect | detect-only |
| `deim`      | `("detect",)` (default)             | detect | detect-only |
| `deimv2`    | `("detect",)` (default)             | detect | detect-only |
| `rtdetr`    | `("detect",)` (default)             | detect | detect-only |
| `picodet`   | `("detect",)` (default)             | detect | detect-only |
| `rfdetr`    | `("detect", "segment")`             | detect | seg uses smaller sizes |
| `yolonas`   | `("detect", "pose")`                | detect | pose adds size `n` |
| `ec`     | `("detect", "pose", "segment")`     | detect | all three tasks |
| `l2cs`      | `("gaze",)`                         | gaze   | inference-only; two-stage (face detector + gaze head); not trainable in LibreYOLO |

Families that override `SUPPORTED_TASKS` also declare `TASK_INPUT_SIZES` so
each task can use a different per-size input resolution (relevant for RF-DETR).

## Examples by family + task

### Detection only

```text
LibreYOLOXn.pt
LibreYOLO9s.pt
LibreYOLO9E2Es.pt
LibreYOLONASm.pt
LibreDFINEl.pt
LibreDEIMx.pt
LibreDEIMv2s.pt
LibreRTDETRr50.pt
LibreRFDETRn.pt
LibrePICODETs.pt
LibreECs.pt
```

### Multi-task families

```text
# yolonas — detect + pose
LibreYOLONASs.pt           # detect (default)
LibreYOLONASn-pose.pt      # pose (note: size n only ships for pose)
LibreYOLONASs-pose.pt
LibreYOLONASm-pose.pt
LibreYOLONASl-pose.pt

# rfdetr — detect + segment
LibreRFDETRn.pt            # detect
LibreRFDETRn-seg.pt        # segment

# ec — detect + pose + segment
LibreECs.pt             # detect (default)
LibreECs-pose.pt        # pose
LibreECs-seg.pt         # segment
```

### Gaze (inference-only)

```text
LibreL2CSr50.pt           # L2CS gaze estimation (ResNet-50, Gaze360 weights)
```

`gaze` is L2CS's only task, so — like `detect` for the detection families —
it carries no suffix in the canonical filename; `-gaze` is accepted but
redundant. L2CS weights are not hosted by LibreYOLO (the Gaze360 dataset
license forbids redistribution); see `libreyolo/models/l2cs/model.py`.

## Resolution precedence

When loading via `LibreYOLO("...")`, the task is resolved with this priority
(see `libreyolo/tasks.py:resolve_task` and the factory in
`libreyolo/models/__init__.py`):

```
explicit task=    →    checkpoint["task"]    →    filename suffix    →    family DEFAULT_TASK
```

Official LibreYOLO v1.0 checkpoints must carry `task` metadata; see
[`checkpoint_schema.md`](checkpoint_schema.md). State-dict key inspection is a
legacy compatibility path for old LibreYOLO checkpoints, not the standard for
new artifacts.

## Filename regex

`BaseModel._filename_regex` builds the canonical pattern as:

```
<prefix>(?P<size>{size_alternation})(?P<task>{task_suffixes})?\.pt
```

with `task_suffixes` derived from `SUPPORTED_TASKS` via
`libreyolo.tasks.task_suffix_pattern`. This is the single source of truth for
parsing a filename back into `(family, size, task)`.
