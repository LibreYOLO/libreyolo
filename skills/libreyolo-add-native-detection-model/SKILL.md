---
name: libreyolo-add-native-detection-model
description: >-
  Add a detection model to LibreYOLO. Covers both YOLO-grid families (YOLOX,
  YOLOv9, YOLO-NAS) and DETR-style families (RF-DETR, D-FINE; future RT-DETR).
  Includes the universal contract (BaseModel / BaseTrainer / TrainConfig /
  BaseValPreprocessor), the per-family integration patterns from real ports,
  and the trade-off LibreYOLO has explicitly made: best-in-class fine-tuning
  UX, not from-scratch paper-recipe reproduction.
---

# Adding a detection model to LibreYOLO

## 1. What LibreYOLO is

A single `LibreYOLO("Libre<Family><size>.pt")` factory dispatches to family-local
implementations. Each family lives entirely inside `libreyolo/models/<family>/`
and plugs into 4 shared ABCs. The factory iterates `BaseModel._registry` (auto-
populated via `__init_subclass__`) and the first `can_load(state_dict)` match wins.

```
libreyolo/
├── models/<family>/      ← all family code lives here
├── training/             ← BaseTrainer, schedulers, EMA, TrainConfig
├── validation/           ← DetectionValidator + BaseValPreprocessor subclasses
├── backends/             ← ONNX / TensorRT / OpenVINO / NCNN / TorchScript runtime
├── export/               ← BaseExporter and per-format export helpers
└── data/                 ← YOLODataset / COCODataset / dataloader / collate
```

## 2. Currently supported

| Family   | Sizes              | Paradigm | Native code | License (upstream) |
|----------|--------------------|----------|-------------|---------------------|
| YOLOX    | n / t / s / m / l / x | YOLO-grid | yes        | Apache-2.0         |
| YOLOv9   | t / s / m / c      | YOLO-grid | yes         | MIT (`MultimediaTechLab/YOLO`) |
| YOLO-NAS | s / m / l          | YOLO-grid | yes         | Apache-2.0 code; weights non-redistributable (Deci CDN) |
| RF-DETR  | n / s / m / l      | DETR      | wrapper     | Apache-2.0         |
| D-FINE   | n / s / m / l / x  | DETR      | yes         | Apache-2.0         |

⚠️ **YOLOv9 license is *not* the same as the original paper repo.** LibreYOLO's
YOLOv9 port follows `MultimediaTechLab/YOLO` (MIT, by Kin-Yiu Wong and Hao-Tang Tsui),
**not** `WongKinYiu/yolov9` (GPL-3.0, the paper's reference implementation). When
you encounter a model with multiple upstream forks, check each — the permissive
fork is what makes integration into MIT-licensed LibreYOLO clean.

"Native code" means the model + loss + trainer live in `libreyolo/models/<family>/`.
"Wrapper" means the family delegates to a separate PyPI package (RF-DETR uses
`rfdetr` directly; the LibreYOLO code is mostly an adapter).

## 3. License check — do this first

LibreYOLO core is **MIT**, and the project explicitly tries to stay MIT-compatible.
Before porting anything, check **both** licenses on the upstream:

1. **The code** — read the upstream's `LICENSE` file (don't trust the README).
2. **The weights** — check the HuggingFace model card / release page / project site.
   Code and weights can have *different* licenses (e.g. Apache code + GPL weights, or
   permissive code + a custom commercial-restriction license on weights).

| Upstream license | Code in main? | Rehost weights under `LibreYOLO/` HF? | Action |
|---|---|---|---|
| MIT / Apache-2.0 / BSD | ✅ | ✅ | normal integration; ship NOTICE per upload-hf-model skill |
| GPL-3.0 | ⚠️ ship as plugin only | ❌ never (forces GPL on users) | `libreyolo-<family>` separate package; user downloads weights from upstream |
| AGPL-3.0 | ⚠️ plugin only | ❌ never (network-use clause is broader) | same as GPL but stricter |
| Custom / non-redistributable | case-by-case | ❌ usually | link to upstream CDN, like YOLO-NAS does for Deci's bucket |

Code-license rationale: GPL is "viral" — putting GPL code inside MIT-licensed
LibreYOLO would force the entire library to become GPL, breaking every
downstream user who relies on MIT terms.

Weights-license rationale: the legal status of weights under GPL is unsettled,
but the conservative interpretation says any code that links a loaded GPL weight
becomes a combined work and inherits GPL on distribution. Don't expose users
to that without an explicit opt-in.

**Already-shipped examples**:
- D-FINE — Apache-2.0 code + Apache-2.0 weights → clean integration in core, weights rehosted on `LibreYOLO/LibreDFINE*`.
- RF-DETR — Apache-2.0 → clean.
- YOLOv9 — MIT code + MIT weights, via `MultimediaTechLab/YOLO` (Kin-Yiu Wong & Hao-Tang Tsui). Ported from the permissive fork, **not** `WongKinYiu/yolov9` (GPL-3.0).
- YOLO-NAS — Apache-2.0 code + custom Deci CDN for weights → linked rather than rehosted.
- YOLO-World — GPL-3.0 code + GPL-3.0 weights → flagged in #108; **plugin-only is the right call** even though wondervictor (paper first author) actively distributes.

When unsure, fetch the actual LICENSE file:

```
curl -sL https://raw.githubusercontent.com/<org>/<repo>/<branch>/LICENSE | head -3
```

The first line is canonical. "GNU GENERAL PUBLIC LICENSE Version 3" or
"GNU AFFERO GENERAL PUBLIC LICENSE Version 3" → copyleft, treat carefully.
"MIT License" or "Apache License Version 2.0" → permissive, proceed.

For weights, check the HF model card YAML at the top:

```yaml
license: gpl-3.0    # ← copyleft
license: apache-2.0 # ← permissive
license: cc-by-nc-4.0 # ← non-commercial: usually a hard "no" for inclusion
```

## 4. The 4 ABCs every family plugs into

| ABC | File | Required overrides |
|---|---|---|
| `BaseModel` | `models/base/model.py` | `can_load`, `detect_size`, `detect_nb_classes`, `_init_model`, `_get_available_layers`, `_preprocess`, `_forward`, `_postprocess`, `_get_preprocess_numpy` |
| `BaseTrainer` | `training/trainer.py` | `_config_class`, `get_model_family`, `get_model_tag`, `create_transforms`, `create_scheduler`, `get_loss_components` |
| `TrainConfig` | `training/config.py` | dataclass subclass with `kw_only=True`, override only fields that differ |
| `BaseValPreprocessor` | `validation/preprocessors.py` | `__call__`, `normalize`, optionally `uses_letterbox` and `custom_normalization` |

Auto-registration kicks in on import: `models/__init__.py` adds one line per
family. **Import order = `can_load` priority** when heuristics overlap.

## 5. Two architectural patterns

Pick one. The contracts diverge non-trivially.

### YOLO-grid pattern (YOLOX, YOLOv9, YOLO-NAS)

- Model output: per-scale tensor list. Shape and contents differ across families: YOLOX is `(B, 5+nc, H, W)` per scale (4 reg + 1 objectness + nc classes), with grid offsets and exp applied only in export mode; YOLOv9 / YOLO-NAS drop the objectness channel and emit `(B, 4+nc, N)`-shaped tensors. Confirm your family's exact shape against the upstream head.
- Training targets: `(B, max_labels, 5)` padded `[class, cx, cy, w, h]` pixel coords.
- Loss: per-anchor + assignment (SimOTA / TaskAlignedAssigner / DFL).
- Augmentation: numpy/cv2, mosaic + mixup central. Lives in `training/augment.py`
  (`MosaicMixupDataset`, `random_affine`, etc.) — reused across YOLO families.
- ONNX: 1 output named `"output"`, opset 13 default works.
- Detection head exposes `self.export: bool` flipped by the exporter at
  `export/exporter.py:_model_context`.
- NCNN / OpenVINO / TensorRT / TorchScript all work out of the box.

### DETR pattern (RF-DETR, D-FINE; future RT-DETR)

- Model output: dict `{"pred_logits": (B, Q, nc), "pred_boxes": (B, Q, 4)}` cxcywh in [0, 1].
- Training targets: `list[dict{labels, boxes_cxcywh_normalized}]` per image — no padding.
- Loss: Hungarian matching + auxiliary outputs across decoder layers (FGL, GO-LSD for D-FINE).
- Augmentation: torchvision v2 transforms with `tv_tensors.Image` + `tv_tensors.BoundingBoxes`.
- Multi-scale: per-batch random resize via a custom collate (`BatchImageCollateFunction`-style).
- Backbone LR multiplier (0.1× or 0.5×) is standard. Implies per-group LR application
  in `_train_epoch` — needs to be overridden, hooks aren't enough.
- Gradient clipping (`max_norm=0.1`) is standard.
- ONNX: 2 outputs `["pred_logits", "pred_boxes"]`, opset ≥ 16 (for `grid_sample`).
- Export wrapper: a small `nn.Module` that calls `model.deploy()` (recursive
  `convert_to_deploy` on every submodule that defines it) and flattens dict→tuple.
- **NCNN does not work** for DETR-family models — its op registry lacks `topk`.
- EMA mid-training decay change (`set_decay`) is sometimes used to stabilize the
  final phase after augmentation stops.

### Sibling architectures

When your family is the *same architecture* as an existing family with a
different training objective (e.g. a port whose architecture is identical
to an existing port, with only the loss / matcher changed), landmine #3's
"match on tokens unique to your architecture" advice fails because the
architectures are literally identical and both `can_load` checks fire on
the same checkpoint. The disambiguation pattern:

1. Embed an explicit `model_family` field in the converted checkpoint's
   metadata (the conversion script's job). This is the strongest signal.
2. Use the family's `FILENAME_PREFIX` (e.g. `LibreDFINE`, `LibreDEIM`)
   in `detect_size_from_filename` as a fallback hint when metadata is
   absent, e.g. someone hands you a raw upstream `.pth` they renamed.
   Each sibling abstains on the other's prefix.
3. Order the registry imports in `libreyolo/models/__init__.py` so the
   more-specific family loads first; `BaseModel._registry` is walked
   in import order and the first `can_load` match wins.
4. Raise an explicit "ambiguous between {A, B}" error on a true tie
   (architecture-equal checkpoint, no metadata, no filename hint)
   rather than silently picking one.

This pattern is reusable for any descendant family that inherits an
existing port's architecture.

## 6. The training-recipe trade-off (read this before claiming a port is "done")

LibreYOLO has explicitly chosen **not to reproduce upstream paper recipes for
from-scratch training**. From-scratch reproduction would require sponsoring
hundreds of GPU-hours per family and matching every augmentation, EMA quirk,
loss weight, and warmup detail. That's not what 99% of users want.

What LibreYOLO *does* aim for:

1. **Inference parity** — bit-equivalent outputs vs. upstream on the released
   checkpoints. This is non-negotiable.
2. **Best-possible fine-tuning** — a user can load the upstream pretrained
   checkpoint and fine-tune on a small custom dataset, getting within ~1 mAP of
   what `python -m upstream.train ...` would have produced.

Concretely, this means **every existing family has gaps relative to its upstream
training recipe**, and that's by design. Examples we've already accepted:

- D-FINE skips Objects365 pretraining (not relevant for fine-tune users).
- D-FINE used to skip `RandomZoomOut` / `RandomIoUCrop` / `RandomPhotometricDistort`
  in v1 (added in a later commit, ~+1.5 mAP gain).
- YOLOv9 doesn't ship the auxiliary "branch" head used during from-scratch training.
- YOLO-NAS uses LibreYOLO's letterbox preprocessing (close but not identical to
  SuperGradients' exact pipeline) — documented as a known parity gap.

**When you port a new model**:

1. **Spend agentic time reading the upstream training code first.** The model's
   forward pass is the easy 30%. Augmentations, optimizer param groups, LR
   schedule shape, EMA dynamics, multi-scale collation, gradient clipping —
   that's the other 70%, and it dominates fine-tune quality.
2. **Decide explicitly which pieces you skip.** Document them in a commit
   message or in the family's docstring.
3. **Aim for fine-tune parity, not paper parity.** Test by loading upstream
   weights, fine-tuning on `coco128` or `marbles` for ~10 epochs, and verifying
   mAP improves.
4. **Don't pretend gaps don't exist.** If the augmentation chain is half what
   upstream uses, say so. A 5-line `transforms.py` that says "TODO: port
   `RandomZoomOut`" is more honest than silently shipping a degraded recipe.

A useful agent prompt:
*"In `<upstream-repo>/`, identify every augmentation, loss weight, optimizer
param group, LR schedule, and EMA behavior used during training. Output a
concrete checklist of what would need to be ported."*

### The minimum bar for "this port loads upstream weights correctly"

Before claiming inference is correct, run a tensor-equivalence check:

1. Import the upstream model class side-by-side with yours.
2. Build both with the same config / size; cross-load the upstream
   `state_dict` into yours and inspect the missing/unexpected key
   diff. Use `strict=True` only if your port loads the *full* upstream
   state dict (this is the case for D-FINE/DEIM-style DETR ports that
   mirror upstream attribute naming exactly). For ports that
   intentionally drop upstream layers (YOLOX strips training-state
   buffers; YOLOv9 drops the auxiliary head and remaps legacy
   `detect.*` -> `head.*` keys; YOLO-NAS unwraps SuperGradients EMA
   buffers), use `strict=False` and assert that the missing/unexpected
   key set matches a *documented expected set*. Silent unexpected
   drift, not the strict mode itself, is the thing to catch. See
   landmine #14 for `_strict_loading()`.
3. Run identical inputs through both at FP32 and assert
   `max_abs_diff == 0` on the output tensors that come from layers
   present in both models, in both `eval()` and `train()` modes.

This recipe validates architectural fidelity, attribute naming, and
state-dict compatibility in one shot. Save the script as a one-off
under the family's test directory; it pays for itself on every future
upstream version bump. Fine-tune sanity (load -> 10 epochs on coco128
or marbles -> mAP improves) is the second gate, not the first.

If your family is a wrapper around an upstream PyPI package (RF-DETR
pattern), this check reduces to "import the upstream package and verify
it produces the documented outputs"; substitute accordingly.

## 7. Per-family integration: what each one actually shipped

### YOLOX (`models/yolox/`)
- Files: `__init__.py`, `model.py`, `nn.py`, `trainer.py`, `utils.py`, `loss.py` (6 files).
- Pattern: YOLO-grid. Pixel cxcywh targets, BGR 0–255 inference, letterbox preprocessing.
- Augmentations: reuses shared `libreyolo/training/augment.py` (mosaic + mixup); no family-local `transforms.py`.
- Recipe gaps: minimal. Closest to upstream of any family.

### YOLOv9 (`models/yolo9/`)
- Files: 7 files. Largest YOLO-grid port (~2.3k LoC) due to ELAN/RepNCSPELAN modules.
- Pattern: YOLO-grid. RGB 0-1 normalisation. Validation path letterboxes; the default
  inference path uses plain `Image.resize` (`utils.py:_postprocess` defaults
  `letterbox=False`), a known asymmetry called out in the val-preprocessor docstring.
- Recipe gaps: from-scratch auxiliary head dropped; mixup disabled; trainer builds
  three param groups (BN / Conv / Bias) at the same `lr` rather than upstream's
  three at distinct LRs (no backbone-LR split).

### YOLO-NAS (`models/yolonas/`)
- Files: 7 files. Native nn but state-dict-compatible with SuperGradients' SG checkpoints.
- Pattern: YOLO-grid.
- Recipe gaps: SG's exact augmentation pipeline replaced by LibreYOLO's standard letterbox path (documented).
- Quirks: weights download from Deci's CDN, not LibreYOLO's HF org (license).

### RF-DETR (`models/rfdetr/`)
- Files: 6 files (`__init__.py`, `config.py`, `model.py`, `nn.py`, `trainer.py`, `utils.py`).
  RF-DETR is the only family that keeps its `<Family>Config` family-local
  (in `models/rfdetr/config.py`) rather than appending it to
  `libreyolo/training/config.py`.
- Pattern: DETR. **Wrapper, not native** — delegates to the `rfdetr` PyPI package.
- Subprocess isolation lives in `tests/e2e/test_rf1_training.py`, not in
  the trainer itself; the trainer calls upstream `model.train()` directly.
- Recipe gaps: training is upstream's, so few. Inference path adapts upstream's
  postprocessor (cxcywh → xyxy, COCO 91→80 class remap).

### D-FINE (`models/dfine/`)
- Files: 16 files. Largest port (~4k LoC).
  - Architecture: `nn.py`, `backbone.py`, `encoder.py`, `decoder.py`, `common.py`, `ms_deform.py`
  - Numerical core: `fdr.py` (FDR math, separately parity-tested), `box_ops.py`
  - Loss: `loss.py`, `matcher.py`, `denoising.py`
  - Wrapper + IO: `model.py`, `utils.py`, `transforms.py`, `trainer.py`
- Pattern: DETR.
- Recipe gaps: from-scratch HGNetV2 pretrained backbone download disabled (users start
  from D-FINE's own COCO/obj2coco checkpoints). Backbone-LR multiplier added in v2.
  Fine-tune now closely matches upstream's recipe.

### Shared conversion helpers (`weights/_conversion_utils.py`)

When a family does need a conversion script, use the shared helpers
rather than reinventing the plumbing. They cover the parts that every
script ends up needing:

| Helper | Purpose |
|---|---|
| `add_repo_root_to_path()` | so `python weights/convert_*.py` can import `libreyolo.*` cleanly |
| `load_checkpoint(path)` | `torch.load` with `map_location="cpu"` and `weights_only=False`; centralises the post-PyTorch-2.6 default change so it cannot bite per-script |
| `extract_state_dict(ckpt, *, prefer_ema=True)` | unwraps the common upstream layouts: `{"ema": {"module": ...}}`, `{"model": ...}`, `{"state_dict": ...}`, raw dicts, or anything with a `.state_dict()` method |
| `strip_state_dict_prefix(state_dict, prefix)` | drops a leading prefix when the upstream wrapper is `model.model.<...>` |
| `wrap_libreyolo_checkpoint(state_dict, *, model_family, size, nc, names=None)` | builds the canonical metadata-wrapped LibreYOLO format; `build_class_names(nc)` falls back to COCO-80 names for `nc=80`, generic `class_<i>` otherwise |
| `save_checkpoint(checkpoint, output_path)` | creates parent dirs and writes |

`weights/README.md` classifies each shipped conversion as one of three
tiers, which is the right framing for a new one too:

- **metadata-wrap** (D-FINE, DEIM): module names already match,
  `extract_state_dict` -> `wrap_libreyolo_checkpoint` -> `save_checkpoint`,
  ~50-100 LoC end-to-end.
- **light structural** (RT-DETR HGNetv2): EMA unwrap + a small set of
  encoder/decoder key remaps + drop-list for tensors absent in
  LibreYOLO's port. Saves a flat converted `state_dict` (no metadata
  wrap on this path historically).
- **heavy structural** (YOLOv9): translate numbered upstream layer
  indices into LibreYOLO semantic module names, remap sublayer names
  for ELAN / RepNCSPELAN / AConv / ADown / SPP / heads, skip the
  auxiliary head, inject fixed DFL weights. Hundreds of LoC.

Reference test for the helpers: `tests/unit/test_weight_conversion_utils.py`.

### Files-touched matrix (universal centralizing files)

Every family edits these:

| File | Why |
|---|---|
| `libreyolo/models/<family>/{__init__.py, model.py, nn.py, trainer.py, utils.py}` | family-local code |
| `libreyolo/models/__init__.py` | one-line family import (drives auto-registration order) |
| `libreyolo/__init__.py` | `LibreYOLO<Family>` export + `__all__` |
| `libreyolo/training/config.py` | append `<Family>Config(TrainConfig)` (RF-DETR is the exception: keeps `RFDETRConfig` family-local at `models/rfdetr/config.py`) |
| `libreyolo/validation/preprocessors.py` | append `<Family>ValPreprocessor` |
| `tests/unit/test_<family>_*.py` | parity / shape / loss / smoke tests against upstream — done before claiming inference is correct |
| `tests/e2e/conftest.py` | append rows to `MODEL_CATALOG` |

Conditional edits depending on family:

| File | When |
|---|---|
| `libreyolo/models/<family>/loss.py` | non-trivial loss (everyone except RF-DETR) |
| `libreyolo/models/<family>/transforms.py` | augmentation diverges from the shared `training/augment.py` mosaic+mixup default (D-FINE adds tv2-based ops; YOLOv9 / YOLO-NAS subclass) |
| `libreyolo/training/scheduler.py` | paper recipe needs an LR shape that doesn't exist yet — e.g. D-FINE added `FlatCosineScheduler` (warmup → flat → cosine tail). Add a new generic `BaseScheduler` subclass; do **not** put schedulers under `models/<family>/` |
| `libreyolo/training/ema.py` | EMA decay needs to change at runtime (e.g. mid-train restart). D-FINE added `set_decay(decay, ramp=False)` — generic enough to leave shared |
| `libreyolo/backends/base.py` | output shape diverges from YOLO grid (DETR families need it) |
| `libreyolo/backends/tensorrt.py` | output names differ from `"output"` (DETR families) |
| `libreyolo/export/exporter.py` | needs an `_model_context` branch (D-FINE has one for the deploy wrapper) |
| `libreyolo/export/onnx.py` | output count differs from 1 or 3 (DETR's 2-output case) |
| `weights/convert_<family>_weights.py` | needed when upstream ships a checkpoint format LibreYOLO can't load directly (extra wrapping, EMA buffer drops, key remaps, or just no `model_family` metadata). **Skip it** when (a) your family is a wrapper that consumes upstream checkpoints in-process (RF-DETR), (b) your top-level module attributes mirror upstream's so SG/upstream `state_dict`s load with `_strict_loading=False` plus an in-process unwrap helper (YOLO-NAS, YOLOX), or (c) the conversion is trivial enough to keep inside `LibreYOLO("upstream.pt")`. When you *do* write one: wrap with metadata (`model_family`, `size`, `nc`, `names`) so the factory routes without filename heuristics; print a missing/unexpected-key diff after loading the wrapped dict into a fresh model (silent drops are a frequent source of slow-burn fine-tune bugs); write atomically (`.tmp` + rename) so an interrupted run can't half-write a corrupt `.pt`; fail loudly on shape mismatches. The script can stay under ~100 LoC if your top-level attributes mirror upstream's names so no key remapping is needed (YOLO-NAS is the canonical example of this design). |
| `pyproject.toml` | **mandatory for wrapper integrations** (RF-DETR's `[rfdetr]` extra is required, not optional — the wrapper is non-functional without the dep). For native ports, only if you genuinely can't avoid a new dep. |

## 8. The integration-proof tests

You're integrated when both pass for every size of your family.

| Test | What it proves | Notes |
|---|---|---|
| `tests/e2e/test_val_coco128.py` | Inference loads + runs; preprocessing + class mapping + postprocessing are correct. | Runs `model.val(data="coco128.yaml")`, asserts mAP50-95 ≥ 0.18. |
| `tests/e2e/test_rf1_training.py` | Training improves the model on a real dataset (marbles). | Trains 10 epochs, asserts post-mAP > pre-mAP and post-mAP ≥ 0.05. |

To wire your family into both: append rows to `MODEL_CATALOG` in
`tests/e2e/conftest.py`. Both tests parametrize over the catalog.

### Optional faithfulness gate

`test_val_coco128`'s mAP50-95 ≥ 0.18 floor is a sanity check that
preprocessing and class mapping are wired correctly; it is not a
faithfulness check. A silent regression in the conversion script or
numerical drift in the model port can still leave the floor intact while
losing several mAP. The faithfulness gate is loading the converted
official checkpoint and asserting full-COCO `mAP >= published - 0.5`.

Recommended pattern when users care about matching upstream's published
numbers: `tests/nightly/test_<family>_official_ckpt_map.py`, gated on a
`<FAMILY>_OFFICIAL_CKPT_DIR` env var, kept out of the default suite to
avoid pulling multi-GB weights in CI. This is opt-in by design.

**DETR families**: skip the `last_loss < first_loss` assertion in `test_rf1_training`.
DETR total loss is the sum of ~38 weighted aux terms (per-decoder-layer + pre +
encoder-aux + DN paths) and is too noisy on small datasets for monotonic-decrease
to be reliable. RF-DETR's branch and the D-FINE branch both exempt themselves.

## 9. Silent-corruption landmines (in priority order)

The ones below have actually burned integrations in this repo. Each line is a one-shot:
*[which family hit it]* — what to do.

1. **Color space mismatch between training transform and val preprocessor**
   *(YOLOX BGR vs YOLOv9 RGB)* — pin the convention in both docstrings, cross-check.
2. **Target format mismatch** *(D-FINE)* — DETR criteria want `list[dict]` but the
   data pipeline yields padded `(B, max_labels, 5)`; translate in `on_forward`.
3. **`can_load()` too greedy** *(RF-DETR almost stole D-FINE checkpoints)* — match
   on tokens unique to your architecture; never `"backbone"` or `"weight"`.
4. **Backbone LR multiplier missing** *(DETR families)* — silent ~0.5 mAP loss in
   fine-tuning. Implies per-group LR + `_train_epoch` override.
5. **Multi-scale collate epoch propagation** *(D-FINE)* — collate needs `set_epoch()`
   called from the trainer at each epoch start.
6. **Stop-epoch augmentation policy** *(D-FINE)* — disable `RandomZoomOut`/`RandomIoUCrop`
   etc. at epoch N. Different from `no_aug_epochs` (which kills mosaic for last N).
7. **`labels_getter=lambda` is unpicklable** under Python 3.14's `forkserver`
   *(D-FINE on macOS)* — use a module-level function for `SanitizeBoundingBoxes`.
8. **`RandomIoUCrop` has no `p` parameter** in torchvision v2 — wrap with `RandomApply`.
9. **MPS-specific torch bugs in DETR backward** *(D-FINE)* — provide a `_setup_device`
   override that falls back to CPU; CUDA path stays unchanged.
10. **Post-train device drift** *(D-FINE)* — when the trainer fell back to CPU, the
    wrapper's `self.device` is still MPS; `model.val()` after `model.train()` hits
    a device mismatch. End `train()` with `self.model.to(self.device)`.
11. **ONNX opset 13 default** — DETR families with deformable attention need ≥ 16.
    Set per-family default in `BaseExporter.__call__`.
12. **NCNN can't handle DETR ops** *(D-FINE)* — block the export early with
    `NotImplementedError` instead of producing a graph the runtime can't load.
13. **`head.export` flag missing** *(YOLO-grid families)* — without it, ONNX bakes
    static shapes that work only at the exact resolution exported.
14. **`strict=True` state-dict loading** — override `_strict_loading() = False` if
    upstream checkpoints carry EMA buffers, profiling state, or auxiliary heads.
15. **Cross-family rejection on cross-family transfer** — when intentionally
    splicing weights, pop `model_family` from the donor checkpoint dict.
16. **EMA decay too low** — if you lower it from 0.9999 without good reason,
    early-epoch evals show flat or decreasing mAP because the EMA hasn't settled.
17. **Letterbox vs. plain resize** — `uses_letterbox` property must match the
    training transform; `DetectionValidator` reads it for target rescaling.
18. **`_train_epoch` override drift** *(DETR families)* — when you copy the parent
    loop to add per-group LR + grad clip + epoch propagation, leave a comment
    "kept in sync with `BaseTrainer._train_epoch` as of <commit>" so drift is
    auditable. Promote to shared hooks if a third family needs the same overrides.
19. **Per-size defaults copy-pasted from one size to all.** When upstream
    ships per-size YAMLs, build a side-by-side table of every override per
    size before assuming the s config applies to n. DETR examples:
    `freeze_at`, `freeze_norm`, backbone-LR multiplier, EMA decay,
    `lr_gamma`, peak `lr`. YOLO-grid examples: BN epsilon / momentum
    overrides on the smallest size, `INPUT_SIZES`, depth/width
    multipliers, head reg-max. The default is rarely uniform across
    sizes. A single-row table that's wrong on n/m is a silent ~1 mAP
    regression on those sizes. Cross-check each row against the actual
    upstream YAML, not just the first one you ported.
20. **`<Family>Config.min_lr_ratio` is one knob trying to cover several
    upstream `lr_gamma` values.** `FlatCosineScheduler` computes
    `min_lr = lr * min_lr_ratio`, so the ratio is the upstream `lr_gamma`
    by another name. Different families pick different values
    (D-FINE: `1.0` = no cosine decay, by design; DEIM: `0.5`; ECDet:
    `0.5`); within a family, sizes can disagree too (DEIM-N overrides
    to `1.0`). The trap is picking *any* uniform value without
    cross-checking. A ratio of `1.0` is correct when upstream's
    `lr_gamma` is `1.0` and a silent bug otherwise; the ratio is
    not the issue, the cross-check is.

## 10. Workflow

1. **Check both licenses** — code (upstream `LICENSE` file) and weights
   (HF model card). If either is GPL/AGPL, plan for a plugin-only ship; if
   weights are non-redistributable, link to the upstream CDN like YOLO-NAS.
2. **Pick the pattern**: YOLO-grid or DETR. Skim the existing family that's closest.
3. **Audit upstream's training recipe** with an agent. Decide what you skip.
4. **Implement family-local code** (`models/<family>/`) — the model, postprocess,
   and inference wrapper first. Verify inference parity using the recipe
   in §6 (cross-load upstream `state_dict`, assert `max_abs_diff == 0`
   on identical inputs over the layers present in both models). For
   wrapper integrations like RF-DETR, substitute "verify the wrapped
   package produces its documented outputs."
5. **Wire central files** — `models/__init__.py`, `__init__.py`, `config.py`,
   `validation/preprocessors.py`. Family must load via `LibreYOLO("Libre<Family>s.pt")`.
6. **Implement the trainer** — `trainer.py`, `transforms.py`, `loss.py`. Verify
   the loss matches upstream on synthetic inputs (parity test, 1e-5 tolerance).
7. **Wire into `MODEL_CATALOG`** and run `test_val_coco128` + `test_rf1_training`.
8. **Test ONNX export** at minimum. If the rest of the export formats are family-
   compatible, run them too.
9. **Upload weights to HuggingFace** under `LibreYOLO/Libre<Family><size>/` —
   see the separate `libreyolo-upload-hf-model` skill (skip if upstream license
   forbids redistribution; link to upstream CDN instead).
