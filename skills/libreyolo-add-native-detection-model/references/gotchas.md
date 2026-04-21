# Gotchas: Silent-Corruption Landmines

These produce wrong results, not errors. Each section lists:
- **What it is**
- **How it breaks**
- **How to avoid**

Triaged by likelihood × impact. The first six are the ones that have burned integrators in this repo before.

---

## 1. Color space mismatch between training transform and ValPreprocessor

**What it is.** YOLOX trains on BGR, 0-255 range. YOLOv9 trains on RGB, 0-1 range. Other YOLO-family variants vary.

**How it breaks.** If your `<Family>ValPreprocessor` normalizes or color-converts differently from your training transform, validation mAP silently collapses. Inference on test images may still look plausible because the model is robust to small channel swaps — but the reported metrics are wrong.

**How to avoid.** Decide the color convention once at design time. Write it in the docstring of both the training transform and the ValPreprocessor. Cross-check:

- Training transform: what format do batches arrive in the model at train time?
- ValPreprocessor `normalize` property: does it divide by 255? Must match.
- ValPreprocessor `custom_normalization` property: does it apply mean/std? If yes, DetectionValidator skips further rescaling.

Compare `YOLOXValPreprocessor` (BGR, 0-255, no normalization) at preprocessors.py:89-134 with `YOLO9ValPreprocessor` (RGB, 0-1, no custom normalization) at preprocessors.py:188-232. Note the `[:, :, ::-1]` BGR→RGB swap in YOLOv9 but not YOLOX.

---

## 2. Box target format (cxcywh vs xyxy, normalized vs pixel)

**What it is.** Loss functions expect targets in a specific format:
- YOLOX loss: `[class, cx, cy, w, h]` in **pixel** coordinates, converted to normalized inside the loss function.
- YOLOv9 loss: `[class, x1, y1, x2, y2]` in **normalized** (0-1) coordinates.

**How it breaks.** If your training transform outputs targets in a different format than your loss expects, the loss still decreases (gradient descent finds *something* to optimize) but the model learns to predict boxes at wrong coordinates. A trained model produces detections that are offset, shrunk, or rotated relative to ground truth.

**How to avoid.** Pin the target format in the docstring of `<Family>TrainTransform.__call__` and of your loss class. Write a unit test that feeds a known (image, targets) pair through the transform and asserts the target format.

---

## 3. head.export flag missing or not respected

**What it is.** The detection head should expose `self.export: bool`. When `export = True`, the head skips dynamic shape operations (anchor-grid construction via `torch.arange`, DFL soft-argmax decoding, etc.) and returns raw per-anchor outputs — ONNX-friendly.

**How it breaks.** Without the flag, ONNX tracing bakes in static shapes that work only at the exact resolution you exported at. The exported model produces plausible-looking detections at that resolution and garbage at any other.

**How to avoid.** Give the head an `export` attribute, default `False`. The exporter (`libreyolo/export/exporter.py:260-266`) sets it to `True` automatically during ONNX export. Test the exported model at a resolution **different** from what you exported at to catch this before release.

---

## 4. Config inheritance silently carrying dangerous defaults

**What it is.** `TrainConfig` has defaults tuned for YOLOX (`mixup_prob=1.0`, `mosaic_scale=(0.1, 2.0)`, `degrees=10.0`, etc.). If your family's recipe calls for different values, you must explicitly override them in `<Family>Config`.

**How it breaks.** Example: YOLOv9 paper uses **no mixup**. If you start from a copy of YOLOXConfig without reviewing each field, you inherit `mixup_prob=1.0`, and training silently fights itself — the mixup augmentation conflicts with DFL loss dynamics.

**How to avoid.** In `<Family>Config`, explicitly set every hyperparameter that appears in your family's paper recipe, even if the inherited value is technically OK. It's a cheap way to freeze the recipe against future base-class changes. See `YOLO9Config` at config.py:122-138 as a template — it explicitly sets `mixup_prob: float = 0.0` even though one could argue that's a "non-default" override.

---

## 5. `can_load()` heuristic too greedy

**What it is.** `BaseModel._registry` is walked in import order. `LibreYOLO(path)` calls `can_load()` on each class until one returns True.

**How it breaks.** If your new model's `can_load()` matches keys that also appear in other families (e.g. checking just for `"backbone"`), and your class imports before the real family, you will silently hijack load attempts for other models. The user loads a YOLOv9 checkpoint, your class says "yes, that's mine," returns `size=None`, and the factory blows up — or worse, the factory falls through to filename-based detection and loads it wrong.

**How to avoid.** Check for tokens **unique to your architecture**. Examples:
- YOLOX checks `head.stems` (specific to its decoupled head)
- YOLOv9 checks `repncspelan`, `adown`, `sppelan` (specific to its ELAN blocks)

Never check generic tokens (`backbone`, `head`, `weight`, `bias`). When unsure, test by constructing a minimal dict of another family's top-level keys and asserting `your_class.can_load(that_dict) == False`.

---

## 6. Test subprocess isolation expectation

**What it is.** `make test_e2e` runs each e2e test file in a separate subprocess (Makefile line 72: `$(UV) pytest "$$f"` per file). This is intentional — it prevents CUDA context corruption between tests that export to different backends.

**How it breaks.** If you write a test that relies on state persisting across files (e.g., a model cached in a global variable, a temp directory cleaned up by another test's teardown), it will pass under `make test_e2e` but fail when someone runs `pytest tests/e2e/`. Worse: it may appear flaky — passing sometimes depending on test order.

**How to avoid.** Design tests to be subprocess-friendly. Use pytest fixtures within-file, not globals. Clean up after yourself in each test. If you need cross-file coordination, use files on disk, not Python state.

---

## 7. Letterbox vs. plain resize mismatch with validator

**What it is.** YOLO-family detectors typically use letterbox (aspect-preserving resize with gray padding, pad_value=114). DETR-family detectors use plain square resize. `BaseValPreprocessor` exposes a `uses_letterbox` property that `DetectionValidator` reads to decide how to rescale target boxes.

**How it breaks.** If your training uses letterbox but `uses_letterbox` returns `False`, the validator applies the wrong coordinate correction and mAP comes out wrong.

**How to avoid.** The `uses_letterbox` and `normalize` properties must be the first thing you set in the preprocessor, and they must match your training transform. Unit-test them together: run a sample image through the training transform and the val preprocessor, and assert their output shapes and value ranges match.

---

## 8. Scheduler: hardcoded vs. config-driven

**What it is.** YOLOX's trainer hardcodes `WarmupCosineScheduler` (yolox/trainer.py:38-47). YOLOv9's trainer reads `self.config.scheduler` and supports `"linear"` or `"cos"` (yolo9/trainer.py:38-59).

**How it breaks.** Not a bug in the existing models — a design consistency issue for new ones. If you hardcode but users expect to be able to swap schedulers via YAML, they'll silently get whatever you hardcoded regardless of config. Conversely, if you make it config-driven but the "alternative" schedulers were never tested against your recipe, users can produce broken training runs by picking the wrong one.

**How to avoid.** Pick intentionally. If the paper recipe requires a specific scheduler, hardcode it (document in a comment) and reject config attempts to override. If you've tested multiple schedulers, expose the choice.

---

## 9. Input size asymmetry across size codes

**What it is.** YOLOX uses `{"n": 416, "t": 416, "s": 640, ...}` — smaller variants run at lower resolution. YOLOv9 uses 640 for all sizes.

**How it breaks.** If any downstream code hardcodes `640` instead of reading `self.input_size` (which is set to `INPUT_SIZES[size]` in `BaseModel.__init__`), your smaller variants silently run at the wrong resolution — input shape mismatches propagate through the head and produce either errors (if stride checks catch it) or silent accuracy loss (if not).

**How to avoid.** Everywhere in your code (preprocessing, postprocessing, tests) read `self.input_size` or `INPUT_SIZES[size]`, never hardcode a number. The one place it's OK to hardcode is the YAML `sizes:` dict in `tests/e2e/configs/<family>.yaml` — but then it must match `INPUT_SIZES` exactly.

---

## 10. strict=True state-dict loading

**What it is.** `BaseModel._strict_loading()` returns `True` by default. The shared loader then passes that flag into `self.model.load_state_dict(state_dict, strict=...)`.

**How it breaks.** If your model has any optional modules (EMA state, reparam-able convs, profiling buffers, auxiliary heads), or if upstream weight formats occasionally include debug keys, strict loading raises `RuntimeError` on an otherwise-valid checkpoint.

**How to avoid.** If you suspect any flexibility is needed, override `_strict_loading()` to return `False` (see YOLOX's override at yolox/model.py:108-109). If legacy keys need renaming, do not rely on `_prepare_state_dict()` alone — the current shared loader does not call it. Either remap keys before calling `load_state_dict()`, override `_load_weights()`, or do the remap in family-specific init/load code before the shared loader runs.

---

## 11. Cross-family checkpoint rejection

**What it is.** When `BaseTrainer` saves a checkpoint, it stores `model_family` metadata (trainer.py around line 562). When `BaseModel` loads, it compares this against `self._get_model_name()` and raises if they mismatch (base/model.py:316-324).

**How it breaks.** Intentional — this is a guardrail that prevents loading a YOLOX checkpoint into a YOLOv9 wrapper. But if you manually splice weights (e.g. initializing a new family from another family's backbone for transfer learning), the `model_family` key in the donor checkpoint will trip the rejection.

**How to avoid.** When doing intentional cross-family splicing, pop `model_family` from the checkpoint dict before loading. Also drop `nc`, `size`, `names` — they're metadata, not weights.

---

## 12. Filename regex surprises with -seg suffix

**What it is.** `BaseModel._filename_regex()` (base/model.py:221-228) builds a regex: `<prefix>([<sizes>])(-seg)?<ext>`. The `(-seg)?` group is optional but always there.

**How it breaks.** If someone (you, upstream, a user) names a weight file `Libre<Family>s-seg.pt`, `detect_task_from_filename()` returns `"seg"` — and `LibreYOLO(path)` will try to load it as a segmentation model. For a detection-only family, this path isn't supported.

**How to avoid.** Don't name any weight file with `-seg` for a detection-only family. If you want to be paranoid, add a guard in your wrapper's `__init__`:

```python
task = self.detect_task_from_filename(Path(model_path).name)
if task == "seg":
    raise ValueError(f"<Family> does not support segmentation")
```

Note: this skill scopes segmentation out. When native-seg becomes a thing, a separate skill will cover the `-seg` conventions.

---

## 13. EMA decay too low destroys convergence early

**What it is.** Default `ema_decay = 0.9998` in `TrainConfig`. YOLOv9 uses `0.9999`. The EMA model tracks a running average of model weights for cleaner evaluation.

**How it breaks.** If you set `ema_decay` too low (e.g. 0.99 copy-pasted from some older recipe), the EMA model lags noisily behind the training model. Early-epoch evals show flat or decreasing mAP because the EMA hasn't settled.

**How to avoid.** Use the paper's recommended value for your family. When in doubt, start with `0.9998` (YOLOX default). Only lower it if batch size is unusually large (effective sample rate increases).

---

## 14. `workers=0` vs. `workers=N` and multiprocessing start method

**What it is.** `tests/e2e/conftest.py:20` forces `multiprocessing.set_start_method("spawn", force=True)` because fork()+CUDA segfaults. This applies to e2e tests but not necessarily to user training runs.

**How it breaks.** If your DataLoader workers initialize CUDA tensors at startup, fork-based multiprocessing can corrupt the parent CUDA context. Symptoms: segfaults (exit code 139, signal 11) minutes into training, typically during the first validation pass.

**How to avoid.** Don't initialize CUDA in dataset `__init__` or `__getitem__` — keep those CPU-only. Tensors move to GPU inside the training loop, not inside the worker.

---

## Cross-reference

For the interface signatures that these gotchas talk about, see [contract.md](contract.md).

For where each file sits, see [files-checklist.md](files-checklist.md).
