---
name: libreyolo-model-porting-full
description: Port a model family into LibreYOLO with training, export, backend-runtime, and test-suite parity.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [libreyolo, object-detection, model-porting, training, export, backends, testing]
    related_skills: [libreyolo-model-porting-mvp, test-driven-development, systematic-debugging]
---

# LibreYOLO Model Porting (Full)

Use this when the goal is not just loading a new model family, but making it behave like a first-class LibreYOLO family across:
- factory loading
- native inference
- validation
- training
- checkpoints / resume
- export
- exported-runtime inference
- tests

This skill assumes you already understand the MVP integration path. If the user only wants PyTorch inference/validation support, use `libreyolo-model-porting-mvp` instead.

## Success criteria for a full port

A full port should support most or all of the following:
- `LibreYOLO("weights.pt")` chooses the right family
- native inference works
- `model.val(...)` works
- `model.train(...)` works and saves reloadable checkpoints
- checkpoint metadata includes `nc`, `names`, `size`, and `model_family`
- exported models can be produced in the formats the family claims to support
- exported models can be loaded back through `LibreYOLO(exported_path)` when the backend supports that family
- the family can be added to the repo's e2e catalog and relevant tests pass

## Files you will likely need to touch

Always:
- `libreyolo/models/<family>/__init__.py`
- `libreyolo/models/<family>/model.py`
- `libreyolo/models/<family>/nn.py`
- `libreyolo/models/__init__.py`

Usually:
- `libreyolo/models/<family>/utils.py`
- `libreyolo/models/<family>/trainer.py`
- `libreyolo/models/<family>/loss.py`
- `libreyolo/validation/preprocessors.py`
- `libreyolo/__init__.py`
- `pyproject.toml`

For export/runtime parity:
- `libreyolo/backends/base.py`
- possibly `libreyolo/backends/tensorrt.py`
- possibly export metadata handling if the family has special needs

For tests:
- `tests/e2e/conftest.py`
- `tests/e2e/configs/<family>.yaml`
- new or updated unit tests under `tests/unit/`
- possibly new e2e coverage files if existing generic tests are not enough

## Recommended workflow

### 1. Start with MVP integration first

Do not begin with training or export.

First get these working:
- family detection
- size detection
- class-count detection
- native PyTorch inference
- validation

Only once those pass should you move to training and export.

### 2. Implement training support

There are two patterns in this repo:

1. Shared `BaseTrainer` pattern
- used by YOLOX / YOLOv9 style families
- best when the PyTorch module itself can train in-process and returns structured losses

2. Upstream-wrapper pattern
- used by RF-DETR
- best when the original project has its own training loop and checkpoint format

Pick one intentionally.

## Shared `BaseTrainer` path

Create `libreyolo/models/<family>/trainer.py` and subclass `BaseTrainer`.

Implement:
- `_config_class()`
- `get_model_family()`
- `get_model_tag()`
- `create_transforms()`
- `create_scheduler(iters_per_epoch)`
- `get_loss_components(outputs)`
- usually `on_forward(...)`
- optionally `on_setup()` and `on_mosaic_disable()`

Then add `train(...)` on the wrapper model with the same pattern used by `LibreYOLOX` and `LibreYOLO9`:
- resolve dataset yaml via `load_data_config(..., autodownload=True)`
- rebuild architecture if dataset `nc` differs
- seed RNGs when requested
- instantiate trainer with `wrapper_model=self`
- support resume if appropriate
- reload `best.pt` after training finishes

### Training contract to satisfy

The shared trainer expects the training forward pass to produce something containing:
- `outputs["total_loss"]`

Your `get_loss_components(outputs)` should expose family-specific terms for logging.

### Checkpoint contract to satisfy

`BaseTrainer._save_checkpoint()` writes metadata used elsewhere in the repo. A full port should preserve this behavior.

At minimum, checkpoints should remain reloadable with:
- `model`
- `optimizer`
- `config`
- `loss`
- `nc`
- `size`
- `model_family`
- `names`

If your architecture needs special logic when `nc` changes, override `_rebuild_for_new_classes()` in the wrapper.

## Validation during training

Training is not complete unless validation works too.

The shared trainer's validation path relies on the generic `DetectionValidator`, which in turn depends on:
- `val_preprocessor_class`
- `_forward(...)`
- `_postprocess(...)`

So training support is coupled to validation correctness. If validation is wrong, `best.pt` selection and reported mAP will also be wrong.

## 3. Add export support

The exporter classes are shared across families. You usually do not add a new exporter subclass.

Instead make sure your family is compatible with the shared export path:
- the underlying `nn.Module` must be traceable/exportable
- `model.export(...)` should work through `BaseExporter`
- `_get_preprocess_numpy()` must reflect deployment preprocessing because INT8 calibration uses it
- if the model head needs export mode, follow the repo convention of exposing `head.export`

Important:
- RF-DETR needs `opset >= 17`
- your family may also need family-specific export kwargs or an override of `export(...)`

## 4. Add exported-runtime support

This is the most common hidden blocker.

A family being exportable is NOT the same as the exported model being usable through LibreYOLO's runtime backends.

Read:
- `libreyolo/backends/base.py`

Current backend parsing/preprocessing is family-aware only for:
- `yolox`
- `rfdetr`
- everything else falls back to yolo9-like behavior

So if your new family has different preprocessing or exported tensor layout, you must extend backend parsing.

At minimum review and patch:
- `_preprocess(...)`
- `_parse_outputs(...)`
- any family-specific detection in backend loaders

If TensorRT has family-specific assumptions, patch those too.

## 5. Add tests

### Add the family to the e2e catalog

If the port should participate in standard e2e coverage, update:
- `tests/e2e/conftest.py`

Add entries to `MODEL_CATALOG` once the family is stable enough.

### Add a training config

If config-driven training tests or benchmarks should include the family, add:
- `tests/e2e/configs/<family>.yaml`

Use existing files as structure references, but verify the options actually match the trainer you implemented.
Do not cargo-cult stale settings.

### Add or update unit tests

At minimum consider tests for:
- family registration / factory detection
- size detection
- class-count detection
- checkpoint reload with metadata
- export metadata round-trip
- backend output parsing if you extended it

Useful examples:
- `tests/unit/test_factory.py`
- `tests/unit/test_export.py`
- `tests/unit/test_export_ncnn.py`
- `tests/unit/test_coco_validation.py`
- `tests/unit/test_validation_metrics.py`

### Relevant e2e tests to run

Depending on what the family claims to support:
- `tests/e2e/test_val_coco128.py`
- `tests/e2e/test_rf1_training.py`
- `tests/e2e/test_torchscript.py`
- `tests/e2e/test_onnx.py`
- `tests/e2e/test_openvino.py`
- `tests/e2e/test_ncnn.py`
- `tests/e2e/test_tensorrt.py`

## Repo-specific pitfalls

1. `can_load(...)` collisions
- registry matching is first-hit-wins
- broad heuristics will misclassify checkpoints

2. filename-based size detection assumes single-character size codes
- override `detect_size_from_filename()` if your sizes are `tiny`, `small`, etc.

3. `_prepare_state_dict()` is currently not used in shared loading
- if key remapping is needed, do not assume overriding `_prepare_state_dict()` is enough

4. export backends are not generic
- native PyTorch support can work while ONNX/OpenVINO/NCNN/TensorRT still fail

5. validation coordinate conventions are easy to break
- validator passes `original_size` to `_postprocess(...)` as `(width, height)`
- dataset internals may track image info as `(height, width)`
- letterbox vs direct resize changes how GT and predictions must be rescaled

6. some repo configs are imperfect references
- use existing configs for shape, not as unquestioned truth

7. optional dependencies need full wiring
- add lazy import gates, extras, and clear error messages if the family depends on external libraries

## Recommended verification order

Run these in order and stop at the first failure.

1. native wrapper smoke test
2. validation on a tiny dataset
3. short training run on a tiny dataset
4. reload saved checkpoint in a fresh process
5. TorchScript export
6. ONNX export
7. exported-model round-trip load
8. optional backend-specific tests: OpenVINO, NCNN, TensorRT
9. add to `MODEL_CATALOG` and run family-targeted e2e coverage

## Practical commands

Typical fast loop:

```bash
pytest tests/unit/test_factory.py -v
pytest tests/unit/test_export.py -v
pytest tests/e2e/test_val_coco128.py -k '<family>' -v -m e2e
pytest tests/e2e/test_rf1_training.py -k '<family>' -v -m e2e
pytest tests/e2e/test_onnx.py -k '<family>' -v -m e2e
```

If the family uses heavy CUDA code or unstable upstream training, isolate with a fresh subprocess, following the RF-DETR pattern used by the repo's tests.

## Definition of done

Do not call the port complete until you can answer yes to these:
- can `LibreYOLO(...)` identify the family reliably?
- can native PyTorch inference run?
- can validation produce sane metrics?
- can training save and reload usable checkpoints?
- can the claimed export formats actually round-trip?
- were backend parsers updated if exported outputs differ from YOLOX/YOLOv9/RF-DETR?
- is there at least one focused unit test and one family-relevant e2e test path?

## When to stop and narrow scope

If exported-runtime support becomes a large detour, explicitly split the work:
1. native PyTorch family integration
2. training/checkpoint parity
3. exported-runtime parity

That keeps the repo shippable while avoiding half-broken backend support.
