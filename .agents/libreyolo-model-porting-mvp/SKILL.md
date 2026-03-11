---
name: libreyolo-model-porting-mvp
description: Port a new PyTorch model family into LibreYOLO with unified factory, inference, and validation support.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [libreyolo, object-detection, pytorch, model-porting, inference, validation]
    related_skills: [libreyolo-model-porting-full, code-review, systematic-debugging]
---

# LibreYOLO Model Porting (MVP)

Use this when you need to add a new model family to the LibreYOLO repo with native PyTorch loading, unified `LibreYOLO(...)` factory support, inference, and validation.

This skill is for the minimum viable port. If the user wants training parity, exported-runtime parity, or full test-suite participation, use `libreyolo-model-porting-full` instead.

## What "MVP" means in this repo

A successful MVP port can:
- load through `LibreYOLO(...)`
- auto-detect the family from checkpoint keys
- auto-detect size and class count when possible
- run native PyTorch inference
- run generic validation through `model.val(...)`

An MVP port does NOT automatically guarantee:
- training support
- ONNX/TensorRT/OpenVINO/NCNN round-trip inference
- INT8 export/calibration
- inclusion in the full e2e catalog

## First: inspect the closest existing family

Before writing code, choose the nearest template:
- `libreyolo/models/yolox/model.py` for anchor-free CNN detectors with YOLOX-like preprocessing/output shapes
- `libreyolo/models/yolo9/model.py` for YOLO-style dense detectors with DFL/head-specific rebuild logic
- `libreyolo/models/rfdetr/model.py` for transformer-style detectors, optional dependencies, or family-specific special-casing

Read these files first:
- `libreyolo/models/base/model.py`
- `libreyolo/models/__init__.py`
- `libreyolo/models/yolox/model.py`
- `libreyolo/models/yolo9/model.py`
- `libreyolo/models/rfdetr/model.py`
- `libreyolo/validation/detection_validator.py`
- `libreyolo/validation/preprocessors.py`

## Required files to create

At minimum add:
- `libreyolo/models/<family>/__init__.py`
- `libreyolo/models/<family>/model.py`

In practice you will usually also need:
- `libreyolo/models/<family>/nn.py`
- `libreyolo/models/<family>/utils.py`

## Required wrapper contract

In `libreyolo/models/<family>/model.py`, define a subclass of `BaseModel`.

You must set these class attributes:
- `FAMILY`
- `FILENAME_PREFIX`
- `INPUT_SIZES`
- `val_preprocessor_class`

Optional:
- `WEIGHT_EXT` if not `.pt`

Example conventions already used by the repo:
- `FAMILY = "yolox"`
- `FILENAME_PREFIX = "LibreYOLOX"`
- `INPUT_SIZES = {"n": 416, "s": 640, ...}`

## Required classmethods for the unified factory

Your subclass must implement:
- `can_load(cls, weights_dict) -> bool`
- `detect_size(cls, weights_dict) -> str | None`
- `detect_nb_classes(cls, weights_dict) -> int | None`

Important:
- `BaseModel.__init_subclass__` only auto-registers real subclasses that implement `can_load`
- `LibreYOLO(...)` walks `BaseModel._registry` in import order
- first `can_load(...)` match wins

So `can_load(...)` must be specific. Do not use vague substring checks that could also match YOLOX, YOLOv9, or RF-DETR.

## Required instance methods from `BaseModel`

Implement all abstract methods:
- `_init_model(self)`
- `_get_available_layers(self)`
- `_get_preprocess_numpy()`
- `_preprocess(self, image, color_format="auto", input_size=None)`
- `_forward(self, input_tensor)`
- `_postprocess(self, output, conf_thres, iou_thres, original_size, max_det=300, ratio=1.0, **kwargs)`

The repo expects the following runtime contracts.

### `_preprocess(...)` contract

Return exactly:
- input tensor
- original PIL image
- original size as `(width, height)`
- resize ratio used for box restoration, or `1.0` if not needed

### `_postprocess(...)` contract

Return a dict with:
- `boxes`: xyxy boxes in original image coordinates
- `scores`: confidence scores
- `classes`: class ids
- `num_detections`: integer count

Also accept extra kwargs even if ignored, especially:
- `input_size`
- `letterbox`

Validation passes those through.

## Constructor pattern to follow

Use the same pattern as YOLOX/YOLOv9:
1. set any family-specific attributes first
2. call `super().__init__(...)`
3. if `model_path` is a string, call `self._load_weights(model_path)`

This is important because the factory may instantiate your class with:
- a path string
- an extracted `state_dict` dict
- `None`

## Factory wiring

After creating the wrapper, register it.

Edit:
- `libreyolo/models/__init__.py`

Add an import so the class is registered via `__init_subclass__`.

If the family should be public from the top-level package, also update:
- `libreyolo/__init__.py`

If the family depends on an optional extra package, copy the RF-DETR pattern:
- add a lazy registration helper in `libreyolo/models/__init__.py`
- gate imports in `libreyolo/__init__.py`
- add an extra in `pyproject.toml`

## Validation wiring

Choose the right validation preprocessor.

If an existing preprocessor works, reuse one of:
- `YOLOXValPreprocessor`
- `YOLO9ValPreprocessor`
- `RFDETRValPreprocessor`

If none fits, add a new one in:
- `libreyolo/validation/preprocessors.py`

The preprocessor choice matters because the validator uses it to decide:
- resize behavior
- letterboxing
- normalization
- ground-truth box rescaling

## Repo-specific gotchas

1. Import order matters
- `LibreYOLO(...)` picks the first matching family from the registry.

2. `detect_size_from_filename()` assumes single-character size codes
- BaseModel builds a regex from `INPUT_SIZES.keys()` as a character class.
- If your family uses `tiny`, `small`, etc., override `detect_size_from_filename()`.

3. `_prepare_state_dict()` is not wired into shared loading
- `BaseModel._load_weights()` does not currently call `_prepare_state_dict()`.
- If your port needs key remapping, either override `_load_weights()` or patch the shared path.

4. Rebuild logic may need an override
- If changing `nb_classes` only requires replacing the final detection head, override `_rebuild_for_new_classes()` like YOLOv9 does.

5. Strict loading is often too rigid for ports
- Existing families usually return `False` from `_strict_loading()`.

## Recommended implementation order

1. Add `libreyolo/models/<family>/nn.py`
2. Add preprocessing/postprocessing helpers in `utils.py`
3. Implement `model.py`
4. Import the class in `libreyolo/models/__init__.py`
5. Optionally expose it in `libreyolo/__init__.py`
6. Run a native smoke test
7. Run validation on a tiny dataset

## Smoke test checklist

Run a small Python check before touching exports or training:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("path/to/weights.pt")
print(type(model).__name__, model.size, model.nb_classes)

results = model("libreyolo/assets/parkour.jpg")
print(results)
```

Then validate:

```python
metrics = model.val(data="coco8", batch=2)
print(metrics)
```

## Minimum code review checklist

Before calling the port done, verify:
- `can_load(...)` does not collide with existing families
- size detection works from weights and, if needed, from filename
- class-count detection is correct
- `_postprocess(...)` returns original-image coordinates
- validation works without shape/letterbox bugs
- top-level import and factory registration work

## If the user asks for more than MVP

Escalate to `libreyolo-model-porting-full` when they want any of:
- `model.train(...)`
- e2e training participation
- ONNX/TensorRT/OpenVINO/NCNN runtime support
- exported-model round-trip loading via `LibreYOLO(exported_path)`
- metadata sidecars / backend parsing changes

## Linked templates

Use the linked wrapper skeleton as a starting point, then adapt it to the closest existing family rather than filling it blindly.
