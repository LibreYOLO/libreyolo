---
name: libreyolo-transformer-detector-porting
description: Port transformer-based object detectors into LibreYOLO, following the RF-DETR-style integration path for factory loading, validation, training, export, and backend support.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [libreyolo, transformer-detectors, detr, object-detection, model-porting, export, validation]
    related_skills: [libreyolo-model-porting-mvp, libreyolo-model-porting-full, systematic-debugging]
---

# Porting Transformer Detectors into LibreYOLO

Use this skill when porting a transformer-style detector into LibreYOLO, especially DETR-like models with:
- encoder/decoder blocks
- query embeddings
- set-based predictions
- direct resize preprocessing instead of YOLO-style letterbox
- exported outputs that do NOT look like YOLOX or YOLOv9

Examples:
- DETR
- Deformable DETR
- DINO / Grounding-DINO-like detector variants
- RT-DETR-like transformer detectors
- RF-DETR-like models

This skill is specifically about the transformer path. For generic CNN detector ports, use `libreyolo-model-porting-mvp` or `libreyolo-model-porting-full`.

## Core principle

In LibreYOLO, transformer detectors are usually NOT just another YOLO-family wrapper.
They tend to need RF-DETR-style special handling in at least four places:
- family detection
- preprocessing / validation
- training
- exported-runtime backend parsing

Do not assume the YOLOX or YOLOv9 path will work unchanged.

## First inspect the RF-DETR path

Read these files before changing anything:
- `libreyolo/models/rfdetr/model.py`
- `libreyolo/models/__init__.py`
- `libreyolo/__init__.py`
- `libreyolo/backends/base.py`
- `libreyolo/validation/preprocessors.py`
- `libreyolo/export/exporter.py`
- `tests/e2e/test_rf1_training.py`
- `tests/e2e/conftest.py`

Treat RF-DETR as the reference architecture for transformer-family ports.

## Decide the integration style up front

Choose one of these paths before coding.

### Path A: native LibreYOLO wrapper around an upstream transformer model

Use this when:
- you can construct the model cleanly inside LibreYOLO
- you can control preprocessing/postprocessing directly
- you can reload checkpoints without depending on the original training CLI

This is better when long-term maintenance inside LibreYOLO matters.

### Path B: upstream-wrapper integration

Use this when:
- the upstream project has a non-trivial training pipeline
- weights/checkpoints depend on upstream config objects or custom loaders
- rewriting training inside `BaseTrainer` would be fragile or expensive

This is what RF-DETR effectively does.

If unsure, start with Path B for complex DETR-family models.

## Files you will usually need

At minimum:
- `libreyolo/models/<family>/__init__.py`
- `libreyolo/models/<family>/model.py`
- `libreyolo/models/<family>/nn.py`
- `libreyolo/models/<family>/utils.py`

Often also:
- `libreyolo/models/<family>/trainer.py`
- `libreyolo/models/__init__.py`
- `libreyolo/__init__.py`
- `libreyolo/backends/base.py`
- `pyproject.toml`
- tests under `tests/unit/` and `tests/e2e/`

## Wrapper requirements

Create a `BaseModel` subclass and set:
- `FAMILY`
- `FILENAME_PREFIX`
- `INPUT_SIZES`
- `val_preprocessor_class`

Important for transformer families:
- `INPUT_SIZES` may map sizes to different square resolutions, like RF-DETR does
- if your size names are not single-character codes, override `detect_size_from_filename()`

## Factory detection strategy

Implement:
- `can_load(cls, weights_dict)`
- `detect_size(cls, weights_dict, state_dict=None)` if full checkpoint context is needed
- `detect_nb_classes(cls, weights_dict)`

For transformer detectors, `can_load(...)` often should look for keys like:
- `transformer`
- `encoder`
- `decoder`
- `query_embed`
- `class_embed`
- `bbox_embed`
- model-specific backbone markers

Important:
- make detection specific enough to avoid collisions with RF-DETR if both families coexist
- first match wins in `BaseModel._registry`

If size detection needs checkpoint args/config, copy the RF-DETR pattern where the factory passes the full checkpoint for size detection.
That may require changing `libreyolo/models/__init__.py`.

## Preprocessing rules for transformer detectors

Most transformer detectors in this family do NOT want YOLO-style letterbox preprocessing.
Instead they usually expect:
- image load
- tensor conversion
- normalization, often ImageNet mean/std
- direct resize to a fixed resolution

That means you should usually:
- add or reuse a transformer-friendly validation preprocessor
- return `ratio = 1.0` from `_preprocess(...)` unless your postprocess explicitly relies on a resize ratio

Follow the RF-DETR pattern if applicable:
- direct resize
- ImageNet normalization
- no letterbox

## Postprocess contract

Your `_postprocess(...)` must return:
- `boxes`
- `scores`
- `classes`
- `num_detections`

For transformer detectors, outputs are often something like:
- `pred_boxes`: normalized cxcywh
- `pred_logits`: raw logits over classes

Typical conversion steps:
1. convert logits to probabilities or scores
2. select top class per query
3. threshold by confidence
4. convert normalized boxes to xyxy pixel coordinates in original image space
5. clamp to image bounds
6. map class ids if the checkpoint uses a different label convention

RF-DETR includes an important example of COCO 91 -> COCO 80 mapping. Copy that pattern if the upstream model uses a non-contiguous or background-inclusive class space.

## Validation support

Validation is critical because training and checkpoint selection depend on it.

Ensure your family's `val_preprocessor_class` matches the model's true inference preprocessing.
If none of these work:
- `YOLOXValPreprocessor`
- `YOLO9ValPreprocessor`
- `RFDETRValPreprocessor`

then add a new preprocessor in:
- `libreyolo/validation/preprocessors.py`

For transformer detectors, check these carefully:
- normalization behavior
- direct resize vs letterbox
- image-size metadata passed to postprocessing
- whether class logits include background

## Training decision tree

### Use `BaseTrainer` only if all are true
- the model trains cleanly inside a normal PyTorch loop
- forward(images, targets) can return a dict with `total_loss`
- you can define transforms/scheduler/loss extraction without wrapping an upstream training stack

### Otherwise use an upstream-wrapper training path
- implement `train(...)` directly on the wrapper model
- call the upstream trainer or training function
- reload the best checkpoint manually afterward
- adapt checkpoint/class-count logic explicitly

Transformer detectors often fit the second case better.

## Export considerations

Transformer detectors frequently need special export handling.

Before claiming export support, verify:
- the nn.Module is traceable
- unsupported ops are not blocking ONNX
- required opset is set high enough
- outputs remain interpretable by LibreYOLO backends

Like RF-DETR, your family may need to override:
- `export(...)`

to force a higher opset or family-specific kwargs.

## Backend-runtime support is not optional for a full port

This is the hidden failure mode.

Even if ONNX export succeeds, LibreYOLO backend inference may still fail because `libreyolo/backends/base.py` currently only knows how to parse:
- YOLOX-style outputs
- YOLOv9-style outputs
- RF-DETR-style outputs

If your transformer family produces a different output layout, you MUST extend backend parsing.

Review and patch:
- `_preprocess(...)`
- `_parse_outputs(...)`
- add a new family-specific parse helper if needed

If the runtime metadata uses `model_family`, make sure your exported files or sidecars preserve it so the backend can dispatch correctly.

## Optional dependency pattern

Transformer detectors often rely on external packages.

If so, follow the RF-DETR style:
- add an optional dependency group in `pyproject.toml`
- add lazy registration / import checks in `libreyolo/models/__init__.py`
- add lazy top-level access in `libreyolo/__init__.py`
- emit a clear installation hint when the dependency is missing

## Testing strategy

### Minimum test set
- factory detection test
- size detection test
- class-count detection test
- native inference smoke test
- validation smoke test

### Full test set
- checkpoint reload test
- short training improvement test
- TorchScript export if supported
- ONNX export
- exported-model round-trip load
- backend-specific tests for OpenVINO/NCNN/TensorRT if claimed

Use these files as references:
- `tests/unit/test_factory.py`
- `tests/unit/test_export.py`
- `tests/e2e/test_rf1_training.py`
- `tests/e2e/test_onnx.py`
- `tests/e2e/test_val_coco128.py`

If the family is stable enough for the main e2e matrix, add it to:
- `tests/e2e/conftest.py` -> `MODEL_CATALOG`

## Common transformer-specific pitfalls

1. Background class mismatch
- some DETR-family heads predict `num_classes + 1`
- LibreYOLO user-facing outputs may expect contiguous foreground classes only

2. Box format mismatch
- exported/native outputs are often normalized cxcywh, not xyxy pixels

3. Direct resize vs letterbox mismatch
- wrong validation preprocessor can silently wreck metrics

4. Full-checkpoint metadata dependence
- size detection may require upstream args/config, not just tensor shapes

5. Export success but runtime failure
- ONNX file writes successfully, but backend parser does not know how to read outputs

6. Over-broad family detection
- your `can_load(...)` can accidentally swallow RF-DETR checkpoints or vice versa

7. Class mapping assumptions
- COCO-91, COCO-80, background-inclusive, or custom label maps must be handled explicitly

## Recommended build order

1. implement native wrapper
2. get `LibreYOLO(weights)` family detection working
3. get native inference working
4. get validation working
5. add training path
6. verify checkpoint reload
7. add ONNX export
8. add runtime backend parsing
9. only then add the family to the broader e2e matrix

## Definition of done

A transformer detector port is not done until you have verified:
- correct family detection
- correct class-count handling
- correct box scaling back to original image size
- correct handling of background / label mappings
- validation metrics that make sense
- checkpoint reload in a fresh process
- export support only for formats that actually round-trip
- backend parsing support for any claimed exported-runtime inference

## When to narrow scope

If the upstream model is very complex, split the work into phases:
1. native inference + validation
2. training/checkpoint support
3. export
4. exported-runtime backend support

That is better than shipping a half-working "full" integration.
