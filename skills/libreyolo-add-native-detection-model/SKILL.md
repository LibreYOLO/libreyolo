---
name: libreyolo-add-native-detection-model
description: Add a native detection model family to LibreYOLO. Use when integrating a model in the YOLOX/YOLOv9 style: local nn implementation, BaseModel wrapper, BaseTrainer integration, validation preprocessor, factory registration, export compatibility, and tests. Not for upstream-wrapper models or segmentation.
---

# Add a Native Detection Model to LibreYOLO

Use this skill for **native detection families only**.

- **Native** means the model implementation, loss, and training loop live inside `libreyolo/`.
- **Detection** means boxes only. Segmentation is out of scope.
- Canonical patterns on this branch are `YOLOX` and `YOLOv9`.
- `RF-DETR` is useful as contrast, but it is not the template for this skill.

## Hard requirements

Every new native detection family must satisfy these:

1. `LibreYOLO()` can recognize the family from weights.
2. Size and class count auto-detection work.
3. The family has a `BaseModel` wrapper in `libreyolo/models/<family>/model.py`.
4. The family has a `BaseTrainer` subclass in `libreyolo/models/<family>/trainer.py`.
5. Inference works and returns correct `Results`.
6. Training checkpoints can be reloaded correctly.
7. At least one export path works if the family is expected to support export.
8. The family is wired into the test catalog and has targeted tests for its real failure modes.

## Triggered requirements

Only do these when the model actually needs them:

- Add a new validation preprocessor implementation only if neither the YOLOX nor YOLOv9 preprocessing pattern is correct. Still set `val_preprocessor_class` explicitly for the family.
- Add head-only class rebuild logic if full model rebuild is wrong or wasteful.
- Add state-dict key remapping if upstream keys do not match local module names.
- Add backend parser branches if exported outputs are not compatible with existing YOLO-style parsing.
- Add optional dependencies only if the model truly needs them.

Do not turn these into blanket checklist items for every family.

## Workflow

1. Decide the family identifier, filename prefix, size codes, input sizes, and output tensor shape.
2. Read `references/contract.md` for the required wrapper and trainer interfaces.
3. Create `libreyolo/models/<family>/` with the local `nn.py`, `model.py`, `trainer.py`, and supporting files.
4. Register the family in `libreyolo/models/__init__.py` and export it from `libreyolo/__init__.py`.
5. Add config defaults and an explicit `val_preprocessor_class`. Reuse an existing preprocessor implementation only if the semantics match exactly.
6. Update the test catalog and add the smallest test set that proves the integration is real.

## Hot files

These are the files most native-family ports touch:

- `libreyolo/__init__.py`
- `libreyolo/models/__init__.py`
- `libreyolo/training/config.py`
- `libreyolo/validation/preprocessors.py`
- `tests/e2e/conftest.py`

## Common traps

Read `references/gotchas.md` when you touch any of these:

- color space or normalization
- target box format
- export behavior
- scheduler and augmentation defaults
- `can_load()` heuristics
- class-count rebuilding
- checkpoint key remapping

Important: `BaseModel._prepare_state_dict()` exists, but the current shared loader does **not** call it automatically. If your family depends on key remapping, wire that behavior explicitly.

## Troubleshooting

- Wrong family selected by `LibreYOLO(path)`: `can_load()` is too greedy or collides with an earlier family.
- Validation mAP collapses while inference still looks plausible: training transform and validation preprocessing disagree on color space, scaling, or box semantics.
- Exported model produces garbage detections: the head export path or backend parser does not match the model's true output format.
- `load_state_dict` missing/unexpected keys: `_strict_loading()` is too strict, or key remapping was implemented through `_prepare_state_dict()` without wiring it into the real load path.

## Minimal validation

Do not require full backend parity by default. Run the smallest set that proves the family is integrated correctly:

- one factory/autodetect test
- one forward/inference smoke test
- one trainer checkpoint reload test
- one export round-trip if export is in scope
- one family-specific unit test for the part most likely to silently go wrong

Use `references/files-checklist.md` to decide which files are actually required for this family.
