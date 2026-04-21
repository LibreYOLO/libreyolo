# LibreYOLO Agentic Features — TODO

Tracking per the CLAUDE.md workflow convention. Source of truth for progress. Complements the design doc at `docs/agentic-features/specs/2026-04-22-agentic-features-design.md`.

## Phase 1 — All three features

### Setup
- [x] Fork `LibreYOLO/libreyolo` → `aalvsz/libreyolo`
- [x] Clone fork locally at `/Users/ander.alvarez/dev/perso/libreyolo`
- [x] Investigate upstream feature branches
- [x] Write design doc (this file's sibling)
- [ ] Commit design + todo
- [ ] Set up Python env (install package in editable mode, run existing tests)

### Feature A — YOLOv9 instance segmentation (#47)
- [ ] Branch `agentic/a-yolo9-segmentation` off upstream `47-add-instance-segmentation-to-yolo9`
- [ ] Smoke-test the upstream seg code end-to-end (tiny synthetic set, 1-2 iters)
- [ ] Write `scripts/train_yolo9_seg.py` (HF Jobs–ready)
- [ ] Write `notebooks/yolo9_segmentation_tutorial.ipynb`
- [ ] Write `docs/agentic-features/blog/yolo9-instance-segmentation.md`
- [ ] Add `tests/smoke/test_yolo9_seg_smoke.py`
- [ ] Push branch, open draft PR in fork

### Feature B — Distillation (#41/#42)
- [ ] Branch `agentic/b-distillation` off upstream `41-research-mgd-and-cwd-distillation`
- [ ] Smoke-test upstream distill code (2 iters, teacher ≠ student, loss finite)
- [ ] Write `scripts/train_distill_yolo9.py` (HF Jobs–ready)
- [ ] Write `notebooks/yolo9_distillation_tutorial.ipynb`
- [ ] Write `docs/agentic-features/blog/yolo9-distillation.md`
- [ ] Add `tests/smoke/test_distillation_smoke.py`
- [ ] Push branch, open draft PR in fork

### Feature C — VisDrone fine-tune (SAM-style demo)
- [ ] Branch `agentic/c-visdrone-finetune` off `main`
- [ ] Write `scripts/finetune_yolo9_visdrone.py` (HF Jobs–ready)
- [ ] Write `notebooks/yolo9_visdrone_finetune.ipynb`
- [ ] Write `docs/agentic-features/blog/yolo9-visdrone-finetune.md`
- [ ] Add `tests/smoke/test_visdrone_finetune_smoke.py`
- [ ] Push branch, open draft PR in fork

## Phase 2 — Real training (Feature C)

- [ ] Confirm HF Jobs access for user `ander2221`; fallback plan if blocked
- [ ] Submit HF Jobs training run for VisDrone fine-tune
- [ ] Monitor and babysit to completion
- [ ] Push weights to `ander2221/libreyolo-yolo9-visdrone`
- [ ] Finalize blog post with real metrics
- [ ] Update notebook to load the uploaded weights

## Review section (to be filled after completion)

<!-- Summary of what shipped, what didn't, and why. -->

## Lessons

Consolidated in `tasks/lessons.md`.
