# LibreYOLO Agentic Features — Design

**Date:** 2026-04-22
**Status:** Approved for implementation
**Scope:** Apply an ml-intern-style agentic workflow to LibreYOLO (Claude Code subagents, not the ml-intern package). Deliver three useful features that extend the library's research-artifact layer.

## Background

ml-intern (HuggingFace) is an agentic tool that autonomously researches, codes, trains, and publishes ML artifacts. The canonical demo: fine-tune SAM on a medical dataset, push weights + tutorial notebook + blog post to HF Hub. We replicate that value using Claude Code subagents rather than ml-intern itself, targeting LibreYOLO.

Investigation of the upstream repo revealed that the most impactful features on the roadmap (YOLOv9 instance segmentation, MGD/CWD distillation) **already have near-complete branches** (~85% done). What's missing is the last-mile research layer: real trained weights, tutorial notebooks, and blog posts. That's precisely where an agentic workflow adds value.

## The three features

### A. YOLOv9 instance segmentation (issue #47 / branch `47-add-instance-segmentation-to-yolo9`)
Upstream branch already contains: seg architecture, `MaskLoss`, polygon storage in dataset, mask targets wired through training loop, mosaic fixes for seg labels, differential LR, unit + e2e tests.

**Our additions in this feature branch:**
- Validate the upstream branch builds and runs end-to-end (smoke-test training on 2-image synthetic set, 1 iter)
- Tutorial notebook: `notebooks/yolo9_segmentation_tutorial.ipynb` — loads pretrained detection weights, fine-tunes on a small COCO-seg subset, runs inference, visualizes masks
- Training script ready for HF Jobs: `scripts/train_yolo9_seg.py`
- Blog draft: `docs/agentic-features/blog/yolo9-instance-segmentation.md`

### B. Knowledge distillation (issues #41, #42 / branch `41-research-mgd-and-cwd-distillation`)
Upstream branch already contains: `libreyolo/distillation/` module with `distiller.py`, `hooks.py`, `losses.py` (MGD + CWD), distill config fields, wiring into base trainer, unit tests with convergence checks.

**Our additions in this feature branch:**
- Validate the upstream branch builds and runs (2-iter smoke-test, teacher ≠ student, loss finite)
- Tutorial notebook: `notebooks/yolo9_distillation_tutorial.ipynb` — distill YOLOv9-medium → YOLOv9-tiny on a small COCO subset, compare before/after mAP and latency
- Training script ready for HF Jobs: `scripts/train_distill_yolo9.py`
- Blog draft: `docs/agentic-features/blog/yolo9-distillation.md`

### C. Domain fine-tune + reference tutorial (no upstream branch, closest to the SAM example)
Pick a realistic domain (recommend: **VisDrone aerial-imagery detection** — publicly available, 10 classes, non-trivial, good for a detection library's story). Using `main` branch (no new features needed).

**Deliverables:**
- Tutorial notebook: `notebooks/yolo9_visdrone_finetune.ipynb` — downloads dataset, converts to YOLO format, fine-tunes from a COCO checkpoint, evaluates, visualizes
- Training script ready for HF Jobs: `scripts/finetune_yolo9_visdrone.py`
- Blog draft: `docs/agentic-features/blog/yolo9-visdrone-finetune.md`
- **Phase 2 real training (primary target):** run this end-to-end on HF Jobs, push trained weights to `ander2221/libreyolo-yolo9-visdrone` on HF Hub, finalize the blog with real metrics

## Architecture and boundaries

- **Isolation:** three branches in `aalvsz/libreyolo`, one per feature. Each is independently testable, pushable, and (optionally) upstream-PR-able. No interdependencies.
- **Shared infrastructure:** `docs/agentic-features/` houses the specs, blog drafts, and a shared `README.md` indexing the three features. `scripts/` and `notebooks/` follow LibreYOLO's existing conventions.
- **Smoke-test harness:** each feature branch adds a CPU/MPS-runnable integration test under `tests/smoke/` that validates the full pipeline with tiny synthetic data — no GPU, runs in <60s.

## Phase split

**Phase 1 — All three features (this session):**
- Fork and clone (done)
- Write spec, commit (this doc)
- Per feature: branch, validate upstream code, add tutorial + training script + blog draft + smoke test
- Push branches + open draft PRs against `aalvsz/libreyolo` for review

**Phase 2 — Real training (one feature):**
- Target: feature C (VisDrone fine-tune) — cheapest, closest to SAM example, demonstrates end-to-end story
- Submit HF Jobs training run using the prepared script
- Gate: HF Jobs requires billing. Token scope has `job.write` but account is free tier. If Jobs is blocked, fallback = prepare the job script for the user to submit with their own compute, document training command in the blog

## Success criteria

1. Fork exists at `github.com/aalvsz/libreyolo` with three feature branches
2. Each branch passes its smoke test on macOS/MPS
3. Each branch has a runnable tutorial notebook
4. Each branch has a blog draft
5. For the chosen Phase 2 target: real weights exist on HF Hub, blog is finalized with real metrics, notebook runs against the uploaded weights

## Out of scope

- Actual upstream PRs to `LibreYOLO/libreyolo` (the user may submit later; fork is the deliverable)
- Training the other two features to production quality (documented but not executed)
- New model architectures (YOLO-NAS etc.) — deferred to future work

## Risks and mitigations

| Risk | Mitigation |
|------|-----------|
| Upstream feature branch doesn't build / has broken tests | Smoke-test first, fix minor issues in our branch, escalate if deep |
| HF Jobs not available on free tier | Document the training command; user runs on own GPU |
| MPS float/dtype issues on Mac smoke-tests | Force `device="cpu"` in smoke tests; integration-test on MPS only if trivially available |
| Dataset download fails in Phase 2 | Use tiny sample already cached in `huggingface_hub` or synthetic alternatives |

## References

- Upstream repo: https://github.com/LibreYOLO/libreyolo
- Fork: https://github.com/aalvsz/libreyolo
- ml-intern: https://github.com/huggingface/ml-intern
- SAM medical demo (reference style): huggingface.co/Mayank022/sam-medical
- Target HF user: `ander2221`
