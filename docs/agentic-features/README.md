# LibreYOLO Agentic Features — Index

This directory tracks an ml-intern-style effort to add three useful features to LibreYOLO using Claude Code subagents instead of the ml-intern package. Design, rationale, and per-feature blog posts all live here.

**Fork:** https://github.com/aalvsz/libreyolo

## The three branches

| Branch | Based on | What it delivers |
|--------|----------|------------------|
| [`agentic/a-yolo9-segmentation`](https://github.com/aalvsz/libreyolo/tree/agentic/a-yolo9-segmentation) | upstream `47-add-instance-segmentation-to-yolo9` | **Fix:** mask-target resolution was hardcoded at 160×160 (only valid for imgsz=640); now scales with input. Plus smoke test, tutorial notebook, blog, training script, HF-Hub-ready publish. |
| [`agentic/b-distillation`](https://github.com/aalvsz/libreyolo/tree/agentic/b-distillation) | upstream `41-research-mgd-and-cwd-distillation` | Smoke tests proving MGD + CWD distillation works end-to-end on CPU. Tutorial notebook, blog, training script. |
| [`agentic/c-visdrone-finetune`](https://github.com/aalvsz/libreyolo/tree/agentic/c-visdrone-finetune) | `main` | SAM-style reference fine-tune: VisDrone aerial-imagery pipeline end-to-end. 10 smoke tests. HF Jobs–ready training script. |

## Artifacts per branch

Each feature branch ships the same shape of deliverables:

| File | Purpose |
|------|---------|
| `tests/smoke/test_<feature>_smoke.py` | Offline validation on synthetic data — CPU-runnable in under a few seconds |
| `scripts/<train_or_finetune>.py` | HF Jobs–ready CLI; optional `--push` to HF Hub with auto-generated model card |
| `notebooks/<feature>_tutorial.ipynb` | Walkthrough that runs on a laptop |
| `docs/agentic-features/blog/<feature>.md` | Narrative with rationale, bug fixes, scale-up recipe |

## Design doc

[`specs/2026-04-22-agentic-features-design.md`](./specs/2026-04-22-agentic-features-design.md) — the spec this work implements, including Phase 1 (all three features) vs Phase 2 (real HF Jobs training) split.

## Phase 2 status

Phase 2 (real training → pushed weights on HF Hub) is **blocked on account credits**:

```
402 Payment Required — Pre-paid credit balance is insufficient
```

The HF Jobs API is reachable with the existing token (fine-grained scope includes `job.write`), but the account has no pre-paid balance. See [`PHASE2_STATUS.md`](./PHASE2_STATUS.md) on the `agentic/c-visdrone-finetune` branch for the exact commands to unblock (~$5 buys the smoke check plus the full ~3 hr a10g-small VisDrone training run).

## Why this is "really useful"

Investigating the repo surfaced that two of the three feature areas (`#47` seg, `#41` distillation) were already ~85% implemented on upstream branches — architecture, losses, dataset handling, tests. What was missing was the research-artifact layer: a smoke test that proves end-to-end convergence, a tutorial a user can run, a training recipe ready for managed compute, and a target for published weights.

The `agentic/a-yolo9-segmentation` branch additionally includes a real upstream bug fix — the smoke test immediately exposed that mask-target rasterization was hardcoded at the imgsz=640 proto resolution (`160×160`), making any other imgsz crash with a BCE shape mismatch. The fix is five edits introducing a named `MASK_STRIDE = 4` constant.

Together these three branches could be opened as PRs against `LibreYOLO/libreyolo` to close real gaps.
