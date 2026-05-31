# LibreYOLO Review Guide

Use this file as context for agents performing PR reviews. This is not general
implementation guidance for agents writing code. Keep review comments specific,
evidence-based, and scoped to the PR under review.

## Core axioms

- Humans own issues, PRs, reviews, merges.
- One PR solves one problem.
- Shared changes need blast-radius justification.
- Keep PRs as small as possible.
- Read `/docs` before reviewing contracts.
- Metadata is the model-loading source of truth.
- Filenames are not model identity.
- State-dict sniffing is legacy compatibility.
- Foreign weights require converter scripts.
- Official checkpoints use flat v1 metadata.
- Checkpoints store state dicts, not objects.
- Metadata changes update docs and helpers.
- License compatibility is non-negotiable.
- One factory routes model families.
- Family constants define names, sizes, tasks.
- Task is a first-class axis.
- Detect is the suffixless default task.
- Task resolution is explicit, metadata, suffix, default.
- Cross-family checkpoint loads fail.
- Cross-task checkpoint loads fail.
- Not every family supports every task.
- YOLO9 and RF-DETR anchor coverage.
- Public APIs follow Ultralytics-style conventions.
- Explicit user kwargs beat defaults.
- CLI defaults are family-derived.
- Config dataclasses define training truth.
- Trainers orchestrate; families own recipes.
- RF-DETR ignores generic YOLO augmentation knobs.
- RF-DETR learning rate is absolute.
- DDP batch means global batch.
- Per-rank loaders divide global batch.
- Python multi-GPU training auto-spawns DDP.
- Torchrun owns rank and device environment.
- DDP loss scales by world size.
- Rank zero owns side effects.
- Autobatch returns rank-divisible global batches.
- Unit tests prove CPU-safe API behavior.
- Install smoke proves clean environment importability.
- GPU nightly proves real-model behavior.
- Nightly-selected skips are failures.
- Original-canvas coordinates are canonical.
- YOLO labels feed COCO metrics.
- Preprocessing is family-local.
- Validation is shared and task-aware.
- Results stay flat and API-compatible.
- Backends must behave like models.
- Exported runtimes round-trip metadata.
- DETR outputs use top-k, not NMS.
- Segmentation metrics are mask-first.
- Pose validation uses COCO OKS semantics.

## Review focus

- Flag unrelated changes bundled into bugfix PRs.
- Flag shared-code changes from model-specific bugs.
- Flag metadata behavior that bypasses `/docs`.
- Flag filename heuristics replacing metadata contracts.
- Flag API behavior that silently accepts ignored options.
- Flag changes that weaken normal single-GPU training.
- Flag DDP fixes that regress non-DDP paths.
- Flag CI marker changes that hide tests.
- Flag heavyweight tests in the fast unit suite.
- Flag GPL, AGPL, LGPL, proprietary, or unknown derivations.
- Flag PR descriptions that omit meaningful behavior changes.

## Contract references

- `docs/checkpoint_schema.md`: checkpoint metadata and loading rules.
- `docs/nomenclature.md`: model names, tasks, suffixes, resolution order.
- `docs/testing.md`: test tiers and validation expectations.
- `docs/adr/`: architecture decisions and design contracts.
- `CONTRIBUTING.md`: contribution and metadata-change policy.
- `AGENTS.md`: agent and PR policy.
