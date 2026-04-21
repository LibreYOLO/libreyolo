# Lessons Learned — LibreYOLO Agentic Features

Per the user's CLAUDE.md: capture corrections and non-obvious validated choices so future sessions stay coherent.

<!-- Add entries as: date, context, lesson, how to apply. -->

## 2026-04-22

- **Check upstream branches before implementing a "new" feature.** The LibreYOLO repo had near-complete branches for YOLOv9 segmentation and MGD/CWD distillation. What looked like three features-to-build was actually two features-to-finish + one feature-to-demo. Always `git branch -r` against upstream before assuming greenfield work.
- **Subagents have no GPU.** Claude Code subagents run in-process — they can't train models. "Make it work" means code + smoke-test + notebook + blog, with real training deferred to HF Jobs or the user's own compute.
