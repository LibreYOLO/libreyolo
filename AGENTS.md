# Agent Instructions

- Agents must not open GitHub issues.
- Agents must not open pull requests.
- Agents must not post issue comments or PR comments unless a human explicitly
  asks for it.
- Humans handle issue creation, PR creation, review submission, and final merge
  decisions.
- Agents may reply with a one-click GitHub URL (no description pre-filled) so
  the human can open the PR or issue themselves.
- When possible, work in git worktrees
- The default branch is dev
## Commit policy
- Do not add LLMs or agent tools as co-authors in commits.
- Keep commit messages short and factual.
- Avoid pushing docs, artifacts, helper scripts, or anything that should not go into the upstream LibreYOLO library

## Documentation

- Contributor-facing policy lives in `CONTRIBUTING.md`.
- Exceptionally important schemas and contracts such as the checkpoint metadata standard live under /docs in the libreyolo repository. They are short and factual.
- `/docs/checkpoint_schema.md` documents checkpoint metadata rules used for
  loading, identifying, and validating model checkpoints.
- `/docs/nomenclature.md` documents canonical model names, filename rules,
  family/size/task conventions, and task-resolution behavior.
- `/docs/testing.md` documents test tiers, CI expectations, smoke tests, nightly
  tests, and manual validation policy.
- `/docs/adr/` documents architecture decisions and design contracts.

## Licensing policy

- LibreYOLO faithfully respects open-source licenses.
- Agents must not copy, adapt, paraphrase, or derive code from any third-party
  project unless that project is explicitly licensed under MIT or Apache-2.0
  and is compatible with LibreYOLO's licensing requirements.
- If an agent may have been exposed to, influenced by, or contaminated by code
  under GPL, AGPL, LGPL, proprietary, unknown, or otherwise incompatible terms,
  the agent must immediately flag the contamination risk to the developer and
  avoid contributing the affected code.

## Review guidelines

- These guidelines apply to agents performing PR reviews, not agents
  implementing code changes.
- PR-review agents must read `REVIEW.md` before reviewing pull requests.
- Treat `REVIEW.md` as repository context for scope, contracts, and common
  regression risks.
- If a PR conflicts with `REVIEW.md`, flag the conflict with concrete file
  evidence.

## Pull Request (PR) policy

- Before reviewing or changing PRs, read the relevant files under /docs,
  especially documented schemas, contracts, and architecture decisions.
- Prefer one PR per problem, or per small group of tightly related problems.
- When debugging a specific model or issue, avoid changing global behavior or
  shared code unless the shared change is genuinely necessary.
- Shared-code changes are allowed when they are the clean software engineering
  solution, but the PR must explain why the shared change is necessary and what
  other models or workflows it may affect.
- Keep PRs to the least code needed to solve the stated problem.
- Do not mention other computer vision libraries in PR titles or descriptions
  unless the comparison is necessary to explain compatibility or API behavior.
- Encourage humans to write PR titles and descriptions in their own words so the
  history is easier for reviewers and future readers to follow.

## General library constraints
- Generally every user facing API (Python, yamls, etc) has to follow the Ultralytics YOLO standard
- The Flagship models of LibreYOLO are YOLO9 (CNNs) and RF-DETR (transformers), and new features have to at least cover this two
