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

## General library constraints
- Generally every user facing API (Python, yamls, etc) has to follow the Ultralytics YOLO standard
- The Flagship models of LibreYOLO are YOLO9 (CNNs) and RF-DETR (transformers), and new features have to at least cover this two
