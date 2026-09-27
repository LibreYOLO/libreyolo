---
name: libreyolo-release
description: >-
  Cut a LibreYOLO version release end to end: evidence-backed changelog built
  by fan-out agents over the release..dev diff, a gate scoreboard (unit CI,
  GPU e2e on Modal reusing the nightly, weight autodownload probe, wheel
  smoke, notices sync, docs drift, breaking-change sweep), the branch/tag/PyPI
  mechanics, post-publish verification, and the handoff to the marketing
  repo's announcement pipeline. Use whenever the user says "release",
  "cut vX.Y.Z", "ship a version", "prepare the changelog for the release",
  or "is dev ready to release?".
---

# LibreYOLO release

Turn "let's release" into a published PyPI version with a real changelog,
proven gates, and marketing assets queued, without re-deriving the process
each time. The engineering source of truth is `RELEASING.md`; this skill
wraps it with everything that document assumes you already know.

**Golden rule: nothing goes in the changelog without evidence (a commit SHA
or PR number), and nothing ships without a green scoreboard.** The changelog
is built before the gates run so that gate failures can be traced back to the
change that caused them.

## The lay of the land (read once, saves an hour)

- Branches are `dev` and `release`. There is **no `main`**.
- `pyproject.toml` `version` is the source of truth. `dev` carries a
  `X.Y.Z.dev0` marker; `release` carries the clean `X.Y.Z`. The bump commit
  lives on the release side. It is the **only** file that carries the
  version: `CITATION.cff` and `.zenodo.json` are intentionally version-less
  and frozen so all academic citations cluster on one Scholar/Zenodo record.
  Never bump, date, or retitle them, even if they look stale.
- **The version label is the human's call, not yours.** Do not infer the
  number from the size of the diff. The `.dev0` marker on `dev` is a hint,
  not a decision: the team may ship a large feature range under a patch
  label (e.g. everything since v1.3.0 released as **v1.3.1**), or hold
  features and cut a narrow patch. Ask; see "Minor vs patch" below.
- CI gotcha: `unit-tests.yml` and `install-smoke.yml` trigger only on
  push/PR to **`dev`** (plus `workflow_dispatch`). A `dev -> release` PR
  shows **no checks**; the real gate is CI on the dev push. Do not read the
  empty checks list as green; "no checks" on a release PR means untested.
- `publish.yml` fires on `v*` tags, rejects tags not reachable from
  `release`, and ends in a manual approval on the **`pypi` GitHub
  Environment** (the job shown as "Publish to PyPI"). It builds with
  `uv build` and publishes via Trusted Publishing / OIDC; there is no PyPI
  token anywhere.

### Minor vs patch (decide with the user, up front)

The mechanics below default to a minor (`X.Y.0`) cut from `dev`, because
that is the common case. A **patch/hotfix (`X.Y.Z`, Z>0)** differs:

- **Feature drop labeled as a patch** (what happened with v1.3.1: the whole
  `v1.3.0..dev` range shipped under `1.3.1`): mechanically identical to the
  minor cut from `dev` below; only the number differs. Set both the `dev`
  marker and the release version to that patch number instead of the next
  `X.Y.0`. The changelog still lists everything in the range.
- **True narrow hotfix**: branch off `release` (not `dev`), cherry-pick or
  commit only the fix(es), bump `release` to `X.Y.Z`, tag. Then merge that
  back into `dev` so the fix is not stranded (this is the mirror of
  preflight check 3). The changelog lists only the fixes, not unrelated
  dev features.

Confirm which of these you are doing before Phase 1, because it changes
what the changelog contains and which branch you cut from.
- GPU e2e lives on Modal. `tools/ci/modal_nightly.py` takes `--ref <sha>`,
  so the same harness that runs the nightly can test the exact release
  candidate commit.
- The announcement side (GitHub release body polish, LinkedIn carousel,
  Reddit GIF post) is the **marketing repo's** `new-version-release` skill
  (`../marketing/skills/new-version-release/`). This skill produces the
  facts file and changelog it consumes; do not rebuild its pipelines here.

## Phase 0: Preflight and scoreboard

Create a working directory **outside the repo tree** (scratchpad or
`~/release-<version>/`) and start `SCOREBOARD.md` in it. Every phase below
appends one line: `| gate | PASS/FAIL/SKIPPED | evidence link or note |`.
The scoreboard is the deliverable you show the user before asking for the
go/no-go; a release with SKIPPED rows needs their explicit sign-off per row.

Preflight checks (all read-only, run in parallel):

```bash
git fetch upstream --tags
# 1. What is shipping and from where
git log -1 upstream/dev --oneline
grep '^version' pyproject.toml                       # on dev: expect X.Y.Z.dev0 (e.g. 1.3.1.dev0)
# 2. Last released version = base ref for everything
git tag --sort=-creatordate | head -5
# 3. Hotfixes stranded on release that dev never got (MUST be merged back first)
git log upstream/dev..upstream/release --oneline
# 4. CI on dev HEAD is actually green
gh run list -R LibreYOLO/libreyolo --branch dev --limit 10
# 5. Anything already labeled/queued as release-blocking
gh pr list -R LibreYOLO/libreyolo --base dev --state open --limit 30
```

Confirm with the user: version label (`vX.Y.Z`), base tag, and which gates
to run (default: all except RF5). RF5 is skipped by standing decision
(issue #744 item 18), not by accident; a SKIPPED RF5 row is not an open
defect. Then proceed without further check-ins until the go/no-go.

## Phase 1: Changelog by fan-out (facts first, prose second)

Range is `vLAST..upstream/dev`. Fan out parallel read-only agents, one lens
each. Every agent returns **numbered fact entries with evidence**: one-line
claim + commit SHA or PR number + file path. No pointer, no fact. Uncertain
items get marked uncertain, never dropped.

**Verify against code, not commit messages.** This is the discipline that
separates an accurate changelog from a plausible one. A commit titled "add
X family" proves someone intended X, not that X is wired, exported, and
usable. For every headline claim, confirm in the actual source: is the
class exported from `libreyolo/__init__.py` or a registry, or is it an
unwired native port that nobody imports (dead code from a user's view)? Is
a "new family" actually new, or did it already exist at `vLAST` and only
get modified? Is the thing a user-facing `Libre*` class, or an internal
backbone (bert/swin-style) consumed only by another model? Is it a class or
just a factory function? Commit messages routinely overstate all of these.

Lenses:

1. **New models and tasks**: model families, task types, weight variants
   added. Check `libreyolo/models/`, registry/config files, `WEIGHT_VARIANTS`,
   `libreyolo/config/datasets/`, docs pages added.
2. **Features and API surface**: new CLI commands/flags, new public API
   (exports in `libreyolo/__init__.py`), export formats, new skills shipped
   in `skills/`.
3. **Bug fixes**: every fix, including obscure ones. `git log --grep` for
   fix/bug plus a pass over merged PRs
   (`gh pr list -R LibreYOLO/libreyolo --state merged --search "merged:>=<base-date>"`).
4. **Breaking changes and deprecations**: changed signatures, renamed or
   removed public names, changed defaults, CLI arg changes. Diff
   `libreyolo/__init__.py`, `libreyolo/cli/`, and anything in
   `docs/*_schema.md` between the two refs.
5. **Changed output semantics** (do not fold into "bug fixes"): changes
   that make the same call return a *different number or shape* than
   `vLAST` did, even when the code looks like a fix. Examples that bit the
   v1.3.1 range: validation metric keys renamed (old precision/recall were
   never a real P/R pair; became honest `map_5095`/`ar_100` with deprecated
   aliases), depth metrics switched per-pixel to per-image (RMSE changes),
   a double letterbox-scaling bug in DETR val preprocessors that had been
   mis-scaling GT boxes and therefore mAP. Users comparing numbers across
   versions must be told; these belong in a **Changed** section, called out
   loudly.
6. **Training / internals / performance**: trainer, losses, augmentations,
   postprocess moves, refactors big enough to mention.
7. **Tests, CI, docs, packaging, licensing**: new test files/counts,
   workflow changes, dependency/extras changes, NOTICE / THIRD_PARTY_NOTICES
   entries. Count precisely: "N new test *modules*" is not the same as "N
   new files" (fixtures/`.npz` goldens are not modules); state which.
8. **People and numbers**: `git shortlog -sn vLAST..upstream/dev`, commit
   count, files touched, closed issues, PR count.

Merge into `facts.md` (numbered `F1..Fn`, grouped by lens).

### Fact-check pass (before any of it becomes prose)

The first fan-out finds candidates; a second pass is what makes them true.
Run one more round of read-only agents whose only job is to **hunt false
positives** in `facts.md`: for each fact, is the claim exactly what the
code does, or an overstatement? This pass is where "native port" gets
downgraded to "reimplementation that still runs through transformers", an
unwired family gets struck from the shipped list, a modified-not-new family
gets removed, and miscounts get corrected. Also lock attribution here:
**get PR numbers from the merge-commit text** (`Merge pull request #N`).
Something that landed as a direct commit with no merge-PR line is
attributed as "issue #N" or "commit `<sha>`", **not** "PR #N". Mark each
fact `verified` or `struck (reason)`. Only verified facts may appear in the
changelog.

Also record each surviving capability's validation evidence, including any
RF1 exclusion and its exact reason. The changelog must describe callable
behavior and known limits separately.

Out of scope for the changelog (do not spend the pass on these): raw
performance/throughput deltas and docs/website prose content. Everything
that changes code behavior or public API is in scope.

### Write the changelog

Where it lives: there is currently **no tracked `CHANGELOG.md`** on
`upstream/dev` (releases have shipped through GitHub Releases only). Ask
the user whether to (a) start a root `CHANGELOG.md` in
[keepachangelog](https://keepachangelog.com) format, committed via the
release PR, and/or (b) just produce `changelog.md` in the working dir for
the GitHub release body. Default to both: a tracked file is worth starting.

Format (keepachangelog sections; only include sections that have content):

```markdown
## [X.Y.Z] - YYYY-MM-DD

One-paragraph summary, leading with the strongest verified item.

### Added        New models (sizes + license posture), tasks, features, CLI, skills.
### Changed      Changed-output-semantics (loud), behavior/default changes.
### Fixed        Every non-omitted fix, one line each, plain language.
### Deprecated   Renamed-with-alias, soft-removed names.
### Removed      Hard removals (what to do instead).

Contributors: everyone in the range, alphabetized, with what they did.
Stats: N commits, N files, +N/-N lines, N new test modules.

`pip install --upgrade libreyolo`
```

Tone: factual, concrete, numbers over adjectives, zero hype. No em dashes.
Every line traces to a **verified** fact id. Show the changelog to the user
**now**; they curate (a fix can advertise the bug), you never silently drop
items.

## Phase 2: Gates

Run these in parallel where possible; append each result to the scoreboard.

### Gate A: unit CI on the exact candidate SHA

Already covered by preflight check 4 if dev HEAD is the candidate. If the
latest dev run is stale or red, stop here.

### Gate B: GPU e2e on Modal (reuse the nightly, don't reinvent it)

The nightly harness is the canonical GPU gate. Two ways to run it against
the candidate; prefer the first (no local credentials needed):

```bash
# 1. Dispatch the existing workflow; force bypasses the tested-SHA cache
gh workflow run e2e-nightly-dev.yml -R LibreYOLO/libreyolo -f force=true
gh run watch -R LibreYOLO/libreyolo <run-id>   # ~up to 3h; check step summary

# 2. Or run the Modal app directly (needs MODAL_TOKEN_ID/SECRET in env)
uvx modal run tools/ci/modal_nightly.py --ref <candidate-sha> --target test_nightly
```

Both print a `MODAL_NIGHTLY_RESULT {json}` line with `status`, runtime, and
estimated GPU cost; record status **and cost** on the scoreboard.

**A green nightly in the run list does not mean the candidate was tested.**
The workflow skips when the target already passed this week, and a skipped run
still reports success. Counting a skip as a pass is how an untested commit
ships. Two questions, in order, and both must be answered from the run itself:

**1. Did the suite run at all, on this SHA?** Read the `e2e` job's
**conclusion**, not whether the job is listed: a skipped run still lists
`e2e (dev)`, with conclusion `skipped`. The guard's step summary on every run
names the target it resolved.

```bash
gh api repos/LibreYOLO/libreyolo/actions/runs/<run-id>/jobs \
  --jq '.jobs[] | "\(.name) \(.conclusion)"'
# e2e (dev) success  -> the suite ran and the job passed
# e2e (dev) skipped  -> it never ran; this run is not evidence about any SHA
#                       (a companion "Skipped (dev already tested)" job says so)
```

**2. Did it pass?** For the dev workflow, do not stop at the artifact name.
The upload is `if: always()`, so `modal-nightly-<sha>` also exists for failed,
timed-out, and unknown-status runs. Read the status out of it:

```bash
gh run download <run-id> -R LibreYOLO/libreyolo \
  -n modal-nightly-<candidate-sha> -D /tmp/nightly
jq -r '.status, .ref, .estimated_gpu_cost_usd' /tmp/nightly/modal-nightly-result.json
# status must be exactly "passed", and ref must be the candidate SHA
```

Only `status == "passed"` on the candidate SHA counts; then skip the re-run,
that is the point of reusing the nightly. Anything else, including three
consecutive green runs, is not evidence about the candidate.

**A green nightly on an ancestor is not a pass on the tag.** v1.5.0 Gate B
ran on `7fd0c5df`; the tag is `c25f6dff` (version bump, docs, and the
Windows `os.replace` retry). The next cut must re-run (or wait for a
nightly that already targeted) the SHA that will be tagged, not the last
green ancestor. After the version-bump commit lands, Gate B evidence older
than that commit is stale.

Note the artifact exists **only** for the dev workflow. The release and PyPI
workflows run on the self-hosted runner and upload nothing, so for those the
evidence is the job list plus the run conclusion from question 1.

For gates beyond the nightly contract (RF5 training benchmark, a heavier
suite, a specific GPU), use the `launch-serverless-gpu-job` skill
(Vast / Modal / Beam) instead of hand-rolling anything.

### Gate C: risk-targeted e2e subset (local GPU, optional but cheap)

Map Phase-1 facts to test files: any model family touched in the range gets
its `tests/e2e/test_<family>*.py` run locally via the
`libreyolo-run-e2e-tests` skill (one file per process, `-m "e2e and not
rf5"`). This catches family-specific regressions the general nightly case
may not exercise. Skip if no local GPU; note it on the scoreboard.

### Gate D: weight autodownload probe

Every weight variant that is new or renamed in the range must actually
resolve. For each, HEAD-request its Hugging Face URL (or instantiate the
model with autodownload in a temp cache) and record HTTP 200. A release
that advertises a model whose weights 404 is the most embarrassing failure
this project can have; this gate exists because of that.

### Gate E: wheel build + fresh-venv smoke

Build with `uv build` to smoke-test the **same artifact path CI uses**
(`publish.yml` builds with `uv build`, not `python -m build`). Then install
the wheel into a throwaway venv and import it. Paths below are POSIX; on
this Windows box use the scratchpad dir and `Scripts/` layout (the sibling
`merge-to-dev` skill has the Windows form):

```bash
uv build                     # sdist + wheel; MANIFEST.in must keep weights out
# POSIX:
python -m venv /tmp/relsmoke && /tmp/relsmoke/bin/pip install dist/*.whl
/tmp/relsmoke/bin/python -c "import libreyolo; print(libreyolo.__version__)"
/tmp/relsmoke/bin/libreyolo --help
# Windows (PowerShell): use a scratchpad venv, then
#   <venv>\Scripts\python.exe -c "import libreyolo; print(libreyolo.__version__)"
#   <venv>\Scripts\libreyolo.exe --help
```

Check the sdist/wheel size is sane (no accidentally bundled weights or
datasets) and the version string matches expectation.

### Gate F: notices and license sync

If Phase-1 lens 1 found new model families or ported code, verify `NOTICE`,
`THIRD_PARTY_NOTICES.txt`, and `weights/LICENSE_NOTICE.txt` gained the
matching entries in the same range. A new family with no notices diff is a
red flag; surface it as a licensing decision for the user, never patch it
silently.

### Gate G: docs drift

Every headline changelog item should be reachable in docs: repo `docs/`
and, where relevant, the website repo. List headline facts with no doc
mention; the user decides whether that blocks or ships as follow-up.

## Phase 3: Go/no-go, then cut it

Present the scoreboard + curated changelog. On explicit "go":

This is the minor / feature-drop path (cut from `dev`). For a true narrow
hotfix, use the "Minor vs patch" branch-off-`release` variant instead.

1. **Merge `release` -> `dev` first** (recovers release-only hotfixes;
   preflight check 3 told you if there are any). Resolve conflicts on dev;
   the version line auto-merges to the clean release value, so **manually
   set dev back to the next `X.Y.Z.dev0`** (the number the user chose).
2. **Bump + open the release PR.** Branch off merged dev, set
   `pyproject.toml` to the clean `X.Y.Z` the user chose, and push. Per
   `AGENTS.md`, the agent **may open the `dev -> release` PR itself**. The
   description must follow `.github/pull_request_template.md` with a
   `## Code provenance` section that is accurate for the actual diff, and
   must state that the PR shows no CI checks by design and must be merged
   with a **merge commit, never squash** (squash collapses the whole
   cycle's history). Opening the PR is where the agent's authority stops:
   the human reviews, approves and merges. Handing over the one-click
   compare URL
   `https://github.com/LibreYOLO/libreyolo/compare/release...<branch>?expand=1`
   instead of opening the PR remains a valid alternative, and is the better
   one when the human wants the description in their own words.
3. **GitHub release (human action).** Cutting the release is the user's. By
   default the agent creates **no** release object: hand over the changelog
   file and the New Release URL and let the human fill it in, so no
   agent-created draft or tag target is left behind if the release is
   cancelled or retargeted.

   ```
   https://github.com/LibreYOLO/libreyolo/releases/new?target=release&tag=vX.Y.Z
   ```

   The human sets the title, pastes `changelog.md`, and publishes; that is
   what creates the tag and fires `publish.yml`.

   **Prefill the form so the human only reviews and clicks Publish.** The
   `releases/new` page accepts `title` and `body` query params, so hand
   over a URL with everything filled in. Prefilling creates nothing (no tag,
   no draft) until the human clicks Publish, so it stays inside the
   "tag creation is a human action" rule below. Build it by URL-encoding
   `changelog.md`:

   ```python
   import urllib.parse, pathlib
   body = pathlib.Path("changelog.md").read_text(encoding="utf-8")
   q = urllib.parse.urlencode(
       {"target": "release", "tag": "vX.Y.Z", "title": "vX.Y.Z", "body": body}
   )
   print("https://github.com/LibreYOLO/libreyolo/releases/new?" + q)
   ```

   GitHub accepts a few KB of `body` in the URL (a full changelog is ~3 KB,
   well within limits). Still hand over the raw `changelog.md` too, in case
   the human wants to paste manually.

   **Standardise what you hand over (past releases drifted: "v1.3.0",
   "v1.2.0 release", "LibreYOLO v1.1.1", "Release v1.0.0" are all real
   past titles).** When you give the human the New Release URL, also give
   them the exact fields:
   - **Title**: the bare tag `vX.Y.Z` (matches the most recent release).
   - **Description**: the `changelog.md` from Phase 1, which must open with
     a bold lead line `LibreYOLO **vX.Y.Z**, <one-line summary>` followed by
     `##` sections (bolded item names + PR/issue refs), matching the v1.3.0
     body shape.
   - **Pre-release**: unchecked. **Set as latest**: leave on.
   - **Binaries**: attach none; the wheel ships via `publish.yml`.

   Do **not** run `gh release create` yourself, even with `--draft`. The gh
   CLI creates the `vX.Y.Z` git tag up front for a draft ("if a matching git
   tag does not yet exist, one will automatically get created"), and
   `publish.yml` fires on `v*` tag pushes, so an agent-created draft can push
   a release tag and kick off the publish workflow before the human has
   decided anything. Tag creation stays a human action.
4. **Approve the publish.** The run pauses on the `pypi` GitHub Environment
   for a manual approval (the job shows as "Publish to PyPI"). This is a
   human-required click; the agent never approves it.

## Phase 4: Post-publish verification (do not skip)

```bash
# PyPI index catches up within minutes; poll, don't assume
pip index versions libreyolo
# POSIX (on Windows use a scratchpad venv + Scripts/python.exe):
python -m venv /tmp/pypismoke && /tmp/pypismoke/bin/pip install libreyolo==X.Y.Z
/tmp/pypismoke/bin/python -c "import libreyolo; assert libreyolo.__version__ == 'X.Y.Z'"
# GPU e2e against the tagged SHA (mandatory dispatch; do not skip).
# Pass the tag: the workflow otherwise resolves refs/heads/release, which
# can move after the tag is cut. github.run.headSha is the default branch
# at dispatch time and is not the evidence.
gh workflow run e2e-nightly-release.yml -R LibreYOLO/libreyolo \
  -f ref=vX.Y.Z -f force=true
# Confirm the guard resolved that tag. The step summary line is
# "Target: `tag vX.Y.Z@<sha>`" and <sha> must equal `git rev-parse vX.Y.Z`.
# github.run.headSha is the default branch at dispatch time; ignore it.
```

Append results to the scoreboard. Dispatching the release nightly is
required; the user may OK not waiting for it to go green, but they may
not OK skipping the dispatch. The 1.5.0 cut credited Gate B on an
ancestor of the tag and never dispatched this workflow (issue #744
item 16). The release is "done" when PyPI installs clean and the
release nightly has been dispatched against the tag SHA.

## Phase 5: Marketing handoff

Hand `facts.md` + `changelog.md` to the marketing repo's
`new-version-release` skill (checkout at `../marketing`). It owns curation
for the announcement, the GitHub release body polish, the LinkedIn carousel
pipeline (generate -> audit -> simulator -> final review -> queue), and the
Reddit GIF post pack. Your facts file matches its expected format (numbered
facts with evidence), so it can skip its own fact-finding phase and start
at curation. Never post to social platforms yourself; assets are queued for
the user to post by hand.

## Anti-patterns

- Writing the changelog from memory or from PR titles alone; titles lie,
  diffs don't. Verify each headline claim against the actual source.
- Counting an unwired native port, a modified-not-new family, or an
  internal backbone as a shipped user-facing family. Verify it's exported
  and reachable first.
- Shipping the changelog without the false-positive fact-check pass.
- Replacing exact validation evidence with a blanket capability label.
- Attributing a direct-commit change as "PR #N" when it has no merge-PR.
- Inferring the version number from the diff size instead of asking; the
  label is the human's call.
- Treating the empty checks list on the `dev -> release` PR as green.
- Re-running a 3-hour Modal nightly when the last green run already tested
  the candidate SHA.
- Reading a green nightly as a pass on the candidate. Skipped runs are green
  too, and the dev artifact is uploaded on failure as well, so neither the tick
  nor the artifact name is the evidence. The evidence is an `e2e` job that
  concluded `success`, plus `status == "passed"` inside
  `modal-nightly-result.json`.
- Crediting Gate B on a SHA that is not the tag. An ancestor nightly does
  not measure the bump commit. Dispatch `e2e-nightly-release.yml` after
  publish with `-f ref=vX.Y.Z` and check that the guard resolved that tag
  SHA, not `refs/heads/release`.
- Squash-merging the release PR.
- Bumping the version on dev instead of on the release-PR branch.
- Skipping Phase 4 because "publish.yml was green" (green publish and a
  broken install have coexisted before in other projects; verify the
  artifact users actually get).
- Fixing a missing license notice yourself instead of surfacing it.
- "Helpfully" bumping the version or date in `CITATION.cff` / `.zenodo.json`.
  They are frozen on purpose; a version there fragments the citation record.
  This exact mistake has already been made and reverted once.

## Related

- `RELEASING.md`: the minimal engineering process this skill wraps.
- `skills/libreyolo-run-e2e-tests/`: how to run e2e correctly (markers,
  one-file-per-process rule).
- `skills/launch-serverless-gpu-job/`: rented/serverless GPUs for gates
  beyond the nightly contract.
- `../marketing/skills/new-version-release/`: the announcement pipeline
  (curation, GitHub release body, LinkedIn carousel, Reddit GIF).
- `.github/workflows/`: `e2e-nightly-dev.yml`, `e2e-nightly-release.yml`,
  `publish.yml`, `unit-tests.yml`, `install-smoke.yml`.
