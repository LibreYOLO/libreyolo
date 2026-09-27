# LibreYOLO Testing Strategy

Version: 3.1

This is the CI/test contract for LibreYOLO. Times are UTC.

## Layers

| Layer | Workflow / owner | Runs on | Trigger | Green means |
| --- | --- | --- | --- | --- |
| Unit | `.github/workflows/unit-tests.yml` | GitHub Linux, macOS, Windows; Python 3.10. Parallel via pytest-xdist; `distributed`-marked gloo tests run in a separate serial Linux-only job. | push to `dev`, PR to `dev`, manual | CPU-safe library and CLI/API behavior works. |
| Install smoke | `.github/workflows/install-smoke.yml` | GitHub clean VMs; Python 3.10 | push to `dev`, PR to `dev`, manual, daily, publish | A clean user env can install, import, and start LibreYOLO. |
| GPU e2e nightly | `.github/workflows/e2e-nightly-dev.yml` | GitHub-hosted controller; Modal L4 GPU worker | daily schedule, manual | Selected real-model GPU tests execute and pass on latest `dev`. |
| GPU e2e manual | `.github/workflows/e2e-nightly-{release,pypi}.yml` | self-hosted `gpu`, `libreyolo-e2e` tower runner | manual | Selected real-model GPU tests execute and pass on `release` or latest PyPI when explicitly requested. |
| Manual QA | humans | human machine | before releases/demos/hackathons | Representative user behavior was checked by a human. |

Boundaries:

- CLI/API correctness: unit tests.
- Clean install/import/package data: install smoke.
- Model loading, inference, training, validation, tracking, video: GPU e2e.
- Visual quality and release workflow confidence: manual QA.

## Unit

Command:

```bash
make test_pr_gate

# Equivalent commands used by the GitHub workflow. The main cross-platform job
# runs the suite in parallel, excluding the multi-rank gloo tests:
LIBREYOLO_PR_GATE=1 uv run --no-sync pytest tests/unit -m "unit and not external_data and not network and not distributed" -n auto --dist loadfile

# The `distributed`-marked gloo tests run serially, in a separate Linux-only
# job (each test mp.spawns its own ranks; gloo process-group init costs ~105s
# per test on the macOS runner, and the gradient-parity tolerances are
# BLAS-sensitive across platforms):
LIBREYOLO_PR_GATE=1 uv run --no-sync pytest tests/unit -m "unit and not external_data and not network and distributed"
```

Scope: CPU-safe behavior, config, parsing, errors, serialization, and CLI/API
logic.

### PR Gate v1.0

The PR gate is the merge-blocking unit-test contract for pushes and pull
requests to `dev`.

Contract:

- No external HTTP/network access. Localhost sockets remain allowed so local
  distributed-training unit tests can run.
- No live dataset, model-weight, Hugging Face, GitHub release, cloud bucket, or
  CDN downloads.
- No dependency on staged external datasets, staged external weights, secrets,
  GPU hardware, CUDA, or vendor export runtimes.
- Tests that intentionally require staged datasets/weights must use
  `@pytest.mark.external_data` and remain outside the PR-gate marker expression.
- Tests that intentionally require non-local network access must use
  `@pytest.mark.network` and remain outside the PR-gate marker expression.

New unit tests are in the PR gate by default. If a test cannot satisfy this
contract, prefer a local fixture or mock. If real external data is essential,
move the coverage to the appropriate e2e, nightly, or manual suite.

### Native-port parity gates

ConvNeXt V2: `tests/unit/test_convnextv2_parity.py` compares all eight
official 224px classifiers to the pinned Meta graph (set
`CONVNEXTV2_UPSTREAM` and `CONVNEXTV2_CHECKPOINTS`). Local unit tests cover
checkpoint, training/resume, and ONNX/TorchScript behavior.
`tests/e2e/test_convnextv2.py -m e2e` checks real weights and smoke10
fine-tuning; it is not added to the detection-only nightly catalog.
Evidence and limits: [ConvNeXt V2](provenance/convnextv2.md).

A ported architecture must have a pinned-reference tensor parity test in
addition to ordinary shape and API tests. The reference checkout and any
checkpoint remain external and the test is marked `external_data`; the PR gate
still exercises the port with local random weights plus exported-runtime
parity.

The edge-specialist gate is `tests/unit/test_edge_models_parity.py`:

```bash
LIBREYOLO_TEED_UPSTREAM=/path/to/TEED \
LIBREYOLO_TEED_CHECKPOINT=/path/to/local/teed.pth \
LIBREYOLO_DEXINED_UPSTREAM=/path/to/DexiNed \
pytest -q tests/unit/test_edge_models_parity.py
```

The pinned commits and source licenses are recorded in the per-family NOTICE
files. External checkpoints retain their own data/weight terms and must not be
added to the repository as test fixtures.

### LibreMODUS external-weight gates

LibreMODUS is marked `modus`. Its PR-gate coverage is entirely local:

```bash
pytest -q tests/unit/test_modus.py
```

That test builds a two-layer random MoT model and a fake tokenizer. It checks
the released 196,840-token ordering, learned checkpoint-key surface,
Accelerate dispatch with checkpoint-absent deterministic buffers, constrained
COCO/grounding grammars, dense payloads, the supported matrix, chaining,
self-verification, and local FP8 cache arithmetic. It performs no HTTP request
and loads no external weight.

The dense-boundary tests also pin the public camera-facing normal convention,
relative-depth normalization, aligned multi-input canvas requirement, separate
standard text/image guidance scales (`4.0` / `2.0`), and authenticated-only
upstream download policy.

The real checkpoint is intentionally excluded from scheduled CI: it is about
30 GB, is loaded directly from an external custom-term repository, and needs
hardware larger than the standard nightly L4 for the BF16 reference. Before a
release claims full LibreMODUS validation, a maintainer runs the following
manual gates with:

- MODUS source pinned to
  `c299ef0fbba1cfe7c93336c45d7085afd770c0fa`;
- checkpoint revision
  `8428a81602c19141e422b1e1795dddcb5d2bc14b`;
- the same GPU architecture, image preprocessing, seeds, noise tensors, and
  ten flow steps for upstream/LibreYOLO parity;
- externally obtained datasets that are never added to this repository.

Required BF16 results:

| Gate | Required result |
| --- | --- |
| Step parity, five images per task | velocity-field and autoregressive-logit maximum absolute difference `< 1e-3` |
| NYUv2 depth | AbsRel within `0.005` of `0.065` |
| NYUv2 normals | mean angular error within `0.5°` of `19.92°` |
| RefCOCO val, fixed 500-example subset | grounding accuracy within `1.0` point of `54.5` |
| COCO detection | record the exact subset, mAP, and a qualitative grid; upstream publishes no reference number |

After recording this port's BF16 numbers, run the identical samples and seeds
with `LibreMODUS(dtype="fp8")`:

| FP8 delta versus this port's BF16 result | Acceptance |
| --- | --- |
| NYUv2 depth AbsRel | `<= +0.002` |
| NYUv2 normal mean angle | `<= +0.15°` |
| RefCOCO grounding | `>= -0.3` point |
| BIPED edge ODS | `>= -0.005` |
| 512px end-to-end peak VRAM | `< 12 GB` |

If an FP8 quality gate fails, widen the high-precision decoder-block exemption
and record the resulting recipe; do not weaken the threshold silently.
Metric/parity/VRAM gates are **not yet passed** unless a PR or release record
contains the hardware, software versions, commands, dataset subset manifests,
and raw results. A CPU unit green run is not evidence for them.

## Install Smoke

Scripts:

- `tests/smoke/run_install_smoke.py`
- `tests/smoke/install_surface.py`

Matrix:

| Mode | Trigger | Runners |
| --- | --- | --- |
| editable install from checkout | push to `dev`, PR to `dev`, manual | Linux, macOS, Windows |
| wheel build/install | push to `dev`, PR to `dev`, manual | Linux |
| sdist build/install | push to `dev`, PR to `dev`, manual | Linux |
| PyPI install | daily `03:00`, manual, after PyPI publish | Linux, macOS, Windows |

Checks: fresh venv, selected install mode, `pip check`, `import libreyolo`,
`LibreYOLO`, `Results`, `SAMPLE_IMAGE`, bundled sample image exists,
lazy VLM, promptable-segmentation, and open-vocabulary family exports,
`libreyolo --help`, `libreyolo version --json --quiet`,
`libreyolo checks --json --quiet`, and import location check.

Reproduce:

```bash
python tests/smoke/run_install_smoke.py --mode editable
python tests/smoke/run_install_smoke.py --mode wheel
python tests/smoke/run_install_smoke.py --mode sdist
python tests/smoke/run_install_smoke.py --mode pypi
```

Non-goals: weights, datasets, inference, training, validation, export, CUDA,
and visual inspection.

## GPU E2E Nightly

Files: `.github/workflows/e2e-nightly-release.yml`,
`.github/workflows/e2e-nightly-dev.yml`,
`.github/workflows/e2e-nightly-pypi.yml`,
`tools/ci/modal_nightly.py`,
`tests/e2e/nightly_contract.py`, `tests/e2e/conftest.py`,
`tests/e2e/test_deterministic_inference.py`,
`tests/e2e/test_rf1_training.py`, `Makefile`.

Execution: scheduled nightly targets latest `dev` only; manual workflows can
target `release` and latest PyPI. The scheduled `dev` workflow runs from a
GitHub-hosted controller job and executes GPU work on Modal L4 via
`tools/ci/modal_nightly.py`; release and PyPI manual workflows still use the
self-hosted `gpu`, `libreyolo-e2e` runner. Each target has a 180 minute timeout.
SHA/version cache skips unchanged targets; manual `force=true` runs the selected
target. The scheduled `dev` run starts at `04:00` UTC. Do not add a
`pull_request` trigger.

Reading the cache correctly matters, because a skipped run is still green:

- The cache key is the target SHA (or PyPI version) plus the ISO week, so an
  unchanged target is retested once a week rather than never. Without the week
  component a quiet branch is tested once and the environment underneath it
  (Modal image, published weights, transitive dependencies) drifts untested.
- A run that skips reports success. The green tick means the guard resolved,
  not that the target was tested in that run. Every run states which it was in
  its step summary, and a skipping run shows a `Skipped (...)` job instead of
  the `e2e` job.
- To answer "was this exact commit tested, and did it pass?", read the `e2e`
  job's conclusion, not the run's. A skipped run still lists the `e2e` job with
  conclusion `skipped`, while the overall run reports success. Only
  `e2e` concluding `success` means the suite ran and passed.
- The `dev` workflow additionally uploads `modal-nightly-<sha>`, but that upload
  is `if: always()`, so the artifact exists for failed and timed-out runs too.
  Its name proves the suite executed on that SHA, not that it passed; the pass
  is `"status": "passed"` inside `modal-nightly-result.json`. The release and
  PyPI workflows upload no artifact at all.

The Modal-backed `dev` run is serialized with a GitHub Actions concurrency group
because it writes a shared Modal volume. The remote GPU function has a 180 minute
timeout and the GitHub controller leaves timeout headroom so logs and result
artifacts can still be parsed after a Modal-side timeout. It stores
`modal-nightly.log` and `modal-nightly-result.json` as GitHub Actions artifacts
and writes runtime, GPU, and estimated GPU cost to the step summary. Exact
billing remains authoritative in Modal; the GitHub value is a GPU-runtime
estimate.

Every e2e test also carries a per-test timeout, `E2E_TIMEOUT` (default 900
seconds, `0` disables), enforced by `pytest-timeout` in thread mode so that a
test wedged inside a native call, such as a stalled weight download, is killed
with a stack trace instead of burning the whole nightly budget in silence. The
suite-level Modal timeout is the last resort, not the first line of defence.

Weight reuse: the Modal volume caches loose weight files, and Hugging Face
snapshots (open-vocab, SAM, VLM families) as whole directories, since their
loaders only skip a download when the `.libreyolo_snapshot_complete` marker and
the config sit beside the weights. Verified snapshots are committed even when
the suite fails, so a run that dies mid-suite does not leave the next one
downloading them cold again. Set the `HF_TOKEN` repository secret to lift the
Modal container off anonymous Hub rate limits; without it the run stays
anonymous.

Commands:

```bash
make test_general_nightly
make test_flagship_nightly
make test_training_nightly
make test_nightly
make test_e2e E2E_TIMEOUT=1800
```

V3.1 contract:

- `general_nightly`: a curated matrix of the smallest public checkpoint for 16
  detector families. It checks stable native inference and batched/sequential
  parity, plus two open-vocabulary smoke cases; currently 34 tests.
- `flagship_nightly`: native YOLO9/RF-DETR validation, input equivalence
  (path/PIL/NumPy sources, mixed-size lists, original-image coordinates), video,
  tracking, CLI, and one RF1 training/reload size per flagship family; currently
  58 tests. The full
  RF1 size matrix remains available under `-m rf1` for manual or future
  full-matrix runs.
- `training_nightly`: opt-in training-time GPU coverage for CUDA graph capture.
  It keeps 14 representative mechanism, lifecycle, and fallback cases available
  through `make test_training_nightly`, but `make test_nightly` does not invoke
  this advanced suite. The full per-family sweep stays under `-m e2e`.
- L2CS gaze is non-redistributable (no public download route), so it runs as a
  non-gated per-family suite
  (`tests/e2e/test_l2cs_gaze.py`) that skips when the weight is not staged
  locally, rather than gating the nightly.
- `training_nightly`, export backends, ExecuTorch, inference CUDA graph
  matrices, and extended-task training suites are opt-in and outside the
  default nightly. The Make targets exclude their markers defensively, and
  collection rejects tests that combine those opt-in markers with a
  default-nightly marker.
- Nightly-selected skips are failures.

Collect:

```bash
uv pip install --group dev -e ".[rfdetr,onnx]"
pytest tests/e2e --collect-only -q -m "general_nightly and not export_backend and not executorch and not cuda_graph and not extended_training and not training_nightly"
pytest tests/e2e --collect-only -q -m "flagship_nightly and not export_backend and not executorch and not cuda_graph and not extended_training and not training_nightly"
```

Advanced suites remain available explicitly, for example:

```bash
make test_training_nightly
pytest tests/e2e/test_cuda_graph_families.py -m cuda_graph
pytest tests/e2e/test_executorch.py -m executorch
```

Missing local weights before full green: `weights/LibreDEIM*.pt`,
`weights/LibreRTDETRv2r18.pt`, `weights/LibreRTDETRv4s.pt`. YOLO-NAS now
auto-downloads from Deci's CDN (checksum-verified), and L2CS gaze is non-gated
and skips when `weights/LibreL2CSr50.pt` is absent.

## Versioning

Patch: wording only. Minor: added coverage/platform/threshold/runtime change.
Major: green run means materially different confidence.
