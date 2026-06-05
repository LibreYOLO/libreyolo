# LibreYOLO Testing Strategy

Version: 2.1

This is the CI/test contract for LibreYOLO. Times are UTC.

## Layers

| Layer | Workflow / owner | Runs on | Trigger | Green means |
| --- | --- | --- | --- | --- |
| Unit | `.github/workflows/unit-tests.yml` | GitHub Linux, macOS, Windows; Python 3.10 | push to `dev`, PR to `dev`, manual | CPU-safe library and CLI/API behavior works. |
| Install smoke | `.github/workflows/install-smoke.yml` | GitHub clean VMs; Python 3.10 | push to `dev`, PR to `dev`, manual, daily, publish | A clean user env can install, import, and start LibreYOLO. |
| GPU e2e nightly | `.github/workflows/e2e-nightly-{release,dev,pypi}.yml` | self-hosted `gpu`, `libreyolo-e2e` tower runner | daily staggered schedule, manual | Selected real-model GPU tests execute and pass. |
| Manual QA | humans | human machine | before releases/demos/hackathons | Representative user behavior was checked by a human. |

Boundaries:

- CLI/API correctness: unit tests.
- Clean install/import/package data: install smoke.
- Model loading, inference, training, validation, tracking, video: GPU e2e.
- Visual quality and release workflow confidence: manual QA.

## Unit

Command:

```bash
uv run --no-sync pytest tests/unit -m unit
```

Scope: CPU-safe behavior, config, parsing, errors, serialization, and CLI/API
logic.

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
`tests/e2e/nightly_contract.py`, `tests/e2e/conftest.py`,
`tests/e2e/test_deterministic_inference.py`,
`tests/e2e/test_rf1_training.py`, `Makefile`.

Execution: targets `dev`, `release`, latest PyPI; 180 minute timeout per target;
SHA/version cache skips unchanged targets; manual `force=true` runs all targets.
Schedules are staggered at `03:00` UTC for `release`, `04:00` UTC for `dev`,
and `05:00` UTC for PyPI. Do not add a `pull_request` trigger.

Commands:

```bash
make test_general_nightly
make test_flagship_nightly
make test_nightly
```

V2.1 contract:

- `general_nightly`: one smallest native inference case for every public
  detector family that has a public auto-download route (LibreYOLO HF, or Deci's
  CDN for YOLO-NAS); currently 15 tests.
- `flagship_nightly`: heavier YOLO9/RF-DETR native validation, video, tracking,
  CLI, and one RF1 training/reload size per flagship family; currently 48 tests
  with `not export_backend`. The full RF1 size matrix remains available under
  `-m rf1` for manual or future full-matrix runs.
- Detector families cover detection. L2CS gaze is non-redistributable (no public
  download route), so it runs as a non-gated per-family suite
  (`tests/e2e/test_l2cs_gaze.py`) that skips when the weight is not staged
  locally, rather than gating the nightly.
- Export backends are outside default nightly.
- Nightly-selected skips are failures.

Collect:

```bash
uv pip install --group dev -e ".[rfdetr,onnx]"
pytest tests/e2e --collect-only -q -m general_nightly
pytest tests/e2e --collect-only -q -m "flagship_nightly and not export_backend"
```

Missing local weights before full green: `weights/LibreDEIM*.pt`,
`weights/LibreRTDETRv2r18.pt`, `weights/LibreRTDETRv4s.pt`. YOLO-NAS now
auto-downloads from Deci's CDN (checksum-verified), and L2CS gaze is non-gated
and skips when `weights/LibreL2CSr50.pt` is absent.

## Versioning

Patch: wording only. Minor: added coverage/platform/threshold/runtime change.
Major: green run means materially different confidence.
