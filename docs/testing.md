# LibreYOLO Testing Strategy

Version: 1.0

This is the CI/test contract for LibreYOLO. Times are UTC.

## Layers

| Layer | Workflow / owner | Runs on | Trigger | Green means |
| --- | --- | --- | --- | --- |
| Unit | `.github/workflows/unit-tests.yml` | GitHub Linux, macOS, Windows; Python 3.10 | push to `dev`, PR to `dev`, manual | CPU-safe library and CLI/API behavior works. |
| Install smoke | `.github/workflows/install-smoke.yml` | GitHub clean VMs; Python 3.10 | push to `dev`, PR to `dev`, manual, daily, publish | A clean user env can install, import, and start LibreYOLO. |
| GPU e2e nightly | `.github/workflows/e2e-nightly.yml` | self-hosted `gpu`, `libreyolo-e2e` tower runner | daily `03:00`, manual | Selected real-model GPU tests execute and pass. |
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

Files: `tests/e2e/nightly_contract.py`, `tests/e2e/conftest.py`,
`tests/e2e/test_deterministic_inference.py`, `Makefile`.

Execution: targets `dev`, `main`, latest PyPI; 180 minute timeout per target;
SHA/version cache skips unchanged targets; manual `force=true` runs all targets.
Do not add a `pull_request` trigger.

Commands:

```bash
make test_general_nightly
make test_flagship_nightly
make test_nightly
```

V1 contract:

- `general_nightly`: one smallest native inference case for every public family;
  currently 16 tests.
- `flagship_nightly`: heavier YOLO9/RF-DETR native validation, training, video,
  tracking, and CLI; currently 57 tests with `not export_backend`.
- L2CS covers gaze inference; detector families cover detection.
- Export backends are outside default nightly.
- Nightly-selected skips are failures.

Collect:

```bash
pytest tests/e2e --collect-only -q -m general_nightly
pytest tests/e2e --collect-only -q -m "flagship_nightly and not export_backend"
```

Missing local weights before full green: `downloads/yolonas/yolo_nas_s_coco.pth`,
`weights/LibreDEIM*.pt`, `weights/LibreRTDETRv2r18.pt`,
`weights/LibreRTDETRv4s.pt`, possibly `weights/LibreL2CSr50.pt`.

## Versioning

Patch: wording only. Minor: added coverage/platform/threshold/runtime change.
Major: green run means materially different confidence.
