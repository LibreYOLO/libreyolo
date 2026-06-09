"""Doctor smoke test on the bundled coco8 dataset.

Usage:
    pytest tests/e2e/test_doctor_coco8.py -v -m e2e
"""

import json

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.external_data, pytest.mark.network]


def test_doctor_coco8_runs_clean():
    from libreyolo.doctor import diagnose

    report = diagnose("coco8.yaml", progress=False, autodownload=True)
    # coco8 is a healthy sample dataset: no errors, JSON contract holds.
    assert not report.errors
    payload = json.loads(json.dumps(report.to_dict()))
    assert payload["stats"]["splits"]["train"]["images"] == 4
    assert payload["stats"]["splits"]["val"]["images"] == 4
