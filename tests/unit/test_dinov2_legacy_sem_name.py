"""``LibreYOLO("LibreRFDETRn-sem.pt")`` resolves to the published DINOv2 semantic weight."""

from __future__ import annotations

import pytest

from libreyolo.models.dinov2.model import LibreDINOv2

pytestmark = pytest.mark.unit


def test_legacy_semantic_name_resolves():
    assert LibreDINOv2.detect_size_from_filename("weights/LibreRFDETRn-sem.pt") == "n"
    assert LibreDINOv2.detect_task_from_filename("LibreRFDETRn-sem.pt") == "semantic"
    assert LibreDINOv2.get_download_url("LibreRFDETRn-sem.pt") == (
        "https://huggingface.co/LibreYOLO/LibreRFDETRn-sem/resolve/main/LibreRFDETRn-sem.pt"
    )
    # Other names still go through the regular DINOv2 rules.
    assert LibreDINOv2.detect_size_from_filename("LibreRFDETRs-sem.pt") is None
