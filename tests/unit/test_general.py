import logging
from types import SimpleNamespace

import pytest

from libreyolo.utils.general import log_saved_result

pytestmark = pytest.mark.unit


def test_log_saved_result_attaches_path_and_logs(caplog, tmp_path):
    result = SimpleNamespace()
    save_path = tmp_path / "annotated.jpg"

    caplog.set_level(logging.INFO, logger="libreyolo.utils.general")

    saved_path = log_saved_result(result, save_path)

    assert saved_path == str(save_path)
    assert result.saved_path == str(save_path)
    assert f"Results saved to {save_path}" in caplog.text
