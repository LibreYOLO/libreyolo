import pytest


pytestmark = pytest.mark.unit


def test_codex_review_smoke_branch_exists():
    assert True


def test_codex_review_smoke_second_push():
    assert 1 + 1 == 2
