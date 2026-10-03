"""Unit tests for libreyolo.utils.general.increment_path."""

from pathlib import Path

import pytest

from libreyolo.utils.general import increment_path

pytestmark = pytest.mark.unit


def test_free_path_is_returned_unchanged(tmp_path):
    target = tmp_path / "runs" / "exp"
    result = increment_path(target)
    assert result == target
    assert isinstance(result, Path)
    assert not target.exists()  # mkdir defaults to False


def test_accepts_str_and_returns_path(tmp_path):
    result = increment_path(str(tmp_path / "exp"))
    assert result == tmp_path / "exp"
    assert isinstance(result, Path)


def test_exist_ok_reuses_existing_path(tmp_path):
    target = tmp_path / "exp"
    target.mkdir()
    assert increment_path(target, exist_ok=True) == target


def test_existing_directory_gets_counter_from_two(tmp_path):
    (tmp_path / "exp").mkdir()
    assert increment_path(tmp_path / "exp") == tmp_path / "exp2"
    (tmp_path / "exp2").mkdir()
    assert increment_path(tmp_path / "exp") == tmp_path / "exp3"


def test_first_gap_is_used(tmp_path):
    for name in ("exp", "exp2", "exp4"):
        (tmp_path / name).mkdir()
    assert increment_path(tmp_path / "exp") == tmp_path / "exp3"


def test_sep_goes_between_name_and_counter(tmp_path):
    (tmp_path / "exp").mkdir()
    assert increment_path(tmp_path / "exp", sep="_") == tmp_path / "exp_2"
    (tmp_path / "exp_2").mkdir()
    assert increment_path(tmp_path / "exp", sep="_") == tmp_path / "exp_3"


def test_existing_file_keeps_its_extension(tmp_path):
    result_file = tmp_path / "result.json"
    result_file.write_text("{}")
    assert increment_path(result_file) == tmp_path / "result2.json"
    (tmp_path / "result2.json").write_text("{}")
    assert increment_path(result_file) == tmp_path / "result3.json"
    assert increment_path(result_file, sep="-") == tmp_path / "result-2.json"


def test_existing_file_without_extension(tmp_path):
    (tmp_path / "LOG").write_text("")
    assert increment_path(tmp_path / "LOG") == tmp_path / "LOG2"


def test_directory_with_dot_in_name_is_not_split(tmp_path):
    (tmp_path / "run.v1").mkdir()
    assert increment_path(tmp_path / "run.v1") == tmp_path / "run.v12"


def test_mkdir_creates_returned_directory_with_parents(tmp_path):
    target = tmp_path / "a" / "b" / "exp"
    first = increment_path(target, mkdir=True)
    assert first == target and first.is_dir()
    second = increment_path(target, mkdir=True)
    assert second == tmp_path / "a" / "b" / "exp2" and second.is_dir()


def test_mkdir_with_exist_ok_on_existing_directory(tmp_path):
    target = tmp_path / "exp"
    target.mkdir()
    assert increment_path(target, exist_ok=True, mkdir=True) == target
    assert target.is_dir()


def test_counter_has_no_upper_bound(tmp_path):
    (tmp_path / "exp").mkdir()
    for n in range(2, 1200):
        (tmp_path / f"exp{n}").mkdir()
    result = increment_path(tmp_path / "exp")
    assert result == tmp_path / "exp1200"
    assert not result.exists()
