"""train()/val() called outside a main guard run with workers=0, and re-run workers exit with a clear message (#925)."""

from __future__ import annotations

import dataclasses
import multiprocessing
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from libreyolo.backends.base import BaseBackend
from libreyolo.models.base.model import BaseModel, _wrap_train_with_cfg
from libreyolo.utils import main_guard
from libreyolo.utils.main_guard import (
    exit_if_reimporting_main,
    guard_main_reimport,
    is_main_guarded,
    workers_for_main_process,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def reimporting_main(monkeypatch):
    monkeypatch.setattr(
        multiprocessing.current_process(), "_inheriting", True, raising=False
    )


def test_main_process_is_not_stopped():
    exit_if_reimporting_main("train()")


def test_reimporting_worker_exits_with_the_fix(reimporting_main, capsys):
    with pytest.raises(SystemExit) as exc:
        exit_if_reimporting_main("val()")

    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "val() was called at the top level" in err
    assert 'if __name__ == "__main__":' in err


def test_decorator_runs_the_check_once(reimporting_main):
    calls = []
    guarded = guard_main_reimport("val()")(lambda: calls.append(1))

    assert guard_main_reimport("val()")(guarded) is guarded
    with pytest.raises(SystemExit):
        guarded()
    assert calls == []


def test_reimporting_worker_lets_workers_zero_through(reimporting_main):
    exit_if_reimporting_main("val()", workers=0)
    guarded = guard_main_reimport("val()")(lambda data=None, workers=4: "ran")
    assert guarded(workers=0) == "ran"
    with pytest.raises(SystemExit):
        guarded()


class _TrainOnly:
    task = None
    TRAIN_CONFIG = None

    @staticmethod
    def _raw_train(self, data=None, workers=4, **kwargs):
        return "ran"


_TrainOnly.train = _wrap_train_with_cfg(_TrainOnly._raw_train)


def test_reimporting_worker_stops_train(reimporting_main, capsys):
    with pytest.raises(SystemExit):
        _TrainOnly().train(data="data.yaml")
    assert "train() was called at the top level" in capsys.readouterr().err


def test_reimporting_worker_runs_train_without_workers(reimporting_main, tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("workers: 0\n", encoding="utf-8")

    assert _TrainOnly().train(data="data.yaml", workers=0) == "ran"
    assert _TrainOnly().train(data="data.yaml", cfg=str(cfg)) == "ran"


def test_every_val_and_train_is_guarded():
    from libreyolo.models.rfdetr.model import LibreRFDETR
    from libreyolo.models.yolo9.model import LibreYOLO9

    for method in (BaseModel.val, BaseBackend.val, LibreRFDETR.val):
        assert getattr(method, "_libreyolo_main_guarded", False), method
    assert getattr(LibreYOLO9.train, "_libreyolo_cfg_wrapped", False)


def _tiny_dataset(root: Path) -> Path:
    from PIL import Image

    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (64, 64), (60 * i, 90, 120)).save(
                root / "images" / split / f"{i}.jpg"
            )
            (root / "labels" / split / f"{i}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    data = root / "data.yaml"
    data.write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\n"
        "names:\n  0: thing\n",
        encoding="utf-8",
    )
    return data


def _run_script(
    tmp_path: Path, body: str, start_method: str = "spawn"
) -> subprocess.CompletedProcess:
    data = _tiny_dataset(tmp_path / "ds")
    script = tmp_path / "script.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import multiprocessing
            multiprocessing.set_start_method({start_method!r}, force=True)
            from libreyolo import LibreYOLO9

            DATA = {str(data)!r}
            """
        )
        + textwrap.dedent(body),
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    return subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
        timeout=600,
        check=False,
    )


@pytest.mark.parametrize(
    "start_method",
    [
        "spawn",
        pytest.param(
            "forkserver",
            marks=pytest.mark.skipif(
                "forkserver" not in multiprocessing.get_all_start_methods(),
                reason="forkserver is unavailable on this platform",
            ),
        ),
    ],
)
def test_unguarded_script_runs_without_workers(tmp_path, start_method):
    result = _run_script(
        tmp_path,
        """
        model = LibreYOLO9(None, size="t", device="cpu")
        model.val(data=DATA, imgsz=64, batch=2, plots=False, workers=1)
        model.val(data=DATA, imgsz=64, batch=2, plots=False, workers=1)
        print("SCRIPT_DONE")
        """,
        start_method,
    )

    assert result.returncode == 0, result.stderr[-4000:]
    assert "SCRIPT_DONE" in result.stdout
    assert result.stderr.count("so it runs with workers=0") == 1
    assert "was called at the top level" not in result.stderr
    assert "bootstrapping phase" not in result.stderr


def test_unguarded_script_trains_without_workers(tmp_path):
    result = _run_script(
        tmp_path,
        """
        model = LibreYOLO9(None, size="t", device="cpu")
        model.train(
            data=DATA, epochs=1, imgsz=64, batch=2, workers=2, amp=False,
            project=str(__import__("pathlib").Path.cwd() / "runs"), plots=False,
        )
        print("SCRIPT_DONE")
        """,
    )

    assert result.returncode == 0, result.stderr[-4000:]
    assert "SCRIPT_DONE" in result.stdout
    assert "train() is called at the top level of your script" in result.stderr


def test_guarded_spawn_script_keeps_its_workers(tmp_path):
    result = _run_script(
        tmp_path,
        """
        if __name__ == "__main__":
            model = LibreYOLO9(None, size="t", device="cpu")
            model.val(data=DATA, imgsz=64, batch=2, plots=False, workers=1)
            print("SCRIPT_DONE")
        """,
    )

    assert result.returncode == 0, result.stderr[-4000:]
    assert "SCRIPT_DONE" in result.stdout
    assert "top level of your script" not in result.stderr


def test_unguarded_workers_zero_script_keeps_its_own_processes(tmp_path):
    result = _run_script(
        tmp_path,
        """
        model = LibreYOLO9(None, size="t", device="cpu")
        model.val(data=DATA, imgsz=64, batch=2, plots=False, workers=0)

        def child():
            print("CHILD_RAN", flush=True)

        if __name__ == "__main__":
            process = multiprocessing.Process(target=child)
            process.start()
            process.join()
            assert process.exitcode == 0
            print("SCRIPT_DONE")
        """,
    )

    assert result.returncode == 0, result.stderr[-4000:]
    assert "CHILD_RAN" in result.stdout
    assert "SCRIPT_DONE" in result.stdout


GUARDED = """\
import x

def main():
    x.train()

if __name__ == "__main__":
    main()
    x.val(
        workers=4,
    )
if "__main__" == __name__:
    x.train()
main()
if __name__ == "__main__" and ENABLED:
    x.train()
"""


@pytest.mark.parametrize(
    "line, guarded",
    [(4, False), (7, True), (8, True), (10, True), (12, True), (13, False), (1, False), (15, True)],
)
def test_main_guard_is_read_from_the_source(line, guarded):
    assert is_main_guarded(GUARDED, line) is guarded


def test_workers_zero_and_fork_are_left_alone(monkeypatch):
    monkeypatch.setattr(main_guard, "_reimports_main", lambda: False)
    assert workers_for_main_process("val()", 4) == 4
    monkeypatch.setattr(main_guard, "_reimports_main", lambda: True)
    assert workers_for_main_process("val()", 0) == 0


def test_a_call_from_outside_the_script_keeps_its_workers(monkeypatch):
    # Under pytest the call comes from the test runner, not from top-level
    # code of the __main__ script, so nothing is changed.
    monkeypatch.setattr(main_guard, "_reimports_main", lambda: True)
    monkeypatch.setattr(main_guard, "_main_line", lambda: None)
    assert workers_for_main_process("val()", 4) == 4


@dataclasses.dataclass
class _NoWorkersConfig:
    workers: int = 0


class _ResumingTrain:
    """Defaults to workers=0 and restores a saved count on resume, like RF-DETR."""

    task = None
    TRAIN_CONFIG = _NoWorkersConfig

    def _restore_resume_interpolation(self, resume):
        pass

    def _checkpoint_train_config(self, resume):
        return {}

    @staticmethod
    def _raw_train(self, data=None, resume=None, **kwargs):
        return kwargs.get("workers", 2 if resume else 0)


_ResumingTrain.train = _wrap_train_with_cfg(_ResumingTrain._raw_train)


def test_unguarded_train_passes_workers_zero_explicitly(monkeypatch, caplog):
    import libreyolo.models.base.model as base_model

    monkeypatch.setattr(main_guard, "_warned", set())
    monkeypatch.setattr(base_model, "runs_unguarded", lambda: True)
    with caplog.at_level("WARNING"):
        assert _ResumingTrain().train(data="d.yaml") == 0
        assert _ResumingTrain().train(data="d.yaml", resume="last.pt") == 0
    assert caplog.text.count("so it runs with workers=0") == 1
    monkeypatch.setattr(base_model, "runs_unguarded", lambda: False)
    assert _ResumingTrain().train(data="d.yaml", resume="last.pt") == 2


class _KwargsTrain:
    task = None
    TRAIN_CONFIG = None

    @staticmethod
    def _raw_train(self, data=None, **kwargs):
        return kwargs.get("workers"), kwargs.get("num_workers")


_KwargsTrain.train = _wrap_train_with_cfg(_KwargsTrain._raw_train)


def test_unguarded_train_zeroes_both_worker_spellings(monkeypatch):
    import libreyolo.models.base.model as base_model

    monkeypatch.setattr(main_guard, "_warned", set())
    monkeypatch.setattr(base_model, "runs_unguarded", lambda: True)
    assert _KwargsTrain().train(data="d.yaml", num_workers=4) == (0, 0)
    assert _KwargsTrain().train(data="d.yaml", workers=4) == (0, None)


def test_an_unguarded_line_drops_the_workers(monkeypatch, tmp_path, caplog):
    script = tmp_path / "script.py"
    script.write_text(GUARDED, encoding="utf-8")
    monkeypatch.setattr(main_guard, "_reimports_main", lambda: True)
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(script), raising=False)
    monkeypatch.setattr(main_guard, "_warned", set())
    monkeypatch.setattr(main_guard, "_main_line", lambda: 13)
    with caplog.at_level("WARNING"):
        assert workers_for_main_process("train()", 8) == 0
        assert workers_for_main_process("train()", None) == 0
    assert caplog.text.count("so it runs with workers=0") == 1
    monkeypatch.setattr(main_guard, "_main_line", lambda: 7)
    assert workers_for_main_process("train()", 8) == 8
