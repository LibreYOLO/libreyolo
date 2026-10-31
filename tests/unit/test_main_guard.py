"""train()/val() called outside a main guard exit workers with a clear message (#925)."""

from __future__ import annotations

import multiprocessing
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from libreyolo.backends.base import BaseBackend
from libreyolo.models.base.model import BaseModel, _wrap_train_with_cfg
from libreyolo.utils.main_guard import exit_if_reimporting_main, guard_main_reimport

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
def test_unguarded_script_names_the_fix(tmp_path, start_method):
    result = _run_script(
        tmp_path,
        """
        model = LibreYOLO9(None, size="t", device="cpu")
        model.val(data=DATA, imgsz=64, batch=2, plots=False, workers=1)
        """,
        start_method,
    )

    assert result.returncode != 0
    assert "val() was called at the top level of your script" in result.stderr
    assert "bootstrapping phase" not in result.stderr


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
