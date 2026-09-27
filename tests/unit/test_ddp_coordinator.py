"""Regression tests for the private automatic-DDP coordinator."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import textwrap
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from libreyolo.training._ddp_coordinator import (
    CLEANUP_SENTINEL_NAME,
    COORDINATOR_BOOTSTRAP_ENV,
    JOB_PROTOCOL,
    JOB_PROTOCOL_VERSION,
    MANIFEST_NAME,
    PAYLOAD_NAME,
    PAYLOAD_SIZE_WARNING_BYTES,
    STATUS_NAME,
    __file__ as coordinator_file,
    _cleanup_abandoned_job,
    _coordinator_launch,
    _load_manifest,
    _load_payload,
    _pop_coordinator_bootstrap,
    coordinator_main,
    job_workspace,
    write_job,
)
from libreyolo.training.ddp_spawn import _build_init_kw, ddp_aware
from libreyolo.training.ddp_spawn import spawn_for_model
from libreyolo.training.distributed import spawn_ddp_train
from libreyolo.training.distributed import _spawn_standard_pickle_fallback


pytestmark = pytest.mark.unit


def _recording_worker(
    rank: int,
    nprocs: int,
    master_addr: str,
    master_port: int,
    result_path: str,
    token: str,
) -> None:
    record = {
        "rank": rank,
        "world": nprocs,
        "rank_env": os.environ["RANK"],
        "local_rank_env": os.environ["LOCAL_RANK"],
        "world_env": os.environ["WORLD_SIZE"],
        "master_addr": os.environ["MASTER_ADDR"],
        "master_port": os.environ["MASTER_PORT"],
        "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cwd": os.getcwd(),
        "token": token,
        "argv": sys.argv,
        "main_file": getattr(sys.modules.get("__main__"), "__file__", None),
        "bootstrap_env": os.environ.get(COORDINATOR_BOOTSTRAP_ENV),
    }
    rank_path = Path(result_path).with_name(f"rank-{rank}.json")
    rank_path.write_text(json.dumps(record), encoding="utf-8")
    if rank == 0:
        Path(result_path).write_text(json.dumps(record), encoding="utf-8")


def _failing_worker(
    rank: int,
    nprocs: int,
    master_addr: str,
    master_port: int,
    result_path: str,
) -> None:
    if rank == 1:
        raise RuntimeError("intentional rank failure")


def _slow_worker(
    rank: int,
    nprocs: int,
    master_addr: str,
    master_port: int,
    result_path: str,
) -> None:
    import time

    Path(result_path).write_text("started", encoding="utf-8")
    time.sleep(30)


def _job_dirs() -> set[Path]:
    root = Path(tempfile.gettempdir())
    return set(root.glob("libreyolo-ddp-job-*"))


@pytest.fixture
def private_tempdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Give this test its own system temp dir, inherited by the coordinator.

    The cleanup checks compare job dirs in the temp root; with the shared root,
    jobs from other xdist workers made the before/after sets differ.
    """
    root = tmp_path / "tmp"
    root.mkdir()
    monkeypatch.setenv("TMPDIR", str(root))
    monkeypatch.setattr(tempfile, "tempdir", None)
    return root


def test_job_protocol_round_trip(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_job(
            job_dir,
            worker_fn=_recording_worker,
            spawn_args=("protocol-token",),
            nprocs=2,
            result_path=str(tmp_path / "result.json"),
            master_addr="127.0.0.1",
            master_port=12345,
            import_paths=[str(tmp_path)],
        )

    manifest = _load_manifest(job_dir)
    payload = _load_payload(job_dir, manifest)
    assert manifest["protocol"] == JOB_PROTOCOL
    assert manifest["version"] == JOB_PROTOCOL_VERSION
    assert payload["job_id"] == manifest["job_id"]
    assert payload["spawn_args"] == ("protocol-token",)
    assert manifest["import_paths"] == [str(tmp_path)]
    assert payload["caller_argv"] == sys.argv
    assert manifest["caller_argv"] == sys.argv
    assert payload["caller_main_file"] == getattr(
        sys.modules.get("__main__"), "__file__", None
    )
    assert manifest["caller_main_file"] == payload["caller_main_file"]
    assert (job_dir / PAYLOAD_NAME).stat().st_size < PAYLOAD_SIZE_WARNING_BYTES
    assert caught == []


@pytest.mark.parametrize("main_file", [Path("program.py"), b"program.py"])
def test_job_identity_normalizes_pathlike_and_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    main_file,
) -> None:
    main_module = sys.modules["__main__"]
    monkeypatch.setattr(sys, "argv", [Path("program.py"), b"--bytes-flag"])
    monkeypatch.setattr(main_module, "__file__", main_file, raising=False)
    job_dir = tmp_path / "job"

    write_job(
        job_dir,
        worker_fn=_recording_worker,
        spawn_args=("identity-normalization",),
        nprocs=1,
        result_path=str(tmp_path / "result.json"),
        master_addr="127.0.0.1",
        master_port=12345,
        import_paths=[str(tmp_path)],
    )

    manifest = _load_manifest(job_dir)
    payload = _load_payload(job_dir, manifest)
    expected_argv = ["program.py", "--bytes-flag"]
    expected_main_file = "program.py"
    assert manifest["caller_argv"] == expected_argv
    assert payload["caller_argv"] == expected_argv
    assert manifest["caller_main_file"] == expected_main_file
    assert payload["caller_main_file"] == expected_main_file


def test_successful_oversized_payload_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "libreyolo.training._ddp_coordinator.PAYLOAD_SIZE_WARNING_BYTES", 1
    )
    job_dir = tmp_path / "job"

    with pytest.warns(RuntimeWarning, match="automatic DDP payload is"):
        write_job(
            job_dir,
            worker_fn=_recording_worker,
            spawn_args=("large-enough-for-test",),
            nprocs=1,
            result_path=str(tmp_path / "result.json"),
            master_addr="127.0.0.1",
            master_port=12345,
            import_paths=[str(tmp_path)],
        )

    assert (job_dir / PAYLOAD_NAME).is_file()


def test_coordinator_bootstrap_stays_off_command_line_and_is_consumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    command, coordinator_env = _coordinator_launch(
        tmp_path / "job",
        env={"PRESERVED": "value"},
        parent_port=23456,
        parent_token="private-parent-token",
        cleanup_token="private-cleanup-token",
    )

    assert command == [
        sys.executable,
        str(Path(coordinator_file).resolve()),
        str(tmp_path / "job"),
    ]
    rendered_command = " ".join(command)
    assert "23456" not in rendered_command
    assert "private-parent-token" not in rendered_command
    assert "private-cleanup-token" not in rendered_command
    assert coordinator_env["PRESERVED"] == "value"

    monkeypatch.setenv(
        COORDINATOR_BOOTSTRAP_ENV,
        coordinator_env[COORDINATOR_BOOTSTRAP_ENV],
    )
    bootstrap = _pop_coordinator_bootstrap()
    assert bootstrap["parent_port"] == 23456
    assert bootstrap["parent_token"] == "private-parent-token"
    assert bootstrap["cleanup_token"] == "private-cleanup-token"
    assert COORDINATOR_BOOTSTRAP_ENV not in os.environ


def test_job_protocol_rejects_future_version(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / MANIFEST_NAME).write_text(
        json.dumps(
            {
                "protocol": JOB_PROTOCOL,
                "version": JOB_PROTOCOL_VERSION + 1,
                "job_id": "future",
                "nprocs": 1,
            }
        ),
        encoding="utf-8",
    )

    assert coordinator_main(job_dir) == 1
    status = json.loads((job_dir / STATUS_NAME).read_text(encoding="utf-8"))
    assert status["outcome"] == "error"
    assert "Unsupported DDP job manifest version" in status["message"]


def test_coordinator_propagates_env_result_mask_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, private_tempdir: Path
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    before = _job_dirs()
    result_path = tmp_path / "result.json"

    spawn_ddp_train(
        _recording_worker,
        spawn_args=("round-trip",),
        nprocs=2,
        result_path=str(result_path),
        devices=[0, 1],
    )

    assert _job_dirs() == before
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2,3"
    for rank in range(2):
        record = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert record["rank"] == rank
        assert record["rank_env"] == str(rank)
        assert record["local_rank_env"] == str(rank)
        assert record["world"] == 2
        assert record["world_env"] == "2"
        assert record["cvd"] == "2,3"
        assert record["token"] == "round-trip"
        assert record["bootstrap_env"] is None
        assert all("libreyolo-ddp-job-" not in arg for arg in record["argv"])
    assert json.loads(result_path.read_text())["rank"] == 0


def test_coordinator_surfaces_worker_traceback_and_cleans_up(
    tmp_path: Path, private_tempdir: Path
) -> None:
    before = _job_dirs()
    with pytest.raises(RuntimeError, match="intentional rank failure"):
        spawn_ddp_train(
            _failing_worker,
            spawn_args=(),
            nprocs=2,
            result_path=str(tmp_path / "unused.json"),
        )
    assert _job_dirs() == before


def test_coordinator_keeps_standard_pickle_rejection(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="standard pickle"):
        spawn_ddp_train(
            lambda *args: None,
            spawn_args=(),
            nprocs=1,
            result_path=str(tmp_path / "unused.json"),
        )


def test_coordinator_uses_selected_package_when_cwd_contains_shadow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shadow_root = tmp_path / "shadow"
    shadow_package = shadow_root / "libreyolo"
    shadow_package.mkdir(parents=True)
    shadow_marker = tmp_path / "shadow-imported.txt"
    (shadow_package / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(shadow_marker)!r}).write_text('bad')\n"
        "raise RuntimeError('shadow LibreYOLO was imported')\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(shadow_root)
    result_path = tmp_path / "result.json"

    spawn_ddp_train(
        _recording_worker,
        spawn_args=("shadow-check",),
        nprocs=1,
        result_path=str(result_path),
    )

    assert not shadow_marker.exists()
    assert json.loads(result_path.read_text(encoding="utf-8"))["cwd"] == str(
        shadow_root
    )


def test_pathlike_sys_path_entry_is_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "path", [Path("."), *sys.path])
    result_path = tmp_path / "result.json"

    spawn_ddp_train(
        _recording_worker,
        spawn_args=("pathlike",),
        nprocs=1,
        result_path=str(result_path),
    )

    assert json.loads(result_path.read_text(encoding="utf-8"))["token"] == "pathlike"


def test_parent_liveness_socket_stops_workers_and_cleans_job(tmp_path: Path) -> None:
    import time

    with job_workspace() as (job_dir, cleanup_token):
        temp_root = job_dir.parent
        started_path = tmp_path / "worker-started.txt"
        import_paths = []
        for entry in sys.path:
            try:
                import_paths.append(
                    os.path.abspath(os.fsdecode(os.fspath(entry)) or os.getcwd())
                )
            except (OSError, TypeError, ValueError):
                continue
        write_job(
            job_dir,
            worker_fn=_slow_worker,
            spawn_args=(),
            nprocs=1,
            result_path=str(started_path),
            master_addr="127.0.0.1",
            master_port=12345,
            import_paths=list(dict.fromkeys(import_paths)),
        )
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(15)
        parent_token = "parent-liveness-test-token"
        command, coordinator_env = _coordinator_launch(
            job_dir,
            env=os.environ.copy(),
            parent_port=listener.getsockname()[1],
            parent_token=parent_token,
            cleanup_token=cleanup_token,
        )
        process = subprocess.Popen(
            command,
            env=coordinator_env,
            stdin=subprocess.DEVNULL,
        )
        parent_connection, _ = listener.accept()
        assert parent_connection.recv(len(parent_token)).decode("ascii") == parent_token
        listener.close()
        deadline = time.monotonic() + 20
        while not started_path.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                process.kill()
                pytest.fail("coordinator worker did not start")
            time.sleep(0.1)

        parent_connection.close()
        assert process.wait(timeout=15) == 1
        assert not temp_root.exists()


def test_abandoned_cleanup_preserves_caller_chosen_directory(tmp_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="libreyolo-ddp-job-") as root:
        temp_root = Path(root).resolve()
        job_dir = temp_root / "job"
        job_dir.mkdir()
        valuable = temp_root / "valuable-sibling.txt"
        valuable.write_text("preserve me", encoding="utf-8")
        manifest = {"job_id": "forged-job-id"}
        (temp_root / CLEANUP_SENTINEL_NAME).write_text(
            json.dumps(
                {
                    "protocol": JOB_PROTOCOL,
                    "version": JOB_PROTOCOL_VERSION,
                    "job_id": manifest["job_id"],
                    "job_dir": str(job_dir.resolve()),
                    "workspace": str(temp_root),
                    "token_sha256": "forged-token-hash",
                }
            ),
            encoding="utf-8",
        )

        _cleanup_abandoned_job(job_dir, manifest, "real-launcher-token")

        assert valuable.read_text(encoding="utf-8") == "preserve me"
        assert job_dir.is_dir()


class _DispatchProbe:
    @ddp_aware()
    def train(self, *, device="auto", batch=4):
        return {"device": device, "batch": batch}


class _BatchSizeDispatchProbe:
    @ddp_aware(batch_key="batch_size")
    def train(self, *, device="auto", batch_size=4, resume=False):
        raise AssertionError("multi-device call did not dispatch")


def test_normal_model_class_remains_imported_by_name() -> None:
    init_kw = _build_init_kw(_DispatchProbe())

    assert init_kw["_module"] == __name__
    assert init_kw["_class"] == "_DispatchProbe"
    assert "_class_object" not in init_kw


@pytest.mark.parametrize("device", [None, "auto", "cpu", "mps", 0, "0", [0]])
def test_single_device_forms_do_not_launch(
    device, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_launch(*args, **kwargs):
        raise AssertionError("single-device call reached automatic DDP")

    monkeypatch.setattr("libreyolo.training.ddp_spawn.spawn_for_model", fail_launch)
    assert _DispatchProbe().train(device=device, batch=7) == {
        "device": device,
        "batch": 7,
    }


def test_existing_torchrun_env_does_not_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "0")

    def fail_launch(*args, **kwargs):
        raise AssertionError("explicit torchrun call reached automatic DDP")

    monkeypatch.setattr("libreyolo.training.ddp_spawn.spawn_for_model", fail_launch)
    assert _DispatchProbe().train(device="0,1")["device"] == "0,1"


@pytest.mark.parametrize("device", [[0, 1], "0,1"])
def test_multi_device_forms_forward_batch_key_and_resume(
    device, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = {"dispatched": True}
    launch = MagicMock(return_value=expected)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("libreyolo.training.ddp_spawn.spawn_for_model", launch)

    result = _BatchSizeDispatchProbe().train(
        device=device,
        batch_size=-1,
        resume=True,
    )

    assert result is expected
    call = launch.call_args
    assert call.args[1]["device"] == device
    assert call.args[1]["batch_size"] == -1
    assert call.args[1]["resume"] is True
    assert call.args[2] == 2
    assert call.kwargs == {"devices": [0, 1], "batch_key": "batch_size"}


def _wrapper(model_path=None):
    instance = MagicMock()
    instance.model = nn.Linear(4, 2)
    instance.model_path = model_path
    return instance


def test_spawn_for_model_preserves_resume_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "resume.pt"
    checkpoint.write_bytes(b"checkpoint placeholder")
    instance = _wrapper(str(checkpoint))
    captured = {}

    def fake_spawn(worker_fn, spawn_args, nprocs, result_path, **kwargs):
        captured["spawn_args"] = spawn_args
        Path(result_path).write_text("{}", encoding="utf-8")

    with (
        patch("torch.save") as save,
        patch("libreyolo.training.distributed.spawn_ddp_train", side_effect=fake_spawn),
        patch("libreyolo.training.ddp_spawn._build_init_kw", return_value={}),
    ):
        spawn_for_model(
            instance,
            train_kw={"resume": True, "batch": 8},
            nprocs=2,
            devices=[0, 1],
        )

    save.assert_not_called()
    assert captured["spawn_args"][0] == str(checkpoint)
    assert captured["spawn_args"][2]["resume"] is True
    assert captured["spawn_args"][2]["batch"] == 8


def test_spawn_for_model_resolves_batch_size_autobatch_before_launch() -> None:
    instance = _wrapper()
    captured = {}

    def fake_spawn(worker_fn, spawn_args, nprocs, result_path, **kwargs):
        captured["train_kw"] = spawn_args[2]
        Path(result_path).write_text("{}", encoding="utf-8")

    with (
        patch("torch.save"),
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda.empty_cache"),
        patch(
            "libreyolo.training.autobatch.resolve_auto_batch", return_value=12
        ) as resolve,
        patch("libreyolo.training.distributed.spawn_ddp_train", side_effect=fake_spawn),
        patch("libreyolo.training.ddp_spawn._build_init_kw", return_value={}),
    ):
        spawn_for_model(
            instance,
            train_kw={"batch_size": -1, "imgsz": 320, "amp": False},
            nprocs=2,
            devices=[0, 1],
            batch_key="batch_size",
        )

    assert captured["train_kw"]["batch_size"] == 12
    assert resolve.call_args.kwargs["world_size"] == 2
    assert resolve.call_args.kwargs["imgsz"] == 320


def test_spawn_for_model_returns_result_and_reloads_parent_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "last.pt"
    checkpoint.write_bytes(b"checkpoint placeholder")
    instance = _wrapper()

    def fake_spawn(worker_fn, spawn_args, nprocs, result_path, **kwargs):
        Path(result_path).write_text(
            json.dumps({"last_checkpoint": str(checkpoint), "metric": 0.5}),
            encoding="utf-8",
        )

    with (
        patch("torch.save"),
        patch("torch.cuda.is_available", return_value=False),
        patch("libreyolo.training.distributed.spawn_ddp_train", side_effect=fake_spawn),
        patch("libreyolo.training.ddp_spawn._build_init_kw", return_value={}),
    ):
        result = spawn_for_model(
            instance,
            train_kw={"batch": 8},
            nprocs=2,
            devices=[0, 1],
        )

    assert result["metric"] == 0.5
    assert instance.model_path == str(checkpoint)
    instance._load_weights.assert_called_once_with(str(checkpoint))
    assert instance.device == torch.device("cpu")
    assert instance.model.training is False


def _run_model_script(tmp_path: Path, *, guarded: bool) -> dict:
    helper = tmp_path / "ddp_script_support.py"
    helper.write_text(
        textwrap.dedent(
            """
            import json
            import os
            from pathlib import Path

            from libreyolo.training.ddp_spawn import ddp_aware

            def script_worker(rank, nprocs, master_addr, master_port, result_path, train_kw):
                callback = train_kw["callbacks"]
                if rank == 0:
                    callback({"rank": rank, "world": nprocs})
                    Path(result_path).write_text(json.dumps({
                        "rank": rank,
                        "world": nprocs,
                        "batch": train_kw["batch"],
                        "logger": train_kw["loggers"],
                    }))

            class ScriptModel:
                @ddp_aware()
                def train(self, *, device="auto", batch=4, callbacks=None, loggers=None):
                    raise AssertionError("multi-device call did not dispatch")
            """
        ),
        encoding="utf-8",
    )

    launch_path = tmp_path / (
        "guarded-launches.txt" if guarded else "unguarded-launches.txt"
    )
    callback_path = tmp_path / (
        "guarded-callback.json" if guarded else "unguarded-callback.json"
    )
    output_path = tmp_path / (
        "guarded-result.json" if guarded else "unguarded-result.json"
    )
    script = tmp_path / ("guarded.py" if guarded else "unguarded.py")
    script_body = textwrap.dedent(
        f"""
            import json
            from pathlib import Path
            from unittest.mock import patch

            import torch
            from ddp_script_support import ScriptModel, script_worker
            from libreyolo.training.distributed import spawn_ddp_train

            LAUNCH_PATH = Path({str(launch_path)!r})
            CALLBACK_PATH = Path({str(callback_path)!r})
            OUTPUT_PATH = Path({str(output_path)!r})
            with LAUNCH_PATH.open("a", encoding="utf-8") as handle:
                handle.write("launch\\n")

            def module_callback(event):
                CALLBACK_PATH.write_text(json.dumps(event), encoding="utf-8")

            def launch(model, train_kw, nprocs, *, devices, batch_key):
                result_path = OUTPUT_PATH.with_name("worker-result.json")
                spawn_ddp_train(
                    script_worker,
                    spawn_args=(train_kw,),
                    nprocs=nprocs,
                    result_path=str(result_path),
                    devices=devices,
                )
                return json.loads(result_path.read_text(encoding="utf-8"))

            def run():
                with patch("torch.cuda.is_available", return_value=True), patch(
                    "libreyolo.training.ddp_spawn.spawn_for_model", side_effect=launch
                ):
                    result = ScriptModel().train(
                        device="0,1", batch=8, callbacks=module_callback, loggers="mlflow"
                    )
                OUTPUT_PATH.write_text(json.dumps(result), encoding="utf-8")
            """
    )
    script_body += (
        "\nif __name__ == '__main__':\n    run()\n" if guarded else "\nrun()\n"
    )
    script.write_text(script_body, encoding="utf-8")

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), str(tmp_path), env.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert launch_path.read_text(encoding="utf-8").splitlines() == ["launch"]
    assert json.loads(callback_path.read_text(encoding="utf-8")) == {
        "rank": 0,
        "world": 2,
    }
    return json.loads(output_path.read_text(encoding="utf-8"))


def test_unguarded_top_level_model_train_matches_guarded_script(tmp_path: Path) -> None:
    unguarded = _run_model_script(tmp_path, guarded=False)
    guarded = _run_model_script(tmp_path, guarded=True)
    assert (
        unguarded
        == guarded
        == {
            "rank": 0,
            "world": 2,
            "batch": 8,
            "logger": "mlflow",
        }
    )


def test_guarded_main_model_class_and_program_identity_reach_model_worker(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "guarded-main-result.json"
    side_effect_path = tmp_path / "guarded-main-side-effect.txt"
    script = tmp_path / "guarded_main_model.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import json
            import os
            import sys
            from pathlib import Path

            import torch
            import torch.nn as nn

            from libreyolo.training._ddp_coordinator import COORDINATOR_BOOTSTRAP_ENV
            from libreyolo.training.ddp_spawn import spawn_for_model

            OUTPUT_PATH = Path({str(output_path)!r})
            SIDE_EFFECT_PATH = Path({str(side_effect_path)!r})
            with SIDE_EFFECT_PATH.open("a", encoding="utf-8") as marker:
                marker.write("parent-only\\n")

            class GuardedMainModel:
                def __init__(self, model_path=None, device="auto"):
                    self.model = nn.Linear(2, 1)
                    self.model_path = model_path
                    self.device = torch.device("cpu")
                    if model_path:
                        state = torch.load(
                            model_path, map_location="cpu", weights_only=True
                        )
                        self.model.load_state_dict(state)

                def train(self, *, device="auto"):
                    import __main__

                    return {{
                        "rank": int(os.environ["LOCAL_RANK"]),
                        "class_name": type(self).__name__,
                        "class_module": type(self).__module__,
                        "argv": list(sys.argv),
                        "main_file": getattr(__main__, "__file__", None),
                        "bootstrap_env": os.environ.get(
                            COORDINATOR_BOOTSTRAP_ENV
                        ),
                    }}

            def main():
                result = spawn_for_model(
                    GuardedMainModel(),
                    train_kw={{}},
                    nprocs=2,
                )
                OUTPUT_PATH.write_text(json.dumps(result), encoding="utf-8")

            if __name__ == "__main__":
                main()
            """
        ),
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), str(tmp_path), env.get("PYTHONPATH", "")]
    )
    original_args = [str(script), "--user-flag", "value with spaces"]

    completed = subprocess.run(
        [sys.executable, *original_args],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result == {
        "rank": 0,
        "class_name": "GuardedMainModel",
        "class_module": "__main__",
        "argv": original_args,
        "main_file": str(script),
        "bootstrap_env": None,
    }
    assert side_effect_path.read_text(encoding="utf-8").splitlines() == ["parent-only"]
    assert all("libreyolo-ddp-job-" not in arg for arg in result["argv"])


def test_guarded_standard_pickle_callback_uses_compatibility_launcher(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "ddp_callback_support.py"
    helper.write_text(
        textwrap.dedent(
            """
            import json
            import os
            from pathlib import Path

            def callback_worker(
                rank, nprocs, master_addr, master_port, result_path, callback
            ):
                if rank == 0:
                    callback(f"rank={rank}")
                    Path(result_path).write_text(json.dumps({
                        "rank": rank,
                        "world": nprocs,
                        "cvd": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    }), encoding="utf-8")
            """
        ),
        encoding="utf-8",
    )
    callback_path = tmp_path / "callback.txt"
    import_mask_dir = tmp_path / "import-masks"
    result_path = tmp_path / "fallback-result.json"
    script = tmp_path / "guarded_fallback.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import multiprocessing
            import os
            from pathlib import Path

            from ddp_callback_support import callback_worker
            from libreyolo.training.distributed import spawn_ddp_train

            if multiprocessing.current_process().name != "MainProcess":
                mask_dir = Path({str(import_mask_dir)!r})
                mask_dir.mkdir(exist_ok=True)
                (mask_dir / f"{{os.getpid()}}.txt").write_text(
                    os.environ.get("CUDA_VISIBLE_DEVICES", "<missing>"),
                    encoding="utf-8",
                )

            CALLBACK_HANDLE = Path({str(callback_path)!r}).open(
                "a", encoding="utf-8"
            )

            def module_callback(message):
                CALLBACK_HANDLE.write(message)
                CALLBACK_HANDLE.flush()

            def main():
                os.environ["CUDA_VISIBLE_DEVICES"] = "4,7"
                spawn_ddp_train(
                    callback_worker,
                    spawn_args=(module_callback,),
                    nprocs=2,
                    result_path={str(result_path)!r},
                    devices=[1, 0],
                )
                CALLBACK_HANDLE.close()
                assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,7"

            if __name__ == "__main__":
                main()
            """
        ),
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), str(tmp_path), env.get("PYTHONPATH", "")]
    )

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, (
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    assert "guarded-script compatibility launcher" in completed.stderr
    assert callback_path.read_text(encoding="utf-8") == "rank=0"
    assert sorted(
        path.read_text(encoding="utf-8") for path in import_mask_dir.iterdir()
    ) == ["7,4", "7,4"]
    assert json.loads(result_path.read_text(encoding="utf-8")) == {
        "rank": 0,
        "world": 2,
        "cvd": "7,4",
    }


def test_compatibility_launcher_restores_parent_mask_after_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,7")
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = "7,4"
    observed = []

    def failing_spawn(*args, **kwargs):
        observed.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
        raise RuntimeError("intentional compatibility launch failure")

    monkeypatch.setattr("torch.multiprocessing.spawn", failing_spawn)

    with (
        pytest.warns(RuntimeWarning, match="compatibility launcher"),
        pytest.raises(RuntimeError, match="intentional compatibility launch failure"),
    ):
        _spawn_standard_pickle_fallback(
            _recording_worker,
            ("unused",),
            2,
            str(tmp_path / "unused.json"),
            "127.0.0.1",
            12345,
            child_env,
        )

    assert observed == ["7,4"]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,7"


def test_unguarded_compatibility_fallback_stops_without_recursing(
    tmp_path: Path,
) -> None:
    helper = tmp_path / "ddp_fallback_support.py"
    helper.write_text(
        "def callback_worker(rank, world, host, port, result_path, callback):\n"
        "    callback(str(rank))\n",
        encoding="utf-8",
    )
    launches = tmp_path / "launches.txt"
    callback_path = tmp_path / "callback.txt"
    script = tmp_path / "unguarded_fallback.py"
    script.write_text(
        textwrap.dedent(
            f"""
            from pathlib import Path

            from ddp_fallback_support import callback_worker
            from libreyolo.training.distributed import spawn_ddp_train

            with Path({str(launches)!r}).open("a", encoding="utf-8") as marker:
                marker.write("launch\\n")
            CALLBACK_HANDLE = Path({str(callback_path)!r}).open(
                "w", encoding="utf-8"
            )

            def module_callback(message):
                CALLBACK_HANDLE.write(message)
                CALLBACK_HANDLE.flush()

            spawn_ddp_train(
                callback_worker,
                spawn_args=(module_callback,),
                nprocs=1,
                result_path={str(tmp_path / "unused.json")!r},
            )
            """
        ),
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), str(tmp_path), env.get("PYTHONPATH", "")]
    )

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )

    assert completed.returncode != 0
    assert launches.read_text(encoding="utf-8").splitlines() == [
        "launch",
        "launch",
    ]
    assert "from a spawned subprocess" in completed.stderr
    assert "if __name__ == '__main__'" in completed.stderr


def test_rank_children_import_the_coordinator_not_the_user_script(tmp_path, monkeypatch):
    """Guardless DDP hung: DataLoader workers a rank starts with spawn ran the
    user's unguarded script as ``__mp_main__`` (from the restored ``__file__``)
    and re-entered train(). Spawn must set up main from the coordinator."""
    import types
    from multiprocessing import spawn

    from libreyolo.training import _ddp_coordinator as coordinator

    user_script = tmp_path / "train_unguarded.py"
    user_script.write_text("raise SystemExit('user script must not run')\n")
    fake_main = types.ModuleType("__mp_main__")
    fake_main.__file__ = "/path/to/_ddp_coordinator.py"
    fake_main.__spec__ = None
    monkeypatch.setitem(sys.modules, "__main__", fake_main)
    monkeypatch.setitem(sys.modules, "__mp_main__", fake_main)
    monkeypatch.setattr(sys, "argv", list(sys.argv))

    coordinator._restore_caller_identity(
        {"caller_argv": [str(user_script)], "caller_main_file": str(user_script)}
    )

    assert fake_main.__file__ == str(user_script)
    prep = spawn.get_preparation_data("worker")
    assert prep.get("init_main_from_name") == "libreyolo.training._ddp_coordinator"
    assert "init_main_from_path" not in prep
