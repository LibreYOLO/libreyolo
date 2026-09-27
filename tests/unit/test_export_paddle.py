"""Unit tests for Paddle export and runtime integration."""

from __future__ import annotations

import multiprocessing
import queue
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from libreyolo.export.exporter import PaddleExporter

pytestmark = pytest.mark.unit
_PROCESS_SYNC_TIMEOUT = 30
# Window used to assert the second process stays blocked. Only meaningful
# after both children have signalled readiness, so it measures the lock and
# not interpreter startup.
_LOCK_CONTENTION_WINDOW = 1.0


def _wrapper(family: str = "yolo9", task: str = "detect") -> MagicMock:
    model = MagicMock()
    model._get_model_name.return_value = family
    model.task = task
    return model


def _hold_paddle_install_lock(output_path, ready, entered, release):
    from libreyolo.export.paddle import _paddle_install_lock

    # Announce readiness *before* contending. A spawned child has to boot a new
    # interpreter and import libreyolo first, which costs far more than the
    # window the mutual-exclusion assertion uses. Signalling here lets the test
    # pay that cost explicitly, so the window afterwards measures the lock
    # rather than process startup.
    ready.put(True)
    with _paddle_install_lock(Path(output_path)):
        entered.put(True)
        release.wait(timeout=_PROCESS_SYNC_TIMEOUT)


def test_paddle_exporter_rejects_dynamic_batch_and_unsimplified_graph():
    exporter = PaddleExporter(_wrapper())
    with pytest.raises(ValueError, match="dynamic=False"):
        exporter(dynamic=True)
    with pytest.raises(ValueError, match="batch=1"):
        exporter(batch=2)
    with pytest.raises(ValueError, match="simplify=True"):
        exporter(simplify=False)
    with pytest.raises(ValueError, match="opset=15"):
        exporter(opset=14)


def test_rfdetr_block_happens_before_dependency_check(monkeypatch):
    from libreyolo.export import paddle as paddle_export

    dependency_check = MagicMock(side_effect=AssertionError("must not run"))
    monkeypatch.setattr(
        paddle_export, "check_paddle_export_available", dependency_check
    )
    exporter = PaddleExporter(_wrapper("rfdetr"))
    with pytest.raises(NotImplementedError, match="GridSample"):
        exporter._preflight(half=False, int8=False, data=None)
    dependency_check.assert_not_called()


def test_export_paddle_keeps_only_runtime_artifacts_and_metadata(monkeypatch, tmp_path):
    from libreyolo.export import paddle as paddle_export

    calls = {}

    def fake_convert(model_path, save_dir, **kwargs):
        calls.update(model_path=model_path, save_dir=save_dir, kwargs=kwargs)
        inference = Path(save_dir) / "inference_model"
        inference.mkdir(parents=True)
        (inference / "model.pdmodel").write_bytes(b"model")
        (inference / "model.pdiparams").write_bytes(b"parameters")
        (inference / "model.pdiparams.info").write_bytes(b"info")
        (Path(save_dir) / "x2paddle_code.py").write_text("generated")

    package = types.ModuleType("x2paddle")
    convert = types.ModuleType("x2paddle.convert")
    convert.onnx2paddle = fake_convert
    package.convert = convert
    monkeypatch.setitem(sys.modules, "x2paddle", package)
    monkeypatch.setitem(sys.modules, "x2paddle.convert", convert)
    monkeypatch.setattr(paddle_export, "check_paddle_export_available", lambda: None)
    monkeypatch.setattr(
        paddle_export, "_normalize_onnx_for_x2paddle", lambda path: None
    )

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    output = tmp_path / "model_paddle"
    result = paddle_export.export_paddle(
        str(onnx_path),
        str(output),
        metadata={"model_family": "yolo9", "task": "detect"},
    )

    assert result == str(output)
    assert {path.name for path in output.iterdir()} == {
        "metadata.yaml",
        "model.pdiparams",
        "model.pdiparams.info",
        "model.pdmodel",
    }
    metadata = yaml.safe_load((output / "metadata.yaml").read_text())
    assert metadata == {"model_family": "yolo9", "task": "detect"}
    assert calls["kwargs"] == {
        "enable_optim": False,
        "disable_feedback": True,
        "enable_onnx_checker": True,
    }


def test_export_paddle_restores_previous_artifact_when_install_fails(
    monkeypatch, tmp_path
):
    from libreyolo.export import paddle as paddle_export

    def fake_convert(model_path, save_dir, **kwargs):
        inference = Path(save_dir) / "inference_model"
        inference.mkdir(parents=True)
        (inference / "model.pdmodel").write_bytes(b"new-model")
        (inference / "model.pdiparams").write_bytes(b"new-parameters")

    package = types.ModuleType("x2paddle")
    convert = types.ModuleType("x2paddle.convert")
    convert.onnx2paddle = fake_convert
    package.convert = convert
    monkeypatch.setitem(sys.modules, "x2paddle", package)
    monkeypatch.setitem(sys.modules, "x2paddle.convert", convert)
    monkeypatch.setattr(paddle_export, "check_paddle_export_available", lambda: None)
    monkeypatch.setattr(
        paddle_export, "_normalize_onnx_for_x2paddle", lambda path: None
    )

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"onnx")
    output = tmp_path / "model_paddle"
    output.mkdir()
    (output / "model.pdmodel").write_bytes(b"old-model")
    real_replace = paddle_export.os.replace

    def fail_new_install(source, destination):
        if Path(source).name == "artifact":
            raise OSError("simulated install failure")
        real_replace(source, destination)

    monkeypatch.setattr(paddle_export.os, "replace", fail_new_install)
    with pytest.raises(OSError, match="simulated install failure"):
        paddle_export.export_paddle(str(onnx_path), str(output))

    assert (output / "model.pdmodel").read_bytes() == b"old-model"
    assert not (tmp_path / ".model_paddle.previous").exists()


def test_paddle_install_lock_serializes_processes(tmp_path):
    """A second process must not enter the install lock while the first holds it.

    Two things this test has to avoid, both of which it got wrong before:

    * The mutual-exclusion assertion is a *negative* one ("the second process
      does not signal"), so it only means something once that process is
      actually ready to contend. A spawned child spends over a second booting
      an interpreter and importing libreyolo, which is longer than the window
      itself, so without a readiness handshake the assertion held whatever the
      lock did -- it passed even against a no-op lock.
    * Only started processes may be joined. Building the list as they start
      keeps a timeout here from being replaced by "can only join a started
      process" from the cleanup path, which hid the real cause.
    """
    ctx = multiprocessing.get_context("spawn")
    ready = ctx.Queue()
    entered = ctx.Queue()
    release_first = ctx.Event()
    release_second = ctx.Event()
    output = tmp_path / "model_paddle"
    started = []

    def spawn(release):
        process = ctx.Process(
            target=_hold_paddle_install_lock,
            args=(str(output), ready, entered, release),
        )
        process.start()
        started.append(process)
        # Wait out interpreter startup and imports before relying on any timing.
        ready.get(timeout=_PROCESS_SYNC_TIMEOUT)
        return process

    try:
        first = spawn(release_first)
        entered.get(timeout=_PROCESS_SYNC_TIMEOUT)

        second = spawn(release_second)
        # The second child is past its imports and about to take the lock, so
        # this window now measures the lock. Against a no-op lock the child
        # signals within milliseconds and this assertion fails, which is the
        # point.
        with pytest.raises(queue.Empty):
            entered.get(timeout=_LOCK_CONTENTION_WINDOW)

        release_first.set()
        entered.get(timeout=_PROCESS_SYNC_TIMEOUT)
        release_second.set()
        first.join(timeout=_PROCESS_SYNC_TIMEOUT)
        second.join(timeout=_PROCESS_SYNC_TIMEOUT)
        assert first.exitcode == 0
        assert second.exitcode == 0
    finally:
        release_first.set()
        release_second.set()
        for process in started:
            if process.is_alive():
                process.terminate()
            process.join(timeout=_PROCESS_SYNC_TIMEOUT)


def test_x2paddle_onnx_normalization_removes_only_default_dilation(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from libreyolo.export.paddle import _normalize_onnx_for_x2paddle

    nodes = [
        helper.make_node(
            "MaxPool", ["input"], ["middle"], kernel_shape=[3, 3], dilations=[1, 1]
        ),
        helper.make_node(
            "MaxPool", ["middle"], ["output"], kernel_shape=[3, 3], dilations=[2, 1]
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "maxpool",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 16, 16])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 10, 12])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=7
    )
    path = tmp_path / "model.onnx"
    onnx.save(model, path)

    _normalize_onnx_for_x2paddle(path)

    normalized = onnx.load(path)
    first_names = {attribute.name for attribute in normalized.graph.node[0].attribute}
    second = {
        attribute.name: list(attribute.ints)
        for attribute in normalized.graph.node[1].attribute
    }
    assert "dilations" not in first_names
    assert second["dilations"] == [2, 1]


def test_x2paddle_onnx_normalization_rewrites_static_sizes_resize(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    from libreyolo.export.paddle import _normalize_onnx_for_x2paddle

    sizes = numpy_helper.from_array(
        np.asarray([1, 3, 8, 8], dtype=np.int64), name="sizes"
    )
    resize = helper.make_node(
        "Resize",
        ["input", "", "", "sizes"],
        ["output"],
        mode="linear",
        coordinate_transformation_mode="half_pixel",
    )
    graph = helper.make_graph(
        [resize],
        "resize",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])],
        initializer=[sizes],
        value_info=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 8, 8])
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 15)], ir_version=8
    )
    path = tmp_path / "resize.onnx"
    onnx.save(model, path)

    _normalize_onnx_for_x2paddle(path)

    normalized = onnx.load(path)
    node = normalized.graph.node[0]
    assert list(node.input[:2]) == ["input", ""]
    assert len(node.input) == 3
    rewritten = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in normalized.graph.initializer
    }
    np.testing.assert_array_equal(rewritten[node.input[2]], [1.0, 1.0, 2.0, 2.0])
    attributes = {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }
    assert attributes["coordinate_transformation_mode"] == b"pytorch_half_pixel"
    reshape = normalized.graph.node[1]
    assert reshape.op_type == "Reshape"
    assert list(reshape.input) == [node.output[0], "sizes"]
    assert reshape.output[0] == "output"


def test_x2paddle_onnx_normalization_decomposes_clip(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    from libreyolo.export.paddle import _normalize_onnx_for_x2paddle

    lower = numpy_helper.from_array(np.asarray(0, dtype=np.int64), name="lower")
    upper = numpy_helper.from_array(np.asarray(31, dtype=np.int64), name="upper")
    graph = helper.make_graph(
        [helper.make_node("Clip", ["input", "lower", "upper"], ["output"])],
        "clip",
        [helper.make_tensor_value_info("input", TensorProto.INT64, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.INT64, [1, 4])],
        initializer=[lower, upper],
    )
    path = tmp_path / "clip.onnx"
    onnx.save(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 15)], ir_version=8
        ),
        path,
    )

    _normalize_onnx_for_x2paddle(path)

    normalized = onnx.load(path)
    assert [node.op_type for node in normalized.graph.node] == ["Max", "Min"]
    assert normalized.graph.node[0].input[:] == ["input", "lower"]
    assert normalized.graph.node[1].output[0] == "output"


def test_x2paddle_onnx_normalization_resolves_negative_gather_indices(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    from libreyolo.export.paddle import _normalize_onnx_for_x2paddle

    indices = numpy_helper.from_array(
        np.asarray([-1, 0], dtype=np.int64), name="indices"
    )
    graph = helper.make_graph(
        [helper.make_node("Gather", ["input", "indices"], ["output"], axis=1)],
        "gather",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 2])],
        initializer=[indices],
    )
    path = tmp_path / "gather.onnx"
    onnx.save(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 15)], ir_version=8
        ),
        path,
    )

    _normalize_onnx_for_x2paddle(path)

    normalized = onnx.load(path)
    node = normalized.graph.node[0]
    rewritten = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in normalized.graph.initializer
    }
    np.testing.assert_array_equal(rewritten[node.input[1]], [3, 0])


@pytest.mark.parametrize(
    ("equation", "inputs"),
    (
        ("bchw,bnc->bnhw", ("features", "queries")),
        ("bqc,bchw->bqhw", ("queries", "features")),
    ),
)
def test_x2paddle_onnx_normalization_lowers_mask_projection_einsum(
    tmp_path, equation, inputs
):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from libreyolo.export.paddle import _normalize_onnx_for_x2paddle

    graph = helper.make_graph(
        [
            helper.make_node(
                "Einsum",
                inputs,
                ["masks"],
                equation=equation,
            )
        ],
        "einsum",
        [
            helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 8, 4, 4]),
            helper.make_tensor_value_info("queries", TensorProto.FLOAT, [1, 3, 8]),
        ],
        [helper.make_tensor_value_info("masks", TensorProto.FLOAT, [1, 3, 4, 4])],
    )
    path = tmp_path / "einsum.onnx"
    onnx.save(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 15)], ir_version=8
        ),
        path,
    )

    _normalize_onnx_for_x2paddle(path)

    normalized = onnx.load(path)
    assert [node.op_type for node in normalized.graph.node] == [
        "Reshape",
        "MatMul",
        "Reshape",
    ]
    assert normalized.graph.node[-1].output[0] == "masks"


def test_x2paddle_onnx_normalization_rejects_newer_opset(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    from libreyolo.export.paddle import _normalize_onnx_for_x2paddle

    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "newer",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    path = tmp_path / "model.onnx"
    onnx.save(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 16)], ir_version=7
        ),
        path,
    )
    with pytest.raises(NotImplementedError, match="opset 15 or lower"):
        _normalize_onnx_for_x2paddle(path)


class _InputHandle:
    def __init__(self):
        self.value = None
        self.shape = None

    def reshape(self, shape):
        self.shape = tuple(shape)

    def copy_from_cpu(self, value):
        self.value = value.copy()


class _OutputHandle:
    def __init__(self, value):
        self.value = value

    def copy_to_cpu(self):
        return self.value.copy()


class _Predictor:
    def __init__(self):
        self.input = _InputHandle()
        self.output = np.ones((1, 84, 4), dtype=np.float32)

    def get_input_names(self):
        return ["images"]

    def get_output_names(self):
        return ["output"]

    def get_input_handle(self, name):
        assert name == "images"
        return self.input

    def get_output_handle(self, name):
        assert name == "output"
        return _OutputHandle(self.output)

    def run(self):
        return True


def _install_fake_paddle(monkeypatch):
    predictor = _Predictor()

    class Config:
        def __init__(self, model, parameters):
            self.model = model
            self.parameters = parameters
            self.cpu = False
            self.mkldnn_disabled = False
            self.ir_optim = True
            self.memory_optimized = False

        def disable_gpu(self):
            self.cpu = True

        def disable_mkldnn(self):
            self.mkldnn_disabled = True

        def switch_ir_optim(self, enabled):
            self.ir_optim = enabled

        def enable_memory_optim(self):
            self.memory_optimized = True

    inference = types.ModuleType("paddle.inference")
    inference.Config = Config
    inference.create_predictor = lambda config: predictor
    paddle = types.ModuleType("paddle")
    paddle.__path__ = []
    paddle.inference = inference
    monkeypatch.setitem(sys.modules, "paddle", paddle)
    monkeypatch.setitem(sys.modules, "paddle.inference", inference)
    return predictor


def _paddle_artifact(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "model.pdmodel").write_bytes(b"model")
    (artifact / "model.pdiparams").write_bytes(b"parameters")
    (artifact / "metadata.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "model_family": "yolo9",
                "size": "t",
                "task": "detect",
                "supported_tasks": ["detect"],
                "default_task": "detect",
                "nc": 2,
                "names": {"0": "first", "1": "second"},
                "imgsz": 320,
                "imgsz_h": 320,
                "imgsz_w": 320,
            }
        ),
        encoding="utf-8",
    )
    return artifact


def test_paddle_backend_reads_metadata_and_runs_cpu(monkeypatch, tmp_path):
    predictor = _install_fake_paddle(monkeypatch)
    from libreyolo.backends.paddle import PaddleBackend

    backend = PaddleBackend(_paddle_artifact(tmp_path), device="cpu")
    blob = np.zeros((1, 3, 320, 320), dtype=np.float32)
    outputs = backend._run_inference(blob)

    assert backend.model_family == "yolo9"
    assert backend.model_size == "t"
    assert backend.task == "detect"
    assert backend.imgsz == 320
    assert backend.names == {0: "first", 1: "second"}
    assert predictor.input.shape == blob.shape
    assert np.array_equal(predictor.input.value, blob)
    assert np.array_equal(outputs[0], predictor.output)


def test_factory_routes_paddle_artifact(monkeypatch, tmp_path):
    _install_fake_paddle(monkeypatch)
    import libreyolo.backends.paddle as paddle_backend
    from libreyolo.models import LibreYOLO

    artifact = _paddle_artifact(tmp_path)
    sentinel = object()
    factory = MagicMock(return_value=sentinel)
    monkeypatch.setattr(paddle_backend, "PaddleBackend", factory)

    assert LibreYOLO(str(artifact), device="cpu") is sentinel
    factory.assert_called_once_with(
        str(artifact), nb_classes=None, device="cpu", task=None
    )
