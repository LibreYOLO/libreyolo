"""Offline unit tests for the LibreVLA tier (no lerobot, no weights, no network).

Covers the ``act`` task registration, the ``Actions`` payload and its
``Results`` integration, the observation contract (camera mapping, state
coercion, instruction), the factory, the checkpoint contract, the offline
metric, and the predict surface end to end through a fake family.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import Actions, LibreSmolVLA, LibreVLA, Results
from libreyolo.models.vla import _load_checkpoint
from libreyolo.models.vla.base import LibreVLAModel
from libreyolo.models.vla.checkpoint import (
    CONTRACT_FILENAME,
    is_vla_checkpoint,
    read_contract,
    write_contract,
)
from libreyolo.models.vla.metrics import action_error
from libreyolo.models.vla.observation import (
    Observation,
    action_names_from_features,
    coerce_state,
    frame_to_tensor,
    map_cameras,
)
from libreyolo.tasks import TASKS, normalize_task, suffix_to_task, task_to_suffix
from libreyolo.utils.drawing import draw_actions

pytestmark = [pytest.mark.unit, pytest.mark.vla]


# ---------------------------------------------------------------------------
# Task registration
# ---------------------------------------------------------------------------


def test_act_task_is_canonical():
    assert "act" in TASKS
    for alias in ("act", "action", "actions", "vla", "policy", "robot-policy"):
        assert normalize_task(alias) == "act"
    assert task_to_suffix("act") == "act"
    assert suffix_to_task("-act") == "act"


# ---------------------------------------------------------------------------
# Actions payload
# ---------------------------------------------------------------------------


class TestActions:
    def test_shape_and_metadata(self):
        a = Actions(
            torch.zeros(5, 3), (10, 20), names=["x", "y", "g"], fps=30, instruction="go"
        )
        assert a.horizon == 5 and a.dim == 3 and len(a) == 5
        assert a.first.shape == (3,)
        assert a.names == ["x", "y", "g"] and a.fps == 30.0 and a.instruction == "go"
        assert "horizon=5" in repr(a)

    def test_one_dimensional_becomes_single_step(self):
        a = Actions(np.array([1.0, 2.0]))
        assert a.horizon == 1 and a.dim == 2

    def test_rejects_bad_input(self):
        with pytest.raises(ValueError):
            Actions(torch.zeros(2, 2, 2))
        with pytest.raises(ValueError):
            Actions(torch.tensor([[float("nan"), 0.0]]))
        with pytest.raises(ValueError):
            Actions(torch.zeros(2, 3), names=["a", "b"])
        with pytest.raises(ValueError):
            Actions(torch.zeros(2, 3), fps=0)

    def test_slice_and_move_keep_metadata(self):
        a = Actions(
            torch.arange(12.0).reshape(4, 3),
            names=["a", "b", "c"],
            fps=10,
            instruction="t",
        )
        s = a[1:3]
        assert (
            s.horizon == 2
            and s.names == ["a", "b", "c"]
            and s.fps == 10
            and s.instruction == "t"
        )
        one = a[0]
        assert one.horizon == 1 and float(one.first[1]) == 1.0
        n = a.numpy()
        assert isinstance(n.data, np.ndarray) and n.names == a.names
        c = a.cpu().to(torch.float64)
        assert c.data.dtype == torch.float64 and c.instruction == "t"

    def test_to_dict_rounds(self):
        a = Actions(
            torch.tensor([[0.123456, 1.0]]), names=["p", "q"], fps=5, instruction="i"
        )
        d = a.to_dict(decimals=3)
        assert d == {
            "instruction": "i",
            "horizon": 1,
            "dim": 2,
            "fps": 5.0,
            "names": ["p", "q"],
            "actions": [[0.123, 1.0]],
        }


class TestResultsWithActions:
    def _result(self):
        a = Actions(
            torch.arange(6.0).reshape(3, 2), names=["a", "b"], fps=2, instruction="pick"
        )
        return Results(None, (8, 6), path=None, names={}, actions=a)

    def test_len_summary_json_repr(self):
        r = self._result()
        assert len(r) == 3
        assert r.boxes is None and r.actions.horizon == 3
        rows = json.loads(r.to_json())
        assert (
            len(rows) == 1 and rows[0]["instruction"] == "pick" and rows[0]["dim"] == 2
        )
        assert "actions=" in repr(r)

    def test_slice_moves_and_update(self):
        r = self._result()
        first = r[0]
        assert first.actions.horizon == 1 and first.actions.names == ["a", "b"]
        assert isinstance(r.numpy().actions.data, np.ndarray)
        r2 = self._result()
        r2.update(actions=Actions(torch.zeros(1, 2)))
        assert r2.actions.horizon == 1

    def test_plot_appends_strip(self):
        r = self._result()
        img = Image.new("RGB", (6, 8), (0, 0, 0))
        out = r.plot(img)
        assert out.size[0] == 6 and out.size[1] > 8
        strip = draw_actions(Image.new("RGB", (300, 100)), r.actions, panel_height=120)
        assert strip.size == (300, 220)

    def test_plot_needs_source_when_path_unset(self):
        with pytest.raises(ValueError):
            self._result().plot()


# ---------------------------------------------------------------------------
# Observation contract
# ---------------------------------------------------------------------------


class TestMapCameras:
    SLOTS = ["camera1", "camera2", "camera3"]

    def test_slot_names_take_their_slot_and_others_fill_in_order(self):
        out = map_cameras({"wrist": 1, "camera1": 2}, self.SLOTS)
        assert out == {"camera1": 2, "camera2": 1}

    def test_explicit_cameras_map_by_position(self):
        out = map_cameras(
            {"front": 1, "side": 2}, self.SLOTS, cameras=["side", "front"]
        )
        assert out == {"camera1": 2, "camera2": 1}

    def test_unknown_name_with_explicit_cameras_raises(self):
        with pytest.raises(ValueError, match="not in cameras"):
            map_cameras({"top": 1}, self.SLOTS, cameras=["front"])

    def test_too_many_frames_raises(self):
        with pytest.raises(ValueError, match="camera slot"):
            map_cameras({"a": 1, "b": 2}, ["camera1"])
        with pytest.raises(ValueError):
            map_cameras({}, self.SLOTS)


class TestCoerceState:
    def test_array_callable_and_none(self):
        warnings = []
        assert coerce_state([1, 2, 3], 3, warn=warnings.append).tolist() == [
            1.0,
            2.0,
            3.0,
        ]
        assert coerce_state(lambda: np.ones(3), 3).dtype == np.float32
        zeros = coerce_state(None, 3, warn=warnings.append)
        assert zeros.tolist() == [0.0, 0.0, 0.0] and len(warnings) == 1

    def test_wrong_length_or_nonfinite_raises(self):
        with pytest.raises(ValueError, match="expects 3"):
            coerce_state([1, 2], 3)
        with pytest.raises(ValueError, match="finite"):
            coerce_state([1, float("inf"), 2], 3)


def test_frame_to_tensor_and_action_names():
    t = frame_to_tensor(Image.new("RGB", (4, 2), (255, 0, 0)))
    assert (
        t.shape == (3, 2, 4) and float(t[0, 0, 0]) == 1.0 and float(t[1, 0, 0]) == 0.0
    )
    assert action_names_from_features({"action": {"names": ["j1", "j2"]}}) == [
        "j1",
        "j2",
    ]
    assert action_names_from_features({"action": {"names": {"motors": ["m1"]}}}) == [
        "m1"
    ]
    assert action_names_from_features({}) is None


# ---------------------------------------------------------------------------
# Factory and install gating
# ---------------------------------------------------------------------------


class TestFactory:
    def test_default_alias_and_lazy_load(self):
        model = LibreVLA(device="cpu")
        assert isinstance(model, LibreSmolVLA)
        assert model.size == "base" and model.task == "act"
        assert model._policy is None  # nothing downloaded or loaded yet
        assert model.model_path == "lerobot/smolvla_base"

    def test_reserved_and_unknown_aliases(self):
        with pytest.raises(ValueError, match="reserved"):
            LibreVLA("pi0")
        with pytest.raises(ValueError, match="Unknown VLA model"):
            LibreVLA("not-a-policy")

    def test_pinned_revision_is_a_commit_sha(self):
        for size, sha in LibreSmolVLA.HF_REVISIONS.items():
            assert size in LibreSmolVLA.HF_REPOS
            assert len(sha) == 40 and int(sha, 16) >= 0

    def test_instruction_and_task_validation(self):
        model = LibreVLA(device="cpu", instruction="  push the cube ")
        assert model.instruction == "push the cube"
        with pytest.raises(ValueError):
            model.set_instruction("   ")
        with pytest.raises(ValueError, match="act"):
            LibreSmolVLA(task="detect", device="cpu")
        with pytest.raises(AttributeError, match="set_instruction"):
            model.set_classes(["cube"])

    def test_missing_lerobot_raises_install_hint(self, monkeypatch, tmp_path):
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("lerobot"):
                raise ImportError("no lerobot")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        model = LibreVLA(device="cpu")
        monkeypatch.setattr(model, "_ensure_weights", lambda: str(tmp_path))
        with pytest.raises(ImportError, match=r"libreyolo\[vla\]"):
            model.predict(Image.new("RGB", (8, 8)), instruction="go")

    def test_unsupported_surfaces_raise(self):
        model = LibreVLA(device="cpu")
        with pytest.raises(NotImplementedError):
            model.export("onnx")
        with pytest.raises(NotImplementedError):
            model.track("x.mp4")
        with pytest.raises(ValueError, match="data="):
            model.train()
        with pytest.raises(ValueError, match="data="):
            model.val()


# ---------------------------------------------------------------------------
# Checkpoint contract
# ---------------------------------------------------------------------------


def test_checkpoint_contract_roundtrip(tmp_path):
    assert not is_vla_checkpoint(tmp_path)
    assert not is_vla_checkpoint(123)
    path = write_contract(
        tmp_path,
        family="smolvla",
        size="base",
        base_repo="lerobot/smolvla_base",
        base_revision="c" * 40,
        data="lerobot/svla_so101_pickplace",
        fps=30,
        cameras=["up", "side"],
        action_names=["a"] * 6,
        state_names=None,
        chunk_size=50,
    )
    assert path.name == CONTRACT_FILENAME and is_vla_checkpoint(tmp_path)
    contract = read_contract(tmp_path)
    assert contract["cameras"] == ["up", "side"] and contract["fps"] == 30
    assert contract["schema"] == 1 and contract["libreyolo_version"]

    model = LibreVLA(str(tmp_path), device="cpu")
    assert isinstance(model, LibreSmolVLA)
    assert model.camera_slots == ["up", "side"]
    assert model.action_names == ["a"] * 6 and model.fps == 30
    assert model.model_path == str(tmp_path)

    (tmp_path / CONTRACT_FILENAME).write_text(
        json.dumps({"schema": 99, "family": "smolvla", "size": "base"})
    )
    with pytest.raises(ValueError, match="schema"):
        read_contract(tmp_path)
    (tmp_path / CONTRACT_FILENAME).write_text(
        json.dumps({"schema": 1, "family": "other", "size": "base"})
    )
    with pytest.raises(ValueError, match="unknown family"):
        _load_checkpoint(tmp_path)


# ---------------------------------------------------------------------------
# Offline metric
# ---------------------------------------------------------------------------


class TestActionError:
    def test_perfect_prediction_is_zero(self):
        target = torch.arange(24.0).reshape(2, 4, 3)
        m = action_error(target.clone(), target, names=["a", "b", "c"])
        assert m["val/action_l1"] == 0.0 and m["val/action_mse"] == 0.0
        assert m["val/action_l1_dims"] == {"a": 0.0, "b": 0.0, "c": 0.0}
        assert m["val/steps"] == 8

    def test_mask_and_values(self):
        target = torch.zeros(1, 2, 2)
        pred = torch.tensor([[[1.0, 3.0], [100.0, 100.0]]])
        m = action_error(pred, target, mask=torch.tensor([[True, False]]))
        assert m["val/action_l1"] == 2.0 and m["val/action_l1_first"] == 2.0
        assert m["val/action_mse"] == 5.0 and m["val/steps"] == 1
        assert m["val/action_l1_dims"] == {"a0": 1.0, "a1": 3.0}

    def test_shape_errors(self):
        with pytest.raises(ValueError):
            action_error(torch.zeros(2, 3), torch.zeros(3, 3))
        with pytest.raises(ValueError, match="mask"):
            action_error(
                torch.zeros(1, 2, 2),
                torch.zeros(1, 2, 2),
                mask=torch.zeros(3, dtype=torch.bool),
            )


# ---------------------------------------------------------------------------
# Predict surface through a fake family
# ---------------------------------------------------------------------------


class FakeVLA(LibreVLAModel):
    FAMILY = "fakevla"
    FILENAME_PREFIX = "LibreFakeVLA"
    HF_REPOS = {"tiny": "fake/tiny"}
    INPUT_SIZES = {"tiny": 8}

    def __init__(self, **kwargs):
        super().__init__("tiny", **kwargs)
        self.seen = []

    def _ensure_weights(self):
        return "fake-dir"

    def _load_policy(self, snapshot_dir):
        self.loaded_from = snapshot_dir
        self._policy = SimpleNamespace(reset=lambda: None, to=lambda device: None)

    @property
    def camera_slots(self):
        return ["camera1", "camera2"]

    @property
    def state_dim(self):
        return 2

    @property
    def action_dim(self):
        return 3

    @property
    def chunk_size(self):
        return 4

    def _predict_chunk(self, observation: Observation):
        self.seen.append(observation)
        base = torch.arange(12.0).reshape(4, 3)
        return base + float(observation.state.sum())


@pytest.fixture
def fake():
    return FakeVLA(device="cpu")


def _img(w=6, h=4):
    return Image.new("RGB", (w, h), (10, 20, 30))


class TestPredict:
    def test_single_frame_goes_to_first_slot(self, fake):
        r = fake.predict(_img(), state=[1, 2], instruction="go")
        assert isinstance(r, Results) and r.orig_shape == (4, 6)
        assert r.actions.horizon == 4 and r.actions.dim == 3
        assert float(r.actions.first[0]) == 3.0  # state sum added
        obs = fake.seen[-1]
        assert list(obs.frames) == ["camera1"] and obs.instruction == "go"
        assert fake.loaded_from == "fake-dir"

    def test_dict_maps_cameras_and_sticky_instruction(self, fake):
        fake.set_instruction("stack")
        r = fake.predict({"wrist": _img(3, 3), "camera1": _img()}, state=np.zeros(2))
        assert r.orig_shape == (4, 6)  # primary is slot camera1
        assert list(fake.seen[-1].frames) == ["camera1", "camera2"]
        assert r.actions.instruction == "stack"

    def test_missing_instruction_raises(self, fake):
        with pytest.raises(ValueError, match="instruction"):
            fake.predict(_img(), state=[0, 0])

    def test_state_none_warns_once_and_callable_is_called_per_frame(self, fake, caplog):
        with caplog.at_level("WARNING"):
            fake.predict([_img(), _img()], instruction="go")
        assert sum("state=None" in m for m in caplog.messages) == 1
        counter = {"n": 0}

        def read_state():
            counter["n"] += 1
            return [counter["n"], 0]

        results = fake.predict(
            [_img(), _img(), _img()], state=read_state, instruction="go"
        )
        assert counter["n"] == 3 and [r.frame_idx for r in results] == [0, 1, 2]
        assert float(results[2].actions.first[0]) == 3.0

    def test_list_directory_and_stream_shapes(self, fake, tmp_path):
        for i in range(2):
            _img().save(tmp_path / f"f{i}.png")
        results = fake.predict(str(tmp_path), state=[0, 0], instruction="go")
        assert isinstance(results, list) and len(results) == 2
        assert results[0].path.endswith("f0.png")
        gen = fake.predict(
            [_img(), {"camera2": _img()}], state=[0, 0], instruction="go", stream=True
        )
        assert hasattr(gen, "__next__")
        assert len(list(gen)) == 2
        with pytest.raises(ValueError, match="stream=True"):
            fake.predict("rtsp://camera/1", state=[0, 0], instruction="go")

    def test_reset_and_wrong_state(self, fake):
        fake.reset()  # before load: no-op
        with pytest.raises(ValueError, match="expects 2"):
            fake.predict(_img(), state=[1, 2, 3], instruction="go")

    def test_save_writes_render(self, fake, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        r = fake.predict(_img(40, 30), state=[0, 0], instruction="go", save=True)
        saved = Path(r.saved_path)
        assert saved.is_file()
        assert (
            saved.resolve().parent == (tmp_path / "runs" / "act" / "predict").resolve()
        )
        out = tmp_path / "explicit.png"
        fake.predict(
            _img(40, 30),
            state=[0, 0],
            instruction="go",
            save=True,
            output_path=str(out),
        )
        assert out.is_file()
        with pytest.raises(ValueError, match="output_path"):
            fake.predict(_img(), state=[0, 0], instruction="go", output_path="x.png")
