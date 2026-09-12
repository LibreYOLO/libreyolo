"""LeRobot adapter contract checks with no optional runtime or downloads."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo.models.vla.observation import Observation
from libreyolo.models.vla.smolvla import LibreSmolVLA

pytestmark = [pytest.mark.unit, pytest.mark.vla]


def test_smolvla_load_processors_and_predict(monkeypatch):
    calls = {}
    config = SimpleNamespace(
        type="smolvla",
        image_features={"observation.images.camera1": object()},
        robot_state_feature=SimpleNamespace(shape=(2,)),
        action_feature=SimpleNamespace(shape=(3,)),
        chunk_size=4,
    )

    class Policy:
        @classmethod
        def from_pretrained(cls, path, config):
            calls["load"] = (path, config)
            return cls()

        def eval(self):
            calls["eval"] = True

        def to(self, device):
            calls["device"] = str(device)

        def reset(self):
            calls["reset"] = True

        def predict_action_chunk(self, batch):
            assert not torch.is_grad_enabled()
            calls["batch"] = batch
            return torch.ones(1, 4, 6)

    def processors(config, **kwargs):
        calls["processors"] = kwargs
        return lambda raw: raw, lambda chunk: chunk * 2

    pieces = (
        SimpleNamespace(from_pretrained=lambda path: config),
        None,
        Policy,
        processors,
    )
    monkeypatch.setattr(
        LibreSmolVLA, "_require_lerobot", classmethod(lambda cls: pieces)
    )
    model = LibreSmolVLA(device="cpu")
    model._load_policy("saved-policy")
    assert calls["load"] == ("saved-policy", config)
    assert calls["eval"] and calls["reset"] and calls["device"] == "cpu"
    assert calls["processors"]["pretrained_path"] == "saved-policy"
    assert calls["processors"]["preprocessor_overrides"]["device_processor"] == {
        "device": "cpu"
    }
    assert model.camera_slots == ["camera1"]
    assert (model.state_dim, model.action_dim, model.chunk_size) == (2, 3, 4)
    observation = Observation(
        {"camera1": Image.new("RGB", (8, 8))},
        np.zeros(2),
        "move",
    )
    chunk = model._predict_chunk(observation)
    assert chunk.shape == (1, 4, 3) and (chunk == 2).all()
    assert calls["batch"]["observation.images.camera1"].shape == (3, 8, 8)
    assert calls["batch"]["task"] == "move"
    model.contract = {"cameras": ["wrist"]}
    assert model.camera_slots == ["wrist"]
    config.type = "other"
    with pytest.raises(ValueError, match="not 'smolvla'"):
        model._pretrained_config("other-policy")


def test_optional_instruction_preserves_supplied_text():
    class OptionalInstruction(LibreSmolVLA):
        REQUIRES_INSTRUCTION = False

    model = OptionalInstruction(device="cpu")
    assert model._resolve_instruction(None) == ""
    assert model._resolve_instruction("  lift ") == "lift"
    model.set_instruction("push")
    assert model._resolve_instruction(None) == "push"
    with pytest.raises(ValueError, match="No instruction"):
        LibreSmolVLA(device="cpu")._resolve_instruction(None)
