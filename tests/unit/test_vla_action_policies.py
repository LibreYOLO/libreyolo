"""Action-policy aliases, history and offline timestamp alignment."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreVLA
from libreyolo.models.vla.act_policy import LibreACT
from libreyolo.models.vla.diffusion_policy import LibreDiffusionPolicy
from libreyolo.models.vla.observation import Observation

pytestmark = [pytest.mark.unit, pytest.mark.vla]


@pytest.mark.parametrize(
    "alias,cls",
    [
        ("act", LibreACT),
        ("act-policy", LibreACT),
        ("act_policy", LibreACT),
        ("diffusion", LibreDiffusionPolicy),
        ("diffusion-policy", LibreDiffusionPolicy),
        ("diffusion_policy", LibreDiffusionPolicy),
    ],
)
def test_scratch_policy_aliases_are_lazy_and_untrained(alias, cls):
    model = LibreVLA(alias, device="cpu")
    assert isinstance(model, cls)
    assert model.task == "act" and model.model_path is None
    assert model._resolve_instruction(None) == ""
    with pytest.raises(ValueError, match="untrained"):
        model.predict(Image.new("RGB", (8, 8)))


def test_diffusion_history_pads_first_observation_and_resets():
    model = LibreDiffusionPolicy(device="cpu")
    model._config = SimpleNamespace(n_obs_steps=2, n_action_steps=4)
    frame = Image.new("RGB", (8, 8))
    first = Observation({"front": frame}, np.array([1.0, 2.0]), "")
    second = Observation({"front": frame}, np.array([3.0, 4.0]), "")
    raw = model._raw_observation(first)
    assert raw["observation.state"].shape == (1, 2, 2)
    torch.testing.assert_close(
        raw["observation.state"][0], torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    )
    raw = model._raw_observation(second)
    torch.testing.assert_close(
        raw["observation.state"][0], torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    )
    assert raw["observation.images.front"].shape == (1, 2, 3, 8, 8)
    with pytest.raises(ValueError, match="camera keys changed"):
        model._raw_observation(Observation({"wrist": frame}, np.zeros(2), ""))
    model.reset()
    raw = model._raw_observation(second)
    torch.testing.assert_close(
        raw["observation.state"][0], torch.tensor([[3.0, 4.0], [3.0, 4.0]])
    )


def test_diffusion_validation_starts_at_current_timestep():
    model = LibreDiffusionPolicy(device="cpu")
    model._config = SimpleNamespace(n_obs_steps=2, n_action_steps=4)
    target = torch.arange(8.0).view(1, 8, 1)
    pad = torch.tensor([[True, False, False, True, False, False, False, True]])
    aligned, mask = model._validation_targets(target, pad, 4)
    assert aligned.flatten().tolist() == [1.0, 2.0, 3.0, 4.0]
    assert mask.tolist() == [[False, False, True, False]]
    assert model.chunk_size == 4
    with pytest.raises(ValueError, match="shorter"):
        model._validation_targets(target, pad, 8)
