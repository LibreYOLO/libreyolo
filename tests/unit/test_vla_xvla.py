"""X-VLA registration, checkpoint and action-mode contracts without weights."""

from types import SimpleNamespace

import pytest

from libreyolo import LibreVLA, LibreXVLA
from libreyolo.models.vla import _RESERVED_ALIASES
from libreyolo.models.vla.checkpoint import write_contract

pytestmark = [pytest.mark.unit, pytest.mark.vla]


@pytest.mark.parametrize("alias", ["xvla", "xvla-base", "xvla_base"])
def test_xvla_alias_is_lazy(alias):
    model = LibreVLA(alias, device="cpu")
    assert isinstance(model, LibreXVLA) and model._policy is None
    assert model.task == "act"
    assert "xvla" not in _RESERVED_ALIASES
    with pytest.raises(ValueError, match="instruction"):
        model._resolve_instruction(None)


def test_xvla_checkpoint_round_trip(tmp_path):
    write_contract(
        tmp_path,
        family="xvla",
        size="base",
        base_repo="lerobot/xvla-base",
        base_revision=LibreXVLA.HF_REVISIONS["base"],
        data="fake/data",
        fps=30,
        cameras=["front", "wrist"],
        action_names=None,
        state_names=None,
        chunk_size=30,
    )
    model = LibreVLA(tmp_path, device="cpu")
    assert isinstance(model, LibreXVLA)
    assert model.camera_slots == ["front", "wrist"]
    assert model._ensure_weights() == str(tmp_path)
    assert model.contract["chunk_size"] == 30


def test_xvla_base_dimensions_and_training_representation():
    model = LibreXVLA(device="cpu")
    config = SimpleNamespace(
        image_features={
            f"observation.images.{name}": object()
            for name in ["image", "image2", "image3"]
        },
        robot_state_feature=SimpleNamespace(shape=(8,)),
        action_feature=SimpleNamespace(shape=(20,)),
        chunk_size=30,
        action_mode="ee6d",
    )
    model._config = config
    assert model.camera_slots == ["image", "image2", "image3"]
    assert (model.state_dim, model.action_dim, model.chunk_size) == (8, 20, 30)
    assert config.action_mode == "ee6d"
    assert model._prepare_training_config(config).action_mode == "auto"
    assert LibreXVLA.get_download_url("LibreXVLAbase") == (
        "https://huggingface.co/lerobot/xvla-base/tree/cdb7964e4fe842935d671bfab5a5ebe00a96648c"
    )
