"""Offline CPU smoke for ACT and Diffusion, using the real LeRobot runtime.

Run on Python 3.12+: ``python tests/smoke/vla_action_policies.py`` after
installing ``libreyolo[vla]`` and ``lerobot[diffusion]>=0.6.1``. Only the
dataset reader is replaced with synthetic in-memory observations. Policies,
optimizers, processors, saved weights and checkpoint reload are real.
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import ClassVar


def run_smoke():
    started = time.monotonic()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from PIL import Image
    from torch.utils.data import Dataset

    from libreyolo import LibreVLA
    from libreyolo.models.vla.training import trainer

    torch.set_num_threads(2)
    configs = {
        "act": ACTConfig(
            device="cpu",
            pretrained_backbone_weights=None,
            chunk_size=4,
            n_action_steps=4,
            dim_model=32,
            dim_feedforward=64,
            n_heads=4,
            n_encoder_layers=1,
            n_decoder_layers=1,
            n_vae_encoder_layers=1,
        ),
        "diffusion": DiffusionConfig(
            device="cpu",
            pretrained_backbone_weights=None,
            horizon=8,
            n_action_steps=4,
            n_obs_steps=2,
            down_dims=(32, 64, 128),
            diffusion_step_embed_dim=32,
            num_train_timesteps=10,
            num_inference_steps=2,
        ),
    }

    class Meta:
        total_episodes = 2
        fps = 10
        camera_keys: ClassVar[list[str]] = ["observation.images.front"]
        features: ClassVar[dict] = {
            "observation.images.front": {
                "dtype": "image",
                "shape": (3, 64, 64),
                "names": ["channels", "height", "width"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (2,),
                "names": ["q0", "q1"],
            },
            "action": {"dtype": "float32", "shape": (2,), "names": ["a0", "a1"]},
        }
        stats: ClassVar[dict] = {}

    for key in Meta.features:
        shape = (3, 1, 1) if "images" in key else (2,)
        Meta.stats[key] = {
            "mean": torch.zeros(shape),
            "std": torch.ones(shape),
            "min": -torch.ones(shape),
            "max": torch.ones(shape),
        }

    original_seam = trainer._lerobot
    upstream = original_seam()
    try:
        with tempfile.TemporaryDirectory(prefix="librevla-smoke-") as root:
            for family, config in configs.items():

                class Data(Dataset):
                    def __init__(self, *args, policy_config=config, **kwargs):
                        self.n_obs = policy_config.n_obs_steps
                        self.horizon = getattr(policy_config, "horizon", 4)

                    def __len__(self):
                        return 8

                    def __getitem__(self, index):
                        shape = () if self.n_obs == 1 else (self.n_obs,)
                        return {
                            "observation.images.front": torch.zeros(*shape, 3, 64, 64),
                            "observation.state": torch.zeros(*shape, 2),
                            "action": torch.ones(self.horizon, 2) * 0.5,
                            "action_is_pad": torch.zeros(
                                self.horizon, dtype=torch.bool
                            ),
                            "task": "",
                        }

                trainer._lerobot = lambda: (Data, lambda *a, **k: Meta(), *upstream[2:])
                model = LibreVLA(family, device="cpu")
                model._scratch_config = lambda meta, policy_config=config: policy_config
                result = model.train(
                    data="synthetic/offline",
                    epochs=1,
                    batch=2,
                    max_steps=3,
                    val_batches=1,
                    output_dir=str(Path(root) / family),
                )
                restored = LibreVLA(result["best"], device="cpu")
                prediction = restored.predict(Image.new("RGB", (64, 64)), state=[0, 0])
                assert prediction.actions.data.shape == (4, 2)
                assert torch.isfinite(prediction.actions.data).all()
                assert prediction.actions.instruction == ""
                metrics = restored.val(max_batches=1, batch=2)
                assert metrics["val/steps"] == 8
                assert restored.contract["base_repo"] is None
                assert restored.contract["base_revision"] is None
                print(f"{family}: train, reload, predict, val passed", flush=True)
    finally:
        trainer._lerobot = original_seam
    print(f"Elapsed: {time.monotonic() - started:.2f}s", flush=True)


if __name__ == "__main__":
    run_smoke()
