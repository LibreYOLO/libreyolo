"""CPU contracts for GTR pose (random weights; real-weight parity is opt-in)."""

import random

import numpy as np
import pytest
import torch

from libreyolo import LibreGTR, LibreYOLO
from libreyolo.models.gtr.pose import GTRPoseDecoderLayer, LibreGTRPoseModel
from libreyolo.utils.serialization import wrap_libreyolo_checkpoint

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    torch.set_num_threads(2)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])


def _pose_checkpoint(size="s", **extra):
    model = LibreGTRPoseModel(size)
    return model, wrap_libreyolo_checkpoint(
        model.state_dict(),
        model_family="gtr",
        size=size,
        nc=1,
        names={0: "person"},
        task="pose",
        imgsz=640,
        num_keypoints=17,
        keypoint_dim=3,
        **extra,
    )


@pytest.mark.parametrize(
    "size, layers, feedforward", [("s", 3, 512), ("m", 4, 512), ("x", 4, 2048)]
)
def test_pose_architecture_matches_published_checkpoints(size, layers, feedforward):
    from libreyolo.models.ec.decoder import ECPoseTransformer

    model = LibreGTRPoseModel(size)
    decoder_layers = model.decoder.decoder.layers
    assert len(decoder_layers) == layers
    assert all(isinstance(layer, GTRPoseDecoderLayer) for layer in decoder_layers)
    assert decoder_layers[0].linear1.out_features == feedforward
    # Tensor names stay those of the shared ECPose decoder (and upstream).
    hidden = model.decoder.hidden_dim
    reference = ECPoseTransformer(
        hidden_dim=hidden,
        num_decoder_layers=layers,
        dim_feedforward=feedforward,
        num_feature_levels=3,
        eval_spatial_size=[640, 640],
    )
    assert set(model.decoder.state_dict()) == set(reference.state_dict())
    assert LibreGTR.detect_size(model.state_dict()) == size


def test_pose_layer_keeps_position_embedding_on_tokens():
    from libreyolo.models.ec.decoder import PoseDeformableTransformerDecoderLayer

    torch.manual_seed(0)
    layer = GTRPoseDecoderLayer(d_model=32, d_ffn=64, n_levels=1, n_heads=4)
    base = PoseDeformableTransformerDecoderLayer(
        d_model=32, d_ffn=64, n_levels=1, n_heads=4
    )
    base.load_state_dict(layer.state_dict())
    tokens = torch.randn(1, 2, 3, 32)
    memory = (torch.randn(4, 8, 16),)
    reference = torch.rand(1, 2, 1, 3, 2)
    args = dict(memory=memory, memory_spatial_shapes=[(4, 4)])
    with torch.no_grad():
        zero = torch.zeros(1, 2, 2, 32)
        # Without an embedding the two layer variants are the same function.
        torch.testing.assert_close(
            layer(tokens, zero, reference, **args),
            base(tokens, zero, reference, **args),
        )
        pos = torch.randn(1, 2, 2, 32)
        # With one, GTR also carries it through the values, residual and gate.
        assert not torch.allclose(
            layer(tokens, pos, reference, **args), base(tokens, pos, reference, **args)
        )


def test_pose_forward_training_and_eval_outputs():
    torch.manual_seed(0)
    model = LibreGTRPoseModel("s", eval_spatial_size=(160, 160)).eval()
    image = torch.randn(1, 3, 160, 160)
    with torch.no_grad():
        out = model(image)
    assert out["pred_logits"].shape == (1, 60, 2)
    assert out["pred_keypoints"].shape == (1, 60, 34)
    with pytest.raises(ValueError, match="square"):
        model(torch.randn(1, 3, 160, 192))


def test_pose_checkpoint_routing_and_prediction(tmp_path):
    model, checkpoint = _pose_checkpoint()
    path = tmp_path / "LibreGTRs-pose.pt"
    torch.save(checkpoint, path)
    loaded = LibreYOLO(str(path), device="cpu")
    assert isinstance(loaded, LibreGTR)
    assert (loaded.task, loaded.size, loaded.nb_classes) == ("pose", "s", 1)
    assert loaded.names == {0: "person"}
    for key, value in model.state_dict().items():
        torch.testing.assert_close(
            value, loaded.model.state_dict()[key], rtol=0, atol=0
        )
    result = loaded.predict(np.zeros((96, 128, 3), np.uint8), conf=0.0)[0]
    assert result.keypoints is not None
    assert tuple(result.keypoints.data.shape[1:]) == (17, 3)

    # A detection instance must not silently accept pose weights.
    with pytest.raises(RuntimeError, match="task"):
        LibreGTR(str(path), size="s", task="detect", device="cpu")


def test_raw_upstream_pose_state_routes_to_gtr():
    from libreyolo.models import _matching_model_classes

    state = LibreGTRPoseModel("s").state_dict()
    assert LibreGTR.can_load(state)
    assert LibreGTR.detect_checkpoint_task(state) == "pose"
    assert LibreGTR.detect_nb_classes(state) == 1
    # First match wins; RF-DETR's discriminator is deliberately broad.
    assert _matching_model_classes(state)[0].FAMILY == "gtr"


def test_pose_download_urls_are_task_aware():
    detect = LibreGTR.get_download_url("LibreGTRs.pt")
    assert detect.endswith(
        "LibreGTRs/resolve/74193dc356e07f51893579211ffdecf0ee2e560a/LibreGTRs.pt"
    )
    pose = LibreGTR.get_download_url("LibreGTRx-pose.pt")
    revision = LibreGTR.HF_TASK_REVISIONS[("x", "pose")] or "main"
    assert pose == (
        "https://huggingface.co/LibreYOLO/LibreGTRx-pose/resolve/"
        f"{revision}/LibreGTRx-pose.pt"
    )
    assert LibreGTR.detect_task_from_filename("LibreGTRm-pose.pt") == "pose"
    assert LibreGTR.get_download_url("LibreGTRm-obb.pt") is None


@pytest.mark.parametrize(
    "size, epochs, backbone_lr_mult, weight_decay",
    [("s", 92, 0.05, 1e-4), ("l", 74, 0.005, 1.25e-4)],
)
def test_pose_recipe_follows_upstream_configs(
    size, epochs, backbone_lr_mult, weight_decay
):
    from libreyolo.models.gtr.pose_trainer import GTRPoseConfig

    config = GTRPoseConfig.from_kwargs(size=size)
    assert (config.epochs, config.backbone_lr_mult, config.weight_decay) == (
        epochs,
        backbone_lr_mult,
        weight_decay,
    )
    assert (config.lr0, config.warmup_iters, config.amp) == (5e-4, 500, False)
    assert GTRPoseConfig.from_kwargs(size=size, epochs=3).epochs == 3


def test_pose_scheduler_linear_warmup_then_constant():
    from types import SimpleNamespace

    from libreyolo.models.gtr.pose_trainer import GTRPoseConfig, GTRPoseTrainer

    trainer = object.__new__(GTRPoseTrainer)
    trainer.config = GTRPoseConfig.from_kwargs(size="s", epochs=10)
    type(trainer).effective_lr = property(lambda self: self.config.lr0)
    scheduler = trainer.create_scheduler(iters_per_epoch=100)
    assert scheduler.update_lr(250) == pytest.approx(2.5e-4)
    assert scheduler.update_lr(500) == pytest.approx(5e-4)
    assert scheduler.update_lr(999) == pytest.approx(5e-4)
    short = SimpleNamespace(config=GTRPoseConfig.from_kwargs(size="s", epochs=1))
    short.effective_lr = 5e-4
    assert GTRPoseTrainer.create_scheduler(short, 10).warmup_iters == 9


def test_pose_training_loss_backpropagates():
    from libreyolo.models.gtr.pose_trainer import GTRPoseConfig, GTRPoseTrainer

    torch.manual_seed(0)
    trainer = object.__new__(GTRPoseTrainer)
    trainer.config = GTRPoseConfig.from_kwargs(size="s")
    trainer.device = torch.device("cpu")
    trainer.model = LibreGTRPoseModel("s", eval_spatial_size=(160, 160)).train()
    trainer.on_setup()
    targets = torch.zeros(1, 2, 5 + 17 * 3)
    targets[0, 0, :5] = torch.tensor([0.0, 80.0, 80.0, 60.0, 90.0])
    xs = torch.linspace(60, 100, 17)
    ys = torch.linspace(40, 120, 17)
    targets[0, 0, 5:] = torch.stack([xs, ys, torch.full_like(xs, 2.0)], 1).flatten()
    losses = trainer.on_forward(torch.randn(1, 3, 160, 160), targets)
    assert torch.isfinite(losses["total_loss"])
    losses["total_loss"].backward()
    grad = trainer.model.backbone.backbone._model.blocks[0].attn.q_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_pose_export_wrapper_traces_and_matches():
    from libreyolo.models.gtr.pose import GTRPoseExportWrapper

    torch.manual_seed(0)
    model = LibreGTRPoseModel("s", eval_spatial_size=(160, 160)).eval()
    image = torch.randn(1, 3, 160, 160)
    with torch.no_grad():
        eager = model(image)
        wrapper = GTRPoseExportWrapper(model)
        traced = torch.jit.trace(wrapper, image, check_trace=False)
        logits, keypoints = traced(image)
    torch.testing.assert_close(logits, eager["pred_logits"], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(keypoints, eager["pred_keypoints"], rtol=1e-4, atol=1e-4)


def test_pose_rejects_lora_and_wrong_keypoint_counts(tmp_path):
    import yaml

    _, checkpoint = _pose_checkpoint()
    path = tmp_path / "LibreGTRs-pose.pt"
    torch.save(checkpoint, path)
    model = LibreYOLO(str(path), device="cpu")
    with pytest.raises(ValueError, match="lora"):
        model.train(data="unused.yaml", lora=True)
    data = tmp_path / "hand.yaml"
    data.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "train": "images",
                "val": "images",
                "names": ["hand"],
                "kpt_shape": [21, 3],
            }
        )
    )
    (tmp_path / "images").mkdir()
    with pytest.raises(ValueError, match="17"):
        model.train(data=str(data), epochs=1)
