"""Class weighting through public config, data, trainer and CLI surfaces."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from libreyolo.training.classification import classification_loss
from libreyolo.training.config import TrainConfig

pytestmark = pytest.mark.unit


def test_config_is_opt_in_and_rejects_ambiguous_values():
    assert TrainConfig().class_weights is False
    assert (
        TrainConfig.from_kwargs(class_weights=True).to_dict()["class_weights"] is True
    )
    for value in ("false", "auto", [1, 2], 1, None):
        with pytest.raises(ValueError, match="class_weights"):
            TrainConfig.from_kwargs(class_weights=value)


def test_weighted_loss_hard_soft_and_distributed_gradient_equivalence():
    logits = torch.tensor(
        [[2.0, -1.0], [-1.0, 2.0], [1.0, 0.0], [0.0, 1.0]], requires_grad=True
    )
    labels = torch.tensor([0, 1, 0, 1])
    weights = torch.tensor([5 / 9, 5.0])
    probabilities = torch.nn.functional.one_hot(labels, 2).float()
    hard = classification_loss(logits, labels, weights)
    soft = classification_loss(logits, probabilities, weights)
    expected = -(logits.log_softmax(-1) * probabilities * weights).sum(-1).mean()
    torch.testing.assert_close(hard, expected)
    torch.testing.assert_close(soft, hard)
    mixed = 0.7 * probabilities + 0.3 * probabilities.flip(0)
    torch.testing.assert_close(
        classification_loss(logits, mixed, weights),
        0.7 * hard + 0.3 * classification_loss(logits, labels.flip(0), weights),
    )
    full_gradient = torch.autograd.grad(hard, logits, retain_graph=True)[0]
    local_means = (
        sum(
            classification_loss(x, y, weights)
            for x, y in zip(logits.chunk(2), labels.chunk(2))
        )
        / 2
    )
    torch.testing.assert_close(
        torch.autograd.grad(local_means, logits)[0], full_gradient
    )
    assert torch.equal(
        classification_loss(logits, labels),
        torch.nn.functional.cross_entropy(logits, labels),
    )
    # A single-class batch must not cancel its weight through the denominator.
    torch.testing.assert_close(
        classification_loss(logits[:1], labels[:1], weights),
        classification_loss(logits[:1], labels[:1]) * weights[0],
    )


@pytest.mark.parametrize(
    "family",
    ["resnet", "convnext", "mobilenetv4", "efficientnetv2", "rfdetr", "dinov2"],
)
def test_all_image_trainers_consume_weights(family):
    import importlib

    classes = {
        "resnet": "ResNetTrainer",
        "convnext": "ConvNeXtTrainer",
        "mobilenetv4": "MobileNetV4Trainer",
        "efficientnetv2": "EfficientNetV2Trainer",
        "rfdetr": "RFDETRTrainer",
        "dinov2": "DINOv2Trainer",
    }
    trainer_cls = getattr(
        importlib.import_module(f"libreyolo.models.{family}.trainer"), classes[family]
    )
    host = object.__new__(trainer_cls)
    host.model = torch.nn.Linear(3, 2)
    host.wrapper_model = SimpleNamespace(task="classify")
    host.class_weights = torch.tensor([5 / 9, 5.0])
    images = torch.randn(4, 3)
    targets = torch.tensor([0, 0, 0, 1])
    for labels in (targets, torch.nn.functional.one_hot(targets, 2).float()):
        result = host.on_forward(images, labels)
        torch.testing.assert_close(
            result["total_loss"],
            classification_loss(host.model(images), labels, host.class_weights),
        )
    assert host.supports_class_weights
    if family not in ("rfdetr", "dinov2"):
        spec = host.cuda_graph_train_spec()
        flat = spec.network(images)
        torch.testing.assert_close(
            spec.assemble(flat, images, targets)["total_loss"],
            host.on_forward(images, targets)["total_loss"],
        )
    else:
        assert host.cuda_graph_train_spec() is None


@pytest.mark.parametrize(
    "rank,mixup,cutmix", [(None, 0.0, 0.0), (0, 1.0, 0.0), (1, 0.0, 1.0)]
)
def test_full_training_counts_and_public_keyword(
    tmp_path, monkeypatch, rank, mixup, cutmix
):
    from libreyolo import LibreResNet
    from libreyolo.models.resnet.trainer import ResNetTrainer

    for c, count in enumerate((9, 1)):
        folder = tmp_path / "data" / "train" / str(c)
        folder.mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (32, 32), color=(255 * c, 20, 20)).save(
                folder / f"{i}.png"
            )
    observed = {}

    def train(host):
        if rank is not None:
            host.is_distributed = True
            host.world_size = 2
            host.rank = rank
        host._setup_classify_data()
        observed["weights"] = host.class_weights
        observed["targets"] = host.train_loader.dataset._impl.targets
        observed["batch_labels"] = next(iter(host.train_loader))[1]
        return {}

    monkeypatch.setattr(ResNetTrainer, "train", train)
    model = LibreResNet(size="18", device="cpu")
    model.train(
        data=str(tmp_path / "data"),
        class_weights=True,
        device="cpu",
        batch=4,
        mixup=mixup,
        cutmix=cutmix,
        imgsz=32,
        workers=0,
        project=str(tmp_path),
        name="run",
    )
    torch.testing.assert_close(observed["weights"], torch.tensor([5 / 9, 5.0]))
    assert observed["targets"].count(0) == 9
    assert observed["targets"].count(1) == 1
    assert observed["batch_labels"].ndim == (2 if mixup or cutmix else 1)


def test_non_classification_rejected_before_setup():
    from libreyolo.models.resnet.trainer import ResNetTrainer

    with pytest.raises(ValueError, match="image-classification"):
        ResNetTrainer(
            model=torch.nn.Linear(2, 2),
            wrapper_model=SimpleNamespace(task="detect"),
            class_weights=True,
        )


def test_validation_loss_uses_training_weights():
    from libreyolo.models.base.classify_validation_loss import ClassifyValidationLoss

    logits = torch.tensor([[1.0, -1.0], [1.0, -1.0]])
    labels = torch.tensor([0, 1])
    weights = torch.tensor([5 / 9, 5.0])
    adapter = ClassifyValidationLoss(
        device=torch.device("cpu"), family="resnet", weights=weights
    )
    torch.testing.assert_close(
        adapter(logits, labels)["loss"], classification_loss(logits, labels, weights)
    )


def test_resume_rejects_changed_weighting_before_loading_weights(tmp_path):
    from libreyolo.models.resnet.trainer import ResNetTrainer
    from libreyolo.training.trainer import BaseTrainer

    path = tmp_path / "checkpoint.pt"
    torch.save({"config": {"class_weights": True}, "model": {}}, path)
    host = object.__new__(ResNetTrainer)
    host.device = torch.device("cpu")
    host.config = TrainConfig(class_weights=False)
    with pytest.raises(ValueError, match="saved class_weights setting"):
        BaseTrainer.resume(host, str(path))


def test_dinov2_classifier_wrapper_returns_logits_for_weighted_training():
    from libreyolo.models.dinov2.model import _DINOv2ClassifierWrapper
    from libreyolo.models.dinov2.trainer import DINOv2Trainer
    from libreyolo.models.rfdetr.nn import RFDETRClassifier

    class Backbone(torch.nn.Module):
        def forward(self, nested):
            return [SimpleNamespace(tensors=nested.tensors)]

    classifier = object.__new__(RFDETRClassifier)
    torch.nn.Module.__init__(classifier)
    classifier.backbone = Backbone()
    classifier.pool = torch.nn.AdaptiveAvgPool2d(1)
    classifier.drop = torch.nn.Identity()
    classifier.linear = torch.nn.Linear(3, 2)
    wrapper = object.__new__(_DINOv2ClassifierWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.classifier = classifier
    host = object.__new__(DINOv2Trainer)
    host.wrapper_model = SimpleNamespace(task="classify")
    host.model = wrapper.train()
    host.class_weights = torch.tensor([5 / 9, 5.0])
    images = torch.randn(4, 3, 8, 8)
    targets = torch.tensor([0, 0, 0, 1])
    result = host.on_forward(images, targets)
    result["total_loss"].backward()
    assert classifier.linear.weight.grad is not None
    torch.testing.assert_close(
        result["total_loss"],
        classification_loss(wrapper(images), targets, host.class_weights),
    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_weighted_loss_precision_and_gradients(dtype):
    logits = torch.tensor([[0.0, 1.0]], dtype=dtype, requires_grad=True)
    # Weight conversion to float16 would overflow before cross-entropy autocasts.
    weights = torch.tensor([1.0, 100000.0])
    loss = classification_loss(logits, torch.tensor([1]), weights)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
