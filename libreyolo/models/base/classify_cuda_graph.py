"""CUDA-graph training capture for the plain-softmax classification families.

ResNet, ConvNeXt, MobileNetV4 and EfficientNetV2 share one training step:
``logits = model(imgs)`` followed by ``F.cross_entropy(logits, targets)``.
The forward never reads the labels, so the whole backbone-plus-head is
static-shaped and capturable, and only the cross-entropy stays eager. One
mixin therefore covers the group; per-family trainers just inherit it.

Cross-entropy is cheap next to the backbone, so these families sit at the
favourable end of the capture trade-off: nearly the entire step is inside
the graph.
"""

from __future__ import annotations

import torch

from ...training.classification import classification_loss

__all__ = ["ClassifyCudaGraphMixin"]


class ClassifyCudaGraphMixin:
    """Provide ``cuda_graph_train_spec`` for cross-entropy classification.

    Mixed in ahead of ``BaseTrainer`` so it overrides the base hook, which
    returns ``None`` (eager) for families that have not opted in.
    """

    supports_class_weights = True

    def cuda_graph_train_spec(self):
        from libreyolo.training.cuda_graph import (
            CudaGraphTrainSpec,
            GraphableNetwork,
        )

        task = getattr(getattr(self, "wrapper_model", None), "task", "classify")
        if task != "classify":
            return None
        model = getattr(self, "model", None)
        if not isinstance(model, torch.nn.Module):
            return None

        network = GraphableNetwork(model)

        def assemble(flat, imgs, targets, polygons=None):
            logits = network.rebuild(flat)
            loss = classification_loss(
                logits, targets, getattr(self, "class_weights", None)
            )
            return {"total_loss": loss, "loss_ce": loss.detach()}

        return CudaGraphTrainSpec(network=network, assemble=assemble)
