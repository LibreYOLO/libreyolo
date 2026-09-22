"""YOLO9's PGI network boundary for optional training compilation.

Uses LibreYOLO's own backbone/neck/head interfaces. The original eager model
forward remains unchanged, including the default auxiliary-loss weighting.
"""

from torch import nn

from libreyolo.training.cuda_graph import CudaGraphTrainSpec, GraphableNetwork


class _PGITrainNetwork(nn.Module):
    """Return the main and auxiliary head maps without target assignment."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, imgs):
        model = self.model
        p3, p4, p5 = model.backbone(imgs)
        main_features = model.neck(p3, p4, p5)
        main = model.head(list(main_features))
        aux_features = model.aux(p3, p4, model.backbone.last_b5)
        auxiliary = model.aux_head(list(aux_features))
        return main, auxiliary


def pgi_compile_spec(trainer):
    """Keep both PGI losses eager and preserve the existing combined outputs."""
    from .nn import AuxNeck, Backbone9, DDetect, LibreYOLO9Model, Neck9

    model = trainer.model
    task = getattr(getattr(trainer, "wrapper_model", None), "task", "detect")
    # Derived heads/necks can have different loss contracts. This adapter is
    # exactly the stock detection forward with its optional PGI branch.
    if (
        task != "detect"
        or type(model) is not LibreYOLO9Model
        or type(model.backbone) is not Backbone9
        or type(model.neck) is not Neck9
        or type(model.head) is not DDetect
        or type(model.aux) is not AuxNeck
        or type(model.aux_head) is not DDetect
        or model.aux_weight <= 0
    ):
        return None

    network = GraphableNetwork(_PGITrainNetwork(model))

    def assemble(flat, imgs, targets, polygons=None):
        main_maps, aux_maps = network.rebuild(flat)
        image_size = [imgs.shape[3], imgs.shape[2]]
        main_loss = model.head._get_loss_fn(imgs.device)
        main_loss.update_anchors(image_size)
        main = main_loss(main_maps, targets)
        aux_loss = model.aux_head._get_loss_fn(imgs.device)
        aux_loss.update_anchors(image_size)
        auxiliary = aux_loss(aux_maps, targets)
        combined = dict(main)
        for key in (
            "total_loss",
            "box_loss",
            "dfl_loss",
            "cls_loss",
            "box",
            "dfl",
            "cls",
        ):
            if key in main and key in auxiliary:
                combined[key] = main[key] + model.aux_weight * auxiliary[key]
        return combined

    return CudaGraphTrainSpec(network=network, assemble=assemble)
