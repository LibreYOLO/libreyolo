"""YOLOv9 end-to-end (NMS-free) network.

:class:`YOLO9E2EHead` is the YOLOv9 detection head of
:mod:`libreyolo.models.yolo9.nn` with a second set of box/class towers that
is trained with one-to-one label assignment. Training supervises both
branches; inference decodes only the one-to-one branch, whose predictions are
reduced by a plain top-K selection instead of NMS
(:mod:`libreyolo.postprocess.yolo9_e2e`).

Provenance:
    * Towers, bias initialization, box decoding and the per-branch loss are
      the MIT YOLOv9 port in :mod:`libreyolo.models.yolo9`
      (MultimediaTechLab/YOLO, https://github.com/MultimediaTechLab/YOLO,
      commit c4cb5f6f, MIT License).
    * The dual-predictor design (a one-to-many and a one-to-one predictor
      with their own towers, losses summed, inference from the one-to-one
      predictor by top-k selection without NMS) follows DATE
      (https://github.com/YiqunChen1999/date, commit 5daf092c, Apache-2.0,
      ``date/models/heads/date.py``) and the dual label assignment idea of
      the YOLOv10 paper (arXiv:2405.14458).
    * The one-to-one towers reading detached features, the module layout and
      the checkpoint glue are LibreYOLO code.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from ..yolo9.nn import YOLO9_CONFIGS, LibreYOLO9Model, YOLO9Head, default_class_neck


class YOLO9E2EHead(YOLO9Head):
    """YOLOv9 head with an extra one-to-one branch for NMS-free inference.

    ``anchor_convs`` / ``class_convs`` are the one-to-many (dense) branch of
    :class:`YOLO9Head`. ``one_to_one_anchor_convs`` / ``one_to_one_class_convs``
    have the same structure and widths and start as copies of the dense
    towers. They read detached features, so the one-to-one loss trains only
    these towers and never the shared backbone and neck.

    Forward contract:
        * ``targets`` given: the summed two-branch loss dict of
          :class:`~libreyolo.models.yolo9_e2e.loss.YOLO9E2ELoss` (needs
          ``img_size=(W, H)``).
        * training, no targets: ``{"one_to_many": raw, "one_to_one": raw}``,
          each a list of per-level ``(B, 4 * reg_max + nc, H, W)`` maps.
        * eval: ``(decoded, raw)`` of the one-to-one branch only, in the
          :class:`YOLO9Head` format.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        num_classes: int,
        *,
        reg_max: int = 16,
        strides: Sequence[int] = (8, 16, 32),
        use_group: bool = True,
        class_neck: Optional[int] = None,
    ):
        super().__init__(
            in_channels,
            num_classes,
            reg_max=reg_max,
            strides=strides,
            use_group=use_group,
            class_neck=class_neck,
        )
        # Registered after the dense towers, so parameters, optimizer state
        # and checkpoint keys stay in dense-then-one-to-one order.
        self.one_to_one_anchor_convs = copy.deepcopy(self.anchor_convs)
        self.one_to_one_class_convs = copy.deepcopy(self.class_convs)
        self.init_bias()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_bias(self) -> None:
        """MultimediaTechLab bias init (box 1.0, class -10) on both branches."""
        super().init_bias()
        if "one_to_one_class_convs" not in self._modules:
            return  # YOLO9Head.__init__ runs this before the copies exist
        for tower in self.one_to_one_anchor_convs:
            tower[-1].bias.fill_(1.0)
        for tower in self.one_to_one_class_convs:
            tower[-1].bias.fill_(-10.0)

    def set_num_classes(self, num_classes: int) -> None:
        """Swap the final 1x1 class convs of both branches for a new class count.

        Hidden tower widths are kept; the new class biases are re-initialized
        and the cached loss is dropped. The dense convs are replaced first.
        """
        super().set_num_classes(num_classes)
        num_classes = int(num_classes)
        for tower in self.one_to_one_class_convs:
            old = tower[-1]
            new = nn.Conv2d(old.in_channels, num_classes, 1)
            tower[-1] = new.to(device=old.weight.device, dtype=old.weight.dtype)
        with torch.no_grad():
            for tower in self.one_to_one_class_convs:
                tower[-1].bias.fill_(-10.0)

    # ------------------------------------------------------------------
    # Forward pieces
    # ------------------------------------------------------------------

    def one_to_one_outputs(self, features: Sequence[Tensor]) -> List[Tensor]:
        """Raw one-to-one maps over detached features (no gradient to the neck)."""
        return self.branch_outputs(
            [feature.detach() for feature in features],
            self.one_to_one_anchor_convs,
            self.one_to_one_class_convs,
        )

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def _get_loss_fn(self, device: Union[str, torch.device]):
        """Lazily build (or rebuild on a device change) the two-branch loss."""
        device = torch.device(device)
        loss_fn = self._loss_fn
        if loss_fn is None or loss_fn.dense_loss.device != device:
            from .loss import YOLO9E2ELoss

            loss_fn = YOLO9E2ELoss(
                num_classes=self.num_classes,
                reg_max=self.reg_max,
                strides=list(self.strides),
                image_size=None,
                device=device,
            )
            self._loss_fn = loss_fn
        return loss_fn

    def forward(
        self,
        features: Sequence[Tensor],
        targets: Optional[Tensor] = None,
        img_size: Optional[Tuple[int, int]] = None,
    ):
        if targets is None and not self.training:
            # Inference needs the one-to-one branch only; the traced export
            # graph is then the plain YOLO9Head graph over these towers.
            raw = self.branch_outputs(
                features, self.one_to_one_anchor_convs, self.one_to_one_class_convs
            )
            return self.decode(raw), raw

        one_to_many = self.branch_outputs(features)
        one_to_one = self.one_to_one_outputs(features)
        if targets is not None:
            if img_size is None:
                raise ValueError(
                    "YOLO9E2EHead needs img_size=(W, H) when targets are given"
                )
            loss_fn = self._get_loss_fn(one_to_many[0].device)
            loss_fn.update_anchors(list(img_size))
            return loss_fn(one_to_many, one_to_one, targets)
        return {"one_to_many": one_to_many, "one_to_one": one_to_one}


class LibreYOLO9E2EModel(LibreYOLO9Model):
    """YOLOv9 backbone and neck with the two-branch :class:`YOLO9E2EHead`.

    Same assembly and forward contract as :class:`LibreYOLO9Model`, except
    that a training forward without targets returns the head's
    ``{"one_to_many", "one_to_one"}`` dict. LibreYOLO's E2E training does
    not attach the PGI auxiliary branch.
    """

    def __init__(self, config="c", reg_max=16, nb_classes=80, img_size=640):
        super().__init__(
            config=config, reg_max=reg_max, nb_classes=nb_classes, img_size=img_size
        )
        # Swap the standard head built by the base assembly for the
        # two-branch head. Building the standard head first keeps the random
        # draws, and so a seeded fresh model's initial weights, identical to
        # earlier LibreYOLO9E2E releases.
        head_channels = YOLO9_CONFIGS[config]["head_channels"]
        self.head = YOLO9E2EHead(
            head_channels,
            nb_classes,
            reg_max=reg_max,
            strides=(8, 16, 32),
            class_neck=default_class_neck(head_channels[0], nb_classes),
        )


__all__ = ["LibreYOLO9E2EModel", "YOLO9E2EHead"]
