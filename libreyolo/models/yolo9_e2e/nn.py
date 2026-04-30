"""Neural network architecture for YOLOv9 end-to-end (NMS-free)."""

import copy
import math

import torch

from ..yolo9.nn import DDetect, LibreYOLO9Model, YOLO9_CONFIGS


class YOLO9E2EDetect(DDetect):
    """YOLOv9 detect head with a one-to-one branch for NMS-free inference.

    Training: both a dense one-to-many branch (standard TAL, topk=10) and a
    one-to-one branch (topk=1) are run in parallel, with gradients blocked on
    the backbone features fed to the one-to-one branch (detach). The dual-
    branch loss sums both sets of losses.

    Inference: only the one-to-one branch is active; top-K selection replaces
    NMS so the model can be used without any post-processing graph ops.

    Color space / normalization: RGB 0–1, same as standard YOLOv9.
    """

    def __init__(self, nc=80, ch=(), reg_max=16, stride=(), use_group=True):
        super().__init__(
            nc=nc, ch=ch, reg_max=reg_max, stride=stride, use_group=use_group
        )
        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)
        self._init_one2one_bias()

    def _init_one2one_bias(self):
        """Initialize biases for the one-to-one branch."""
        for a, b, s in zip(self.one2one_cv2, self.one2one_cv3, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / float(s)) ** 2)

    def _get_loss_fn(self, device):
        """Lazily initialize the dual-branch loss."""
        if self._loss_fn is None:
            from .loss import YOLO9E2ELoss

            self._loss_fn = YOLO9E2ELoss(
                num_classes=self.nc,
                reg_max=self.reg_max,
                strides=self.stride.tolist(),
                image_size=None,
                device=device,
            )
        return self._loss_fn

    def _forward_head(self, x, cv2, cv3):
        outputs = []
        for i in range(self.nl):
            outputs.append(torch.cat((cv2[i](x[i]), cv3[i](x[i])), 1))
        return outputs

    def _inference(self, x):
        shape = x[0].shape

        if self.export or self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                xi.transpose(0, 1) for xi in self._make_anchors(x, self.stride, 0.5)
            )
            if not self.export:
                self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = (
            self._decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        )
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y, x

    def forward(self, x, targets=None, img_size=None):
        """Run dual-branch training or single one-to-one inference."""
        if self.training:
            one2many = self._forward_head(x, self.cv2, self.cv3)
            one2one = self._forward_head(
                [xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3
            )

            if targets is not None:
                loss_fn = self._get_loss_fn(x[0].device)
                if img_size is not None:
                    loss_fn.update_anchors(list(img_size))
                return loss_fn(one2many, one2one, targets)
            return {"one2many": one2many, "one2one": one2one}

        # Inference: use only the one-to-one branch
        one2one = self._forward_head(x, self.one2one_cv2, self.one2one_cv3)
        return self._inference(one2one)


class LibreYOLO9E2EModel(LibreYOLO9Model):
    """YOLOv9 model with a one-to-one head for NMS-free inference."""

    def __init__(self, config="c", reg_max=16, nb_classes=80, img_size=640):
        super().__init__(
            config=config, reg_max=reg_max, nb_classes=nb_classes, img_size=img_size
        )
        head_channels = YOLO9_CONFIGS[config]["head_channels"]
        self.head = YOLO9E2EDetect(
            nc=nb_classes, ch=head_channels, reg_max=reg_max, stride=(8, 16, 32)
        )


__all__ = ["YOLO9E2EDetect", "LibreYOLO9E2EModel"]
