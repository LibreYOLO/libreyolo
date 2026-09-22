"""Grouped proposal selection adapted from RF-DETR (Apache-2.0).

Copyright (c) 2025 Roboflow, Inc.
Source: roboflow/rf-detr, 2d319776673ba840c069b243863a4cbe3a62cb58.
Modified to use LibreYOLO's head MLP and transformer assembly.
"""

from collections.abc import Sequence
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _stack_linear_params(modules: Sequence[nn.Linear]) -> tuple[Tensor, Tensor]:
    """Stack same-shaped ``nn.Linear`` modules' live parameters for a batched matmul.

    Reads ``.weight``/``.bias`` fresh on every call (never cached) so the stacked tensors always
    reflect the current, possibly just-updated, parameter values.

    Args:
        modules: Per-group ``nn.Linear`` layers sharing identical ``in_features``/``out_features``.

    Returns:
        Stacked ``(weight, bias)``, shaped ``(group, out_features, in_features)`` and
        ``(group, out_features)``.
    """
    return torch.stack([m.weight for m in modules]), torch.stack(
        [m.bias for m in modules]
    )


def _batched_group_linear(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
    """Apply ``group`` independent affine transforms via one batched matmul.

    Mathematically equivalent to calling a separate ``nn.Linear`` per group. The batched GEMM may
    use a different floating-point accumulation order, so callers compare within dtype-appropriate
    tolerance rather than requiring bit equality.

    Args:
        x: Input of shape ``(group, ..., in_features)``.
        weight: Stacked per-group weights of shape ``(group, out_features, in_features)``.
        bias: Stacked per-group biases of shape ``(group, out_features)``.

    Returns:
        Output of shape ``(group, ..., out_features)``.
    """
    leading_shape = x.shape[1:-1]
    x_flat = x.reshape(x.shape[0], -1, x.shape[-1])
    out = torch.baddbmm(bias.unsqueeze(1), x_flat, weight.transpose(1, 2))
    return out.reshape(*x.shape[:1], *leading_shape, weight.shape[-2])


def _batched_group_layer_norm(
    x: Tensor, weight: Tensor, bias: Tensor, eps: float
) -> Tensor:
    """Apply ``group`` independent ``nn.LayerNorm`` affines after one shared normalization pass.

    ``LayerNorm``'s mean/variance reduction is computed per position regardless of which group's
    affine follows it, so normalizing once (without affine) and then applying each group's own
    ``weight``/``bias`` is exactly what ``group`` separate ``nn.LayerNorm`` calls compute, just
    without ``group`` separate reduction kernels.

    Args:
        x: Input of shape ``(group, ..., channels)``.
        weight: Stacked per-group scale of shape ``(group, channels)``.
        bias: Stacked per-group shift of shape ``(group, channels)``.
        eps: Shared normalization epsilon; the eligibility guard requires equality across groups.

    Returns:
        Output of shape ``(group, ..., channels)``.
    """
    normalized = F.layer_norm(x, (x.shape[-1],), eps=eps)
    view_shape = (weight.shape[0], *([1] * (x.dim() - 2)), weight.shape[-1])
    return normalized * weight.view(view_shape) + bias.view(view_shape)


class BatchedSelection:
    def _two_stage_batching_eligible(self) -> bool:
        """Return whether the per-group modules can safely share stacked operations.

        :meth:`_two_stage_group_selection` stacks each group's own ``.weight``/``.bias`` directly instead
        of calling the module generically, so it only gives correct results for the exact
        ``nn.Linear``/``nn.LayerNorm``/:class:`~rfdetr.models.math.MLP` types ``LWDETR.__init__`` always
        deepcopies per group, each with its affine parameters present. Exact ``type(...) is ...`` checks
        (not ``isinstance``) are required: a subclass of one of these types would otherwise pass an
        ``isinstance`` check while its own overridden ``forward`` is silently bypassed, since the batched
        path never calls the module -- it only reads ``.weight``/``.bias``. The ``is not None`` checks
        reject a ``bias=False`` ``nn.Linear`` or an ``elementwise_affine=False`` ``nn.LayerNorm``, which
        would otherwise reach ``torch.stack`` over a ``None`` and crash instead of falling back. Modules
        must also agree on parameter shapes, dtypes, devices, LayerNorm epsilon, and MLP depth because one
        stacked operation cannot preserve heterogeneous group contracts. Hooks, instance-level ``forward``
        overrides, and individually compiled children also require the generic call path. A test double, a
        future custom head, or any of these edge configurations all fall back to the per-group loop in
        :meth:`forward`, preserving each module's normal call semantics.
        """
        from .lwdetr import MLP

        assert self.enc_out_class_embed is not None
        assert self.enc_out_bbox_embed is not None
        enc_output = cast(Sequence[nn.Linear], self.enc_output)
        enc_output_norm = cast(Sequence[nn.LayerNorm], self.enc_output_norm)
        class_embeds = cast(Sequence[nn.Linear], self.enc_out_class_embed)
        bbox_mlps = cast(Sequence[MLP], self.enc_out_bbox_embed)
        module_groups = (enc_output, enc_output_norm, class_embeds, bbox_mlps)
        if any(len(modules) != self.group_detr for modules in module_groups):
            return False
        if not (
            all(type(m) is nn.Linear and m.bias is not None for m in enc_output)
            and all(
                type(m) is nn.LayerNorm and m.weight is not None and m.bias is not None
                for m in enc_output_norm
            )
            and all(type(m) is nn.Linear and m.bias is not None for m in class_embeds)
            and all(
                type(m) is MLP
                and all(
                    type(layer) is nn.Linear and layer.bias is not None
                    for layer in m.layers
                )
                for m in bbox_mlps
            )
        ):
            return False

        bbox_layers = [layer for mlp in bbox_mlps for layer in mlp.layers]
        call_modules = [
            *enc_output,
            *enc_output_norm,
            *class_embeds,
            *bbox_mlps,
            *bbox_layers,
        ]
        hook_names = (
            "_forward_hooks",
            "_forward_pre_hooks",
            "_backward_hooks",
            "_backward_pre_hooks",
        )
        if any(getattr(nn.modules.module, f"_global{name}", {}) for name in hook_names):
            return False
        if any(
            "forward" in module.__dict__
            or getattr(module, "_compiled_call_impl", None) is not None
            or any(getattr(module, name, {}) for name in hook_names)
            for module in call_modules
        ):
            return False

        for modules in (enc_output, class_embeds):
            first_weight = modules[0].weight
            first_bias = cast(Tensor, modules[0].bias)
            if any(
                (m.weight.shape, m.weight.dtype, m.weight.device)
                != (first_weight.shape, first_weight.dtype, first_weight.device)
                or (
                    cast(Tensor, m.bias).shape,
                    cast(Tensor, m.bias).dtype,
                    cast(Tensor, m.bias).device,
                )
                != (first_bias.shape, first_bias.dtype, first_bias.device)
                for m in modules[1:]
            ):
                return False

        first_norm = enc_output_norm[0]
        if first_norm.normalized_shape != (self.d_model,) or any(
            m.normalized_shape != first_norm.normalized_shape
            or m.eps != first_norm.eps
            or cast(Tensor, m.weight).dtype != cast(Tensor, first_norm.weight).dtype
            or cast(Tensor, m.weight).device != cast(Tensor, first_norm.weight).device
            or cast(Tensor, m.bias).dtype != cast(Tensor, first_norm.bias).dtype
            or cast(Tensor, m.bias).device != cast(Tensor, first_norm.bias).device
            for m in enc_output_norm[1:]
        ):
            return False

        num_layers = bbox_mlps[0].num_layers
        if any(
            m.num_layers != num_layers or len(m.layers) != num_layers for m in bbox_mlps
        ):
            return False
        for layer_idx in range(num_layers):
            first_layer = cast(nn.Linear, bbox_mlps[0].layers[layer_idx])
            first_weight = first_layer.weight
            first_bias = cast(Tensor, first_layer.bias)
            if any(
                (
                    cast(nn.Linear, m.layers[layer_idx]).weight.shape,
                    cast(nn.Linear, m.layers[layer_idx]).weight.dtype,
                    cast(nn.Linear, m.layers[layer_idx]).weight.device,
                )
                != (first_weight.shape, first_weight.dtype, first_weight.device)
                or (
                    cast(Tensor, cast(nn.Linear, m.layers[layer_idx]).bias).shape,
                    cast(Tensor, cast(nn.Linear, m.layers[layer_idx]).bias).dtype,
                    cast(Tensor, cast(nn.Linear, m.layers[layer_idx]).bias).device,
                )
                != (first_bias.shape, first_bias.dtype, first_bias.device)
                for m in bbox_mlps[1:]
            ):
                return False
        return True

    def _two_stage_group_selection(
        self, output_memory: Tensor, output_proposals: Tensor, group_detr: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run every ``group_detr`` group's two-stage top-k proposal selection in one batched pass.

        Replaces a Python loop that calls each group's own ``enc_output``/``enc_output_norm``/
        ``enc_out_class_embed``/``enc_out_bbox_embed`` on the SAME ``output_memory`` -- the loop issues
        one kernel per op per group, and on a host-dispatch-bound GPU that launch count dominates the
        block's wall time far more than its (small) GEMMs do. Stacking the ``group_detr`` copies of
        each op's weights and running one batched matmul keeps every group's own weights and drops the
        per-group launch count to a constant.

        Only called for ``group_detr > 1``, which the caller (:meth:`forward`) only reaches while
        training -- the ``assert self.training`` below enforces that mechanically. Eval/export always
        pass ``group_detr=1`` and use the original single-group code path, so this method never touches
        ONNX/TorchScript tracing (#1155) or ``torch.compile`` export graphs recorded in eval mode.

        The batched GEMMs can use a different accumulation order than separate calls. The paths match
        within float32 tolerance at real model scale, but under bf16/fp16 a near-tied class score can
        cross the discrete ``topk`` boundary. Tests therefore cover float32 output/gradient parity and
        finite mixed-precision gradients, while the accompanying benchmark checks short-run training
        quality through the public API.

        Args:
            output_memory: Encoder memory of shape ``(bs, S, d_model)``, shared by every group.
            output_proposals: Encoder anchor proposals of shape ``(bs, S, 4)``, shared by every group.
            group_detr: Number of independent groups (``self.group_detr`` while training, and always
                greater than 1 for every call site).

        Returns:
            ``(refpoint_embed_ts, memory_ts, boxes_ts, cls_ts)``, matching the per-group loop's own
            ``torch.cat(parts, dim=1)`` outputs and query order (group 0's queries first, then group
            1's, ...). ``cls_ts`` is ``enc_out_class_embed``'s output at the same selected positions,
            gathered from the ranking pass rather than recomputed by the caller.
        """
        # Training-only by contract, and the contract is load-bearing: the eval/export loop in forward
        # gathers the pre-norm rows so an fp16 CoreML program keeps its Neural Engine placement, while
        # this path norms the full length and gathers post-norm. An eval or exported graph routed here
        # would still produce correct numbers, so nothing would fail -- the model would just silently
        # lose the ANE and fall back to CPU. Assert instead of trusting the caller's guard.
        assert self.training
        from .lwdetr import MLP

        assert self.enc_out_class_embed is not None
        assert self.enc_out_bbox_embed is not None
        bs = output_memory.shape[0]
        class_embeds = cast(Sequence[nn.Linear], self.enc_out_class_embed)
        bbox_mlps = cast(Sequence[MLP], self.enc_out_bbox_embed)
        topk = min(self.num_queries, output_memory.shape[-2])

        enc_output_weight, enc_output_bias = _stack_linear_params(
            cast(Sequence[nn.Linear], self.enc_output)
        )
        norm_weight = torch.stack(
            [cast(nn.LayerNorm, m).weight for m in self.enc_output_norm]
        )
        norm_bias = torch.stack(
            [cast(nn.LayerNorm, m).bias for m in self.enc_output_norm]
        )
        norm_eps = cast(nn.LayerNorm, self.enc_output_norm[0]).eps

        memory_expanded = output_memory.unsqueeze(0).expand(group_detr, -1, -1, -1)
        output_memory_all = _batched_group_linear(
            memory_expanded, enc_output_weight, enc_output_bias
        )
        output_memory_all = _batched_group_layer_norm(
            output_memory_all, norm_weight, norm_bias, norm_eps
        )

        class_weight, class_bias = _stack_linear_params(class_embeds)
        class_logits_all = _batched_group_linear(
            output_memory_all, class_weight, class_bias
        )
        # (group, bs, S) -> (group, bs, nq); torch.topk batches over every leading dim natively.
        topk_proposals_all = torch.topk(class_logits_all.max(-1)[0], topk, dim=-1)[1]

        # enc_out_class_embed is a plain per-position Linear, so gathering its already-computed
        # output at the same indices used below is exactly what re-running it on the gathered
        # hidden state would produce (Linear(x)[idx] == Linear(x[idx])) -- reuse instead of the
        # caller re-running enc_out_class_embed a second time on the gathered subset.
        cls_logits_selected = torch.gather(
            class_logits_all,
            2,
            topk_proposals_all.unsqueeze(-1).expand(
                -1, -1, -1, class_logits_all.shape[-1]
            ),
        )

        tgt_undetach_all = torch.gather(
            output_memory_all,
            2,
            topk_proposals_all.unsqueeze(-1).expand(-1, -1, -1, self.d_model),
        )
        proposals_expanded = output_proposals.unsqueeze(0).expand(
            group_detr, -1, -1, -1
        )
        # See the loop's own comment: gather before the pointwise box MLP, not after.
        output_proposals_all = torch.gather(
            proposals_expanded,
            2,
            topk_proposals_all.unsqueeze(-1).expand(-1, -1, -1, 4),
        )

        hidden = tgt_undetach_all
        num_layers = bbox_mlps[0].num_layers
        for layer_idx in range(num_layers):
            layer_weight, layer_bias = _stack_linear_params(
                cast(Sequence[nn.Linear], [mlp.layers[layer_idx] for mlp in bbox_mlps])
            )
            hidden = _batched_group_linear(hidden, layer_weight, layer_bias)
            if layer_idx < num_layers - 1:
                hidden = F.relu(hidden)
        enc_outputs_coord_delta_all = hidden

        if self.bbox_reparam:
            coord_cxcy_all = (
                enc_outputs_coord_delta_all[..., :2] * output_proposals_all[..., 2:]
                + output_proposals_all[..., :2]
            )
            coord_wh_all = (
                enc_outputs_coord_delta_all[..., 2:].exp()
                * output_proposals_all[..., 2:]
            )
            refpoint_embed_all_undetach = torch.concat(
                [coord_cxcy_all, coord_wh_all], dim=-1
            )
        else:
            refpoint_embed_all_undetach = (
                enc_outputs_coord_delta_all + output_proposals_all
            )
        refpoint_embed_all = refpoint_embed_all_undetach.detach()

        def _merge_groups(t: Tensor) -> Tensor:
            """Flatten the leading group dimension into the query dimension, group 0 first.

            Args:
                t: Tensor of shape ``(group, bs, nq, C)``.

            Returns:
                Tensor of shape ``(bs, group * nq, C)``, matching the per-group loop's own
                ``torch.cat(parts, dim=1)`` order (group 0's queries first, then group 1's, ...).
            """
            return t.permute(1, 0, 2, 3).reshape(bs, group_detr * topk, t.shape[-1])

        return (
            _merge_groups(refpoint_embed_all),
            _merge_groups(tgt_undetach_all),
            _merge_groups(refpoint_embed_all_undetach),
            _merge_groups(cls_logits_selected),
        )
