"""Parity gates for the RF-DETR b49df291 model optimizations (Apache-2.0)."""

import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from libreyolo.models.rfdetr import transformer as tr
from libreyolo.models.rfdetr.math import MLP
from libreyolo.models.rfdetr.tensors import nested_tensor_from_tensor_list

pytestmark = pytest.mark.unit


def tiny_transformer(reparam=True):
    torch.manual_seed(17)
    m = tr.Transformer(
        d_model=16,
        sa_nhead=4,
        ca_nhead=4,
        num_queries=3,
        num_decoder_layers=2,
        dim_feedforward=32,
        group_detr=3,
        two_stage=True,
        num_feature_levels=1,
        bbox_reparam=reparam,
    )
    m.enc_out_class_embed = nn.ModuleList(nn.Linear(16, 4) for _ in range(3))
    m.enc_out_bbox_embed = nn.ModuleList(MLP(16, 16, 4, 3) for _ in range(3))
    return m.double().train()


@pytest.mark.parametrize("reparam", [False, True])
def test_batched_proposals_match_original_full_token_mlp_and_gradients(reparam):
    m = tiny_transformer(reparam)
    baseline = copy.deepcopy(m)
    memory = torch.randn(2, 9, 16, dtype=torch.float64, requires_grad=True)
    reference = memory.detach().clone().requires_grad_()
    proposals = torch.rand(2, 9, 4, dtype=torch.float64).clamp(0.1, 0.9)
    assert m._two_stage_batching_eligible()
    actual = m._two_stage_group_selection(memory, proposals, 3)
    features, boxes, classes = [], [], []
    for i in range(3):
        x = baseline.enc_output_norm[i](baseline.enc_output[i](reference))
        cls = baseline.enc_out_class_embed[i](x)
        ids = cls.max(-1)[0].topk(3, dim=1).indices.unsqueeze(-1)
        delta = baseline.enc_out_bbox_embed[i](x)
        coords = (
            torch.cat(
                [
                    delta[..., :2] * proposals[..., 2:] + proposals[..., :2],
                    delta[..., 2:].exp() * proposals[..., 2:],
                ],
                -1,
            )
            if reparam
            else delta + proposals
        )
        features.append(x.gather(1, ids.expand(-1, -1, 16)))
        boxes.append(coords.gather(1, ids.expand(-1, -1, 4)))
        classes.append(cls.gather(1, ids.expand(-1, -1, 4)))
    expected = (
        torch.cat(boxes, 1).detach(),
        torch.cat(features, 1),
        torch.cat(boxes, 1),
        torch.cat(classes, 1),
    )
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    sum(x.square().sum() for x in actual[1:]).backward()
    sum(x.square().sum() for x in expected[1:]).backward()
    torch.testing.assert_close(memory.grad, reference.grad, rtol=1e-9, atol=1e-9)
    for a, b in zip(m.parameters(), baseline.parameters()):
        if a.grad is not None or b.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=1e-9, atol=1e-9)


def test_custom_proposal_heads_and_hooks_keep_module_path():
    m = tiny_transformer()
    handle = m.enc_out_bbox_embed[0].register_forward_hook(lambda *_: None)
    assert not m._two_stage_batching_eligible()
    handle.remove()
    assert m._two_stage_batching_eligible()

    class CustomLinear(nn.Linear):
        pass

    m.enc_output[0] = CustomLinear(16, 16).double()
    assert not m._two_stage_batching_eligible()


def test_real_box_head_uses_the_shared_mlp_type():
    from libreyolo.models.rfdetr.nn import LibreRFDETRModel

    model = LibreRFDETRModel(config="n", nb_classes=2)
    assert model.model.transformer._two_stage_batching_eligible()


def test_grouped_attention_forward_and_backward():
    torch.manual_seed(23)
    layer = (
        tr.TransformerDecoderLayer(
            16, 4, 4, 32, dropout=0.0, group_detr=3, num_feature_levels=1
        )
        .double()
        .train()
    )
    baseline = copy.deepcopy(layer)
    x = torch.randn(2, 12, 16, dtype=torch.float64, requires_grad=True)
    pos = torch.randn_like(x, requires_grad=True)
    y, p = x.detach().clone().requires_grad_(), pos.detach().clone().requires_grad_()
    actual = layer._grouped_self_attention(x, pos)
    q = k = torch.cat((y + p).split(4, 1), 0)
    v = torch.cat(y.split(4, 1), 0)
    expected = baseline.self_attn(q, k, v, need_weights=False)[0]
    expected = torch.cat(expected.split(2, 0), 1)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(pos.grad, p.grad, rtol=1e-9, atol=1e-9)
    for a, b in zip(layer.self_attn.parameters(), baseline.self_attn.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("order", [1, 2])
def test_linear_relu_autograd(order):
    torch.manual_seed(29)
    inputs = (
        torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True),
        torch.randn(5, 4, dtype=torch.float64, requires_grad=True),
        torch.randn(5, dtype=torch.float64, requires_grad=True),
    )
    check = torch.autograd.gradcheck if order == 1 else torch.autograd.gradgradcheck
    assert check(tr._LinearReLU.apply, inputs)
    torch.testing.assert_close(
        tr._LinearReLU.apply(*inputs), F.relu(F.linear(*inputs)), rtol=0, atol=0
    )


@pytest.mark.parametrize("coords", [2, 4])
def test_interleaved_sine_matches_portable_forward_and_backward(coords):
    torch.manual_seed(31)
    x = torch.rand(2, 5, coords, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    actual = tr._sineembed_interleaved(x, 16, None)
    expected = tr.gen_sineembed_for_position(y, 16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-12, atol=1e-12)


def test_dtype_helpers_backward():
    x = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    y = torch.randn_like(x, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a, b: tr._AddInDtype.apply(a, b, torch.float64), (x, y)
    )
    assert torch.autograd.gradcheck(
        lambda a: tr._CastThenExpand.apply(a, 3, torch.float64), (x,)
    )


def test_dense_nested_tensor_shares_storage_but_padding_does_not():
    x = torch.rand(2, 3, 32, 32, requires_grad=True)
    dense = nested_tensor_from_tensor_list(x)
    assert dense.tensors.data_ptr() == x.data_ptr() and dense.no_padding
    assert not dense.mask.any() and dense.to(torch.device("cpu")).no_padding
    dense.tensors.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    padded = nested_tensor_from_tensor_list(x.detach()[:, :, :31, :31], block_size=32)
    assert not padded.no_padding and padded.tensors.shape == x.shape
    assert padded.mask[:, 31, :].all() and padded.mask[:, :, 31].all()


def test_observed_transformer_retains_legacy_output_contract():
    from libreyolo.models.rfdetr.nn import LibreRFDETRModel

    model = LibreRFDETRModel(config="n", nb_classes=2).train()
    observed = []
    handle = model.model.transformer.register_forward_hook(lambda _m, _args, result: observed.append(len(result)))
    with torch.no_grad():
        model(torch.randn(1, 3, 64, 64))
    handle.remove()
    assert observed == [4]
