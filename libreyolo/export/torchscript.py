"""
TorchScript export implementation.

Exports PyTorch models to TorchScript format via tracing.
"""

import torch


def export_torchscript(nn_model, dummy, *, output_path: str) -> str:
    """Export a PyTorch model to TorchScript format.

    Args:
        nn_model: The PyTorch nn.Module to export.
        dummy: Dummy input tensor for tracing.
        output_path: Destination file path for the .torchscript file.

    Returns:
        The output_path string.
    """
    traced = torch.jit.trace(nn_model, dummy)
    traced.save(output_path)
    return output_path
