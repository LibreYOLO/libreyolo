"""
Activations and Gradients capture class for CAM methods.

This module provides a class for extracting activations and gradients
from targeted intermediate layers in a neural network.
"""

from typing import List, Optional, Callable, Union, Tuple
import torch
import torch.nn as nn


class ActivationsAndGradients:
    """
    Class for extracting activations and registering gradients 
    from targeted intermediate layers.
    
    This is the core infrastructure for all CAM methods that need
    to capture forward activations and/or backward gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: List[nn.Module],
        reshape_transform: Optional[Callable] = None
    ) -> None:
        """
        Initialize the ActivationsAndGradients object.

        Args:
            model: The neural network model.
            target_layers: List of target layers from which to extract 
                          activations and gradients.
            reshape_transform: Optional function to transform the shape of
                              activations and gradients (e.g., for transformers).
        """
        self.model = model
        self.target_layers = target_layers
        self.reshape_transform = reshape_transform
        
        self.gradients: List[torch.Tensor] = []
        self.activations: List[torch.Tensor] = []
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        
        # Register hooks for each target layer
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self._save_activation)
            )
            # Register gradient hook via forward hook
            # (backward hooks have issues: https://github.com/pytorch/pytorch/issues/61519)
            self.handles.append(
                target_layer.register_forward_hook(self._save_gradient)
            )

    def _save_activation(
        self,
        module: nn.Module,
        input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        output: torch.Tensor
    ) -> None:
        """
        Forward hook to save the activation of the targeted layer.

        Args:
            module: The targeted layer module.
            input: The input to the targeted layer.
            output: The output activation of the targeted layer.
        """
        activation = output
        
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        
        self.activations.append(activation.cpu().detach())

    def _save_gradient(
        self,
        module: nn.Module,
        input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        output: torch.Tensor
    ) -> None:
        """
        Forward hook that registers a gradient hook on the output tensor.

        Args:
            module: The targeted layer module.
            input: The input to the targeted layer.
            output: The output activation of the targeted layer.
        """
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # Can only register hooks on tensors that require grad
            return

        def _store_grad(grad: torch.Tensor) -> None:
            """Store gradient when backward pass is computed."""
            if self.reshape_transform is not None:
                grad = self.reshape_transform(grad)
            # Gradients are computed in reverse order, so prepend
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model, capturing activations.

        Args:
            x: The input tensor.

        Returns:
            The model's output.
        """
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self) -> None:
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.release()

