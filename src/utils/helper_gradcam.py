"""Grad-CAM heatmaps (Selvaraju et al., ICCV 2017) for convolutional models.

Grad-CAM weights each activation map A^k of a convolutional layer by the
spatial mean of d y^c / d A^k and keeps the positive part of the weighted
sum: ReLU(sum_k alpha_k^c A^k).
"""

import torch
import torch.nn.functional as F
from torch import nn


class GradCAM:
    """Compute batched Grad-CAM heatmaps for one convolutional layer."""

    def __init__(self, model, target_layer):
        """Register a hook that keeps target_layer activations."""
        self.model = model
        self.activations = None
        self._hook = target_layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        """Store the layer output from the latest forward pass."""
        self.activations = output

    def __call__(self, x, class_idx=None):
        """Return (heatmaps, classes) for a batch of images.

        Explain class_idx per example, or the predicted classes when it is
        None. Heatmaps have shape (N, H, W), are resized to the input, and
        are scaled to [0, 1] per example.
        """
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1)
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        (grads,) = torch.autograd.grad(score, self.activations)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(
            cam, size=x.shape[2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        peak = cam.flatten(1).max(1).values.view(-1, 1, 1)
        return (cam / (peak + 1e-8)).detach(), class_idx.detach()

    def remove_hooks(self):
        """Detach the forward hook from the target layer."""
        self._hook.remove()


def last_conv(model):
    """Return the model's last nn.Conv2d, the usual Grad-CAM target."""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("Grad-CAM needs a model with a convolutional layer")
    return convs[-1]
