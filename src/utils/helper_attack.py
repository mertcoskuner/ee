"""Input gradients and per-example norms for attack updates."""

import torch
import torch.nn.functional as F


def input_grad(model, x, y):
    """Return the input gradient of summed per-example cross-entropy.

    Compute gradients on a clone of x without accumulating model
    parameter gradients; return a tensor with the same shape as x.
    """
    x = x.clone().requires_grad_(True)
    losses = F.cross_entropy(model(x), y, reduction="none")
    (grad,) = torch.autograd.grad(losses.sum(), x)
    return grad


def flat_norm(t):
    """Return per-example L2 norms shaped for broadcasting over the input."""
    return t.flatten(1).norm(dim=1).view(-1, *([1] * (t.dim() - 1)))
