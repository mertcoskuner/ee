"""Untargeted PGD with an L-infinity perturbation budget."""

import torch

from src.utils.helper_attack import input_grad


def pgd_linf(
    model, x_nat, y, eps, alpha, steps, random_start=True, init_noise=None
):
    """Return detached L-infinity PGD examples around x_nat.

    Take steps of size alpha and project onto both the radius-eps box
    and [0, 1]. init_noise overrides the uniform random start, allowing
    comparison with the official implementation using identical noise.
    """
    if init_noise is not None:
        x = (x_nat + init_noise).clamp(0, 1)
    elif random_start:
        x = (x_nat + torch.empty_like(x_nat).uniform_(-eps, eps)).clamp(0, 1)
    else:
        x = x_nat.clone()

    for _ in range(steps):
        grad = input_grad(model, x, y)
        x = x + alpha * grad.sign()
        x = torch.min(torch.max(x, x_nat - eps), x_nat + eps)
        x = x.clamp(0, 1)
    return x.detach()
