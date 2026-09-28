"""Untargeted PGD with an L2 perturbation budget."""

import torch

from src.utils.helper_attack import flat_norm, input_grad


def pgd_l2(model, x_nat, y, eps, alpha, steps, random_start=True):
    """Return detached L2-PGD examples clipped to valid pixel values.

    Take normalized-gradient ascent steps of size alpha, projecting each
    perturbation onto the radius-eps ball around x_nat. A random start
    samples a direction on the eps-sphere before pixel clipping.
    """
    if random_start:
        r = torch.randn_like(x_nat)
        x = (x_nat + eps * r / (flat_norm(r) + 1e-10)).clamp(0, 1)
    else:
        x = x_nat.clone()

    for _ in range(steps):
        grad = input_grad(model, x, y)
        x = x + alpha * grad / (flat_norm(grad) + 1e-10)
        delta = x - x_nat
        delta = delta * torch.clamp(eps / (flat_norm(delta) + 1e-12), max=1.0)
        x = (x_nat + delta).clamp(0, 1)
    return x.detach()
