"""Untargeted PGD with an L-infinity perturbation budget."""

import torch

from src.utils.helper_attack import input_grad

from .registry import ATTACKS


def pgd_linf(model, x_nat, y, eps, alpha, steps, random_start=True):
    """Return detached L-infinity PGD examples around x_nat.

    Take steps of size alpha and project onto both the radius-eps box
    and [0, 1], starting from uniform noise when random_start is set.
    """
    if random_start:
        x = (x_nat + torch.empty_like(x_nat).uniform_(-eps, eps)).clamp(0, 1)
    else:
        x = x_nat.clone()

    for _ in range(steps):
        grad = input_grad(model, x, y)
        x = x + alpha * grad.sign()
        x = torch.min(torch.max(x, x_nat - eps), x_nat + eps)
        x = x.clamp(0, 1)
    return x.detach()


@ATTACKS.register(
    "pgd_linf",
    rank=2,
    label=lambda a: f"PGD-linf (eps={a.eps_linf}, {a.steps} steps)",
)
def run_pgd_linf(model, x, y, a):
    """Run L-infinity PGD with the configured budget and step count."""
    return pgd_linf(model, x, y, a.eps_linf, a.alpha_linf, a.steps)
