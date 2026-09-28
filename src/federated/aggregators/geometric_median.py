"""Robust federated aggregation with the geometric median (Pillutla et al., 2022)."""

import torch

from .registry import AGGREGATORS


class GeometricMedian:
    """Approximate the weighted geometric median with smoothed Weiszfeld steps."""

    def __init__(self, iterations=3, nu=1e-6):
        """Store the number of Weiszfeld iterations and the smoothing nu."""
        self.iterations = iterations
        self.nu = nu

    def __call__(self, updates, weights):
        """Return the (approximate) geometric median of the updates."""
        stacked = torch.stack(updates)
        alphas = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
        alphas = alphas / alphas.sum()
        z = (stacked * alphas[:, None]).sum(0)
        for _ in range(self.iterations):
            dist = (stacked - z).norm(dim=1).clamp(min=self.nu)
            betas = alphas / dist
            z = (stacked * betas[:, None]).sum(0) / betas.sum()
        return z


@AGGREGATORS.register("geometric_median", rank=8)
def build_geometric_median(fl, f):
    """Build the smoothed-Weiszfeld geometric median."""
    return GeometricMedian(fl.gm_iterations)
