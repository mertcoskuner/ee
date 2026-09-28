"""Norm-clipping sanitization of client updates before averaging."""

import torch

from .registry import AGGREGATORS


class NormClipping:
    """Scale each update to norm at most tau, then take the weighted mean.

    When tau is None the round's median update norm is used, so unusually
    large (possibly malicious) updates cannot dominate.
    """

    def __init__(self, tau=None):
        """Store a fixed clipping norm, or None for the median norm."""
        self.tau = tau

    def __call__(self, updates, weights):
        """Return the weighted mean of the clipped updates."""
        stacked = torch.stack(updates)
        norms = stacked.norm(dim=1)
        tau = self.tau if self.tau is not None else norms.median()
        clipped = stacked * (tau / norms.clamp(min=1e-12)).clamp(max=1)[:, None]
        w = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device)
        return (clipped * w[:, None]).sum(0) / w.sum()


@AGGREGATORS.register("norm_clipping", rank=9)
def build_norm_clipping(fl, f):
    """Build norm clipping with a fixed or median-norm threshold."""
    return NormClipping(fl.clip_norm)
