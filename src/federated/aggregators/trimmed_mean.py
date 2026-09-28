"""Coordinate-wise trimmed mean aggregation (Yin et al., ICML 2018)."""

import torch

from .registry import AGGREGATORS


class TrimmedMean:
    """Drop the b largest and b smallest values of every coordinate, then average."""

    def __init__(self, b):
        """Store the number of values trimmed from each end."""
        self.b = b

    def __call__(self, updates, weights):
        """Return the coordinate-wise trimmed mean, shrinking b if needed."""
        b = min(self.b, (len(updates) - 1) // 2)
        ordered = torch.stack(updates).sort(0).values
        return ordered[b : len(updates) - b].mean(0)


@AGGREGATORS.register(
    "trimmed_mean",
    rank=3,
    feasible=lambda n, f: (n > 2 * f, f"needs n > 2f (n={n}, f={f})"),
)
def build_trimmed_mean(fl, f):
    """Build the trimmed mean that drops f values from each end."""
    return TrimmedMean(f)
