"""Coordinate-wise trimmed mean aggregation (Yin et al., ICML 2018)."""

import torch


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
