"""Coordinate-wise median aggregation (Yin et al., ICML 2018)."""

import torch


class CoordinateMedian:
    """Take the median of every coordinate independently.

    For an even number of updates the two middle values are averaged, as
    (median(x) - median(-x)) / 2.
    """

    def __call__(self, updates, weights):
        """Return the coordinate-wise median of the updates."""
        stacked = torch.stack(updates)
        return (stacked.median(0).values - (-stacked).median(0).values) / 2
