"""Outlier detection and elimination of client updates."""

import torch

from src.utils.helper_stats import anomaly_indices

from .registry import AGGREGATORS


class OutlierRemoval:
    """Drop updates whose distance to the coordinate median is a MAD outlier.

    Each update's L2 distance to the coordinate-wise median is scored with
    the MAD anomaly index; updates farther than the median distance with an
    index above threshold are eliminated and the rest are averaged.
    """

    def __init__(self, threshold=2.0):
        """Store the anomaly-index threshold."""
        self.threshold = threshold
        self.removed = []

    def __call__(self, updates, weights):
        """Return the weighted mean of the updates that are not outliers."""
        stacked = torch.stack(updates)
        dist = (stacked - stacked.median(0).values).norm(dim=1).cpu().numpy()
        index = anomaly_indices(dist)
        median = float(sorted(dist)[len(dist) // 2])
        keep = [
            i
            for i in range(len(updates))
            if not (index[i] > self.threshold and dist[i] > median)
        ]
        self.removed = [i for i in range(len(updates)) if i not in keep]
        w = torch.tensor(
            [weights[i] for i in keep], dtype=stacked.dtype, device=stacked.device
        )
        return (stacked[keep] * w[:, None]).sum(0) / w.sum()


@AGGREGATORS.register("outlier_removal", rank=10)
def build_outlier_removal(fl, f):
    """Build MAD-based outlier elimination."""
    return OutlierRemoval(fl.outlier_threshold)
