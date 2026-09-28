"""Krum and Multi-Krum outlier elimination (Blanchard et al., NeurIPS 2017)."""

import torch

from .registry import AGGREGATORS


def krum_scores(updates, f):
    """Return each update's summed squared distance to its n - f - 2 neighbours."""
    stacked = torch.stack(updates)
    dist = torch.cdist(stacked, stacked).pow(2)
    k = max(len(updates) - f - 2, 1)
    dist.fill_diagonal_(float("inf"))
    return dist.topk(k, dim=1, largest=False).values.sum(1)


class Krum:
    """Average the m updates with the lowest Krum scores (m = 1 is Krum)."""

    def __init__(self, f, m=1):
        """Store the assumed number of Byzantine clients f and selection size m."""
        self.f = f
        self.m = m
        self.selected = []

    def __call__(self, updates, weights):
        """Return the mean of the m most central updates."""
        m = min(self.m or len(updates) - self.f, len(updates))
        self.selected = krum_scores(updates, self.f).argsort()[:m].tolist()
        return torch.stack([updates[i] for i in self.selected]).mean(0)


def krum_feasible(n, f):
    """Return whether Krum's n >= 2f + 2 requirement holds, with a reason."""
    return n >= 2 * f + 2, f"needs n >= 2f + 2 (n={n}, f={f})"


@AGGREGATORS.register("krum", rank=4, feasible=krum_feasible)
def build_krum(fl, f):
    """Build Krum, which keeps the single most central update."""
    return Krum(f, m=1)


@AGGREGATORS.register("multi_krum", rank=5, feasible=krum_feasible)
def build_multi_krum(fl, f):
    """Build Multi-Krum, which averages the n - f most central updates."""
    return Krum(f, m=None)
