"""Bulyan aggregation (El Mhamdi et al., ICML 2018)."""

import torch

from .krum import krum_scores


class Bulyan:
    """Select n - 2f updates by repeated Krum, then take a trimmed mean.

    Updates are chosen one at a time, each time the lowest Krum score among
    the remaining ones; the coordinate-wise trimmed mean then drops f values
    from each end (fewer if too few updates remain).
    """

    def __init__(self, f):
        """Store the assumed number of Byzantine clients f."""
        self.f = f
        self.selected = []

    def __call__(self, updates, weights):
        """Return the Bulyan aggregate of the updates."""
        remaining = list(range(len(updates)))
        self.selected = []
        for _ in range(max(len(updates) - 2 * self.f, 1)):
            scores = krum_scores([updates[i] for i in remaining], self.f)
            self.selected.append(remaining.pop(int(scores.argmin())))
        chosen = torch.stack([updates[i] for i in self.selected]).sort(0).values
        b = min(self.f, (len(chosen) - 1) // 2)
        return chosen[b : len(chosen) - b].mean(0)
