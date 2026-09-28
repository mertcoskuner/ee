"""Centered clipping (Karimireddy et al., ICML 2021)."""

import torch


class CenteredClipping:
    """Clip every update's distance to a reference point, then re-centre.

    Starting from the previous round's aggregate v, iterate
    v <- v + mean_i clip_tau(u_i - v), where clip_tau scales a vector down
    to norm tau when it is longer.
    """

    def __init__(self, tau, iterations=1):
        """Store the clipping radius and the number of iterations."""
        self.tau = tau
        self.iterations = iterations
        self.reference = None

    def clip(self, v):
        """Scale v to norm at most tau."""
        norm = v.norm()
        return v if norm <= self.tau else v * (self.tau / norm)

    def __call__(self, updates, weights):
        """Return the centered-clipping aggregate and keep it as reference."""
        v = (
            self.reference
            if self.reference is not None
            else torch.zeros_like(updates[0])
        )
        for _ in range(self.iterations):
            v = v + torch.stack([self.clip(u - v) for u in updates]).mean(0)
        self.reference = v.detach().clone()
        return v
