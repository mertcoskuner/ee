"""Gaussian noise attack: send random updates."""

import torch


class GaussianNoise:
    """Send independent N(0, sigma^2) updates of the model's dimension."""

    omniscient = True

    def __init__(self, sigma, generator=None):
        """Store the noise scale and an optional random generator."""
        self.sigma = sigma
        self.generator = generator

    def craft(self, benign, num_byzantine):
        """Return one independent noise update per Byzantine client."""
        ref = benign[0]
        return [
            self.sigma
            * torch.randn(ref.shape, generator=self.generator).to(ref.device, ref.dtype)
            for _ in range(num_byzantine)
        ]
