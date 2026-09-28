"""Gaussian noise attack: send random updates."""

import torch

from .registry import FL_ATTACKS


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


@FL_ATTACKS.register("gaussian", rank=5)
def build_gaussian(fl):
    """Build the Gaussian noise attack with the configured sigma."""
    return GaussianNoise(fl.gaussian_sigma)
