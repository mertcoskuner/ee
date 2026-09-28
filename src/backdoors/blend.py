"""Blended-injection trigger (Chen et al., 2017)."""

import torch

from .registry import BACKDOORS


@BACKDOORS.register("blend", rank=2)
class Blend:
    """Blend every image with one fixed random pattern: (1 - a) x + a p.

    The pattern is drawn once from a seeded generator, so training and
    evaluation use the same whole-image, low-visibility trigger.
    """

    def __init__(self, bp):
        """Draw the fixed pattern and store the blend ratio."""
        generator = torch.Generator().manual_seed(bp.trigger_seed)
        self.pattern = torch.rand(1, 28, 28, generator=generator)
        self.alpha = bp.blend_alpha

    def apply(self, x):
        """Return the batch x blended with the pattern."""
        pattern = self.pattern.to(x.device, x.dtype)
        return (1 - self.alpha) * x + self.alpha * pattern
