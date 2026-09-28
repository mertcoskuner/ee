"""Dynamic trigger with random location and pattern (Salem et al., 2022)."""

import torch

from .registry import BACKDOORS


@BACKDOORS.register("dynamic", rank=3)
class DynamicTrigger:
    """Stamp a random binary patch at a random position, drawn per image.

    Every call draws a new pattern and location for each image, so a
    model trained on poisoned data sees a different trigger every epoch
    (dynamic trigger training) and must learn the family of triggers
    rather than one fixed patch.
    """

    def __init__(self, bp):
        """Store the patch size and create the trigger random generator."""
        self.size = bp.trigger_size
        self.generator = torch.Generator().manual_seed(bp.trigger_seed)

    def apply(self, x):
        """Return a copy of the batch x with a random patch in every image."""
        x = x.clone()
        s, n = self.size, len(x)
        rows = torch.randint(0, x.shape[-2] - s + 1, (n,), generator=self.generator)
        cols = torch.randint(0, x.shape[-1] - s + 1, (n,), generator=self.generator)
        patches = torch.randint(0, 2, (n, s, s), generator=self.generator).float()
        patches[:, 0, 0] = 1.0
        for i in range(n):
            r, c = int(rows[i]), int(cols[i])
            x[i, ..., r : r + s, c : c + s] = patches[i].to(x.device, x.dtype)
        return x
