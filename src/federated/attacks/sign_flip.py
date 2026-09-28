"""Sign-flipping attack: send the negated benign mean, scaled."""

import torch


class SignFlip:
    """Send -scale times the benign mean update."""

    omniscient = True

    def __init__(self, scale=1.0):
        """Store the scale applied to the flipped update."""
        self.scale = scale

    def craft(self, benign, num_byzantine):
        """Return one malicious update per Byzantine client."""
        update = -self.scale * torch.stack(benign).mean(0)
        return [update.clone() for _ in range(num_byzantine)]
