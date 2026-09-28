"""Sign-flipping attack: send the negated benign mean, scaled."""

import torch

from .registry import FL_ATTACKS


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


@FL_ATTACKS.register("sign_flip", rank=4)
def build_sign_flip(fl):
    """Build the sign-flipping attack with the configured scale."""
    return SignFlip(fl.sign_flip_scale)
