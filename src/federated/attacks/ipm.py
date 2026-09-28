"""Inner Product Manipulation (IPM) attack of Xie et al. (UAI 2019)."""

import torch

from .registry import FL_ATTACKS


class IPM:
    """Send -epsilon times the benign mean so the aggregate reverses direction."""

    omniscient = True

    def __init__(self, epsilon):
        """Store the attack scale epsilon."""
        self.epsilon = epsilon

    def craft(self, benign, num_byzantine):
        """Return one malicious update per Byzantine client."""
        update = -self.epsilon * torch.stack(benign).mean(0)
        return [update.clone() for _ in range(num_byzantine)]


@FL_ATTACKS.register("ipm", rank=3)
def build_ipm(fl):
    """Build IPM with the configured epsilon."""
    return IPM(fl.ipm_epsilon)
