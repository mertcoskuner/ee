"""A Little Is Enough (ALIE) attack of Baruch et al. (NeurIPS 2019)."""

import math

import torch
from scipy.stats import norm


def alie_z(n, m):
    """Return the largest z that keeps a perturbed update inside the majority.

    With n participants of which m are Byzantine, s = floor(n / 2 + 1) - m
    benign updates must be won over, giving z = Phi^-1((n - m - s) / (n - m)).
    """
    s = math.floor(n / 2 + 1) - m
    return float(norm.ppf((n - m - s) / (n - m)))


class ALIE:
    """Send mu - z * sigma, the benign mean shifted by z coordinate-wise stds."""

    omniscient = True

    def __init__(self, z=None):
        """Use a fixed z, or derive it from the round's client counts."""
        self.z = z

    def craft(self, benign, num_byzantine):
        """Return one malicious update per Byzantine client."""
        stacked = torch.stack(benign)
        z = (
            self.z
            if self.z is not None
            else alie_z(len(benign) + num_byzantine, num_byzantine)
        )
        update = stacked.mean(0) - z * stacked.std(0)
        return [update.clone() for _ in range(num_byzantine)]
