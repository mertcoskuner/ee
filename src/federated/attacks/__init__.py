"""Byzantine attacks on federated learning: label flip, ALIE, IPM, sign flip, noise."""

from .alie import ALIE, alie_z
from .gaussian import GaussianNoise
from .ipm import IPM
from .label_flip import LabelFlipClient
from .sign_flip import SignFlip

FL_ATTACKS = ["none", "label_flip", "alie", "ipm", "sign_flip", "gaussian"]


def build_attack(fl):
    """Return the omniscient attack object for fl.attack, or None.

    Label flipping is a data-poisoning attack handled by LabelFlipClient,
    so it (like "none") has no update-crafting object.
    """
    if fl.attack in ("none", "label_flip"):
        return None
    if fl.attack == "alie":
        return ALIE(fl.alie_z)
    if fl.attack == "ipm":
        return IPM(fl.ipm_epsilon)
    if fl.attack == "sign_flip":
        return SignFlip(fl.sign_flip_scale)
    if fl.attack == "gaussian":
        return GaussianNoise(fl.gaussian_sigma)
    raise ValueError(f"Unknown federated attack: {fl.attack}")


__all__ = [
    "ALIE",
    "FL_ATTACKS",
    "GaussianNoise",
    "IPM",
    "LabelFlipClient",
    "SignFlip",
    "alie_z",
    "build_attack",
]
