"""Byzantine attacks on federated learning, registered by name in FL_ATTACKS.

Each module registers a builder build(fl_params) returning an object whose
craft(benign_updates, num_byzantine) returns the malicious updates, or None
for attacks that act through training (registered with client_class, such
as label flipping) and for "none". Every module is imported automatically.
"""

from src.utils.helper_registry import import_submodules

from .alie import ALIE, alie_z
from .gaussian import GaussianNoise
from .ipm import IPM
from .label_flip import LabelFlipClient
from .registry import FL_ATTACKS
from .sign_flip import SignFlip

import_submodules(__name__, __path__)


def build_attack(fl):
    """Return the update-crafting object for fl.attack, or None."""
    return FL_ATTACKS.get(fl.attack)(fl)


def client_class(fl, default):
    """Return the client class Byzantine clients use under fl.attack."""
    return FL_ATTACKS.meta(fl.attack, "client_class", default)


__all__ = [
    "ALIE",
    "FL_ATTACKS",
    "GaussianNoise",
    "IPM",
    "LabelFlipClient",
    "SignFlip",
    "alie_z",
    "build_attack",
    "client_class",
]
