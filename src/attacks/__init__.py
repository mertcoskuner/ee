"""Evasion attacks, registered by name in ATTACKS.

Each module defines an attack and registers a runner
run(model, x, y, attack_params) under its CLI name; every module in this
package is imported automatically.
"""

from src.utils.helper_registry import import_submodules

from .cw import cw_l2
from .fgsm import fgsm
from .lbfgs import lbfgs
from .pgd_l2 import pgd_l2
from .pgd_linf import pgd_linf
from .registry import ATTACKS

import_submodules(__name__, __path__)


def run_attack(model, x, y, name, params):
    """Return x for "clean", otherwise the registered attack's output."""
    if name == "clean":
        return x
    return ATTACKS.get(name)(model, x, y, params.attack)


def attack_label(name, params):
    """Return a display label for name, including its budget settings."""
    if name == "clean":
        return "Clean"
    return ATTACKS.meta(name, "label", lambda a: name)(params.attack)


__all__ = [
    "ATTACKS",
    "attack_label",
    "cw_l2",
    "fgsm",
    "lbfgs",
    "pgd_l2",
    "pgd_linf",
    "run_attack",
]
