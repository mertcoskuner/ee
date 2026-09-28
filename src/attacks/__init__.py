"""Dispatch clean inputs or untargeted attacks by name."""

from .fgsm import fgsm
from .pgd_l2 import pgd_l2
from .pgd_linf import pgd_linf


def run_attack(model, x, y, name, params):
    """Return clean inputs or dispatch the configured untargeted attack.

    Read budgets and step sizes from params.attack. Raise ValueError for
    an unsupported attack name.
    """
    if name == "clean":
        return x
    if name == "fgsm":
        return fgsm(model, x, y, params.attack.eps_linf)
    if name == "pgd_linf":
        return pgd_linf(
            model,
            x,
            y,
            params.attack.eps_linf,
            params.attack.alpha_linf,
            params.attack.steps,
        )
    if name == "pgd_l2":
        return pgd_l2(
            model,
            x,
            y,
            params.attack.eps_l2,
            params.attack.alpha_l2,
            params.attack.steps,
        )
    raise ValueError(f"Unknown attack: {name}")
