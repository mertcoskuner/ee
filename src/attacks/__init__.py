"""Dispatch clean inputs or untargeted attacks by name."""

from .cw import cw_l2
from .fgsm import fgsm
from .lbfgs import lbfgs
from .pgd_l2 import pgd_l2
from .pgd_linf import pgd_linf


def run_attack(model, x, y, name, params):
    """Return clean inputs or dispatch the configured untargeted attack.

    Read budgets, step sizes, and optimizer settings from params.attack.
    Raise ValueError for an unsupported attack name.
    """
    a = params.attack
    if name == "clean":
        return x
    if name == "fgsm":
        return fgsm(model, x, y, a.eps_linf)
    if name == "pgd_linf":
        return pgd_linf(model, x, y, a.eps_linf, a.alpha_linf, a.steps)
    if name == "pgd_l2":
        return pgd_l2(model, x, y, a.eps_l2, a.alpha_l2, a.steps)
    if name == "lbfgs":
        return lbfgs(
            model,
            x,
            y,
            c=a.lbfgs_c,
            search_steps=a.search_steps,
            max_iter=a.lbfgs_iters,
        )
    if name == "cw":
        return cw_l2(
            model,
            x,
            y,
            c=a.cw_c,
            kappa=a.cw_kappa,
            steps=a.cw_steps,
            lr=a.cw_lr,
            search_steps=a.search_steps,
        )
    raise ValueError(f"Unknown attack: {name}")
