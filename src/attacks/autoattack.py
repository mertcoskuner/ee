"""AutoAttack ensemble of parameter-free attacks (Croce and Hein, ICML 2020)."""

import torchattacks

from .registry import ATTACKS


@ATTACKS.register(
    "autoattack",
    rank=7,
    slow=True,
    label=lambda a: f"AutoAttack-{a.autoattack_version} (eps={a.eps_linf})",
)
def run_autoattack(model, x, y, a):
    """Run L-infinity AutoAttack (APGD-CE, APGD-T, FAB-T, Square for standard)."""
    attack = torchattacks.AutoAttack(
        model,
        norm="Linf",
        eps=a.eps_linf,
        version=a.autoattack_version,
        n_classes=10,
        seed=0,
    )
    return attack(x, y).detach()
