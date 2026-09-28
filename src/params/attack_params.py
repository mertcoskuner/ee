"""Evasion attack selection, budgets, iteration counts, and step sizes."""

from dataclasses import dataclass

from src.attacks import ATTACKS

from src.utils.helper_params import option


@dataclass
class AttackParams:
    """Evasion attacks: selection, budgets, and optimizer settings."""

    attack: list[str] | None = option(
        None,
        "attacks to evaluate; none tests clean accuracy only "
        "(default: all in test mode, none after defenses)",
        choices=lambda: ["none"] + ATTACKS.names(),
        allow_all=True,
    )
    eps_linf: float = option(0.3, "L-infinity budget of FGSM and PGD-linf", ge=0)
    eps_l2: float = option(2.0, "L2 budget of PGD-l2", ge=0)
    steps: int = option(100, "PGD iterations", gt=0)
    step_linf: float | None = option(
        None, "PGD-linf step size (default: 2.5 * eps_linf / steps)", gt=0
    )
    search_steps: int = option(5, "binary-search steps of L-BFGS and CW", gt=0)
    lbfgs_c: float = option(1.0, "initial L-BFGS distance weight", gt=0)
    lbfgs_iters: int = option(20, "L-BFGS-B iterations per search step", gt=0)
    cw_c: float = option(1.0, "initial CW loss weight", gt=0)
    cw_kappa: float = option(0.0, "CW confidence margin", ge=0)
    cw_steps: int = option(1000, "CW Adam steps per search step", gt=0)
    cw_lr: float = option(0.01, "CW Adam learning rate", gt=0)

    @property
    def attacks(self) -> list[str]:
        """Return the selected attack names.

        None or "all" selects every registered attack; "none" selects no
        attack, so only clean accuracy is measured.
        """
        if self.attack is not None and "none" in self.attack:
            return []
        return ATTACKS.expand(self.attack)

    @property
    def alpha_linf(self) -> float:
        """Return step_linf, or the default step size 2.5 * eps_linf / steps."""
        if self.step_linf is not None:
            return self.step_linf
        return 2.5 * self.eps_linf / self.steps

    @property
    def alpha_l2(self) -> float:
        """Return the L2 PGD step size, 2.5 * eps_l2 / steps."""
        return 2.5 * self.eps_l2 / self.steps
