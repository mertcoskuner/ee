"""Attack budgets, iteration counts, and derived step sizes."""

from dataclasses import dataclass

ATTACKS = ["fgsm", "pgd_linf", "pgd_l2", "lbfgs", "cw"]


@dataclass
class AttackParams:
    """Store the attack selection, budgets, and optimizer settings."""

    attack: str = "all"
    eps_linf: float = 0.3
    eps_l2: float = 2.0
    steps: int = 100
    search_steps: int = 5
    lbfgs_c: float = 1.0
    lbfgs_iters: int = 20
    cw_c: float = 1.0
    cw_kappa: float = 0.0
    cw_steps: int = 1000
    cw_lr: float = 0.01

    @property
    def attacks(self) -> list[str]:
        """Expand the all selection into every supported attack name."""
        if self.attack == "all":
            return list(ATTACKS)
        return [self.attack]

    @property
    def alpha_linf(self) -> float:
        """Return the L-infinity PGD step size, 2.5 * eps_linf / steps."""
        return 2.5 * self.eps_linf / self.steps

    @property
    def alpha_l2(self) -> float:
        """Return the L2 PGD step size, 2.5 * eps_l2 / steps."""
        return 2.5 * self.eps_l2 / self.steps


def get_attack_params(args) -> AttackParams:
    """Build attack settings from validated CLI arguments."""
    return AttackParams(
        attack=args.attack,
        eps_linf=args.eps_linf,
        eps_l2=args.eps_l2,
        steps=args.steps,
        search_steps=args.search_steps,
        lbfgs_c=args.lbfgs_c,
        lbfgs_iters=args.lbfgs_iters,
        cw_c=args.cw_c,
        cw_kappa=args.cw_kappa,
        cw_steps=args.cw_steps,
        cw_lr=args.cw_lr,
    )
