"""Attack budgets, iteration counts, and derived step sizes."""

from dataclasses import dataclass


@dataclass
class AttackParams:
    """Store the attack selection, norm budgets, and PGD step count."""

    attack: str = "all"
    eps_linf: float = 0.3
    eps_l2: float = 2.0
    steps: int = 100

    @property
    def attacks(self) -> list[str]:
        """Expand the all selection into the three supported attack names."""
        if self.attack == "all":
            return ["fgsm", "pgd_linf", "pgd_l2"]
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
    )
