"""Evasion attack selection, budgets, iteration counts, and step sizes."""

from dataclasses import dataclass

from src.attacks import ATTACKS


@dataclass
class AttackParams:
    """Store the attack selection, budgets, optimizer settings, and surrogate.

    attack None selects every attack (test mode) or none (after defenses);
    ["none"] selects no attack, so only clean accuracy is measured. A
    surrogate_model crafts attacks on another checkpoint for transfer.
    """

    attack: list[str] | None = None
    eps_linf: float = 0.3
    eps_l2: float = 2.0
    steps: int = 100
    step_linf: float | None = None
    search_steps: int = 5
    lbfgs_c: float = 1.0
    lbfgs_iters: int = 20
    cw_c: float = 1.0
    cw_kappa: float = 0.0
    cw_steps: int = 1000
    cw_lr: float = 0.01
    square_queries: int = 1000
    autoattack_version: str = "standard"
    surrogate_model: str | None = None
    surrogate_optimizer: str = "adam"
    surrogate_train_attack: str = "none"

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


def get_attack_params(args) -> AttackParams:
    """Build attack settings from validated CLI arguments."""
    return AttackParams(
        attack=args.attack,
        eps_linf=args.eps_linf,
        eps_l2=args.eps_l2,
        steps=args.steps,
        step_linf=args.step_linf,
        search_steps=args.search_steps,
        lbfgs_c=args.lbfgs_c,
        lbfgs_iters=args.lbfgs_iters,
        cw_c=args.cw_c,
        cw_kappa=args.cw_kappa,
        cw_steps=args.cw_steps,
        cw_lr=args.cw_lr,
        square_queries=args.square_queries,
        autoattack_version=args.autoattack_version,
        surrogate_model=args.surrogate_model,
        surrogate_optimizer=args.surrogate_optimizer,
        surrogate_train_attack=args.surrogate_train_attack,
    )
