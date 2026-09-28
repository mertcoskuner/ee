"""Square Attack: query-based black-box attack (Andriushchenko et al., ECCV 2020)."""

import torchattacks

from .registry import ATTACKS


@ATTACKS.register(
    "square",
    rank=6,
    slow=True,
    black_box=True,
    label=lambda a: f"Square (eps={a.eps_linf}, {a.square_queries} queries)",
)
def run_square(model, x, y, a):
    """Run the L-infinity Square Attack, which only queries model outputs."""
    attack = torchattacks.Square(
        model, norm="Linf", eps=a.eps_linf, n_queries=a.square_queries, seed=0
    )
    return attack(x, y).detach()
