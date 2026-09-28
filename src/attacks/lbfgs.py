"""Box-constrained L-BFGS attack of Szegedy et al. (ICLR 2014)."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from .registry import ATTACKS


def most_likely_wrong_class(model, x, y):
    """Return, per example, the highest-scoring class other than y."""
    with torch.no_grad():
        logits = model(x)
    logits[torch.arange(len(y)), y] = -float("inf")
    return logits.argmax(1)


def lbfgs(model, x, y, c=1.0, search_steps=5, max_iter=20, target=None):
    """Return L-BFGS adversarial examples, or x where the attack fails.

    For each example minimize c * ||x' - x||_2^2 + CE(model(x'), t) over
    x' in [0, 1] with L-BFGS-B, where t defaults to the most likely wrong
    class. A per-example binary search on c keeps the smallest successful
    perturbation: success raises c (stronger distance penalty), failure
    lowers it.
    """
    if target is None:
        target = most_likely_wrong_class(model, x, y)
    n, shape = len(x), x.shape
    x_nat = x.detach().double().cpu().numpy().reshape(n, -1)
    bounds = [(0.0, 1.0)] * x_nat.size
    c_now = np.full(n, float(c))
    lo, hi = np.zeros(n), np.full(n, np.inf)
    best = x.detach().clone()
    best_dist = np.full(n, np.inf)
    dtype, device = x.dtype, x.device

    def objective(flat):
        """Return the summed objective and its gradient for all examples."""
        x_adv = torch.tensor(
            flat.reshape(shape), dtype=dtype, device=device, requires_grad=True
        )
        dist = (x_adv - x).flatten(1).pow(2).sum(1)
        weights = torch.tensor(c_now, dtype=dtype, device=device)
        ce = F.cross_entropy(model(x_adv), target, reduction="none")
        total = (weights * dist + ce).sum()
        (grad,) = torch.autograd.grad(total, x_adv)
        return total.item(), grad.double().cpu().numpy().ravel()

    for _ in range(search_steps):
        result = minimize(
            objective,
            x_nat.ravel(),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter},
        )
        x_adv = torch.tensor(result.x.reshape(shape), dtype=dtype, device=device)
        with torch.no_grad():
            success = (model(x_adv).argmax(1) == target).cpu().numpy()
        dist = (x_adv - x).flatten(1).pow(2).sum(1).cpu().numpy()
        improved = success & (dist < best_dist)
        mask = torch.from_numpy(improved).to(device)
        best[mask] = x_adv[mask]
        best_dist[improved] = dist[improved]
        lo[success] = c_now[success]
        hi[~success] = c_now[~success]
        c_now = np.where(np.isinf(hi), c_now * 10, (lo + hi) / 2)
    return best.detach()


@ATTACKS.register(
    "lbfgs",
    rank=4,
    label=lambda a: f"L-BFGS (c={a.lbfgs_c}, {a.search_steps} searches)",
)
def run_lbfgs(model, x, y, a):
    """Run the L-BFGS attack with the configured search settings."""
    return lbfgs(
        model, x, y, c=a.lbfgs_c, search_steps=a.search_steps, max_iter=a.lbfgs_iters
    )
