"""Untargeted Carlini-Wagner L2 attack (IEEE S&P 2017)."""

import torch

from .registry import ATTACKS


def cw_l2(model, x, y, c=1.0, kappa=0.0, steps=100, lr=0.01, search_steps=5):
    """Return Carlini-Wagner L2 adversarial examples, or x where it fails.

    Optimize w with Adam, where x' = (tanh(w) + 1) / 2 keeps pixels in
    [0, 1], to minimize ||x' - x||_2^2 + c * f(x') with the margin loss
    f(x') = max(Z_y - max_{i != y} Z_i, -kappa) on logits Z. A
    per-example binary search on c keeps the smallest successful
    perturbation: success lowers c, failure raises it.
    """
    n = len(x)
    one_hot = torch.nn.functional.one_hot(y, model(x[:1]).size(1)).bool()
    w_init = torch.atanh((2 * x - 1).clamp(-1 + 1e-6, 1 - 1e-6)).detach()
    c_now = torch.full((n,), float(c), device=x.device)
    lo = torch.zeros(n, device=x.device)
    hi = torch.full((n,), float("inf"), device=x.device)
    best = x.detach().clone()
    best_dist = torch.full((n,), float("inf"), device=x.device)

    for _ in range(search_steps):
        w = w_init.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([w], lr=lr)
        success = torch.zeros(n, dtype=torch.bool, device=x.device)
        for _ in range(steps):
            x_adv = (torch.tanh(w) + 1) / 2
            logits = model(x_adv)
            real = logits[one_hot]
            other = logits.masked_fill(one_hot, -float("inf")).max(1).values
            margin = torch.clamp(real - other, min=-kappa)
            dist = (x_adv - x).flatten(1).pow(2).sum(1)
            loss = (dist + c_now * margin).sum()
            w.grad = torch.autograd.grad(loss, w)[0]
            optimizer.step()
            with torch.no_grad():
                fooled = logits.argmax(1) != y
                improved = fooled & (dist < best_dist)
                best[improved] = x_adv[improved].detach()
                best_dist[improved] = dist[improved].detach()
                success |= fooled
        hi[success] = torch.minimum(hi[success], c_now[success])
        lo[~success] = torch.maximum(lo[~success], c_now[~success])
        c_now = torch.where(torch.isinf(hi), c_now * 10, (lo + hi) / 2)
    return best.detach()


@ATTACKS.register(
    "cw",
    rank=5,
    label=lambda a: f"CW-l2 (c={a.cw_c}, kappa={a.cw_kappa}, {a.cw_steps} steps)",
)
def run_cw(model, x, y, a):
    """Run the Carlini-Wagner L2 attack with the configured settings."""
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
