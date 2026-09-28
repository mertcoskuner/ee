"""Neural Cleanse trigger reverse-engineering (Wang et al., IEEE S&P 2019)."""

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.helper_stats import anomaly_indices


def reverse_trigger(
    model,
    x,
    target,
    steps,
    init_cost,
    lr,
    batch_size=128,
    epoch_steps=10,
    success_threshold=0.99,
    patience=5,
    multiplier=1.5,
):
    """Return the smallest (mask, pattern) that flips clean x to target.

    Stamp inputs as x' = (1 - m) * x + m * p with m and p kept in [0, 1]
    by a tanh parametrization, and minimize CE(model(x'), target) +
    cost * ||m||_1 with Adam on random minibatches. As in the original
    implementation, every epoch_steps steps the cost is multiplied by
    multiplier after patience epochs with success at least
    success_threshold and divided by multiplier ** 1.5 after patience
    epochs below it; the smallest mask reaching the threshold is kept.
    """
    shape = (1, *x.shape[1:])
    mask_raw = torch.zeros(shape, device=x.device, requires_grad=True)
    pattern_raw = torch.zeros(shape, device=x.device, requires_grad=True)
    optimizer = torch.optim.Adam([mask_raw, pattern_raw], lr=lr, betas=(0.5, 0.9))
    labels = torch.full((batch_size,), target, device=x.device)
    cost, up, down = init_cost, 0, 0
    best, best_norm = None, float("inf")
    hits = 0
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(x), (batch_size,), device=x.device)
        mask = torch.tanh(mask_raw) / 2 + 0.5
        pattern = torch.tanh(pattern_raw) / 2 + 0.5
        logits = model((1 - mask) * x[idx] + mask * pattern)
        loss = F.cross_entropy(logits, labels) + cost * mask.sum()
        grads = torch.autograd.grad(loss, [mask_raw, pattern_raw])
        mask_raw.grad, pattern_raw.grad = grads
        optimizer.step()
        hits += (logits.argmax(1) == target).sum().item()
        if step % epoch_steps:
            continue
        rate, hits = hits / (epoch_steps * batch_size), 0
        if rate >= success_threshold and mask.sum().item() < best_norm:
            best_norm = mask.sum().item()
            best = (mask.detach().clone(), pattern.detach().clone())
        up, down = (up + 1, 0) if rate >= success_threshold else (0, down + 1)
        if up >= patience:
            cost, up = cost * multiplier, 0
        elif down >= patience:
            cost, down = cost / multiplier**1.5, 0
    if best is None:
        best = (
            (torch.tanh(mask_raw) / 2 + 0.5).detach(),
            (torch.tanh(pattern_raw) / 2 + 0.5).detach(),
        )
    return best


def neural_cleanse(model, x, num_classes, steps, init_cost, lr, threshold):
    """Reverse-engineer one trigger per class and flag outlier classes.

    Return a dict with per-class mask L1 norms, attack success rates of
    the reversed triggers on x, anomaly indices, the flagged classes
    (anomaly index above threshold and norm below the median), and the
    masks and patterns. A backdoor target class needs an unusually small
    trigger, so its norm is a low outlier.
    """
    masks, patterns, norms, success = [], [], [], []
    for target in range(num_classes):
        mask, pattern = reverse_trigger(model, x, target, steps, init_cost, lr)
        with torch.no_grad():
            stamped = (1 - mask) * x + mask * pattern
            rate = (model(stamped).argmax(1) == target).float().mean().item()
        masks.append(mask.cpu())
        patterns.append(pattern.cpu())
        norms.append(mask.sum().item())
        success.append(rate)
    index = anomaly_indices(norms)
    median = float(np.median(norms))
    flagged = [
        c for c in range(num_classes) if index[c] > threshold and norms[c] < median
    ]
    return {
        "norms": norms,
        "success": success,
        "anomaly_index": index.tolist(),
        "flagged": flagged,
        "masks": masks,
        "patterns": patterns,
    }
