"""Neural Cleanse trigger reverse-engineering (Wang et al., IEEE S&P 2019)."""

import numpy as np
import torch
import torch.nn.functional as F


def reverse_trigger(model, x, target, steps, lam, lr, batch_size=128):
    """Return (mask, pattern) that flips clean inputs x to class target.

    Optimize an unconstrained mask and pattern through a sigmoid so both
    stay in [0, 1], applying x' = (1 - m) * x + m * p and minimizing
    CE(model(x'), target) + lam * ||m||_1 on random minibatches of x.
    """
    shape = (1, *x.shape[1:])
    mask_raw = torch.full(shape, -3.0, device=x.device, requires_grad=True)
    pattern_raw = torch.zeros(shape, device=x.device, requires_grad=True)
    optimizer = torch.optim.Adam([mask_raw, pattern_raw], lr=lr)
    labels = torch.full((batch_size,), target, device=x.device)
    for _ in range(steps):
        idx = torch.randint(0, len(x), (batch_size,), device=x.device)
        mask, pattern = torch.sigmoid(mask_raw), torch.sigmoid(pattern_raw)
        stamped = (1 - mask) * x[idx] + mask * pattern
        loss = F.cross_entropy(model(stamped), labels) + lam * mask.sum()
        grads = torch.autograd.grad(loss, [mask_raw, pattern_raw])
        mask_raw.grad, pattern_raw.grad = grads
        optimizer.step()
    return torch.sigmoid(mask_raw).detach(), torch.sigmoid(pattern_raw).detach()


def anomaly_indices(norms):
    """Return MAD-based anomaly indices |norm - median| / (1.4826 * MAD)."""
    norms = np.asarray(norms, dtype=float)
    median = np.median(norms)
    mad = 1.4826 * np.median(np.abs(norms - median))
    return np.abs(norms - median) / (mad + 1e-12)


def neural_cleanse(model, x, num_classes, steps, lam, lr, threshold):
    """Reverse-engineer one trigger per class and flag outlier classes.

    Return a dict with per-class mask L1 norms, attack success rates of
    the reversed triggers on x, anomaly indices, the flagged classes
    (anomaly index above threshold and norm below the median), and the
    masks and patterns. A backdoor target class needs an unusually small
    trigger, so its norm is a low outlier.
    """
    masks, patterns, norms, success = [], [], [], []
    for target in range(num_classes):
        mask, pattern = reverse_trigger(model, x, target, steps, lam, lr)
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
