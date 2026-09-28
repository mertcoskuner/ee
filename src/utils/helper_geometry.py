"""Loss sweeps, boundary distances, and decision maps along input directions."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def sweep(model, x, y, direction, epsilons):
    """Return mean loss and accuracy of x + eps * direction for each eps."""
    losses, accs = [], []
    for eps in epsilons:
        logits = model((x + eps * direction).clamp(0, 1))
        losses.append(F.cross_entropy(logits, y).item())
        accs.append((logits.argmax(1) == y).float().mean().item())
    return losses, accs


@torch.no_grad()
def flip_distance(model, x, y, direction, epsilons):
    """Return, per image, the smallest positive eps that changes the prediction.

    Images whose prediction never changes within the sweep get infinity.
    """
    dist = torch.full((len(x),), float("inf"))
    for eps in sorted(e for e in epsilons if e > 0):
        wrong = model((x + eps * direction).clamp(0, 1)).argmax(1).cpu() != y.cpu()
        dist[wrong & torch.isinf(dist)] = eps
    return dist


@torch.no_grad()
def decision_map(model, x, u, v, span, steps=61):
    """Return predicted classes on the plane x + a u + b v for a, b in [-span, span]."""
    grid = torch.linspace(-span, span, steps, device=x.device)
    a, b = torch.meshgrid(grid, grid, indexing="ij")
    points = x + a.reshape(-1, 1, 1, 1) * u + b.reshape(-1, 1, 1, 1) * v
    return model(points.clamp(0, 1)).argmax(1).reshape(steps, steps).cpu().numpy()
