"""Batch predictions and feature extraction for attack evaluation."""

import torch

from src.attacks import run_attack


def predict_under_attack(model, x, y, name, params, device):
    """Return CPU class predictions for an attacked dataset in batches.

    Reset the PyTorch seed before each attack for reproducible random
    starts. The model is expected to be in evaluation mode.
    """
    torch.manual_seed(params.run.seed)
    bs, preds = params.data_loader.test_batch_size, []
    for i in range(0, len(x), bs):
        xb, yb = x[i : i + bs].to(device), y[i : i + bs].to(device)
        x_adv = run_attack(model, xb, yb, name, params)
        with torch.no_grad():
            preds.append(model(x_adv).argmax(1).cpu())
    return torch.cat(preds)


@torch.no_grad()
def predict(model, x, device, batch_size=1000):
    """Return CPU class predictions in batches without computing gradients."""
    return torch.cat(
        [
            model(x[i : i + batch_size].to(device)).argmax(1).cpu()
            for i in range(0, len(x), batch_size)
        ]
    )


@torch.no_grad()
def features(model, x, device, batch_size=1000):
    """Return batched penultimate features as a NumPy array on CPU."""
    return torch.cat(
        [
            model.features(x[i : i + batch_size].to(device)).cpu()
            for i in range(0, len(x), batch_size)
        ]
    ).numpy()


def attack(model, x, y, name, params, device):
    """Attack inputs on the requested device and return the result on CPU."""
    return run_attack(model, x.to(device), y.to(device), name, params).cpu()
