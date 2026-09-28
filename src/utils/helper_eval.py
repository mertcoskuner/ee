"""Batch predictions and feature extraction for attack evaluation."""

import torch
from torch import nn

from src.attacks import run_attack
from src.utils.helper_data import load_mnist_tensors


def predict_under_attack(model, x, y, name, params, device, source=None):
    """Return CPU class predictions for an attacked dataset in batches.

    Adversarial examples are crafted on source (a surrogate, for transfer
    attacks) or on model itself when source is None. Reset the PyTorch
    seed before each attack for reproducible random starts. The models are
    expected to be in evaluation mode.
    """
    torch.manual_seed(params.run.seed)
    bs, preds = params.data_loader.test_batch_size, []
    for i in range(0, len(x), bs):
        xb, yb = x[i : i + bs].to(device), y[i : i + bs].to(device)
        attacker = model if source is None else source
        x_adv = run_attack(attacker, xb, yb, name, params)
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
    """Return batched penultimate features as a NumPy array on CPU.

    The features are the inputs of the model's last linear layer, captured
    with a forward hook so any classifier ending in nn.Linear works.
    """
    head = [m for m in model.modules() if isinstance(m, nn.Linear)][-1]
    captured = []
    hook = head.register_forward_hook(
        lambda module, inputs, output: captured.append(inputs[0].cpu())
    )
    try:
        for i in range(0, len(x), batch_size):
            model(x[i : i + batch_size].to(device))
    finally:
        hook.remove()
    return torch.cat(captured).numpy()


def attack(model, x, y, name, params, device):
    """Attack inputs on the requested device and return the result on CPU."""
    return run_attack(model, x.to(device), y.to(device), name, params).cpu()


def accuracy(preds, y):
    """Return the fraction of predictions equal to the labels."""
    return preds.eq(y).float().mean().item()


def evasion_success_rate(clean_preds, adv_preds, y):
    """Return the share of correctly classified inputs the attack turns wrong.

    Only inputs the model gets right without the attack count, so the rate
    measures the attack itself rather than the model's clean errors.
    """
    correct = clean_preds.eq(y)
    if not correct.any():
        return 0.0
    return adv_preds[correct].ne(y[correct]).float().mean().item()


def first_correct_per_class(model, params, device, num_classes=10):
    """Return the first correctly classified test image of every class.

    Every class must have at least one correctly classified image.
    """
    x_all, y_all = load_mnist_tensors(params, train=False)
    ok = predict(model, x_all, device) == y_all
    idx = [int(((y_all == c) & ok).nonzero()[0]) for c in range(num_classes)]
    return x_all[idx], y_all[idx]


def clean_and_robust_accuracy(model, x, y, attack_name, params, device):
    """Return (clean, robust) accuracy of model on (x, y) under attack_name."""
    model.eval()
    clean = accuracy(predict(model, x, device), y)
    robust = accuracy(predict_under_attack(model, x, y, attack_name, params, device), y)
    return clean, robust


def per_class_accuracy(preds, y, num_classes=10):
    """Return the accuracy of preds on every class as a list."""
    return [preds[y == c].eq(c).float().mean().item() for c in range(num_classes)]
