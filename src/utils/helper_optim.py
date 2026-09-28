"""Build the torch.optim optimizer selected for training."""

import torch


def build_optimizer(model, params):
    """Return SGD, SGD with momentum, Adam, or AdamW for model parameters.

    weight_decay is an L2 penalty folded into the gradient for SGD,
    momentum, and Adam, and a decoupled decay for AdamW. Raise ValueError
    for an unknown optimizer name.
    """
    t = params.training
    kwargs = {"lr": t.learning_rate, "weight_decay": t.weight_decay}
    if t.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), **kwargs)
    if t.optimizer == "momentum":
        return torch.optim.SGD(model.parameters(), momentum=t.momentum, **kwargs)
    if t.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    if t.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), **kwargs)
    raise ValueError(f"Unknown optimizer: {t.optimizer}")
