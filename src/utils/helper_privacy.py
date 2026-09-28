"""DP-SGD batch sampling and the privacy budgets of DP training and DP FL."""

import torch
from torch.utils.data import DataLoader

from src.privacy import PoissonBatchSampler, dp_sgd_epsilon


def training_size(params):
    """Return the number of MNIST training images after the validation split."""
    return 60000 - params.data_loader.validation_size


def dp_sgd_rate(params):
    """Return the Poisson sampling rate batch_size / training size."""
    return params.data_loader.batch_size / training_size(params)


def training_epsilon(params, epochs=None):
    """Return the epsilon DP-SGD spends in `epochs` passes (default: all epochs)."""
    p = params.privacy
    q = dp_sgd_rate(params)
    epochs = params.training.epochs if epochs is None else epochs
    steps = epochs * max(1, round(1 / q))
    return dp_sgd_epsilon(q, p.dp_noise, steps, p.dp_delta)


def private_checkpoint(state_dict, epsilon, params):
    """Return a DP-SGD checkpoint: the weights and the budget they spent."""
    return {
        "state_dict": state_dict,
        "dp_epsilon": epsilon,
        "dp_delta": params.privacy.dp_delta,
    }


def checkpoint_budget(path):
    """Return the (epsilon, delta) stored in a DP-SGD checkpoint, or None.

    The budget is recorded at training time, so it reflects the epochs,
    batch size, and training size actually used, whatever the evaluation
    command's options are.
    """
    state = torch.load(path, map_location="cpu")
    if "dp_epsilon" not in state:
        return None
    return state["dp_epsilon"], state["dp_delta"]


def federated_epsilon(fl):
    """Return the epsilon of a DP federated run, or None without DP.

    Central DP (DP-FedAvg) protects each client's participation: a
    subsampled Gaussian mechanism with rate `participation` over `rounds`
    steps. Local DP protects each client against the server: its update
    is clipped to dp_clip, so any change of its data moves it by up to
    2 dp_clip, and noise dp_noise * dp_clip gives multiplier dp_noise / 2,
    composed over the rounds the client is expected to join.
    """
    if fl.dp == "none":
        return None
    if fl.dp == "central":
        return dp_sgd_epsilon(fl.participation, fl.dp_noise, fl.rounds, fl.dp_delta)
    rounds = max(1, round(fl.rounds * fl.participation))
    return dp_sgd_epsilon(1.0, fl.dp_noise / 2, rounds, fl.dp_delta)


def dp_loader(train_loader, params):
    """Return a DataLoader drawing Poisson-sampled batches for DP-SGD."""
    dataset = train_loader.dataset
    generator = torch.Generator().manual_seed(params.run.seed)
    sampler = PoissonBatchSampler(len(dataset), dp_sgd_rate(params), generator)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=params.data_loader.num_workers,
    )
