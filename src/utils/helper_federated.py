"""Create federated clients from partitioned MNIST data."""

import numpy as np
import torch

from src.federated import Client, num_byzantine, partition
from src.federated.attacks import client_class
from src.utils.helper_data import load_mnist_tensors


def build_clients(params, device):
    """Partition MNIST training data and create benign then Byzantine clients.

    The last num_byzantine clients are Byzantine and use the client class
    the attack registers (for example LabelFlipClient); attacks without
    one craft their updates on the server side.
    """
    fl = params.federated
    x, y = load_mnist_tensors(params, train=True)
    shares = partition(
        y.numpy(),
        fl.clients,
        fl.partition,
        fl.alpha,
        fl.shards_per_client,
        params.run.seed,
    )
    first_byzantine = fl.clients - num_byzantine(fl)
    clients = []
    for cid, idx in enumerate(shares):
        idx = torch.from_numpy(idx)
        cls = client_class(fl, Client) if cid >= first_byzantine else Client
        clients.append(cls(cid, x[idx], y[idx], params, device))
    return clients


def label_histogram(clients, num_classes):
    """Return a clients x classes matrix of local label counts."""
    return np.stack([np.bincount(c.y.numpy(), minlength=num_classes) for c in clients])
