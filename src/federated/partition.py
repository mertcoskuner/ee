"""Split a labelled dataset across federated clients, IID or non-IID."""

import numpy as np


def iid_partition(labels, num_clients, rng):
    """Shuffle all samples and deal them into equally sized client shares."""
    return [
        np.sort(p) for p in np.array_split(rng.permutation(len(labels)), num_clients)
    ]


def dirichlet_partition(labels, num_clients, alpha, rng, min_size=10):
    """Give each client a Dirichlet(alpha) share of every class.

    Smaller alpha yields more skewed label distributions. Sampling is
    repeated until every client holds at least min_size samples.
    """
    num_classes = int(labels.max()) + 1
    for _ in range(100):
        parts = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx = rng.permutation(np.flatnonzero(labels == c))
            shares = rng.dirichlet(np.full(num_clients, alpha))
            cuts = (np.cumsum(shares) * len(idx)).astype(int)[:-1]
            for client, chunk in enumerate(np.split(idx, cuts)):
                parts[client].extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.sort(np.array(p, dtype=int)) for p in parts]
    raise ValueError("Dirichlet partition left a client with too few samples")


def shard_partition(labels, num_clients, shards_per_client, rng):
    """Sort samples by label, cut them into shards, and deal shards out.

    Each client receives shards_per_client label-sorted shards, so it sees
    only a few classes (the pathological non-IID split of McMahan et al.).
    """
    order = np.argsort(labels, kind="stable")
    shards = np.array_split(order, num_clients * shards_per_client)
    picks = rng.permutation(len(shards)).reshape(num_clients, shards_per_client)
    return [np.sort(np.concatenate([shards[s] for s in row])) for row in picks]


def partition(labels, num_clients, scheme, alpha, shards_per_client, seed):
    """Return one index array per client for the named partition scheme."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    if scheme == "iid":
        return iid_partition(labels, num_clients, rng)
    if scheme == "dirichlet":
        return dirichlet_partition(labels, num_clients, alpha, rng)
    if scheme == "shards":
        return shard_partition(labels, num_clients, shards_per_client, rng)
    raise ValueError(f"Unknown partition scheme: {scheme}")
