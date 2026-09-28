"""Two-layer (hierarchical) aggregation over groups of clients."""

import torch


class TwoLayer:
    """Aggregate inside random client groups, then across the group results.

    Clients are shuffled into groups of group_size; inner aggregates each
    group (for example FedAvg at an edge server) and outer combines the
    group aggregates (for example a robust rule at the cloud). With FedAvg
    inside and a robust rule outside this is bucketing (Karimireddy et al.,
    ICLR 2022), which reduces the variance a robust rule sees under non-IID
    data.
    """

    def __init__(self, inner, outer, group_size, generator=None):
        """Store the inner and outer aggregators and the group size."""
        self.inner = inner
        self.outer = outer
        self.group_size = group_size
        self.generator = generator

    def __call__(self, updates, weights):
        """Return the outer aggregate of the per-group inner aggregates."""
        order = torch.randperm(len(updates), generator=self.generator).tolist()
        groups = [
            order[i : i + self.group_size]
            for i in range(0, len(order), self.group_size)
        ]
        group_updates = [
            self.inner([updates[i] for i in g], [weights[i] for i in g]) for g in groups
        ]
        group_weights = [sum(weights[i] for i in g) for g in groups]
        return self.outer(group_updates, group_weights)
