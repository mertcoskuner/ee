"""FedAvg aggregation: the sample-weighted mean of client updates."""

import torch


class FedAvg:
    """Average updates weighted by each client's number of samples."""

    def __call__(self, updates, weights):
        """Return sum_i w_i * u_i / sum_i w_i."""
        w = torch.tensor(weights, dtype=updates[0].dtype, device=updates[0].device)
        return (torch.stack(updates) * w[:, None]).sum(0) / w.sum()
