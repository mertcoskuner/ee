"""Federated learning, data heterogeneity, and Byzantine settings."""

from dataclasses import dataclass


@dataclass
class FederatedParams:
    """Store federation, local objective, server, aggregation, and attack settings."""

    rounds: int = 30
    clients: int = 20
    participation: float = 1.0
    partition: str = "iid"
    alpha: float = 0.5
    shards_per_client: int = 2
    local: str = "plain"
    local_steps: int = 10
    local_lr: float = 0.05
    local_momentum: float = 0.0
    local_weight_decay: float = 0.0
    batch_size: int = 32
    mu: float = 0.01
    kd_beta: float = 1.0
    kd_temperature: float = 3.0
    server_opt: str = "sgd"
    server_lr: float = 1.0
    server_momentum: float = 0.9
    server_tau: float = 1e-3
    aggregator: str = "fedavg"
    inner_aggregator: str = "fedavg"
    group_size: int = 1
    byzantine_ratio: float = 0.0
    assumed_byzantine: int | None = None
    attack: str = "none"
    alie_z: float | None = None
    ipm_epsilon: float = 0.5
    sign_flip_scale: float = 1.0
    gaussian_sigma: float = 1.0
    cc_tau: float = 1.0
    cc_iterations: int = 1
    gm_iterations: int = 3
    clip_norm: float | None = None
    outlier_threshold: float = 2.0
    eval_every: int = 1


def get_federated_params(args) -> FederatedParams:
    """Build federated settings from validated CLI arguments."""
    names = FederatedParams.__dataclass_fields__
    return FederatedParams(**{n: getattr(args, f"fl_{n}") for n in names})
