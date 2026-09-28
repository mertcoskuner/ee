"""Federated learning, data heterogeneity, and Byzantine settings.

The list-valued fields in SWEEP_FIELDS are sweep axes: every combination
of their values is run as a separate federated experiment, with each run
holding a single value per field.
"""

from dataclasses import dataclass, field

PARTITIONS = ["iid", "dirichlet", "shards"]
LOCAL_OBJECTIVES = ["plain", "fedprox", "scaffold", "kd"]
SERVER_OPTIMIZERS = ["sgd", "momentum", "nesterov", "adam"]
DP_MODES = ["none", "central", "local"]
SWEEP_FIELDS = ["partition", "local", "server_opt", "aggregator", "attack", "dp"]


@dataclass
class FederatedParams:
    """Store federation, local objective, server, aggregation, and attack settings."""

    rounds: int = 30
    clients: int = 20
    participation: float = 1.0
    partition: list[str] = field(default_factory=lambda: ["iid"])
    alpha: float = 0.5
    shards_per_client: int = 2
    local: list[str] = field(default_factory=lambda: ["plain"])
    local_steps: int = 10
    local_lr: float = 0.05
    local_momentum: float = 0.0
    local_weight_decay: float = 0.0
    batch_size: int = 32
    mu: float = 0.01
    kd_beta: float = 1.0
    kd_temperature: float = 3.0
    server_opt: list[str] = field(default_factory=lambda: ["sgd"])
    server_lr: float = 1.0
    server_momentum: float = 0.9
    server_tau: float = 1e-3
    aggregator: list[str] = field(default_factory=lambda: ["fedavg"])
    inner_aggregator: str = "fedavg"
    group_size: int = 1
    byzantine_ratio: float = 0.0
    assumed_byzantine: int | None = None
    attack: list[str] = field(default_factory=lambda: ["none"])
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
    dp: list[str] = field(default_factory=lambda: ["none"])
    dp_clip: float = 1.0
    dp_noise: float = 1.0
    dp_delta: float = 1e-5


def get_federated_params(args) -> FederatedParams:
    """Build federated settings from validated CLI arguments."""
    names = FederatedParams.__dataclass_fields__
    return FederatedParams(**{n: getattr(args, f"fl_{n}") for n in names})
