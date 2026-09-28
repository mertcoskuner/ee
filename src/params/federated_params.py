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
    return FederatedParams(
        rounds=args.fl_rounds,
        clients=args.fl_clients,
        participation=args.fl_participation,
        partition=args.fl_partition,
        alpha=args.fl_alpha,
        shards_per_client=args.fl_shards_per_client,
        local=args.fl_local,
        local_steps=args.fl_local_steps,
        local_lr=args.fl_local_lr,
        local_momentum=args.fl_local_momentum,
        local_weight_decay=args.fl_local_weight_decay,
        batch_size=args.fl_batch_size,
        mu=args.fl_mu,
        kd_beta=args.fl_kd_beta,
        kd_temperature=args.fl_kd_temperature,
        server_opt=args.fl_server_opt,
        server_lr=args.fl_server_lr,
        server_momentum=args.fl_server_momentum,
        server_tau=args.fl_server_tau,
        aggregator=args.fl_aggregator,
        inner_aggregator=args.fl_inner_aggregator,
        group_size=args.fl_group_size,
        byzantine_ratio=args.fl_byzantine_ratio,
        assumed_byzantine=args.fl_assumed_byzantine,
        attack=args.fl_attack,
        alie_z=args.fl_alie_z,
        ipm_epsilon=args.fl_ipm_epsilon,
        sign_flip_scale=args.fl_sign_flip_scale,
        gaussian_sigma=args.fl_gaussian_sigma,
        cc_tau=args.fl_cc_tau,
        cc_iterations=args.fl_cc_iterations,
        gm_iterations=args.fl_gm_iterations,
        clip_norm=args.fl_clip_norm,
        outlier_threshold=args.fl_outlier_threshold,
        eval_every=args.fl_eval_every,
        dp=args.fl_dp,
        dp_clip=args.fl_dp_clip,
        dp_noise=args.fl_dp_noise,
        dp_delta=args.fl_dp_delta,
    )
