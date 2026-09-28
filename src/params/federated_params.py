"""Federated learning, data heterogeneity, and Byzantine settings.

List-valued fields are sweep axes: every combination of their values is
run as a separate federated experiment.
"""

from dataclasses import dataclass

from src.federated.aggregators import AGGREGATORS
from src.federated.attacks import FL_ATTACKS

from src.utils.helper_params import option

PARTITIONS = ["iid", "dirichlet", "shards"]
LOCAL_OBJECTIVES = ["plain", "fedprox", "scaffold", "kd"]
SERVER_OPTIMIZERS = ["sgd", "momentum", "nesterov", "adam"]
SWEEP_FIELDS = ["partition", "local", "server_opt", "aggregator", "attack"]


def listed(default, help, choices, allow_all=False):
    """Return a list-valued sweep option with the given default values."""
    return option(
        help=help,
        choices=choices,
        allow_all=allow_all,
        default_factory=lambda: list(default),
    )


@dataclass
class FederatedParams:
    """Federated learning: clients, local and server training, robustness."""

    rounds: int = option(30, "communication rounds", gt=0)
    clients: int = option(20, "number of clients", gt=0)
    participation: float = option(1.0, "fraction of clients per round", gt=0, le=1)
    partition: list[str] = listed(["iid"], "data split(s)", PARTITIONS)
    alpha: float = option(0.5, "Dirichlet concentration (smaller: more skewed)", gt=0)
    shards_per_client: int = option(2, "label shards per client", gt=0)
    local: list[str] = listed(["plain"], "local objective(s)", LOCAL_OBJECTIVES)
    local_steps: int = option(10, "local SGD steps per round", gt=0)
    local_lr: float = option(0.05, "local SGD learning rate", gt=0)
    local_momentum: float = option(0.0, "local SGD momentum", ge=0)
    local_weight_decay: float = option(0.0, "local L2 weight decay", ge=0)
    batch_size: int = option(32, "local minibatch size", gt=0)
    mu: float = option(0.01, "FedProx proximal coefficient", ge=0)
    kd_beta: float = option(1.0, "distillation loss weight", ge=0)
    kd_temperature: float = option(3.0, "distillation temperature", gt=0)
    server_opt: list[str] = listed(["sgd"], "server optimizer(s)", SERVER_OPTIMIZERS)
    server_lr: float = option(1.0, "server learning rate", gt=0)
    server_momentum: float = option(0.9, "FedAvgM / Nesterov momentum", ge=0)
    server_tau: float = option(1e-3, "FedAdam adaptivity epsilon", gt=0)
    aggregator: list[str] = listed(
        ["fedavg"], "aggregation rule(s)", AGGREGATORS.names, allow_all=True
    )
    inner_aggregator: str = option(
        "fedavg", "rule inside groups when group_size > 1", choices=AGGREGATORS.names
    )
    group_size: int = option(1, "clients per group for two-layer aggregation", ge=1)
    byzantine_ratio: float = option(0.0, "fraction of Byzantine clients", ge=0, lt=0.5)
    assumed_byzantine: int | None = option(
        None, "attackers robust rules assume (default: from the ratio)", ge=0
    )
    attack: list[str] = listed(
        ["none"], "Byzantine attack(s)", FL_ATTACKS.names, allow_all=True
    )
    alie_z: float | None = option(None, "fixed ALIE z (default: from counts)")
    ipm_epsilon: float = option(0.5, "IPM scale", gt=0)
    sign_flip_scale: float = option(1.0, "sign-flip scale", gt=0)
    gaussian_sigma: float = option(1.0, "Gaussian attack standard deviation", gt=0)
    cc_tau: float = option(1.0, "centered clipping radius", gt=0)
    cc_iterations: int = option(1, "centered clipping iterations", gt=0)
    gm_iterations: int = option(3, "geometric median Weiszfeld iterations", gt=0)
    clip_norm: float | None = option(
        None, "norm clipping threshold (default: median norm)", gt=0
    )
    outlier_threshold: float = option(2.0, "outlier removal MAD threshold", gt=0)
    eval_every: int = option(1, "rounds between test evaluations", gt=0)
