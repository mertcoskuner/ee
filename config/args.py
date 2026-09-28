"""Parse and validate experiment arguments before loading data."""

import argparse
import math

from src.params.attack_params import ATTACKS, AttackParams
from src.params.data_loader_params import DataLoaderParams
from src.params.defense_params import DEFENSES, DefenseParams
from src.params.federated_params import FederatedParams
from src.params.model_params import ModelParams
from src.params.run_params import RunParams
from src.params.training_params import TrainingParams

MODES = [
    "train",
    "test",
    "both",
    "visualize",
    "tsne",
    "gradcam",
    "defense",
    "federated",
]
FL_CHOICES = {
    "partition": ["iid", "dirichlet", "shards"],
    "local": ["plain", "fedprox", "scaffold", "kd"],
    "server_opt": ["sgd", "momentum", "nesterov", "adam"],
    "aggregator": [
        "fedavg",
        "median",
        "trimmed_mean",
        "krum",
        "multi_krum",
        "bulyan",
        "centered_clipping",
        "geometric_median",
        "norm_clipping",
        "outlier_removal",
    ],
    "attack": ["none", "label_flip", "alie", "ipm", "sign_flip", "gaussian"],
}
FL_CHOICES["inner_aggregator"] = FL_CHOICES["aggregator"]


def add_federated_args(parser):
    """Add one --fl_<name> option per FederatedParams field.

    Types and defaults come from the dataclass; fields with a fixed set of
    values get argparse choices, and optional fields accept a number.
    """
    for name, field in FederatedParams.__dataclass_fields__.items():
        default = field.default
        kind = {"int | None": int, "float | None": float}.get(str(field.type))
        kind = kind or type(default)
        parser.add_argument(
            f"--fl_{name}", type=kind, default=default, choices=FL_CHOICES.get(name)
        )


def validate_federated_args(parser, args):
    """Reject inconsistent federated settings through parser.error."""
    for name in ("rounds", "clients", "local_steps", "batch_size", "eval_every"):
        if getattr(args, f"fl_{name}") <= 0:
            parser.error(f"--fl_{name} must be positive")
    if not 0 < args.fl_participation <= 1:
        parser.error("--fl_participation must be in (0, 1]")
    if not 0 <= args.fl_byzantine_ratio < 0.5:
        parser.error("--fl_byzantine_ratio must be in [0, 0.5)")
    if (
        args.fl_attack != "none"
        and round(args.fl_byzantine_ratio * args.fl_clients) == 0
    ):
        parser.error("--fl_attack needs --fl_byzantine_ratio with at least one client")
    for name in (
        "alpha",
        "local_lr",
        "server_lr",
        "server_tau",
        "kd_temperature",
        "cc_tau",
    ):
        if getattr(args, f"fl_{name}") <= 0:
            parser.error(f"--fl_{name} must be positive")
    if args.fl_group_size < 1:
        parser.error("--fl_group_size must be at least 1")


def args_parser(argv=None):
    """Parse optional CLI tokens and return a validated namespace.

    Use process arguments when argv is None. Invalid values or
    incompatible selections terminate through argparse.error.
    """
    parser = argparse.ArgumentParser(
        description="Adversarial attacks and backdoor defenses on MNIST"
    )
    parser.add_argument("--mode", choices=MODES, default=RunParams.mode)
    parser.add_argument(
        "--model", choices=["cnn", "mlp", "transformer"], default=ModelParams.model
    )
    parser.add_argument("--dropout", type=float, default=ModelParams.dropout)
    parser.add_argument("--epochs", type=int, default=TrainingParams.epochs)
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "momentum", "adam", "adamw"],
        default=TrainingParams.optimizer,
    )
    parser.add_argument("--lr", type=float, default=TrainingParams.learning_rate)
    parser.add_argument("--momentum", type=float, default=TrainingParams.momentum)
    parser.add_argument(
        "--weight_decay", type=float, default=TrainingParams.weight_decay
    )
    parser.add_argument("--l1", type=float, default=TrainingParams.l1)
    parser.add_argument(
        "--patience",
        type=int,
        default=TrainingParams.patience,
        help="early-stopping patience in epochs (0 disables it)",
    )
    parser.add_argument("--device", default=RunParams.device)
    parser.add_argument("--seed", type=int, default=RunParams.seed)
    parser.add_argument("--data_dir", default=DataLoaderParams.data_dir)
    parser.add_argument("--results_dir", default=RunParams.results_dir)
    for name in (
        "batch_size",
        "test_batch_size",
        "num_workers",
        "num_samples",
        "validation_size",
    ):
        parser.add_argument(
            f"--{name}", type=int, default=getattr(DataLoaderParams, name)
        )
    parser.add_argument("--adv_train", action="store_true")
    parser.add_argument("--train_steps", type=int, default=TrainingParams.train_steps)
    parser.add_argument("--train_alpha", type=float, default=TrainingParams.train_alpha)
    parser.add_argument("--log_interval", type=int, default=TrainingParams.log_interval)
    parser.add_argument(
        "--attack", choices=ATTACKS + ["all"], default=AttackParams.attack
    )
    parser.add_argument("--eps_linf", type=float, default=AttackParams.eps_linf)
    parser.add_argument("--eps_l2", type=float, default=AttackParams.eps_l2)
    parser.add_argument("--steps", type=int, default=AttackParams.steps)
    parser.add_argument("--search_steps", type=int, default=AttackParams.search_steps)
    parser.add_argument("--lbfgs_c", type=float, default=AttackParams.lbfgs_c)
    parser.add_argument("--lbfgs_iters", type=int, default=AttackParams.lbfgs_iters)
    parser.add_argument("--cw_c", type=float, default=AttackParams.cw_c)
    parser.add_argument("--cw_kappa", type=float, default=AttackParams.cw_kappa)
    parser.add_argument("--cw_steps", type=int, default=AttackParams.cw_steps)
    parser.add_argument("--cw_lr", type=float, default=AttackParams.cw_lr)
    parser.add_argument(
        "--defense", choices=DEFENSES + ["all"], default=DefenseParams.defense
    )
    parser.add_argument("--nc_steps", type=int, default=DefenseParams.nc_steps)
    parser.add_argument("--nc_lambda", type=float, default=DefenseParams.nc_lambda)
    parser.add_argument("--fp_max_drop", type=float, default=DefenseParams.fp_max_drop)
    parser.add_argument("--fp_epochs", type=int, default=DefenseParams.fp_epochs)
    parser.add_argument("--ls_samples", type=int, default=DefenseParams.ls_samples)
    add_federated_args(parser)
    args = parser.parse_args(argv)
    for name in (
        "epochs",
        "batch_size",
        "test_batch_size",
        "train_steps",
        "steps",
        "num_samples",
        "validation_size",
        "log_interval",
        "search_steps",
        "lbfgs_iters",
        "cw_steps",
        "nc_steps",
        "ls_samples",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name} must be positive")
    for name in (
        "lr",
        "train_alpha",
        "eps_linf",
        "eps_l2",
        "momentum",
        "weight_decay",
        "l1",
        "lbfgs_c",
        "cw_c",
        "cw_kappa",
        "cw_lr",
        "nc_lambda",
        "fp_max_drop",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name} must be finite and nonnegative")
    for name in ("lr", "lbfgs_c", "cw_c", "cw_lr"):
        if getattr(args, name) == 0:
            parser.error(f"--{name} must be positive")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    for name in ("num_workers", "patience", "fp_epochs"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be nonnegative")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if args.validation_size >= 60000:
        parser.error("--validation_size must be smaller than 60000")
    if args.mode == "gradcam" and args.model != "cnn":
        parser.error("--mode gradcam requires the convolutional --model cnn")
    validate_federated_args(parser, args)
    return args
