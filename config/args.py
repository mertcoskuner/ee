"""Parse and validate experiment arguments before loading data."""

import argparse
import math

from models import MODELS
from src.attacks import ATTACKS
from src.backdoors import BACKDOORS
from src.defenses import DEFENSES
from src.federated.aggregators import AGGREGATORS
from src.federated.attacks import FL_ATTACKS
from src.params.attack_params import AttackParams
from src.params.backdoor_params import BackdoorParams
from src.params.data_loader_params import DataLoaderParams
from src.params.defense_params import DefenseParams
from src.params.federated_params import (
    DP_MODES,
    LOCAL_OBJECTIVES,
    PARTITIONS,
    SERVER_OPTIMIZERS,
    FederatedParams,
)
from src.params.model_params import ModelParams
from src.params.privacy_params import DP_EPSILONS, PrivacyParams
from src.params.run_params import MODES, RunParams
from src.params.training_params import OPTIMIZERS, TrainingParams


def args_parser(argv=None):
    """Parse optional CLI tokens and return a validated namespace.

    Use process arguments when argv is None. Options that take several
    values (models, optimizers, attacks, defenses, triggers, and the
    federated sweep options) run every combination. Invalid values or
    incompatible selections terminate through argparse.error.
    """
    parser = argparse.ArgumentParser(
        description="Adversarial attacks, backdoors, and Byzantine-robust "
        "federated learning on MNIST"
    )
    parser.add_argument("--mode", choices=MODES, default=RunParams.mode)
    parser.add_argument("--seed", type=int, default=RunParams.seed)
    parser.add_argument(
        "--device",
        default=RunParams.device,
        help="auto (CUDA, then MPS, then CPU) or a torch device such as cuda:1",
    )
    parser.add_argument("--results_dir", default=RunParams.results_dir)
    parser.add_argument("--checkpoint_dir", default=RunParams.checkpoint_dir)

    parser.add_argument(
        "--model", nargs="+", choices=MODELS.names() + ["all"], default=["cnn"]
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=ModelParams.dropout,
        help="dropout rate of the MLP and Transformer (the CNN has no dropout)",
    )
    parser.add_argument(
        "--hidden_sizes",
        type=int,
        nargs="+",
        default=list(ModelParams.hidden_sizes),
    )
    for name in ("patch_size", "dim", "depth", "heads", "mlp_dim"):
        parser.add_argument(f"--{name}", type=int, default=getattr(ModelParams, name))

    parser.add_argument("--data_dir", default=DataLoaderParams.data_dir)
    for name in ("batch_size", "test_batch_size", "num_workers", "validation_size"):
        parser.add_argument(
            f"--{name}", type=int, default=getattr(DataLoaderParams, name)
        )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=DataLoaderParams.num_samples,
        help="use only the first N test images in test, defense and geometry "
        "(default: all; geometry uses 500)",
    )

    parser.add_argument("--epochs", type=int, default=TrainingParams.epochs)
    parser.add_argument(
        "--optimizer", nargs="+", choices=OPTIMIZERS + ["all"], default=["adam"]
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
    parser.add_argument(
        "--train_attack",
        nargs="+",
        choices=["none"] + ATTACKS.names() + ["all"],
        default=["none"],
        help="attack(s) for adversarial training; none trains on clean data",
    )
    parser.add_argument("--train_steps", type=int, default=TrainingParams.train_steps)
    parser.add_argument("--train_alpha", type=float, default=TrainingParams.train_alpha)
    parser.add_argument("--log_interval", type=int, default=TrainingParams.log_interval)
    parser.add_argument("--track_test", action="store_true")
    parser.add_argument(
        "--track_samples", type=int, default=TrainingParams.track_samples
    )

    parser.add_argument(
        "--attack",
        nargs="+",
        choices=["none"] + ATTACKS.names() + ["all"],
        default=AttackParams.attack,
        help="attacks to evaluate (default: fgsm pgd_linf pgd_l2; all adds the "
        "slow lbfgs cw square autoattack; none tests clean accuracy only)",
    )
    parser.add_argument("--eps_linf", type=float, default=AttackParams.eps_linf)
    parser.add_argument("--eps_l2", type=float, default=AttackParams.eps_l2)
    parser.add_argument("--steps", type=int, default=AttackParams.steps)
    parser.add_argument("--step_linf", type=float, default=AttackParams.step_linf)
    parser.add_argument("--search_steps", type=int, default=AttackParams.search_steps)
    parser.add_argument("--lbfgs_c", type=float, default=AttackParams.lbfgs_c)
    parser.add_argument("--lbfgs_iters", type=int, default=AttackParams.lbfgs_iters)
    parser.add_argument("--cw_c", type=float, default=AttackParams.cw_c)
    parser.add_argument("--cw_kappa", type=float, default=AttackParams.cw_kappa)
    parser.add_argument("--cw_steps", type=int, default=AttackParams.cw_steps)
    parser.add_argument("--cw_lr", type=float, default=AttackParams.cw_lr)
    parser.add_argument(
        "--square_queries", type=int, default=AttackParams.square_queries
    )
    parser.add_argument(
        "--autoattack_version",
        choices=["standard", "plus", "rand"],
        default=AttackParams.autoattack_version,
    )
    parser.add_argument(
        "--surrogate_model",
        choices=MODELS.names(),
        default=AttackParams.surrogate_model,
        help="craft attacks on this surrogate and transfer them",
    )
    parser.add_argument(
        "--surrogate_optimizer",
        choices=OPTIMIZERS,
        default=AttackParams.surrogate_optimizer,
    )
    parser.add_argument(
        "--surrogate_train_attack",
        choices=["none"] + ATTACKS.names(),
        default=AttackParams.surrogate_train_attack,
    )

    parser.add_argument(
        "--backdoor",
        nargs="+",
        choices=["none"] + BACKDOORS.names() + ["all"],
        default=["none"],
        help="trigger(s) poisoning the training data; none trains without one",
    )
    parser.add_argument("--poison_rate", type=float, default=BackdoorParams.poison_rate)
    parser.add_argument("--target_class", type=int, default=BackdoorParams.target_class)
    parser.add_argument("--trigger_size", type=int, default=BackdoorParams.trigger_size)
    parser.add_argument("--blend_alpha", type=float, default=BackdoorParams.blend_alpha)
    parser.add_argument("--trigger_seed", type=int, default=BackdoorParams.trigger_seed)

    parser.add_argument(
        "--defense", nargs="+", choices=DEFENSES.names() + ["all"], default=["all"]
    )
    for name in (
        "nc_samples",
        "nc_steps",
        "fp_epochs",
        "fp_eval_samples",
        "ls_samples",
        "ls_components",
    ):
        parser.add_argument(f"--{name}", type=int, default=getattr(DefenseParams, name))
    for name in (
        "nc_lambda",
        "nc_lr",
        "nc_threshold",
        "fp_max_drop",
        "ls_min_fraction",
        "ls_threshold",
    ):
        parser.add_argument(
            f"--{name}", type=float, default=getattr(DefenseParams, name)
        )

    for name in (
        "rounds",
        "clients",
        "shards_per_client",
        "local_steps",
        "batch_size",
        "group_size",
        "cc_iterations",
        "gm_iterations",
        "eval_every",
    ):
        parser.add_argument(
            f"--fl_{name}", type=int, default=getattr(FederatedParams, name)
        )
    for name in (
        "participation",
        "alpha",
        "local_lr",
        "local_momentum",
        "local_weight_decay",
        "mu",
        "kd_beta",
        "kd_temperature",
        "server_lr",
        "server_momentum",
        "server_tau",
        "byzantine_ratio",
        "ipm_epsilon",
        "sign_flip_scale",
        "gaussian_sigma",
        "cc_tau",
        "outlier_threshold",
    ):
        parser.add_argument(
            f"--fl_{name}", type=float, default=getattr(FederatedParams, name)
        )
    parser.add_argument(
        "--fl_partition", nargs="+", choices=PARTITIONS, default=["iid"]
    )
    parser.add_argument(
        "--fl_local", nargs="+", choices=LOCAL_OBJECTIVES, default=["plain"]
    )
    parser.add_argument(
        "--fl_server_opt", nargs="+", choices=SERVER_OPTIMIZERS, default=["sgd"]
    )
    parser.add_argument(
        "--fl_aggregator",
        nargs="+",
        choices=AGGREGATORS.names() + ["all"],
        default=["fedavg"],
    )
    parser.add_argument(
        "--fl_inner_aggregator",
        choices=AGGREGATORS.names(),
        default=FederatedParams.inner_aggregator,
    )
    parser.add_argument(
        "--fl_attack",
        nargs="+",
        choices=FL_ATTACKS.names() + ["all"],
        default=["none"],
    )
    parser.add_argument(
        "--fl_assumed_byzantine", type=int, default=FederatedParams.assumed_byzantine
    )
    parser.add_argument("--fl_alie_z", type=float, default=FederatedParams.alie_z)
    parser.add_argument("--fl_clip_norm", type=float, default=FederatedParams.clip_norm)
    parser.add_argument(
        "--fl_dp",
        nargs="+",
        choices=DP_MODES,
        default=["none"],
        help="differential privacy: none, central (DP-FedAvg), or local",
    )
    for name in ("dp_clip", "dp_noise", "dp_delta"):
        parser.add_argument(
            f"--fl_{name}", type=float, default=getattr(FederatedParams, name)
        )

    parser.add_argument("--dp", action="store_true", help="train with DP-SGD")
    parser.add_argument("--dp_noise", type=float, default=PrivacyParams.dp_noise)
    parser.add_argument("--dp_clip", type=float, default=PrivacyParams.dp_clip)
    parser.add_argument("--dp_delta", type=float, default=PrivacyParams.dp_delta)
    parser.add_argument(
        "--mia", action="store_true", help="also run membership inference in test"
    )
    parser.add_argument("--mia_samples", type=int, default=PrivacyParams.mia_samples)
    parser.add_argument(
        "--dp_epsilons",
        type=float,
        nargs="+",
        default=list(DP_EPSILONS),
        help="privacy budgets compared in --mode dp_mechanisms",
    )
    parser.add_argument("--dp_trials", type=int, default=PrivacyParams.dp_trials)

    args = parser.parse_args(argv)
    for name in (
        "epochs",
        "batch_size",
        "test_batch_size",
        "train_steps",
        "track_samples",
        "steps",
        "num_samples",
        "validation_size",
        "log_interval",
        "search_steps",
        "lbfgs_iters",
        "cw_steps",
        "square_queries",
        "nc_samples",
        "nc_steps",
        "fp_eval_samples",
        "ls_samples",
        "ls_components",
        "trigger_size",
        "patch_size",
        "dim",
        "depth",
        "heads",
        "mlp_dim",
        "fl_rounds",
        "fl_clients",
        "fl_shards_per_client",
        "fl_local_steps",
        "fl_batch_size",
        "fl_group_size",
        "fl_cc_iterations",
        "fl_gm_iterations",
        "fl_eval_every",
        "mia_samples",
        "dp_trials",
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name} must be positive")
    if any(size <= 0 for size in args.hidden_sizes):
        parser.error("--hidden_sizes must be positive")
    for name in (
        "lr",
        "momentum",
        "weight_decay",
        "l1",
        "train_alpha",
        "eps_linf",
        "eps_l2",
        "step_linf",
        "lbfgs_c",
        "cw_c",
        "cw_kappa",
        "cw_lr",
        "nc_lambda",
        "nc_lr",
        "nc_threshold",
        "fp_max_drop",
        "ls_threshold",
        "fl_alpha",
        "fl_local_lr",
        "fl_local_momentum",
        "fl_local_weight_decay",
        "fl_mu",
        "fl_kd_beta",
        "fl_kd_temperature",
        "fl_server_lr",
        "fl_server_momentum",
        "fl_server_tau",
        "fl_ipm_epsilon",
        "fl_sign_flip_scale",
        "fl_gaussian_sigma",
        "fl_cc_tau",
        "fl_clip_norm",
        "fl_outlier_threshold",
    ):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            parser.error(f"--{name} must be finite and nonnegative")
    for name in (
        "lr",
        "lbfgs_c",
        "cw_c",
        "cw_lr",
        "nc_lambda",
        "nc_lr",
        "fl_alpha",
        "fl_local_lr",
        "fl_server_lr",
        "fl_server_tau",
        "fl_kd_temperature",
        "fl_cc_tau",
        "fl_dp_clip",
        "fl_dp_noise",
        "dp_clip",
        "dp_noise",
    ):
        if getattr(args, name) == 0:
            parser.error(f"--{name} must be positive")
    for name in ("dp_noise", "dp_clip", "fl_dp_noise", "fl_dp_clip"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name} must be finite and nonnegative")
    for name in ("dp_delta", "fl_dp_delta"):
        if not 0 < getattr(args, name) < 1:
            parser.error(f"--{name} must be in (0, 1)")
    if any(not math.isfinite(e) or e <= 0 for e in args.dp_epsilons):
        parser.error("--dp_epsilons must be positive")
    if args.dp and args.l1 > 0:
        parser.error("--dp cannot be combined with --l1 (the penalty is not clipped)")
    for name in ("num_workers", "patience", "fp_epochs", "trigger_seed"):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be nonnegative")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if args.validation_size >= 60000:
        parser.error("--validation_size must be smaller than 60000")
    if 28 % args.patch_size:
        parser.error("--patch_size must divide 28")
    if not 0 < args.poison_rate < 1:
        parser.error("--poison_rate must be in (0, 1)")
    if not 0 <= args.target_class < 10:
        parser.error("--target_class must be a digit 0-9")
    if not 0 < args.blend_alpha < 1:
        parser.error("--blend_alpha must be in (0, 1)")
    if not 0 < args.ls_min_fraction <= 0.5:
        parser.error("--ls_min_fraction must be in (0, 0.5]")
    if args.fp_max_drop > 1:
        parser.error("--fp_max_drop must be at most 1")
    if not 0 < args.fl_participation <= 1:
        parser.error("--fl_participation must be in (0, 1]")
    if not 0 <= args.fl_byzantine_ratio < 0.5:
        parser.error("--fl_byzantine_ratio must be in [0, 0.5)")
    if args.fl_assumed_byzantine is not None and args.fl_assumed_byzantine < 0:
        parser.error("--fl_assumed_byzantine must be nonnegative")
    if args.mode == "gradcam":
        flat = [
            m for m in MODELS.expand(args.model) if not MODELS.meta(m, "convolutional")
        ]
        if flat:
            parser.error(f"--mode gradcam needs convolutional models, not {flat}")
    hostile = [
        a
        for a in FL_ATTACKS.expand(args.fl_attack)
        if not FL_ATTACKS.meta(a, "benign", False)
    ]
    if hostile and round(args.fl_byzantine_ratio * args.fl_clients) == 0:
        parser.error(
            f"--fl_attack {' '.join(hostile)} needs --fl_byzantine_ratio "
            "with at least one Byzantine client"
        )
    return args
