"""Parse and validate experiment arguments before loading data."""

import argparse
import math

from src.params.attack_params import AttackParams
from src.params.data_loader_params import DataLoaderParams
from src.params.run_params import RunParams
from src.params.training_params import TrainingParams


def args_parser(argv=None):
    """Parse optional CLI tokens and return a validated namespace.

    Use process arguments when argv is None. Invalid values or
    out-of-range selections terminate through
    argparse.error.
    """
    parser = argparse.ArgumentParser(
        description="Adversarial attacks (FGSM / PGD-linf / PGD-l2) on MNIST"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "both", "visualize", "tsne", "gradcam"],
        default=RunParams.mode,
    )
    parser.add_argument("--epochs", type=int, default=TrainingParams.epochs)
    parser.add_argument("--lr", type=float, default=TrainingParams.learning_rate)
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
        "--attack",
        choices=["fgsm", "pgd_linf", "pgd_l2", "all"],
        default=AttackParams.attack,
    )
    parser.add_argument("--eps_linf", type=float, default=AttackParams.eps_linf)
    parser.add_argument("--eps_l2", type=float, default=AttackParams.eps_l2)
    parser.add_argument("--steps", type=int, default=AttackParams.steps)
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
    ):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name} must be positive")
    for name in ("lr", "train_alpha", "eps_linf", "eps_l2"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f"--{name} must be finite and nonnegative")
    if args.lr == 0:
        parser.error("--lr must be positive")
    if args.num_workers < 0:
        parser.error("--num_workers must be nonnegative")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if args.validation_size >= 60000:
        parser.error("--validation_size must be smaller than 60000")
    return args
