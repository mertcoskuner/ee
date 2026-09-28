"""Optimizer and adversarial training settings."""

from dataclasses import dataclass


@dataclass
class TrainingParams:
    """Store Adam settings and the PGD adversarial training schedule."""

    epochs: int = 10
    learning_rate: float = 1e-4
    adv_train: bool = False
    train_steps: int = 40
    train_alpha: float = 0.01
    log_interval: int = 200


def get_training_params(args) -> TrainingParams:
    """Build training settings from validated CLI arguments."""
    return TrainingParams(
        epochs=args.epochs,
        learning_rate=args.lr,
        adv_train=args.adv_train,
        train_steps=args.train_steps,
        train_alpha=args.train_alpha,
        log_interval=args.log_interval,
    )
