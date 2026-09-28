"""Optimizer, regularization, and adversarial training settings."""

from dataclasses import dataclass


@dataclass
class TrainingParams:
    """Store optimizer, regularization, and PGD adversarial training settings."""

    epochs: int = 10
    optimizer: str = "adam"
    learning_rate: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 0.0
    l1: float = 0.0
    patience: int = 0
    adv_train: bool = False
    train_steps: int = 40
    train_alpha: float = 0.01
    log_interval: int = 200


def get_training_params(args) -> TrainingParams:
    """Build training settings from validated CLI arguments."""
    return TrainingParams(
        epochs=args.epochs,
        optimizer=args.optimizer,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        l1=args.l1,
        patience=args.patience,
        adv_train=args.adv_train,
        train_steps=args.train_steps,
        train_alpha=args.train_alpha,
        log_interval=args.log_interval,
    )
