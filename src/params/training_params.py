"""Optimizer, regularization, and adversarial training settings."""

from dataclasses import dataclass, field

OPTIMIZERS = ["sgd", "momentum", "adam", "adamw"]


@dataclass
class TrainingParams:
    """Store optimizers, regularization, and adversarial training settings.

    optimizer and train_attack list one or more values; each run replaces
    them with a single value. train_attack "none" trains on clean data.
    """

    epochs: int = 10
    optimizer: list[str] = field(default_factory=lambda: ["adam"])
    learning_rate: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 0.0
    l1: float = 0.0
    patience: int = 0
    train_attack: list[str] = field(default_factory=lambda: ["none"])
    train_steps: int = 40
    train_alpha: float = 0.01
    log_interval: int = 200
    track_test: bool = False
    track_samples: int = 1000


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
        train_attack=args.train_attack,
        train_steps=args.train_steps,
        train_alpha=args.train_alpha,
        log_interval=args.log_interval,
        track_test=args.track_test,
        track_samples=args.track_samples,
    )
