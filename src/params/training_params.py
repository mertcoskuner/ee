"""Optimizer, regularization, and adversarial training settings."""

from dataclasses import dataclass

from src.attacks import ATTACKS

from src.utils.helper_params import option

OPTIMIZERS = ["sgd", "momentum", "adam", "adamw"]


@dataclass
class TrainingParams:
    """Training: optimizers, regularization, and adversarial training."""

    epochs: int = option(10, "training epochs", gt=0)
    optimizer: list[str] = option(
        help="torch.optim optimizer(s)",
        choices=OPTIMIZERS,
        allow_all=True,
        default_factory=lambda: ["adam"],
    )
    learning_rate: float = option(1e-4, "learning rate", flag="lr", gt=0)
    momentum: float = option(0.9, "momentum of --optimizer momentum", ge=0)
    weight_decay: float = option(0.0, "L2 penalty (decoupled for AdamW)", ge=0)
    l1: float = option(0.0, "L1 weight penalty coefficient", ge=0)
    patience: int = option(0, "early-stopping patience in epochs (0: off)", ge=0)
    train_attack: list[str] = option(
        help="attack(s) for adversarial training; none trains on clean data",
        choices=lambda: ["none"] + ATTACKS.names(),
        default_factory=lambda: ["none"],
    )
    train_steps: int = option(40, "attack iterations during adversarial training", gt=0)
    train_alpha: float = option(
        0.01, "PGD-linf step size during adversarial training", gt=0
    )
    log_interval: int = option(200, "batches between progress lines", gt=0)
    track_test: bool = option(
        False, "record clean and robust test accuracy after every epoch"
    )
    track_samples: int = option(1000, "test images used by --track_test", gt=0)
