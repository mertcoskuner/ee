"""Backdoor trigger selection and data-poisoning settings."""

from dataclasses import dataclass

from src.backdoors import BACKDOORS
from src.utils.helper_params import option


@dataclass
class BackdoorParams:
    """Backdoor poisoning: trigger, poison rate, and target class."""

    backdoor: list[str] = option(
        help="trigger(s) poisoning the training data; none trains without one",
        choices=lambda: ["none"] + BACKDOORS.names(),
        default_factory=lambda: ["none"],
    )
    poison_rate: float = option(0.1, "fraction of training images poisoned", gt=0, lt=1)
    target_class: int = option(0, "label the trigger maps to", ge=0, le=9)
    trigger_size: int = option(4, "side of BadNets / dynamic patches", gt=0, le=14)
    blend_alpha: float = option(0.2, "blend ratio of the Blend trigger", gt=0, lt=1)
    trigger_seed: int = option(0, "seed of the trigger pattern and poisoned set", ge=0)
