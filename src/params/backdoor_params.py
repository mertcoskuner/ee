"""Backdoor trigger selection and data-poisoning settings."""

from dataclasses import dataclass, field


@dataclass
class BackdoorParams:
    """Store the trigger(s), poison rate, target class, and trigger shape.

    backdoor lists one or more triggers; each run replaces it with a
    single name, and "none" trains without poisoning.
    """

    backdoor: list[str] = field(default_factory=lambda: ["none"])
    poison_rate: float = 0.05
    target_class: int = 0
    trigger_size: int = 4
    blend_alpha: float = 0.2
    trigger_seed: int = 0


def get_backdoor_params(args) -> BackdoorParams:
    """Build backdoor settings from validated CLI arguments."""
    return BackdoorParams(
        backdoor=args.backdoor,
        poison_rate=args.poison_rate,
        target_class=args.target_class,
        trigger_size=args.trigger_size,
        blend_alpha=args.blend_alpha,
        trigger_seed=args.trigger_seed,
    )
