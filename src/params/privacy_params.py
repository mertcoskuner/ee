"""Differential privacy settings: DP-SGD, membership inference, and demos."""

from dataclasses import dataclass, field

DP_EPSILONS = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


@dataclass
class PrivacyParams:
    """Store DP-SGD, membership inference, and DP demonstration settings.

    dp trains with DP-SGD: Poisson sampling at rate batch_size / training
    size, per-example clipping to dp_clip, and Gaussian noise of standard
    deviation dp_noise * dp_clip; dp_delta is the target delta.
    """

    dp: bool = False
    dp_noise: float = 1.0
    dp_clip: float = 1.0
    dp_delta: float = 1e-5
    mia: bool = False
    mia_samples: int = 1000
    dp_epsilons: list[float] = field(default_factory=lambda: list(DP_EPSILONS))
    dp_trials: int = 200


def get_privacy_params(args) -> PrivacyParams:
    """Build differential privacy settings from validated CLI arguments."""
    return PrivacyParams(
        dp=args.dp,
        dp_noise=args.dp_noise,
        dp_clip=args.dp_clip,
        dp_delta=args.dp_delta,
        mia=args.mia,
        mia_samples=args.mia_samples,
        dp_epsilons=args.dp_epsilons,
        dp_trials=args.dp_trials,
    )
