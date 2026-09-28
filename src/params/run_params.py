"""Execution mode, random seed, device, and output settings."""

from dataclasses import dataclass

MODES = [
    "train",
    "test",
    "both",
    "visualize",
    "tsne",
    "gradcam",
    "geometry",
    "defense",
    "federated",
]


@dataclass
class RunParams:
    """Store the execution mode, seed, device, and results directory."""

    mode: str = "both"
    seed: int = 0
    device: str = "auto"
    results_dir: str = "results"


def get_run_params(args) -> RunParams:
    """Build execution settings from validated CLI arguments."""
    return RunParams(
        mode=args.mode,
        seed=args.seed,
        device=args.device,
        results_dir=args.results_dir,
    )
