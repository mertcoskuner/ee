"""Execution mode, random seed, device, and output settings."""

from dataclasses import dataclass

from src.utils.helper_params import option

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
    """Execution: mode, seed, device, and results directory."""

    mode: str = option("both", "experiment to run", choices=MODES)
    seed: int = option(0, "random seed", ge=0, lt=2**32)
    device: str = option("cpu", "torch device, e.g. cpu, mps, cuda:0")
    results_dir: str = option("results", "directory for figures and reports")
