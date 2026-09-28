"""Assemble typed experiment settings from parsed arguments.

PARAM_GROUPS lists every parameter class with its CLI prefix; the parser
and get_params both iterate over it, so a new group only needs an entry
here.
"""

from dataclasses import dataclass

from .attack_params import AttackParams
from .data_loader_params import DataLoaderParams
from .defense_params import DefenseParams
from .federated_params import FederatedParams
from src.utils.helper_params import from_args
from .model_params import ModelParams
from .run_params import RunParams
from .training_params import TrainingParams

PARAM_GROUPS = {
    "run": (RunParams, ""),
    "model": (ModelParams, ""),
    "data_loader": (DataLoaderParams, ""),
    "training": (TrainingParams, ""),
    "attack": (AttackParams, ""),
    "defense": (DefenseParams, ""),
    "federated": (FederatedParams, "fl_"),
}


@dataclass
class ExperimentParams:
    """Group run, model, data, training, attack, defense, and federated settings."""

    run: RunParams
    model: ModelParams
    data_loader: DataLoaderParams
    training: TrainingParams
    attack: AttackParams
    defense: DefenseParams
    federated: FederatedParams


def get_params(args) -> ExperimentParams:
    """Build every parameter group from one validated CLI namespace."""
    return ExperimentParams(
        **{
            name: from_args(cls, args, prefix)
            for name, (cls, prefix) in PARAM_GROUPS.items()
        }
    )
