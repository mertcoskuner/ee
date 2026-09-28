"""Assemble typed experiment settings from parsed arguments."""

from dataclasses import dataclass

from .attack_params import AttackParams, get_attack_params
from .data_loader_params import DataLoaderParams, get_data_loader_params
from .model_params import ModelParams, get_model_params
from .run_params import RunParams, get_run_params
from .training_params import TrainingParams, get_training_params


@dataclass
class ExperimentParams:
    """Group data, training, attack, model, and execution settings."""

    data_loader: DataLoaderParams
    training: TrainingParams
    attack: AttackParams
    model: ModelParams
    run: RunParams


def get_params(args) -> ExperimentParams:
    """Build all parameter groups from one validated CLI namespace."""
    return ExperimentParams(
        data_loader=get_data_loader_params(args),
        training=get_training_params(args),
        attack=get_attack_params(args),
        model=get_model_params(args),
        run=get_run_params(args),
    )
