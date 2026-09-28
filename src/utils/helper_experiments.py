"""Expand list-valued central settings into one experiment per combination.

The model, optimizer, and training-attack options accept several values;
every combination becomes its own run with its own checkpoint name, so one
command can train and evaluate clean and adversarially trained models with
several optimizers and architectures.
"""

import dataclasses
import itertools

from models import MODELS
from src.params.training_params import OPTIMIZERS

CENTRAL_AXES = ["model", "optimizer", "train_attack"]


def checkpoint_tag(model, optimizer, train_attack):
    """Return the checkpoint stem, e.g. best_cnn_adam or best_cnn_adam_adv-fgsm."""
    tag = f"best_{model}_{optimizer}"
    return tag if train_attack == "none" else f"{tag}_adv-{train_attack}"


def expand(values, universe):
    """Return values with "all" replaced by universe, without duplicates."""
    values = universe if "all" in values else values
    return list(dict.fromkeys(values))


def central_runs(params):
    """Yield (settings, params) for every model/optimizer/train_attack combination.

    In federated mode only the model axis applies, because optimizers and
    training attacks of central training are not used there.
    """
    models = expand(params.model.model, MODELS.names())
    optimizers = expand(params.training.optimizer, OPTIMIZERS)
    attacks = list(dict.fromkeys(params.training.train_attack))
    if params.run.mode == "federated":
        optimizers, attacks = optimizers[:1], ["none"]
    for model, optimizer, attack in itertools.product(models, optimizers, attacks):
        tag = checkpoint_tag(model, optimizer, attack)
        yield {"model": model, "optimizer": optimizer, "train_attack": attack}, (
            dataclasses.replace(
                params,
                model=dataclasses.replace(
                    params.model,
                    model=model,
                    tag=tag,
                    save_path=f"{tag}.pth",
                    weights=f"{tag}.pth",
                ),
                training=dataclasses.replace(
                    params.training, optimizer=optimizer, train_attack=attack
                ),
            )
        )
