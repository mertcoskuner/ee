"""Expand list-valued central settings into one experiment per combination.

The model, optimizer, training-attack, and backdoor options accept several
values;
every combination becomes its own run with its own checkpoint name, so one
command can train and evaluate clean and adversarially trained models with
several optimizers and architectures.
"""

import dataclasses
import itertools

from models import MODELS
from src.params.training_params import OPTIMIZERS

CENTRAL_AXES = ["model", "optimizer", "train_attack", "backdoor"]


def checkpoint_tag(model, optimizer, train_attack="none", backdoor="none"):
    """Return the checkpoint stem, e.g. best_cnn_adam_adv-fgsm or best_cnn_adam_bd-blend."""
    tag = f"best_{model}_{optimizer}"
    if train_attack != "none":
        tag += f"_adv-{train_attack}"
    if backdoor != "none":
        tag += f"_bd-{backdoor}"
    return tag


def expand(values, universe):
    """Return values with "all" replaced by universe, without duplicates."""
    values = universe if "all" in values else values
    return list(dict.fromkeys(values))


def central_runs(params):
    """Yield (settings, params) for every central combination.

    In federated mode only the model axis applies, because optimizers and
    training attacks of central training are not used there.
    """
    models = expand(params.model.model, MODELS.names())
    optimizers = expand(params.training.optimizer, OPTIMIZERS)
    attacks = list(dict.fromkeys(params.training.train_attack))
    backdoors = list(dict.fromkeys(params.backdoor.backdoor))
    if params.run.mode == "federated":
        optimizers, attacks, backdoors = optimizers[:1], ["none"], ["none"]
    for model, optimizer, attack, backdoor in itertools.product(
        models, optimizers, attacks, backdoors
    ):
        tag = checkpoint_tag(model, optimizer, attack, backdoor)
        run = {
            "model": model,
            "optimizer": optimizer,
            "train_attack": attack,
            "backdoor": backdoor,
        }
        yield run, (
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
                backdoor=dataclasses.replace(params.backdoor, backdoor=backdoor),
            )
        )
