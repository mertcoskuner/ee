"""Expand list-valued settings into one experiment per combination.

Central runs: the model, optimizer, training-attack, and backdoor options
accept several values, and every combination becomes its own run with its
own checkpoint name, so one command can train and evaluate clean,
adversarially trained, and backdoored models with several optimizers and
architectures. Federated runs: every combination of the SWEEP_FIELDS
values becomes its own federated experiment.
"""

import dataclasses
import itertools

from models import MODELS
from src.attacks import ATTACKS
from src.backdoors import BACKDOORS
from src.federated.aggregators import AGGREGATORS
from src.federated.attacks import FL_ATTACKS
from src.params.federated_params import SWEEP_FIELDS
from src.params.training_params import OPTIMIZERS
from src.utils.helper_run import checkpoint_path

FL_REGISTRIES = {"aggregator": AGGREGATORS, "attack": FL_ATTACKS}

CENTRAL_AXES = ["model", "optimizer", "train_attack", "backdoor"]


def checkpoint_tag(model, optimizer, train_attack="none", backdoor="none", dp=None):
    """Return the checkpoint stem, e.g. best_cnn_adam_adv-fgsm or best_cnn_adam_dp1.

    dp is the DP-SGD noise multiplier, or None without DP.
    """
    tag = f"best_{model}_{optimizer}"
    if train_attack != "none":
        tag += f"_adv-{train_attack}"
    if backdoor != "none":
        tag += f"_bd-{backdoor}"
    if dp is not None:
        tag += f"_dp{dp:g}"
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
    attacks = expand(params.training.train_attack, ATTACKS.names())
    backdoors = expand(params.backdoor.backdoor, BACKDOORS.names())
    if params.run.mode == "federated":
        optimizers, attacks, backdoors = optimizers[:1], ["none"], ["none"]
    for model, optimizer, attack, backdoor in itertools.product(
        models, optimizers, attacks, backdoors
    ):
        dp = params.privacy.dp_noise if params.privacy.dp else None
        tag = checkpoint_tag(model, optimizer, attack, backdoor, dp)
        path = checkpoint_path(params, tag)
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
                    save_path=path,
                    weights=path,
                ),
                training=dataclasses.replace(
                    params.training, optimizer=optimizer, train_attack=attack
                ),
                backdoor=dataclasses.replace(params.backdoor, backdoor=backdoor),
            )
        )


def fl_combinations(fl):
    """Yield one FederatedParams per combination of the federated sweep fields.

    The list-valued SWEEP_FIELDS (partition, local, server_opt, aggregator,
    attack, dp) are expanded, with "all" standing for every registered value.
    """
    axes = []
    for name in SWEEP_FIELDS:
        values = getattr(fl, name)
        registry = FL_REGISTRIES.get(name)
        axes.append(
            registry.expand(values) if registry else list(dict.fromkeys(values))
        )
    for combo in itertools.product(*axes):
        yield dataclasses.replace(fl, **dict(zip(SWEEP_FIELDS, combo)))


def fl_run_tag(fl):
    """Return a short name identifying one federated combination."""
    return "_".join(str(getattr(fl, name)) for name in SWEEP_FIELDS)
