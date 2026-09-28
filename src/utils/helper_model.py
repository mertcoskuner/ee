"""Construct the selected MNIST model and load evaluation checkpoints."""

import dataclasses
import os

import torch

from models import MODELS
from src.utils.helper_experiments import checkpoint_tag
from src.utils.helper_run import checkpoint_path


def build_model(params):
    """Construct the architecture registered under params.model.model."""
    return MODELS.get(params.model.model)(params.model)


def load_weights(model, params, device):
    """Load the selected checkpoint and prepare the model for evaluation.

    Map weights to device, disable parameter gradients, and return the
    model in evaluation mode. Input gradients remain available to
    attacks. Exit with a hint when the checkpoint has not been trained.
    """
    if not os.path.exists(params.model.weights):
        raise SystemExit(
            f"error: checkpoint {params.model.weights} not found. Train it first "
            "with --mode train (or --mode both) and the same --model, --optimizer, "
            "--train_attack, --backdoor, --dp and --checkpoint_dir options."
        )
    model.load_state_dict(torch.load(params.model.weights, map_location=device))
    model.requires_grad_(False)
    return model.eval()


def load_surrogate(params, device):
    """Return the surrogate model for transfer attacks, or None.

    The surrogate is the checkpoint named by --surrogate_model,
    --surrogate_optimizer and --surrogate_train_attack, trained earlier
    in train mode; it is prepared for evaluation like the target model.
    """
    a = params.attack
    if a.surrogate_model is None:
        return None
    tag = checkpoint_tag(
        a.surrogate_model, a.surrogate_optimizer, a.surrogate_train_attack
    )
    path = checkpoint_path(params, tag)
    if not os.path.exists(path):
        raise SystemExit(
            f"error: surrogate checkpoint {path} not found. Train it first with "
            f"--mode train --model {a.surrogate_model} --optimizer "
            f"{a.surrogate_optimizer} --train_attack {a.surrogate_train_attack} "
            "and the same --checkpoint_dir."
        )
    model_params = dataclasses.replace(
        params.model, model=a.surrogate_model, weights=path
    )
    surrogate_params = dataclasses.replace(params, model=model_params)
    model = MODELS.get(a.surrogate_model)(model_params).to(device)
    return load_weights(model, surrogate_params, device)


def load_reference(params, device):
    """Return (model, path) of the clean reference checkpoint, or (None, None).

    The reference is the same architecture and optimizer trained on clean
    data without a backdoor or DP-SGD. It is only loaded for adversarially
    trained, backdoored, or DP-SGD runs whose reference checkpoint exists,
    so their clean accuracy drop can be reported.
    """
    t = params.training
    plain = t.train_attack == "none" and params.backdoor.backdoor == "none"
    if plain and not params.privacy.dp:
        return None, None
    path = checkpoint_path(params, checkpoint_tag(params.model.model, t.optimizer))
    if not os.path.exists(path):
        return None, None
    reference_params = dataclasses.replace(
        params, model=dataclasses.replace(params.model, weights=path)
    )
    model = MODELS.get(params.model.model)(params.model).to(device)
    return load_weights(model, reference_params, device), path
