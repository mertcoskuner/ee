"""Construct the selected MNIST model and load evaluation checkpoints."""

import torch

import dataclasses

from models import MODELS
from src.utils.helper_experiments import checkpoint_tag


def build_model(params):
    """Construct the architecture registered under params.model.model."""
    return MODELS.get(params.model.model)(params.model)


def load_weights(model, params, device):
    """Load the selected checkpoint and prepare the model for evaluation.

    Map weights to device, disable parameter gradients, and return the
    model in evaluation mode. Input gradients remain available to
    attacks.
    """
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
    model_params = dataclasses.replace(
        params.model, model=a.surrogate_model, weights=f"{tag}.pth"
    )
    surrogate_params = dataclasses.replace(params, model=model_params)
    model = MODELS.get(a.surrogate_model)(model_params).to(device)
    return load_weights(model, surrogate_params, device)
