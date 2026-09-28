"""Construct the selected MNIST model and load evaluation checkpoints."""

import torch

from models import MODELS


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
