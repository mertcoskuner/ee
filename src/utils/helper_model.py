"""Construct the MNIST model and load evaluation checkpoints."""

import torch

from models.madry_cnn import MadryCNN


def build_model(params):
    """Construct MadryCNN with the configured number of output classes."""
    return MadryCNN(num_classes=params.model.num_classes)


def load_weights(model, params, device):
    """Load the selected checkpoint and prepare the model for evaluation.

    Map weights to device, disable parameter gradients, and return the
    model in evaluation mode. Input gradients remain available to
    attacks.
    """
    model.load_state_dict(
        torch.load(params.model.weights, map_location=device)
    )
    model.requires_grad_(False)
    return model.eval()
