"""Construct the selected MNIST model and load evaluation checkpoints."""

import torch

from models.CNN import MNIST_CNN
from models.MLP import MLP
from models.Transformer import VisionTransformer


def build_model(params):
    """Construct the classifier named by params.model.model.

    Raise ValueError for an unknown architecture name.
    """
    p = params.model
    if p.model == "cnn":
        return MNIST_CNN(num_classes=p.num_classes)
    if p.model == "mlp":
        return MLP(
            hidden_sizes=p.hidden_sizes,
            num_classes=p.num_classes,
            dropout=p.dropout,
        )
    if p.model == "transformer":
        return VisionTransformer(
            patch_size=p.patch_size,
            num_classes=p.num_classes,
            dim=p.dim,
            depth=p.depth,
            heads=p.heads,
            mlp_dim=p.mlp_dim,
            dropout=p.dropout,
        )
    raise ValueError(f"Unknown model: {p.model}")


def load_weights(model, params, device):
    """Load the selected checkpoint and prepare the model for evaluation.

    Map weights to device, disable parameter gradients, and return the
    model in evaluation mode. Input gradients remain available to
    attacks.
    """
    model.load_state_dict(torch.load(params.model.weights, map_location=device))
    model.requires_grad_(False)
    return model.eval()
