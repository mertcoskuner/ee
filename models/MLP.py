"""Multi-layer perceptron for flattened 28x28 MNIST digits."""

import torch.nn as nn

from .registry import MODELS


class MLP(nn.Module):
    """Classify MNIST images with ReLU hidden layers and dropout.

    Inputs of shape (N, 1, 28, 28) are flattened to (N, 784). Each hidden
    layer applies Linear, ReLU, and Dropout; a final linear head returns
    the class logits.
    """

    def __init__(
        self, input_size=784, hidden_sizes=(512, 256), num_classes=10, dropout=0.1
    ):
        """Create the hidden layers and the linear classification head."""
        super().__init__()
        layers, in_dim = [], input_size
        for size in hidden_sizes:
            layers += [nn.Linear(in_dim, size), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = size
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        """Return unnormalized class logits for a batch of MNIST images."""
        return self.head(self.body(x.flatten(1)))


@MODELS.register("mlp", rank=2)
def build_mlp(p):
    """Build the MLP from model parameters."""
    return MLP(
        hidden_sizes=p.hidden_sizes, num_classes=p.num_classes, dropout=p.dropout
    )
