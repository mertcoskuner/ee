"""PyTorch implementation of the Madry MNIST convolutional network."""

import torch.nn as nn
import torch.nn.functional as F


class MadryCNN(nn.Module):
    """Classify MNIST images using the official checkpoint architecture.

    Inputs have shape (N, 1, 28, 28) and pixel values in [0, 1].
    Flattening uses NHWC order to match the original TensorFlow
    dense-layer weights.
    """

    def __init__(self, num_classes=10):
        """Initialize two convolutional layers and a two-layer classifier."""
        super(MadryCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, padding=2)
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def features(self, x):
        """Return post-ReLU 1024-dimensional features with NHWC flattening."""
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1)
        return F.relu(self.fc1(x))

    def forward(self, x):
        """Return unnormalized class logits for a batch of MNIST images."""
        return self.fc2(self.features(x))
