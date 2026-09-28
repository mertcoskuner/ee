"""Convolutional network for 28x28 single-channel MNIST digits."""

import torch.nn as nn
import torch.nn.functional as F

from .registry import MODELS


class MNIST_CNN(nn.Module):
    """Classify MNIST images with two conv/pool stages and two dense layers.

    Inputs have shape (N, 1, 28, 28). Each 5x5 convolution without padding
    is followed by 2x2 max pooling, so feature maps shrink 28 -> 24 -> 12
    -> 8 -> 4 before the 4 * 4 * 50 features reach the classifier.
    """

    def __init__(self, num_classes=10):
        """Create the two convolutional layers and the dense classifier."""
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        """Return unnormalized class logits for a batch of MNIST images."""
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


@MODELS.register("cnn", rank=1, convolutional=True)
def build_cnn(p):
    """Build MNIST_CNN from model parameters."""
    return MNIST_CNN(num_classes=p.num_classes)
