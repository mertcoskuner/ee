"""Label-flipping data poisoning: train honestly on relabelled data."""

from src.federated.client import Client


class LabelFlipClient(Client):
    """Client that trains on labels mapped y -> num_classes - 1 - y."""

    omniscient = False

    def __init__(self, *args, num_classes=10, **kwargs):
        """Create a client whose local labels are flipped."""
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes

    def labels(self, y):
        """Return flipped labels num_classes - 1 - y."""
        return self.num_classes - 1 - y
