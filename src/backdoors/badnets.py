"""BadNets patch trigger (Gu et al., 2017)."""

from .registry import BACKDOORS


@BACKDOORS.register("badnets", rank=1)
class BadNets:
    """Stamp a white trigger_size x trigger_size square near the bottom-right corner."""

    def __init__(self, bp):
        """Store the trigger size."""
        self.size = bp.trigger_size

    def apply(self, x):
        """Return a copy of the batch x with the patch set to 1."""
        x = x.clone()
        s = self.size
        x[..., -s - 1 : -1, -s - 1 : -1] = 1.0
        return x
