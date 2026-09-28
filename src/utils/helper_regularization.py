"""L1 weight penalty and early stopping for training regularization."""


def l1_penalty(model):
    """Return the summed absolute value of all weight matrices and kernels.

    Biases and normalization parameters (tensors with one dimension) are
    excluded, matching the usual L1 regularization of weights.
    """
    return sum(p.abs().sum() for p in model.parameters() if p.dim() > 1)


class EarlyStopping:
    """Stop training after patience epochs without a better score."""

    def __init__(self, patience):
        """Track the best score; patience 0 disables early stopping."""
        self.patience = patience
        self.best = None
        self.bad_epochs = 0

    def step(self, score):
        """Record an epoch score and return True when training should stop."""
        if self.best is None or score > self.best:
            self.best, self.bad_epochs = score, 0
            return False
        self.bad_epochs += 1
        return 0 < self.patience <= self.bad_epochs
