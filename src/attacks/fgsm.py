"""Single-step untargeted FGSM in pixel space."""

from src.utils.helper_attack import input_grad


def fgsm(model, x, y, eps):
    """Return detached FGSM examples within an L-infinity budget eps.

    Use cross-entropy ascent against labels y and clip pixels to [0, 1].
    """
    return (x + eps * input_grad(model, x, y).sign()).clamp(0, 1).detach()
