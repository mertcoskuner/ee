"""Compare Grad-CAM heatmaps on clean and adversarial MNIST digits.

Grad-CAM (Selvaraju et al., ICCV 2017) weights each activation map A^k of
a convolutional layer by the spatial mean of d y^c / d A^k and keeps the
positive part of the weighted sum: ReLU(sum_k alpha_k^c A^k).
"""

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

from src.utils.helper_data import load_mnist_tensors
from src.utils.helper_eval import attack, predict
from src.utils.helper_model import load_weights
from src.utils.helper_plot import save_fig

matplotlib.use("Agg")


class GradCAM:
    """Compute batched Grad-CAM heatmaps for one convolutional layer."""

    def __init__(self, model, target_layer):
        """Register a hook that keeps target_layer activations."""
        self.model = model
        self.activations = None
        self._hook = target_layer.register_forward_hook(self._save)

    def _save(self, module, inputs, output):
        """Store the layer output from the latest forward pass."""
        self.activations = output

    def __call__(self, x, class_idx=None):
        """Return (heatmaps, classes) for a batch of images.

        Explain class_idx per example, or the predicted classes when it is
        None. Heatmaps have shape (N, H, W), are resized to the input, and
        are scaled to [0, 1] per example.
        """
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1)
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        (grads,) = torch.autograd.grad(score, self.activations)
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(
            cam, size=x.shape[2:], mode="bilinear", align_corners=False
        ).squeeze(1)
        peak = cam.flatten(1).max(1).values.view(-1, 1, 1)
        return (cam / (peak + 1e-8)).detach(), class_idx.detach()

    def remove_hooks(self):
        """Detach the forward hook from the target layer."""
        self._hook.remove()


def last_conv(model):
    """Return the model's last nn.Conv2d, the usual Grad-CAM target."""
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        raise ValueError("Grad-CAM needs a model with a convolutional layer")
    return convs[-1]


def run_gradcam(model, params, device):
    """Save Grad-CAM overlays for clean, PGD-linf, and PGD-L2 digits.

    Use the first correctly classified test image of each class and
    explain the class the model predicts for each input.
    """
    model = load_weights(model, params, device)
    x_all, y_all = load_mnist_tensors(params, train=False)
    ok = predict(model, x_all, device) == y_all
    idx = [int(((y_all == c) & ok).nonzero()[0]) for c in range(10)]
    x, y = x_all[idx], y_all[idx]

    torch.manual_seed(params.run.seed)
    rows = [
        ("Clean", x),
        (
            rf"PGD-$\ell_\infty$ ($\varepsilon$={params.attack.eps_linf:g})",
            attack(model, x, y, "pgd_linf", params, device),
        ),
        (
            rf"PGD-$\ell_2$ ($\varepsilon$={params.attack.eps_l2:g})",
            attack(model, x, y, "pgd_l2", params, device),
        ),
    ]

    cam = GradCAM(model, last_conv(model))
    fig, axes = plt.subplots(2 * len(rows), 10, figsize=(10.5, 7.2))
    try:
        for r, (label, imgs) in enumerate(rows):
            heat, pred = cam(imgs.to(device))
            heat, pred = heat.cpu(), pred.cpu()
            for c in range(10):
                img_ax, cam_ax = axes[2 * r, c], axes[2 * r + 1, c]
                img_ax.imshow(imgs[c, 0], cmap="gray", vmin=0, vmax=1)
                cam_ax.imshow(imgs[c, 0], cmap="gray", vmin=0, vmax=1)
                cam_ax.imshow(heat[c], cmap="jet", alpha=0.5, vmin=0, vmax=1)
                wrong = pred[c].item() != y[c].item()
                img_ax.set_title(
                    f"pred {pred[c].item()}",
                    fontsize=7.5,
                    pad=2,
                    color="#e34948" if wrong else "#0b0b0b",
                    fontweight="bold" if wrong else "normal",
                )
                for ax in (img_ax, cam_ax):
                    ax.set_xticks([])
                    ax.set_yticks([])
            axes[2 * r, 0].set_ylabel(
                label, fontsize=8, rotation=0, ha="right", va="center"
            )
            axes[2 * r + 1, 0].set_ylabel(
                "Grad-CAM", fontsize=8, rotation=0, ha="right", va="center"
            )
    finally:
        cam.remove_hooks()
    save_fig(fig, params, f"gradcam_{params.model.tag}")
