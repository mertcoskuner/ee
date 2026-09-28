"""Compare Grad-CAM heatmaps on clean and adversarial MNIST digits."""

import matplotlib
import matplotlib.pyplot as plt
import torch

from src.utils.helper_eval import attack, first_correct_per_class
from src.utils.helper_gradcam import GradCAM, last_conv
from src.utils.helper_model import load_weights
from src.utils.helper_plot import save_fig

matplotlib.use("Agg")


def run_gradcam(model, params, device):
    """Save Grad-CAM overlays for clean, PGD-linf, and PGD-L2 digits.

    Use the first correctly classified test image of each class and
    explain the class the model predicts for each input.
    """
    model = load_weights(model, params, device)
    x, y = first_correct_per_class(model, params, device)

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
