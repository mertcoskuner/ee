"""Measure the geometry of adversarial perturbations around test images."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.attacks import run_attack
from src.utils.helper_attack import input_grad
from src.utils.helper_data import load_mnist_tensors
from src.utils.helper_eval import predict
from src.utils.helper_geometry import decision_map, flip_distance, sweep
from src.utils.helper_model import load_weights
from src.utils.helper_plot import save_fig, save_json

matplotlib.use("Agg")


def run_geometry(model, params, device):
    """Save loss, accuracy, boundary-distance, and decision-map plots.

    The adversarial direction is the sign of the input gradient (the FGSM
    direction) and the baseline is a random sign vector of the same
    L-infinity size. The report compares how quickly each direction
    raises the loss and flips the prediction, the plane they span for one
    image, and the cosine between PGD perturbations and input gradients.
    """
    model = load_weights(model, params, device)
    x_all, y_all = load_mnist_tensors(params, train=False)
    n = params.data_loader.num_samples or 500
    ok = predict(model, x_all[:n], device) == y_all[:n]
    x, y = x_all[:n][ok].to(device), y_all[:n][ok].to(device)
    torch.manual_seed(params.run.seed)

    grad = input_grad(model, x, y)
    adv_dir = grad.sign()
    rand_dir = torch.randint(0, 2, x.shape, device=device).float() * 2 - 1
    span = 2 * params.attack.eps_linf
    epsilons = np.linspace(-span, span, 41)
    loss_adv, acc_adv = sweep(model, x, y, adv_dir, epsilons)
    loss_rand, acc_rand = sweep(model, x, y, rand_dir, epsilons)
    d_adv = flip_distance(model, x, y, adv_dir, epsilons)
    d_rand = flip_distance(model, x, y, rand_dir, epsilons)

    delta = run_attack(model, x, y, "pgd_linf", params) - x
    cosine = F.cosine_similarity(delta.flatten(1), grad.flatten(1), dim=1)

    report = {
        "images": len(x),
        "flip_rate_adversarial": float(torch.isfinite(d_adv).float().mean()),
        "flip_rate_random": float(torch.isfinite(d_rand).float().mean()),
        "median_flip_eps_adversarial": float(d_adv.median()),
        "median_flip_eps_random": float(d_rand.median()),
        "mean_cosine_pgd_gradient": float(cosine.mean()),
        "mean_pgd_l2": float(delta.flatten(1).norm(dim=1).mean()),
        "mean_pgd_linf": float(delta.abs().flatten(1).max(1).values.mean()),
    }
    print("\n=== Geometry of adversarial perturbations ===")
    for key, value in report.items():
        print(
            f"  {key:30s} {value:.4f}"
            if isinstance(value, float)
            else f"  {key:30s} {value}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    axes[0].plot(epsilons, loss_adv, color="#eb6834", label="gradient sign")
    axes[0].plot(epsilons, loss_rand, color="#2a78d6", label="random sign")
    axes[0].set_xlabel(r"$\varepsilon$ along direction")
    axes[0].set_ylabel("mean cross-entropy")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(epsilons, acc_adv, color="#eb6834", label="gradient sign")
    axes[1].plot(epsilons, acc_rand, color="#2a78d6", label="random sign")
    axes[1].set_xlabel(r"$\varepsilon$ along direction")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0, 1)
    classes = decision_map(model, x[:1], adv_dir[:1], rand_dir[:1], span)
    axes[2].imshow(
        classes.T,
        origin="lower",
        extent=(-span, span, -span, span),
        cmap="tab10",
        vmin=0,
        vmax=9,
    )
    axes[2].plot(0, 0, "k+", ms=10)
    axes[2].set_xlabel("gradient-sign direction")
    axes[2].set_ylabel("random direction")
    axes[2].set_title(f"predicted class around a {int(y[0])}", fontsize=9)
    for ax in axes[:2]:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, params, f"geometry_{params.model.tag}")
    save_json(report, params, f"geometry_{params.model.tag}")
    return report
