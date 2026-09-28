"""Run backdoor defenses on a trained MNIST classifier and save reports."""

import json
import os

import matplotlib
import matplotlib.pyplot as plt
import torch

from src.defenses import fine_pruning, latent_separability, neural_cleanse
from src.utils.helper_data import get_loaders, load_mnist_tensors
from src.utils.helper_eval import features
from src.utils.helper_model import load_weights
from src.utils.helper_plot import save_fig

matplotlib.use("Agg")


def validation_tensors(loader, limit):
    """Return up to limit images and labels from a data loader as tensors."""
    xs, ys, n = [], [], 0
    for imgs, labels in loader:
        xs.append(imgs)
        ys.append(labels)
        n += len(imgs)
        if n >= limit:
            break
    return torch.cat(xs)[:limit], torch.cat(ys)[:limit]


def run_neural_cleanse(model, params, device):
    """Reverse-engineer per-class triggers on clean test images and plot them."""
    d = params.defense
    x, _ = load_mnist_tensors(params, train=False)
    torch.manual_seed(params.run.seed)
    res = neural_cleanse(
        model,
        x[: d.nc_samples].to(device),
        params.model.num_classes,
        d.nc_steps,
        d.nc_lambda,
        d.nc_lr,
        d.nc_threshold,
    )
    print("\n=== Neural Cleanse ===")
    for c in range(params.model.num_classes):
        print(
            f"  Class {c}: mask L1 {res['norms'][c]:7.2f}  "
            f"trigger success {100 * res['success'][c]:6.2f}%  "
            f"anomaly index {res['anomaly_index'][c]:5.2f}"
        )
    print(f"  Flagged target classes: {res['flagged'] or 'none'}")

    fig, axes = plt.subplots(2, params.model.num_classes, figsize=(10.5, 2.6))
    for c in range(params.model.num_classes):
        mask, pattern = res["masks"][c][0, 0], res["patterns"][c][0, 0]
        axes[0, c].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[1, c].imshow(mask * pattern, cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(
            f"{c}: L1 {res['norms'][c]:.0f}",
            fontsize=7,
            color="#e34948" if c in res["flagged"] else "#0b0b0b",
        )
        for ax in axes[:, c]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("mask", fontsize=8)
    axes[1, 0].set_ylabel("trigger", fontsize=8)
    save_fig(fig, params, f"neural_cleanse_{params.model.tag}")
    return {k: v for k, v in res.items() if k not in ("masks", "patterns")}


def run_latent_separability(model, params, device):
    """Cluster each class's training-set features and plot the clusters."""
    d = params.defense
    x, y = load_mnist_tensors(params, train=True)
    x, y = x[: d.ls_samples], y[: d.ls_samples].numpy()
    res = latent_separability(
        features(model, x, device),
        y,
        params.model.num_classes,
        d.ls_components,
        d.ls_min_fraction,
        d.ls_min_silhouette,
        seed=params.run.seed,
    )
    print("\n=== Latent separability ===")
    for s in res["classes"]:
        print(
            f"  Class {s['class']}: minority cluster "
            f"{100 * s['minority_fraction']:5.1f}%  "
            f"silhouette {s['silhouette']:+.3f}"
        )
    print(f"  Flagged classes: {res['flagged'] or 'none'}")

    fig, axes = plt.subplots(2, 5, figsize=(10.5, 4.4))
    for c, ax in enumerate(axes.flat):
        z, assign = res["embeddings"][c]
        ax.scatter(z[:, 0], z[:, 1], c=assign, s=3, cmap="coolwarm", linewidths=0)
        s = res["classes"][c]
        ax.set_title(
            f"{c}: min {100 * s['minority_fraction']:.0f}%  "
            f"sil {s['silhouette']:+.2f}",
            fontsize=8,
            color="#e34948" if c in res["flagged"] else "#0b0b0b",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    save_fig(fig, params, f"latent_separability_{params.model.tag}")
    return {k: v for k, v in res.items() if k != "embeddings"}


def run_fine_pruning(model, params, device):
    """Prune dormant units, fine-tune, and save the pruned checkpoint."""
    d = params.defense
    train_loader, val_loader = get_loaders(params)
    x_val, y_val = validation_tensors(val_loader, d.fp_eval_samples)
    res = fine_pruning(
        model,
        x_val.to(device),
        y_val.to(device),
        train_loader,
        d.fp_max_drop,
        d.fp_epochs,
        params.training.learning_rate,
        device,
    )
    path = f"{params.model.tag}_fine_pruned.pth"
    torch.save(model.state_dict(), path)
    print("\n=== Fine-pruning ===")
    print(f"  Pruned {res['pruned']}/{res['units']} {res['layer']} units")
    print(
        f"  Val accuracy: before {res['acc_before']:.4f}  "
        f"after pruning {res['acc_after_prune']:.4f}  "
        f"after fine-tuning {res['acc_after_finetune']:.4f}"
    )
    print(f"  Saved pruned model ({path})")
    return res


def run_defense(model, params, device):
    """Load the checkpoint, run the selected defenses, and save a JSON report."""
    model = load_weights(model, params, device)
    runners = {
        "neural_cleanse": run_neural_cleanse,
        "latent_separability": run_latent_separability,
        "fine_pruning": run_fine_pruning,
    }
    report = {
        name: runners[name](model, params, device) for name in params.defense.defenses
    }
    os.makedirs(params.run.results_dir, exist_ok=True)
    path = os.path.join(params.run.results_dir, f"defense_{params.model.tag}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  saved {path}")
    return report
