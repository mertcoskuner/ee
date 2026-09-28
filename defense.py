"""Run backdoor defenses on a trained MNIST classifier and save reports."""

import matplotlib
import torch

from src.defenses import fine_pruning, latent_separability, neural_cleanse
from src.utils.helper_data import get_loaders, load_mnist_tensors, loader_tensors
from src.utils.helper_eval import features
from src.utils.helper_model import load_weights
from src.utils.helper_plot import (
    plot_latent_separability,
    plot_neural_cleanse,
    save_json,
)

matplotlib.use("Agg")


def run_neural_cleanse(model, params, device):
    """Reverse-engineer per-class triggers on clean test images."""
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
    plot_neural_cleanse(res, params)
    return {k: v for k, v in res.items() if k not in ("masks", "patterns")}


def run_latent_separability(model, params, device):
    """Cluster each class's training-set features in latent space."""
    d = params.defense
    x, y = load_mnist_tensors(params, train=True)
    x, y = x[: d.ls_samples], y[: d.ls_samples].numpy()
    res = latent_separability(
        features(model, x, device),
        y,
        params.model.num_classes,
        d.ls_components,
        d.ls_min_fraction,
        d.ls_threshold,
        seed=params.run.seed,
    )
    print("\n=== Latent separability ===")
    for s in res["classes"]:
        print(
            f"  Class {s['class']}: minority cluster "
            f"{100 * s['minority_fraction']:5.1f}%  "
            f"silhouette {s['silhouette']:+.3f}  "
            f"anomaly index {s['anomaly_index']:5.2f}"
        )
    print(f"  Flagged classes: {res['flagged'] or 'none'}")
    plot_latent_separability(res, params)
    return {k: v for k, v in res.items() if k != "embeddings"}


def run_fine_pruning(model, params, device):
    """Prune dormant units, fine-tune, and save the pruned checkpoint."""
    d = params.defense
    train_loader, val_loader = get_loaders(params)
    x_val, y_val = loader_tensors(val_loader, d.fp_eval_samples)
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
    save_json(report, params, f"defense_{params.model.tag}")
    return report
