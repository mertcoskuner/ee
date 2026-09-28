"""Save experiment figures, defense plots, and JSON reports."""

import json
import os

import matplotlib.pyplot as plt

ALERT, INK = "#e34948", "#0b0b0b"


def save_fig(fig, params, name):
    """Save a figure as PDF and PNG in results_dir, then close it."""
    os.makedirs(params.run.results_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(params.run.results_dir, f"{name}.{ext}"),
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)
    print(f"  saved {params.run.results_dir}/{name}.pdf|png")


def save_json(report, params, name):
    """Write report as indented JSON to results_dir/name.json."""
    os.makedirs(params.run.results_dir, exist_ok=True)
    path = os.path.join(params.run.results_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  saved {path}")


def plot_neural_cleanse(res, params):
    """Save reversed masks and triggers per class; flagged titles in red."""
    n = len(res["norms"])
    fig, axes = plt.subplots(2, n, figsize=(10.5, 2.6))
    for c in range(n):
        mask, pattern = res["masks"][c][0, 0], res["patterns"][c][0, 0]
        axes[0, c].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[1, c].imshow(mask * pattern, cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title(
            f"{c}: L1 {res['norms'][c]:.0f}",
            fontsize=7,
            color=ALERT if c in res["flagged"] else INK,
        )
        for ax in axes[:, c]:
            ax.set_xticks([])
            ax.set_yticks([])
    axes[0, 0].set_ylabel("mask", fontsize=8)
    axes[1, 0].set_ylabel("trigger", fontsize=8)
    save_fig(fig, params, f"neural_cleanse_{params.model.tag}")


def plot_latent_separability(res, params):
    """Save each class's 2-means split in PCA space; flagged titles in red."""
    fig, axes = plt.subplots(2, 5, figsize=(10.5, 4.4))
    for c, ax in enumerate(axes.flat):
        z, assign = res["embeddings"][c]
        ax.scatter(z[:, 0], z[:, 1], c=assign, s=3, cmap="coolwarm", linewidths=0)
        s = res["classes"][c]
        ax.set_title(
            f"{c}: min {100 * s['minority_fraction']:.0f}%  "
            f"sil {s['silhouette']:+.2f}",
            fontsize=8,
            color=ALERT if c in res["flagged"] else INK,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    save_fig(fig, params, f"latent_separability_{params.model.tag}")
