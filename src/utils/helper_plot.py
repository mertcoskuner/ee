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


def plot_training_curve(curve, params):
    """Save training, validation, and test accuracy per epoch."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    series = [
        ("train_acc", "train (training inputs)", "#52514e", "-"),
        ("test_clean", "test clean", "#2a78d6", "-"),
        ("test_robust", "test robust", "#eb6834", "-"),
        ("val_robust", "validation robust", "#eb6834", ":"),
    ]
    for key, label, color, style in series:
        values = curve.get(key)
        if values and values[0] is not None:
            ax.plot(curve["epoch"], values, style, color=color, label=label)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(params.model.tag, fontsize=9)
    save_fig(fig, params, f"training_curve_{params.model.tag}")


def plot_federated_run(history, counts, params, tag):
    """Save one federated run's accuracy curve and per-client label shares."""
    fig, (ax_acc, ax_lab) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax_acc.plot(
        history["round"], [100 * a for a in history["test_acc"]], color="#2a78d6"
    )
    ax_acc.set_xlabel("round")
    ax_acc.set_ylabel("test accuracy (%)")
    ax_acc.set_ylim(0, 100)
    ax_acc.grid(alpha=0.3)
    ax_acc.set_title(tag, fontsize=8)
    shares = counts / counts.sum(1, keepdims=True)
    ax_lab.imshow(shares.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax_lab.set_xlabel("client")
    ax_lab.set_ylabel("class")
    ax_lab.set_title(f"label shares ({params.federated.partition})", fontsize=9)
    fig.tight_layout()
    save_fig(fig, params, f"federated_{params.model.model}_{tag}")


def print_federated_summary(summary, fields, params):
    """Print and save the final accuracy of every federated combination."""
    print("\n=== Federated summary ===")
    print("  " + "  ".join(f"{n:>18s}" for n in fields) + "  final acc")
    for row in summary:
        acc = "skipped" if row["final_acc"] is None else f"{row['final_acc']:.4f}"
        print("  " + "  ".join(f"{row[n]:>18s}" for n in fields) + f"  {acc}")
    save_json(summary, params, f"federated_summary_{params.model.model}")


def print_test_summary(rows, params):
    """Print and save a table of test metrics, one row per combination.

    Columns are clean accuracy, its drop against the clean reference,
    robust accuracy and attack success rate (_asr) per attack, and the
    backdoor attack success rate.
    """
    seen = list(dict.fromkeys(k for _, res in rows for k in res))
    first = [c for c in ("clean", "clean_drop") if c in seen]
    last = [c for c in ("backdoor_asr",) if c in seen]
    columns = first + [c for c in seen if c not in first + last] + last
    widths = [max(10, len(c) + 2) for c in columns]
    head = f"{'model':>12s} {'optimizer':>10s} {'train_attack':>13s} {'backdoor':>9s}"
    print("\n=== Summary: test metrics ===")
    print(head + "".join(f"{c:>{w}s}" for c, w in zip(columns, widths)))
    for run, res in rows:
        line = (
            f"{run['model']:>12s} {run['optimizer']:>10s} "
            f"{run['train_attack']:>13s} {run['backdoor']:>9s}"
        )
        cells = (res.get(c, float("nan")) for c in columns)
        print(line + "".join(f"{v:>{w}.4f}" for v, w in zip(cells, widths)))
    save_json(
        [{**run, "metrics": res} for run, res in rows], params, "experiment_summary"
    )
