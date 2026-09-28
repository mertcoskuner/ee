"""Run federated training over every requested combination of settings."""

import dataclasses
import itertools

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.federated import Client, Server, assumed_attackers, partition
from src.federated.aggregators import AGGREGATORS, check_feasible
from src.federated.attacks import FL_ATTACKS, client_class
from src.params.federated_params import SWEEP_FIELDS
from src.utils.helper_data import load_mnist_tensors
from src.utils.helper_eval import predict
from src.utils.helper_model import build_model
from src.utils.helper_plot import save_fig, save_json
from src.utils.helper_run import set_seed

matplotlib.use("Agg")

REGISTRIES = {"aggregator": AGGREGATORS, "attack": FL_ATTACKS}


def combinations(fl):
    """Yield one FederatedParams per combination of the sweep fields.

    List-valued fields (partition, local, server_opt, aggregator, attack)
    are expanded, with "all" standing for every registered value.
    """
    axes = []
    for name in SWEEP_FIELDS:
        values = getattr(fl, name)
        registry = REGISTRIES.get(name)
        axes.append(
            registry.expand(values) if registry else list(dict.fromkeys(values))
        )
    for combo in itertools.product(*axes):
        yield dataclasses.replace(fl, **dict(zip(SWEEP_FIELDS, combo)))


def run_tag(fl):
    """Return a short name identifying one combination."""
    return "_".join(str(getattr(fl, name)) for name in SWEEP_FIELDS)


def num_byzantine(fl):
    """Return the number of Byzantine clients; zero for the "none" attack."""
    if FL_ATTACKS.meta(fl.attack, "benign", False):
        return 0
    return round(fl.byzantine_ratio * fl.clients)


def build_clients(params, device):
    """Partition MNIST training data and create benign then Byzantine clients.

    The last num_byzantine clients are Byzantine and use the client class
    the attack registers (for example LabelFlipClient); attacks without
    one craft their updates on the server side.
    """
    fl = params.federated
    x, y = load_mnist_tensors(params, train=True)
    shares = partition(
        y.numpy(),
        fl.clients,
        fl.partition,
        fl.alpha,
        fl.shards_per_client,
        params.run.seed,
    )
    first_byzantine = fl.clients - num_byzantine(fl)
    clients = []
    for cid, idx in enumerate(shares):
        idx = torch.from_numpy(idx)
        cls = client_class(fl, Client) if cid >= first_byzantine else Client
        clients.append(cls(cid, x[idx], y[idx], params, device))
    return clients


def label_histogram(clients, num_classes):
    """Return a clients x classes matrix of local label counts."""
    return np.stack([np.bincount(c.y.numpy(), minlength=num_classes) for c in clients])


def plot_run(history, counts, params, tag):
    """Save one run's accuracy curve and per-client label distribution."""
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


def train_federated(params, device):
    """Run one federated experiment for scalar settings and return its history."""
    fl = params.federated
    set_seed(params.run.seed)
    model = build_model(params).to(device)
    clients = build_clients(params, device)
    server = Server(model, clients, num_byzantine(fl), params, device)
    x_test, y_test = load_mnist_tensors(params, train=False)
    history = {"round": [], "loss": [], "test_acc": []}
    for r in range(1, fl.rounds + 1):
        model.train()
        loss = server.round()
        if r % fl.eval_every == 0 or r == fl.rounds:
            model.eval()
            acc = (predict(model, x_test, device) == y_test).float().mean().item()
            history["round"].append(r)
            history["loss"].append(loss)
            history["test_acc"].append(acc)
            print(f"  Round {r:3d}  loss {loss:.4f}  test acc {acc:.4f}")
    tag = run_tag(fl)
    torch.save(model.state_dict(), f"fl_{params.model.model}_{tag}.pth")
    counts = label_histogram(clients, params.model.num_classes)
    save_json(
        {"settings": dataclasses.asdict(fl), "history": history},
        params,
        f"federated_{params.model.model}_{tag}",
    )
    plot_run(history, counts, params, tag)
    return history


def describe(fl):
    """Return a one-line description of a combination's setup."""
    group = (
        f" in groups of {fl.group_size} ({fl.inner_aggregator} inside)"
        if fl.group_size > 1
        else ""
    )
    return (
        f"partition {fl.partition} | local {fl.local} | server {fl.server_opt} | "
        f"aggregator {fl.aggregator}{group} | attack {fl.attack} "
        f"({num_byzantine(fl)}/{fl.clients} Byzantine, f={assumed_attackers(fl)})"
    )


def run_federated(model, params, device):
    """Run every requested combination and save a summary table.

    The model passed in is not reused: each combination starts from a
    freshly initialized model with the same seed, so results are
    comparable. Infeasible combinations (for example Bulyan without
    n >= 4f + 3) are skipped with the reason.
    """
    summary = []
    runs = list(combinations(params.federated))
    for i, fl in enumerate(runs, 1):
        print(f"\n[{i}/{len(runs)}] {describe(fl)}")
        ok, reason = check_feasible(fl, assumed_attackers(fl))
        row = {name: getattr(fl, name) for name in SWEEP_FIELDS}
        if not ok:
            print(f"  skipped: {fl.aggregator} {reason}")
            summary.append({**row, "final_acc": None, "skipped": reason})
            continue
        history = train_federated(dataclasses.replace(params, federated=fl), device)
        summary.append({**row, "final_acc": history["test_acc"][-1]})

    print("\n=== Federated summary ===")
    print("  " + "  ".join(f"{n:>18s}" for n in SWEEP_FIELDS) + "  final acc")
    for row in summary:
        acc = "skipped" if row["final_acc"] is None else f"{row['final_acc']:.4f}"
        print("  " + "  ".join(f"{row[n]:>18s}" for n in SWEEP_FIELDS) + f"  {acc}")
    save_json(summary, params, f"federated_summary_{params.model.model}")
    return summary
