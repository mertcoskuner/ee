"""Run federated training of an MNIST classifier and save its curves."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.federated import Client, Server, partition
from src.federated.attacks import LabelFlipClient
from src.utils.helper_data import load_mnist_tensors
from src.utils.helper_eval import predict
from src.utils.helper_plot import save_fig, save_json

matplotlib.use("Agg")


def build_clients(params, device):
    """Partition MNIST training data and create benign then Byzantine clients.

    The last round(byzantine_ratio * clients) clients are Byzantine; with
    the label_flip attack they are LabelFlipClient instances, otherwise
    their updates are crafted by the server-side attack.
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
    num_byzantine = round(fl.byzantine_ratio * fl.clients)
    clients = []
    for cid, idx in enumerate(shares):
        idx = torch.from_numpy(idx)
        malicious = cid >= fl.clients - num_byzantine and fl.attack == "label_flip"
        cls = LabelFlipClient if malicious else Client
        clients.append(cls(cid, x[idx], y[idx], params, device))
    return clients, num_byzantine


def label_histogram(clients, num_classes):
    """Return a clients x classes matrix of local label counts."""
    return np.stack([np.bincount(c.y.numpy(), minlength=num_classes) for c in clients])


def plot_federated(history, counts, params):
    """Save the accuracy curve and the per-client label distribution."""
    fig, (ax_acc, ax_lab) = plt.subplots(1, 2, figsize=(10.5, 3.6))
    ax_acc.plot(
        history["round"], [100 * a for a in history["test_acc"]], color="#2a78d6"
    )
    ax_acc.set_xlabel("round")
    ax_acc.set_ylabel("test accuracy (%)")
    ax_acc.set_ylim(0, 100)
    ax_acc.grid(alpha=0.3)
    shares = counts / counts.sum(1, keepdims=True)
    ax_lab.imshow(shares.T, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax_lab.set_xlabel("client")
    ax_lab.set_ylabel("class")
    ax_lab.set_title(f"label shares ({params.federated.partition})", fontsize=9)
    fig.tight_layout()
    save_fig(fig, params, f"federated_{params.model.model}")


def run_federated(model, params, device):
    """Train model with federated learning and return the round history.

    Print the setup, the mean benign training loss and test accuracy every
    eval_every rounds, then save the final global model, a JSON history,
    and plots to results_dir.
    """
    fl = params.federated
    clients, num_byzantine = build_clients(params, device)
    server = Server(model, clients, num_byzantine, params, device)
    x_test, y_test = load_mnist_tensors(params, train=False)
    counts = label_histogram(clients, params.model.num_classes)

    print(
        f"Federated: {fl.clients} clients ({num_byzantine} Byzantine, attack "
        f"{fl.attack}), participation {fl.participation}, partition {fl.partition}"
    )
    print(
        f"Local: {fl.local}, {fl.local_steps} steps, lr {fl.local_lr}  |  "
        f"Server: {fl.server_opt} lr {fl.server_lr}  |  Aggregator: "
        f"{fl.aggregator}"
        + (
            f" over groups of {fl.group_size} ({fl.inner_aggregator} inside)"
            if fl.group_size > 1
            else ""
        )
    )
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

    path = f"fl_{params.model.model}.pth"
    torch.save(model.state_dict(), path)
    print(f"  Saved global model ({path})")
    save_json(
        {"settings": vars(fl), "history": history, "label_counts": counts.tolist()},
        params,
        f"federated_{params.model.model}",
    )
    plot_federated(history, counts, params)
    return history
