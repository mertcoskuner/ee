"""Run federated training over every requested combination of settings."""

import dataclasses

import matplotlib
import torch

from src.federated import Server, assumed_attackers, num_byzantine
from src.federated.aggregators import check_feasible
from src.params.federated_params import SWEEP_FIELDS
from src.utils.helper_data import load_mnist_tensors
from src.utils.helper_eval import accuracy, predict
from src.utils.helper_experiments import fl_combinations, fl_run_tag
from src.utils.helper_federated import build_clients, label_histogram
from src.utils.helper_model import build_model
from src.utils.helper_plot import plot_federated_run, print_federated_summary, save_json
from src.utils.helper_run import set_seed

matplotlib.use("Agg")


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
            acc = accuracy(predict(model, x_test, device), y_test)
            history["round"].append(r)
            history["loss"].append(loss)
            history["test_acc"].append(acc)
            print(f"  Round {r:3d}  loss {loss:.4f}  test acc {acc:.4f}")
    tag = fl_run_tag(fl)
    torch.save(model.state_dict(), f"fl_{params.model.model}_{tag}.pth")
    save_json(
        {"settings": dataclasses.asdict(fl), "history": history},
        params,
        f"federated_{params.model.model}_{tag}",
    )
    counts = label_histogram(clients, params.model.num_classes)
    plot_federated_run(history, counts, params, tag)
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
    runs = list(fl_combinations(params.federated))
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
    print_federated_summary(summary, SWEEP_FIELDS, params)
    return summary
