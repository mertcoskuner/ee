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
from src.utils.helper_federated import build_clients, describe, label_histogram
from src.utils.helper_model import build_model
from src.utils.helper_plot import (
    plot_federated_curves,
    plot_label_shares,
    print_federated_summary,
    save_json,
)
from src.utils.helper_privacy import federated_epsilon
from src.utils.helper_run import checkpoint_path, set_seed

matplotlib.use("Agg")


def train_federated(params, device):
    """Run one federated experiment and return (history, label counts).

    The final global model is saved to the checkpoint directory.
    """
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
    stem = f"fl_{params.model.model}_{fl_run_tag(fl)}"
    torch.save(model.state_dict(), checkpoint_path(params, stem))
    return history, label_histogram(clients, params.model.num_classes)


def run_federated(model, params, device):
    """Run every requested combination and save one report and figure.

    The model passed in is not reused: each combination starts from a
    freshly initialized model with the same seed, so results are
    comparable. Infeasible combinations (for example Bulyan without
    n >= 4f + 3) are skipped with the reason. The results directory gets
    federated_<model>.json (settings, curves, and final accuracy of every
    run), federated_<model>.png (all accuracy curves), and one
    label_shares_<partition>.png per data partition.
    """
    runs, summary, partitions = [], [], set()
    combos = list(fl_combinations(params.federated))
    for i, fl in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}] {describe(fl)}")
        ok, reason = check_feasible(fl, assumed_attackers(fl))
        row = {name: getattr(fl, name) for name in SWEEP_FIELDS}
        if not ok:
            print(f"  skipped: {fl.aggregator} {reason}")
            summary.append({**row, "final_acc": None, "skipped": reason})
            continue
        history, counts = train_federated(
            dataclasses.replace(params, federated=fl), device
        )
        if fl.partition not in partitions:
            partitions.add(fl.partition)
            plot_label_shares(counts, params, fl.partition)
        runs.append({"tag": fl_run_tag(fl), "history": history})
        eps = federated_epsilon(fl)
        if eps is not None:
            print(f"  Privacy spent: epsilon {eps:.3f} (delta {fl.dp_delta})")
        summary.append({**row, "final_acc": history["test_acc"][-1], "dp_epsilon": eps})
    if len(combos) > 1:
        print_federated_summary(summary, SWEEP_FIELDS)
    save_json(
        {
            "settings": dataclasses.asdict(params.federated),
            "runs": [
                {**row, "history": run["history"]}
                for row, run in zip(
                    [r for r in summary if r["final_acc"] is not None], runs
                )
            ],
            "skipped": [r for r in summary if r["final_acc"] is None],
        },
        params,
        f"federated_{params.model.model}",
    )
    if runs:
        plot_federated_curves(runs, params)
    return summary
