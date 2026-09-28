"""Run MNIST experiments for every requested combination of settings.

Each mode maps to one runner. The model, optimizer, and training-attack
options may list several values; every combination is run in turn and,
for modes that evaluate attacks, summarised in one table.
"""

import torch

from config.args import args_parser
from defense import run_defense
from federated import run_federated
from gradcam import run_gradcam
from src.params import get_params
from src.utils.helper_experiments import central_runs
from src.utils.helper_model import build_model
from src.utils.helper_plot import save_json
from src.utils.helper_run import set_seed
from test import run_test
from train import run_training
from visualize import run_tsne, run_visualize


def run_both(model, params, device):
    """Train the model, then evaluate it clean and under attack."""
    run_training(model, params, device)
    return run_test(model, params, device)


RUNNERS = {
    "train": run_training,
    "test": run_test,
    "both": run_both,
    "visualize": run_visualize,
    "tsne": run_tsne,
    "gradcam": run_gradcam,
    "defense": run_defense,
    "federated": run_federated,
}


def print_summary(rows, params):
    """Print and save a table of accuracies, one row per combination."""
    columns = list(dict.fromkeys(k for _, res in rows for k in res))
    head = f"{'model':>12s} {'optimizer':>10s} {'train_attack':>13s}"
    print("\n=== Summary: test accuracy ===")
    print(head + "".join(f"{c:>10s}" for c in columns))
    for run, res in rows:
        line = f"{run['model']:>12s} {run['optimizer']:>10s} {run['train_attack']:>13s}"
        print(line + "".join(f"{res.get(c, float('nan')):>10.4f}" for c in columns))
    save_json(
        [{**run, "accuracy": res} for run, res in rows], params, "experiment_summary"
    )


def main():
    """Run the selected mode for every model/optimizer/train_attack combination."""
    params = get_params(args_parser())
    device = torch.device(params.run.device)
    runs = list(central_runs(params))
    rows = []
    for i, (run, run_params) in enumerate(runs, 1):
        set_seed(run_params.run.seed)
        print(
            f"\n##### [{i}/{len(runs)}] mode {run_params.run.mode} | "
            f"model {run['model']} | optimizer {run['optimizer']} | "
            f"train attack {run['train_attack']} | "
            f"checkpoint {run_params.model.weights} | device {device}"
        )
        model = build_model(run_params).to(device)
        result = RUNNERS[run_params.run.mode](model, run_params, device)
        if run_params.run.mode in ("test", "both"):
            rows.append((run, result))
    if rows:
        print_summary(rows, params)


if __name__ == "__main__":
    main()
