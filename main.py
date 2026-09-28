"""Run MNIST experiments for every requested combination of settings.

Each mode maps to one runner. The model, optimizer, training-attack, and
backdoor options may list several values; every combination is run in
turn and, for modes that evaluate attacks, summarised in one table.
"""

from config.args import args_parser
from defense import run_defense
from federated import run_federated
from geometry import run_geometry
from gradcam import run_gradcam
from privacy import run_dp_accounting, run_dp_mechanisms
from src.params import get_params
from src.utils.helper_experiments import central_runs
from src.utils.helper_model import build_model
from src.utils.helper_plot import print_test_summary
from src.utils.helper_run import resolve_device, set_seed
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
    "geometry": run_geometry,
    "defense": run_defense,
    "dp_mechanisms": run_dp_mechanisms,
    "dp_accounting": run_dp_accounting,
    "federated": run_federated,
}

MODEL_FREE = {"dp_mechanisms", "dp_accounting"}


def describe_run(run, params, device):
    """Return the settings of one run that apply to its mode."""
    mode = params.run.mode
    if mode in MODEL_FREE:
        return f"mode {mode}"
    if mode == "federated":
        return f"mode {mode} | model {run['model']} | device {device}"
    text = (
        f"mode {mode} | model {run['model']} | optimizer {run['optimizer']} | "
        f"train attack {run['train_attack']} | backdoor {run['backdoor']}"
    )
    if params.privacy.dp:
        text += f" | DP-SGD noise {params.privacy.dp_noise:g}"
    return f"{text} | checkpoint {params.model.weights} | device {device}"


def main():
    """Run the selected mode for every central combination of settings.

    Modes in MODEL_FREE use no model, so they run once.
    """
    params = get_params(args_parser())
    device = resolve_device(params.run.device)
    runs = list(central_runs(params))
    if params.run.mode in MODEL_FREE:
        runs = runs[:1]
    rows = []
    for i, (run, run_params) in enumerate(runs, 1):
        set_seed(run_params.run.seed)
        print(f"\n##### [{i}/{len(runs)}] {describe_run(run, run_params, device)}")
        model = None
        if run_params.run.mode not in MODEL_FREE:
            model = build_model(run_params).to(device)
        result = RUNNERS[run_params.run.mode](model, run_params, device)
        if run_params.run.mode in ("test", "both"):
            rows.append((run, result))
    if len(rows) > 1:
        print_test_summary(rows, params)


if __name__ == "__main__":
    main()
