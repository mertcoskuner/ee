"""Run MNIST training, attack evaluation, or visual analysis."""

import torch

from config.args import args_parser
from gradcam import run_gradcam
from src.params import get_params
from src.utils.helper_model import build_model
from src.utils.helper_run import set_seed
from test import run_test
from train import run_training
from visualize import run_tsne, run_visualize


def main():
    """Build the configured model and execute the selected experiment mode."""
    params = get_params(args_parser())

    set_seed(params.run.seed)
    print(f"Seed set to: {params.run.seed}")
    print(f"Mode: {params.run.mode}  |  Weights: {params.model.weights}")

    device = torch.device(params.run.device)
    print(f"Using device: {device}")

    model = build_model(params).to(device)

    if params.run.mode in ("train", "both"):
        run_training(model, params, device)

    if params.run.mode in ("test", "both"):
        run_test(model, params, device)

    if params.run.mode == "visualize":
        run_visualize(model, params, device)

    if params.run.mode == "tsne":
        run_tsne(model, params, device)

    if params.run.mode == "gradcam":
        run_gradcam(model, params, device)


if __name__ == "__main__":
    main()
