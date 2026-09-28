"""Demonstrate differential privacy mechanisms, properties, and accounting."""

import math

import matplotlib
import numpy as np
import torch

from src.privacy import (
    advanced_composition,
    dp_sgd_epsilon,
    gaussian_mechanism,
    laplace_mechanism,
    randomized_response,
    randomized_response_estimate,
    subsampled,
)
from src.utils.helper_data import load_mnist_tensors
from src.utils.helper_plot import plot_dp_accounting, plot_dp_mechanisms, save_json
from src.utils.helper_privacy import dp_sgd_rate, training_size
from src.utils.helper_stats import mean_abs_error

matplotlib.use("Agg")

GROUP = 4
COUNT_IMAGES = 100


def run_dp_mechanisms(model, params, device):
    """Compare basic DP mechanisms and properties on MNIST queries.

    For every epsilon in --dp_epsilons, measure the error of: the Laplace
    and Gaussian mechanisms on a counting query (how many of the first
    COUNT_IMAGES training images show a 0, sensitivity 1); the same Laplace
    answer clamped to the valid range [0, COUNT_IMAGES] (post-processing
    never costs privacy and reduces error when noise is large);
    randomized response estimating the fraction of zeros over all training
    images (local DP); answering GROUP counting queries by splitting
    epsilon among them (sequential composition); and protecting a group of
    GROUP people, whose sensitivity is GROUP (group privacy). The classic
    Gaussian calibration holds only for epsilon < 1, so larger budgets
    have no Gaussian entry. The model is not used.
    """
    p = params.privacy
    _, y = load_mnist_tensors(params, train=True)
    bits = y == 0
    n = COUNT_IMAGES
    count = float(bits[:n].sum())
    generator = torch.Generator().manual_seed(params.run.seed)
    trials = p.dp_trials
    report = {"epsilons": p.dp_epsilons, "true_count": count, "k": GROUP}
    for key in (
        "laplace_error",
        "gaussian_error",
        "laplace_clamped_error",
        "rr_error",
        "composition_error",
        "group_error",
    ):
        report[key] = []
    for eps in p.dp_epsilons:
        noisy = laplace_mechanism(torch.full((trials,), count), 1.0, eps, generator)
        report["laplace_error"].append(float((noisy - count).abs().mean()))
        clamped = noisy.clamp(0, n)
        report["laplace_clamped_error"].append(float((clamped - count).abs().mean()))
        report["gaussian_error"].append(
            mean_abs_error(
                lambda: gaussian_mechanism(
                    count, 1.0, eps, p.dp_delta, generator
                ).item(),
                count,
                trials,
            )
            if eps < 1
            else None
        )
        report["rr_error"].append(
            mean_abs_error(
                lambda: randomized_response_estimate(
                    randomized_response(bits, eps, generator), eps
                ),
                float(bits.float().mean()),
                trials,
            )
        )
        report["composition_error"].append(
            mean_abs_error(
                lambda: laplace_mechanism(count, 1.0, eps / GROUP, generator).item(),
                count,
                trials,
            )
        )
        report["group_error"].append(
            mean_abs_error(
                lambda: laplace_mechanism(count, GROUP, eps, generator).item(),
                count,
                trials,
            )
        )
    print("\n=== Differential privacy mechanisms (mean absolute error) ===")
    print(f"  Counting query: {int(count)} of the first {n} training images are zeros")
    print(
        f"  {'epsilon':>8s} {'Laplace':>9s} {'Gaussian':>9s} {'clamped':>9s} "
        f"{'RR frac':>9s} {'k-comp':>9s} {'group':>9s}"
    )
    for i, eps in enumerate(p.dp_epsilons):
        gauss = report["gaussian_error"][i]
        gauss = "-" if gauss is None else f"{gauss:.2f}"
        print(
            f"  {eps:>8g} {report['laplace_error'][i]:>9.2f} {gauss:>9s} "
            f"{report['laplace_clamped_error'][i]:>9.2f} "
            f"{report['rr_error'][i]:>9.4f} {report['composition_error'][i]:>9.2f} "
            f"{report['group_error'][i]:>9.2f}"
        )
    save_json(report, params, "dp_mechanisms")
    plot_dp_mechanisms(report, params)
    return report


def run_dp_accounting(model, params, device):
    """Compare privacy accounting methods for the configured DP-SGD run.

    With noise --dp_noise, Poisson sampling rate batch_size / training
    size, and --dp_delta, report epsilon after each number of steps up to
    --epochs passes under basic composition and advanced composition of
    the per-step (amplified) Gaussian mechanism, and under RDP without
    and with subsampling. The model is not used.
    """
    p = params.privacy
    q = dp_sgd_rate(params)
    total = params.training.epochs * max(1, round(1 / q))
    steps = sorted({int(s) for s in np.geomspace(1, total, 30)})
    delta_step = p.dp_delta / (2 * total)
    eps_step = math.sqrt(2 * math.log(1.25 / delta_step)) / p.dp_noise
    eps_sub, delta_sub = subsampled(eps_step, delta_step, q)
    report = {
        "noise": p.dp_noise,
        "rate": q,
        "delta": p.dp_delta,
        "training_size": training_size(params),
        "steps": steps,
        "basic": [k * eps_sub for k in steps],
        "advanced": [
            advanced_composition(eps_sub, delta_sub, k, p.dp_delta / 2)[0]
            for k in steps
        ],
        "rdp_full_batch": [
            dp_sgd_epsilon(1.0, p.dp_noise, k, p.dp_delta) for k in steps
        ],
        "rdp_subsampled": [dp_sgd_epsilon(q, p.dp_noise, k, p.dp_delta) for k in steps],
    }
    print("\n=== Privacy accounting ===")
    print(
        f"  noise {p.dp_noise}, sampling rate {q:.4f}, delta {p.dp_delta}, "
        f"{total} steps ({params.training.epochs} epochs)"
    )
    for key, label in (
        ("basic", "basic composition"),
        ("advanced", "advanced composition"),
        ("rdp_full_batch", "RDP, no subsampling"),
        ("rdp_subsampled", "RDP, subsampled (DP-SGD)"),
    ):
        print(f"  {label:28s} epsilon {report[key][-1]:.3f}")
    save_json(report, params, "dp_accounting")
    plot_dp_accounting(report, params)
    return report
