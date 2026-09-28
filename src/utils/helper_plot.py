"""Save experiment figures in PDF and PNG formats."""

import os

import matplotlib.pyplot as plt


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
