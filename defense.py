"""Run backdoor defenses, then optionally evaluate attacks on the defended model."""

import matplotlib

from src.defenses import DEFENSES
from src.utils.helper_model import load_weights
from src.utils.helper_plot import save_json
from test import evaluate_attacks

matplotlib.use("Agg")


def run_defense(model, params, device):
    """Load the checkpoint, run the selected defenses, and save a JSON report.

    Defenses that modify the model run after the analysis-only ones. When
    --attack is given, the selected attacks are evaluated on the model the
    defenses leave behind, so every backdoor defense can be combined with
    every evasion attack.
    """
    model = load_weights(model, params, device)
    names = sorted(
        DEFENSES.expand(params.defense.defense),
        key=lambda n: DEFENSES.meta(n, "modifies_model", False),
    )
    report = {name: DEFENSES.get(name)(model, params, device) for name in names}
    if params.attack.attack is not None:
        report["attacks_after_defense"] = evaluate_attacks(model, params, device)
    save_json(report, params, f"defense_{params.model.tag}")
    return report
