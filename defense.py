"""Run backdoor defenses, then optionally evaluate attacks on the defended model."""

import dataclasses

import matplotlib

from src.defenses import DEFENSES
from src.utils.helper_data import load_test_set
from src.utils.helper_eval import accuracy, predict
from src.utils.helper_model import load_weights
from src.utils.helper_plot import save_json
from test import evaluate_attacks

matplotlib.use("Agg")


def run_defense(model, params, device):
    """Load the checkpoint, run the selected defenses, and save a JSON report.

    Defenses that modify the model run after the analysis-only ones.
    Afterwards the defended model is evaluated: clean accuracy and its drop
    against the model before the defenses, the trigger's attack success
    rate for a backdoored model, and every attack given with --attack, so
    each defense combines with every backdoor and evasion attack.
    """
    model = load_weights(model, params, device)
    x, y = load_test_set(params)
    before = accuracy(predict(model, x, device), y)
    names = sorted(
        DEFENSES.expand(params.defense.defense),
        key=lambda n: DEFENSES.meta(n, "modifies_model", False),
    )
    report = {name: DEFENSES.get(name)(model, params, device) for name in names}
    if params.attack.attack is None:
        params = dataclasses.replace(
            params, attack=dataclasses.replace(params.attack, attack=["none"])
        )
    report["after_defense"] = evaluate_attacks(
        model, params, device, reference_clean=before
    )
    save_json(report, params, f"defense_{params.model.tag}")
    return report
