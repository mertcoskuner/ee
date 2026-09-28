"""Report clean accuracy, its drop, robust accuracy, and attack success rates."""

from src.attacks import attack_label
from src.backdoors import attack_success_rate, build_backdoor
from src.utils.helper_data import load_test_set
from src.utils.helper_eval import (
    accuracy,
    evasion_success_rate,
    per_class_accuracy,
    predict_under_attack,
)
from src.utils.helper_model import load_reference, load_surrogate, load_weights
from src.utils.helper_plot import save_json


def evaluate_attacks(model, params, device, reference_clean=None, save=True):
    """Print and return clean and attacked metrics of an evaluation-mode model.

    The returned dict holds:
    - "clean": clean test accuracy;
    - "clean_drop": reference clean accuracy minus clean accuracy, when a
      reference exists (reference_clean, or the clean checkpoint of the same
      architecture and optimizer for adversarially trained or backdoored
      models);
    - "<attack>": robust accuracy under each selected attack;
    - "<attack>_asr": attack success rate, the share of correctly classified
      images the attack turns wrong;
    - "backdoor_asr": for a backdoored model, the share of non-target images
      the trigger sends to the target class.
    With --surrogate_model the attacks are crafted on the surrogate and
    transferred (grey- or black-box); otherwise they are white-box. With
    save, the metrics and per-class accuracies go to test_<tag>.json.
    """
    x, y = load_test_set(params)
    surrogate = load_surrogate(params, device)
    origin = "white-box"
    if surrogate is not None:
        origin = f"transferred from {params.attack.surrogate_model}"
    subject = f"Test: {params.model.weights}" if save else "Defended model"
    print(f"\n=== {subject}, {len(x)} images, attacks {origin} ===")
    clean_preds = predict_under_attack(model, x, y, "clean", params, device)
    results = {"clean": accuracy(clean_preds, y)}
    per_class = {"clean": per_class_accuracy(clean_preds, y)}
    print(f"  Clean accuracy: {results['clean']:.4f}")
    reference_name = "before defense"
    if reference_clean is None:
        reference, reference_name = load_reference(params, device)
        if reference is not None:
            ref_preds = predict_under_attack(reference, x, y, "clean", params, device)
            reference_clean = accuracy(ref_preds, y)
    if reference_clean is not None:
        results["clean_drop"] = reference_clean - results["clean"]
        print(
            f"  Clean accuracy drop: {results['clean_drop']:+.4f} "
            f"(reference {reference_name}: {reference_clean:.4f})"
        )

    for name in params.attack.attacks:
        preds = predict_under_attack(model, x, y, name, params, device, surrogate)
        results[name] = accuracy(preds, y)
        results[f"{name}_asr"] = evasion_success_rate(clean_preds, preds, y)
        per_class[name] = per_class_accuracy(preds, y)
        print(
            f"  {attack_label(name, params)}: robust accuracy {results[name]:.4f}, "
            f"attack success rate {results[f'{name}_asr']:.4f}"
        )

    trigger = build_backdoor(params.backdoor)
    if trigger is not None:
        target = params.backdoor.target_class
        asr = attack_success_rate(model, x, y, trigger, target, device)
        results["backdoor_asr"] = asr
        print(
            f"  Backdoor {params.backdoor.backdoor} -> class {target}: "
            f"attack success rate {asr:.4f}"
        )
    if not save:
        return results
    name = params.model.tag
    if surrogate is not None:
        name += f"_from-{params.attack.surrogate_model}"
    save_json(
        {
            "checkpoint": params.model.weights,
            "attacks": origin,
            **results,
            "per_class": per_class,
        },
        params,
        f"test_{name}",
    )
    return results


def run_test(model, params, device):
    """Load the selected checkpoint and evaluate it clean and under attack."""
    return evaluate_attacks(load_weights(model, params, device), params, device)
