"""Report clean and adversarial test accuracy, overall and per class."""

from src.attacks import attack_label
from src.backdoors import attack_success_rate, build_backdoor
from src.utils.helper_data import load_test_set
from src.utils.helper_eval import predict_under_attack
from src.utils.helper_model import load_weights


def evaluate_attacks(model, params, device):
    """Print overall and per-class accuracy for clean and attacked inputs.

    Evaluate the prepared (evaluation-mode) model on the test set under
    every selected attack and return overall accuracies keyed by attack
    name, including "clean". For a backdoored model, also report the
    trigger's attack success rate under the key "backdoor_asr".
    """
    x, y = load_test_set(params)
    results = {}
    print(f"\n=== Test Results ({params.model.weights}, {len(x)} images) ===")
    for name in ["clean"] + params.attack.attacks:
        preds = predict_under_attack(model, x, y, name, params, device)
        acc = preds.eq(y).float().mean().item()
        results[name] = acc
        print(
            f"\n{attack_label(name, params)}: accuracy {acc:.4f}  "
            f"({preds.eq(y).sum().item()}/{len(y)})"
        )
        for c in range(params.model.num_classes):
            mask = y == c
            print(
                f"  Class {c}: {preds[mask].eq(c).float().mean().item():.4f}  "
                f"({preds[mask].eq(c).sum().item()}/{mask.sum().item()})"
            )
    trigger = build_backdoor(params.backdoor)
    if trigger is not None:
        target = params.backdoor.target_class
        asr = attack_success_rate(model, x, y, trigger, target, device)
        results["backdoor_asr"] = asr
        print(
            f"\nBackdoor {params.backdoor.backdoor} -> class {target}: "
            f"attack success rate {asr:.4f}"
        )
    return results


def run_test(model, params, device):
    """Load the selected checkpoint and evaluate every selected attack."""
    return evaluate_attacks(load_weights(model, params, device), params, device)
