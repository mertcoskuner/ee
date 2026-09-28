"""Report clean and adversarial test accuracy, overall and per class."""

from src.utils.helper_data import load_test_set
from src.utils.helper_eval import predict_under_attack
from src.utils.helper_model import load_weights


def run_test(model, params, device):
    """Print overall and per-class accuracy for clean and attacked inputs.

    Load the selected checkpoint and return overall accuracies keyed by
    attack name, including clean accuracy.
    """
    x, y = load_test_set(params)
    model = load_weights(model, params, device)

    names = {
        "clean": "Clean",
        "fgsm": f"FGSM      (eps={params.attack.eps_linf})",
        "pgd_linf": f"PGD-linf  (eps={params.attack.eps_linf}, "
        f"{params.attack.steps} steps)",
        "pgd_l2": f"PGD-l2    (eps={params.attack.eps_l2}, "
        f"{params.attack.steps} steps)",
    }
    results = {}

    print(f"\n=== Test Results ({params.model.weights}, {len(x)} images) ===")
    for name in ["clean"] + params.attack.attacks:
        preds = predict_under_attack(model, x, y, name, params, device)
        acc = preds.eq(y).float().mean().item()
        results[name] = acc
        print(
            f"\n{names[name]}: accuracy {acc:.4f}  "
            f"({preds.eq(y).sum().item()}/{len(y)})"
        )
        for c in range(params.model.num_classes):
            mask = y == c
            print(
                f"  Class {c}: {preds[mask].eq(c).float().mean().item():.4f}  "
                f"({preds[mask].eq(c).sum().item()}/{mask.sum().item()})"
            )
    return results
