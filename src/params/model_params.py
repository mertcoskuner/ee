"""Model dimensions, checkpoint paths, and output identifiers."""

from dataclasses import dataclass


@dataclass
class ModelParams:
    """Store class count, checkpoint paths, and output tag."""

    num_classes: int = 10
    save_path: str = "best_model.pth"
    weights: str = "best_model.pth"
    tag: str = "best_model"


def get_model_params(args) -> ModelParams:
    """Name the checkpoint after the training regime."""
    save_path = "best_adv_model.pth" if args.adv_train else "best_model.pth"
    return ModelParams(
        save_path=save_path,
        weights=save_path,
        tag=save_path.removesuffix(".pth"),
    )
