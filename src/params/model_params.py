"""Model dimensions, checkpoint paths, and output identifiers."""

from dataclasses import dataclass


@dataclass
class ModelParams:
    """Store class count, checkpoint source, save path, and output tag."""

    num_classes: int = 10
    pretrained: str = "none"
    save_path: str = "best_model.pth"
    weights: str = "best_model.pth"
    tag: str = "best_model"


def get_model_params(args) -> ModelParams:
    """Resolve checkpoint paths and tags for local or official models."""
    save_path = "best_adv_model.pth" if args.adv_train else "best_model.pth"
    if args.pretrained != "none":
        weights = f"checkpoints/madry_{args.pretrained}.pth"
        tag = args.pretrained
    else:
        weights = save_path
        tag = save_path.removesuffix(".pth")
    return ModelParams(
        pretrained=args.pretrained,
        save_path=save_path,
        weights=weights,
        tag=tag,
    )
