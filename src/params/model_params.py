"""Model choice, architecture sizes, checkpoint paths, and output tags."""

from dataclasses import dataclass


@dataclass
class ModelParams:
    """Store the architecture, its sizes, checkpoint paths, and output tag."""

    model: str = "cnn"
    num_classes: int = 10
    dropout: float = 0.1
    hidden_sizes: tuple[int, ...] = (512, 256)
    patch_size: int = 7
    dim: int = 64
    depth: int = 2
    heads: int = 4
    mlp_dim: int = 128
    save_path: str = "best_cnn.pth"
    weights: str = "best_cnn.pth"
    tag: str = "best_cnn"


def get_model_params(args) -> ModelParams:
    """Name the checkpoint after the model and the training regime."""
    tag = f"best_adv_{args.model}" if args.adv_train else f"best_{args.model}"
    return ModelParams(
        model=args.model,
        dropout=args.dropout,
        save_path=f"{tag}.pth",
        weights=f"{tag}.pth",
        tag=tag,
    )
