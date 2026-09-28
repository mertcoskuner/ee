"""Model choice, architecture sizes, checkpoint paths, and output tags."""

from dataclasses import dataclass, field


@dataclass
class ModelParams:
    """Store the architecture(s), their sizes, and derived checkpoint names.

    model lists one or more architectures; each run replaces it with a
    single name and fills in save_path, weights, and tag.
    """

    model: list[str] = field(default_factory=lambda: ["cnn"])
    num_classes: int = 10
    dropout: float = 0.1
    hidden_sizes: tuple[int, ...] = (512, 256)
    patch_size: int = 7
    dim: int = 64
    depth: int = 2
    heads: int = 4
    mlp_dim: int = 128
    save_path: str = "best_cnn_adam.pth"
    weights: str = "best_cnn_adam.pth"
    tag: str = "best_cnn_adam"


def get_model_params(args) -> ModelParams:
    """Build model settings from validated CLI arguments."""
    return ModelParams(
        model=args.model,
        dropout=args.dropout,
        hidden_sizes=tuple(args.hidden_sizes),
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
    )
