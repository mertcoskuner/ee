"""Model choice, architecture sizes, checkpoint paths, and output tags."""

from dataclasses import dataclass

from models import MODELS

from src.utils.helper_params import option


@dataclass
class ModelParams:
    """Model: architecture, sizes, and derived checkpoint names."""

    model: list[str] = option(
        help="architecture(s) from models/",
        choices=MODELS.names,
        allow_all=True,
        default_factory=lambda: ["cnn"],
    )
    dropout: float = option(0.1, "dropout rate of MLP and Transformer", ge=0, lt=1)
    hidden_sizes: tuple[int, ...] = option((512, 256), "MLP hidden layer sizes", gt=0)
    patch_size: int = option(7, "Transformer patch size (divides 28)", gt=0)
    dim: int = option(64, "Transformer embedding size", gt=0)
    depth: int = option(2, "Transformer encoder layers", gt=0)
    heads: int = option(4, "Transformer attention heads", gt=0)
    mlp_dim: int = option(128, "Transformer feed-forward size", gt=0)
    num_classes: int = option(10, cli=False)
    save_path: str = option("best_cnn.pth", cli=False)
    weights: str = option("best_cnn.pth", cli=False)
    tag: str = option("best_cnn", cli=False)
