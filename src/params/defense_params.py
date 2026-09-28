"""Backdoor defense selection and per-defense hyperparameters."""

from dataclasses import dataclass

from src.defenses import DEFENSES

from src.utils.helper_params import option


@dataclass
class DefenseParams:
    """Backdoor defenses: selection and hyperparameters."""

    defense: list[str] = option(
        help="backdoor defenses to run",
        choices=DEFENSES.names,
        allow_all=True,
        default_factory=lambda: ["all"],
    )
    nc_samples: int = option(1000, "clean images for Neural Cleanse", gt=0)
    nc_steps: int = option(1000, "Neural Cleanse steps per class", gt=0)
    nc_lambda: float = option(1e-3, "Neural Cleanse initial mask cost", gt=0)
    nc_lr: float = option(0.1, "Neural Cleanse Adam learning rate", gt=0)
    nc_threshold: float = option(2.0, "Neural Cleanse anomaly-index threshold", gt=0)
    fp_max_drop: float = option(
        0.04, "Fine-pruning allowed validation accuracy drop", ge=0, le=1
    )
    fp_epochs: int = option(1, "Fine-pruning fine-tuning epochs", ge=0)
    fp_eval_samples: int = option(2000, "Fine-pruning validation images", gt=0)
    ls_samples: int = option(5000, "training images for latent separability", gt=0)
    ls_components: int = option(10, "latent separability PCA dimensions", gt=0)
    ls_min_fraction: float = option(
        0.35, "largest suspicious minority-cluster share", gt=0, le=0.5
    )
    ls_threshold: float = option(2.0, "latent separability anomaly threshold", gt=0)

    @property
    def defenses(self) -> list[str]:
        """Return the selected defense names; "all" selects every one."""
        return DEFENSES.expand(self.defense)
