"""Backdoor defense selection and per-defense hyperparameters."""

from dataclasses import dataclass

DEFENSES = ["neural_cleanse", "latent_separability", "fine_pruning"]


@dataclass
class DefenseParams:
    """Store the selected defenses and their hyperparameters."""

    defense: str = "all"
    nc_samples: int = 1000
    nc_steps: int = 300
    nc_lambda: float = 1e-2
    nc_lr: float = 0.1
    nc_threshold: float = 2.0
    fp_max_drop: float = 0.04
    fp_epochs: int = 1
    fp_eval_samples: int = 2000
    ls_samples: int = 5000
    ls_components: int = 10
    ls_min_fraction: float = 0.35
    ls_min_silhouette: float = 0.15

    @property
    def defenses(self) -> list[str]:
        """Expand the all selection into every supported defense name."""
        if self.defense == "all":
            return list(DEFENSES)
        return [self.defense]


def get_defense_params(args) -> DefenseParams:
    """Build backdoor defense settings from validated CLI arguments."""
    return DefenseParams(
        defense=args.defense,
        nc_steps=args.nc_steps,
        nc_lambda=args.nc_lambda,
        fp_max_drop=args.fp_max_drop,
        fp_epochs=args.fp_epochs,
        ls_samples=args.ls_samples,
    )
