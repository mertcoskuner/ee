"""Backdoor defense selection and per-defense hyperparameters."""

from dataclasses import dataclass, field

from src.defenses import DEFENSES


@dataclass
class DefenseParams:
    """Store the selected defenses and their hyperparameters."""

    defense: list[str] = field(default_factory=lambda: ["all"])
    nc_samples: int = 1000
    nc_steps: int = 1000
    nc_lambda: float = 1e-3
    nc_lr: float = 0.1
    nc_threshold: float = 2.0
    fp_max_drop: float = 0.04
    fp_epochs: int = 1
    fp_eval_samples: int = 2000
    ls_samples: int = 5000
    ls_components: int = 10
    ls_min_fraction: float = 0.4
    ls_threshold: float = 2.0

    @property
    def defenses(self) -> list[str]:
        """Return the selected defense names; "all" selects every one."""
        return DEFENSES.expand(self.defense)


def get_defense_params(args) -> DefenseParams:
    """Build backdoor defense settings from validated CLI arguments."""
    return DefenseParams(
        defense=args.defense,
        nc_samples=args.nc_samples,
        nc_steps=args.nc_steps,
        nc_lambda=args.nc_lambda,
        nc_lr=args.nc_lr,
        nc_threshold=args.nc_threshold,
        fp_max_drop=args.fp_max_drop,
        fp_epochs=args.fp_epochs,
        fp_eval_samples=args.fp_eval_samples,
        ls_samples=args.ls_samples,
        ls_components=args.ls_components,
        ls_min_fraction=args.ls_min_fraction,
        ls_threshold=args.ls_threshold,
    )
