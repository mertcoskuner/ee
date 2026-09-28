"""Backdoor defenses, registered by name in DEFENSES.

Each module implements a defense and registers a runner
run(model, params, device) -> report under its CLI name; every module in
this package is imported automatically. Runners registered with
modifies_model=True change the model in place and run last.
"""

from src.utils.helper_registry import import_submodules

from .fine_pruning import fine_pruning
from .latent_separability import latent_separability
from .neural_cleanse import neural_cleanse
from .registry import DEFENSES

import_submodules(__name__, __path__)

__all__ = ["DEFENSES", "fine_pruning", "latent_separability", "neural_cleanse"]
