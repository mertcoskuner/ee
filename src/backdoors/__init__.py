"""Backdoor (Trojan) triggers and data poisoning, registered in BACKDOORS.

Each trigger module registers a class built from the backdoor parameters
with an apply(batch) -> triggered batch method; every module is imported
automatically. poison.py wraps training data and measures attack success.
"""

from src.utils.helper_registry import import_submodules

from .poison import PoisonedDataset, attack_success_rate, build_backdoor
from .registry import BACKDOORS

import_submodules(__name__, __path__, skip=("registry", "poison"))

__all__ = ["BACKDOORS", "PoisonedDataset", "attack_success_rate", "build_backdoor"]
