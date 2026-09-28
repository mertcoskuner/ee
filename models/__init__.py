"""MNIST classifier architectures, registered by name in MODELS."""

from src.utils.helper_registry import import_submodules

from .registry import MODELS

import_submodules(__name__, __path__)

__all__ = ["MODELS"]
