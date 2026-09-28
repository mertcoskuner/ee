"""Name-to-component registries that make models, attacks, and defenses plug and play.

A package creates a Registry, every module in the package registers its
component with the register decorator, and import_submodules imports all
modules so that dropping a new file into the package is enough to make the
component selectable from the command line.
"""

import importlib
import pkgutil


class Registry:
    """Map names to components (functions or classes) plus optional metadata."""

    def __init__(self, kind):
        """Create an empty registry; kind names the component type in errors."""
        self.kind = kind
        self._items = {}
        self._meta = {}

    def register(self, name, rank=100, **meta):
        """Return a decorator that registers its target under name.

        rank orders names() (and so the expansion of "all"); meta stores
        extra information such as a display label or feasibility checks.
        """

        def decorator(obj):
            """Add obj to the registry and return it unchanged."""
            if name in self._items:
                raise ValueError(f"Duplicate {self.kind}: {name}")
            self._items[name] = obj
            self._meta[name] = {"rank": rank, **meta}
            return obj

        return decorator

    def get(self, name):
        """Return the component registered as name, or raise ValueError."""
        if name not in self._items:
            raise ValueError(
                f"Unknown {self.kind}: {name} (available: {', '.join(self.names())})"
            )
        return self._items[name]

    def meta(self, name, key, default=None):
        """Return the metadata value key of name, or default."""
        self.get(name)
        return self._meta[name].get(key, default)

    def names(self):
        """Return registered names ordered by rank, then by name."""
        return sorted(self._items, key=lambda n: (self._meta[n]["rank"], n))

    def expand(self, selection):
        """Expand a list of names where "all" means every registered name."""
        if selection is None or "all" in selection:
            return self.names()
        for name in selection:
            self.get(name)
        return list(dict.fromkeys(selection))

    def __contains__(self, name):
        """Return whether name is registered."""
        return name in self._items


def import_submodules(package_name, package_path, skip=("registry",)):
    """Import every module of a package so their register decorators run."""
    for info in pkgutil.iter_modules(package_path):
        if info.name not in skip:
            importlib.import_module(f"{package_name}.{info.name}")
