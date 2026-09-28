"""No attack: every client is honest."""

from .registry import FL_ATTACKS


@FL_ATTACKS.register("none", rank=0, benign=True)
def build_none(fl):
    """Return None: there is no Byzantine behaviour to simulate."""
    return None
