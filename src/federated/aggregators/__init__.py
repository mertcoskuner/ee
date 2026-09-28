"""Federated aggregation rules, registered by name in AGGREGATORS.

Each module implements a rule called as rule(updates, weights) and
registers a builder build(fl_params, f) under its CLI name, optionally with
feasible(n, f) -> (ok, reason) for rules that need enough honest clients.
Every module is imported automatically; TwoLayer wraps any two rules.
"""

import math

from src.utils.helper_registry import import_submodules

from .bulyan import Bulyan
from .centered_clipping import CenteredClipping
from .fedavg import FedAvg
from .geometric_median import GeometricMedian
from .krum import Krum
from .median import CoordinateMedian
from .norm_clipping import NormClipping
from .outlier_removal import OutlierRemoval
from .registry import AGGREGATORS
from .trimmed_mean import TrimmedMean
from .two_layer import TwoLayer

import_submodules(__name__, __path__)


def build_aggregator(fl, f, generator=None):
    """Return fl.aggregator, wrapped in two-layer grouping when group_size > 1.

    In two-layer mode fl.inner_aggregator combines each group and
    fl.aggregator combines the groups.
    """
    outer = AGGREGATORS.get(fl.aggregator)(fl, f)
    if fl.group_size > 1:
        inner = AGGREGATORS.get(fl.inner_aggregator)(fl, f)
        return TwoLayer(inner, outer, fl.group_size, generator)
    return outer


def check_feasible(fl, f):
    """Return (ok, reason) for running fl.aggregator with f assumed attackers.

    The outer rule sees one input per participating client, or one per
    group in two-layer mode.
    """
    n = max(1, round(fl.participation * fl.clients))
    if fl.group_size > 1:
        n = math.ceil(n / fl.group_size)
    feasible = AGGREGATORS.meta(fl.aggregator, "feasible")
    return feasible(n, f) if feasible else (True, "")


__all__ = [
    "AGGREGATORS",
    "Bulyan",
    "CenteredClipping",
    "CoordinateMedian",
    "FedAvg",
    "GeometricMedian",
    "Krum",
    "NormClipping",
    "OutlierRemoval",
    "TrimmedMean",
    "TwoLayer",
    "build_aggregator",
    "check_feasible",
]
