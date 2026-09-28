"""Federated aggregation rules: FedAvg, robust statistics, and outlier removal."""

from .bulyan import Bulyan
from .centered_clipping import CenteredClipping
from .fedavg import FedAvg
from .geometric_median import GeometricMedian
from .krum import Krum
from .median import CoordinateMedian
from .norm_clipping import NormClipping
from .outlier_removal import OutlierRemoval
from .trimmed_mean import TrimmedMean
from .two_layer import TwoLayer

FL_AGGREGATORS = [
    "fedavg",
    "median",
    "trimmed_mean",
    "krum",
    "multi_krum",
    "bulyan",
    "centered_clipping",
    "geometric_median",
    "norm_clipping",
    "outlier_removal",
]


def build_rule(name, fl, f):
    """Return the aggregation rule called name, assuming f Byzantine clients."""
    if name == "fedavg":
        return FedAvg()
    if name == "median":
        return CoordinateMedian()
    if name == "trimmed_mean":
        return TrimmedMean(f)
    if name == "krum":
        return Krum(f, m=1)
    if name == "multi_krum":
        return Krum(f, m=None)
    if name == "bulyan":
        return Bulyan(f)
    if name == "centered_clipping":
        return CenteredClipping(fl.cc_tau, fl.cc_iterations)
    if name == "geometric_median":
        return GeometricMedian(fl.gm_iterations)
    if name == "norm_clipping":
        return NormClipping(fl.clip_norm)
    if name == "outlier_removal":
        return OutlierRemoval(fl.outlier_threshold)
    raise ValueError(f"Unknown aggregator: {name}")


def build_aggregator(fl, f, generator=None):
    """Return fl.aggregator, wrapped in two-layer grouping when group_size > 1.

    In two-layer mode fl.inner_aggregator combines each group and
    fl.aggregator combines the groups; the outer rule then assumes at most
    f groups are corrupted.
    """
    if fl.group_size > 1:
        return TwoLayer(
            build_rule(fl.inner_aggregator, fl, f),
            build_rule(fl.aggregator, fl, f),
            fl.group_size,
            generator,
        )
    return build_rule(fl.aggregator, fl, f)


__all__ = [
    "Bulyan",
    "CenteredClipping",
    "CoordinateMedian",
    "FL_AGGREGATORS",
    "FedAvg",
    "GeometricMedian",
    "Krum",
    "NormClipping",
    "OutlierRemoval",
    "TrimmedMean",
    "TwoLayer",
    "build_aggregator",
    "build_rule",
]
