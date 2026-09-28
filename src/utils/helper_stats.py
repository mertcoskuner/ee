"""Robust outlier statistics shared by backdoor defenses and FL aggregators."""

import numpy as np


def anomaly_indices(values):
    """Return MAD-based anomaly indices |v - median| / (1.4826 * MAD).

    The factor 1.4826 makes the median absolute deviation a consistent
    estimate of the standard deviation for normally distributed values.
    """
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    mad = 1.4826 * np.median(np.abs(values - median))
    return np.abs(values - median) / (mad + 1e-12)
