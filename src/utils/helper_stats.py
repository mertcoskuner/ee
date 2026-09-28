"""Small statistics shared by defenses, aggregators, and the DP demonstrations."""

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


def mean_abs_error(release, truth, trials):
    """Return the mean absolute error of `trials` calls of release()."""
    return float(np.mean([abs(release() - truth) for _ in range(trials)]))
