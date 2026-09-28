"""Basic differential privacy mechanisms: Laplace, Gaussian, randomized response."""

import math

import torch


def laplace_mechanism(value, sensitivity, epsilon, generator=None):
    """Return value plus Laplace(sensitivity / epsilon) noise (epsilon-DP)."""
    value = torch.as_tensor(value, dtype=torch.float64)
    u = torch.rand(value.shape, generator=generator, dtype=torch.float64) - 0.5
    scale = sensitivity / epsilon
    return value - scale * torch.sign(u) * torch.log1p(-2 * u.abs())


def gaussian_sigma(sensitivity, epsilon, delta):
    """Return the classic Gaussian-mechanism noise scale for (epsilon, delta)-DP.

    sigma = sqrt(2 ln(1.25 / delta)) * sensitivity / epsilon, valid for
    epsilon < 1 (Dwork and Roth, Theorem A.1).
    """
    return math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon


def gaussian_mechanism(value, sensitivity, epsilon, delta, generator=None):
    """Return value plus Gaussian noise calibrated for (epsilon, delta)-DP."""
    value = torch.as_tensor(value, dtype=torch.float64)
    sigma = gaussian_sigma(sensitivity, epsilon, delta)
    noise = torch.randn(value.shape, generator=generator, dtype=torch.float64)
    return value + sigma * noise


def randomized_response(bits, epsilon, generator=None):
    """Report each bit truthfully with probability e^eps / (1 + e^eps), else flip it.

    Every user randomizes its own bit, so this is epsilon-local-DP.
    """
    bits = torch.as_tensor(bits, dtype=torch.bool)
    keep = math.exp(epsilon) / (1 + math.exp(epsilon))
    truthful = torch.rand(bits.shape, generator=generator) < keep
    return torch.where(truthful, bits, ~bits)


def randomized_response_estimate(reports, epsilon):
    """Return the unbiased estimate of the true fraction of ones from RR reports."""
    keep = math.exp(epsilon) / (1 + math.exp(epsilon))
    observed = torch.as_tensor(reports, dtype=torch.float64).mean().item()
    return (observed - (1 - keep)) / (2 * keep - 1)
