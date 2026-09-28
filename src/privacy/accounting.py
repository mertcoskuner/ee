"""Privacy accounting: basic and advanced composition, subsampling, and RDP."""

import math

ORDERS = list(range(2, 257))


def basic_composition(epsilon, delta, k):
    """Return (k epsilon, k delta) for k adaptive (epsilon, delta)-DP steps."""
    return k * epsilon, k * delta


def advanced_composition(epsilon, delta, k, delta_slack):
    """Return the advanced-composition bound for k (epsilon, delta)-DP steps.

    epsilon' = sqrt(2 k ln(1 / delta_slack)) epsilon + k epsilon (e^epsilon - 1)
    and delta' = k delta + delta_slack (Dwork, Rothblum and Vadhan, 2010).
    """
    eps = math.sqrt(2 * k * math.log(1 / delta_slack)) * epsilon + k * epsilon * (
        math.exp(epsilon) - 1
    )
    return eps, k * delta + delta_slack


def subsampled(epsilon, delta, q):
    """Return the privacy of an (epsilon, delta)-DP step run on a q-subsample.

    Amplification by subsampling: epsilon' = ln(1 + q (e^epsilon - 1)) and
    delta' = q delta.
    """
    return math.log1p(q * math.expm1(epsilon)), q * delta


def rdp_subsampled_gaussian(q, sigma, order):
    """Return the RDP of order `order` of the Poisson-subsampled Gaussian mechanism.

    Uses the exact integer-order expression of Mironov, Talwar and Zhang
    (2019); q = 1 gives the plain Gaussian mechanism, order / (2 sigma^2).
    """
    if q == 0:
        return 0.0
    if q == 1:
        return order / (2 * sigma**2)
    log_a = -math.inf
    for i in range(order + 1):
        log_coef = (
            math.lgamma(order + 1)
            - math.lgamma(i + 1)
            - math.lgamma(order - i + 1)
            + i * math.log(q)
            + (order - i) * math.log1p(-q)
        )
        term = log_coef + (i * i - i) / (2 * sigma**2)
        log_a = max(log_a, term) + math.log1p(math.exp(-abs(log_a - term)))
    return log_a / (order - 1)


def epsilon_from_rdp(rdp, delta, orders=ORDERS):
    """Convert RDP values at orders to the smallest (epsilon, delta)-DP epsilon.

    Uses the conversion of Balle et al. (2020):
    epsilon = rdp - (ln delta + ln alpha) / (alpha - 1) + ln((alpha - 1) / alpha).
    """
    return min(
        r - (math.log(delta) + math.log(a)) / (a - 1) + math.log((a - 1) / a)
        for r, a in zip(rdp, orders)
    )


def dp_sgd_epsilon(q, sigma, steps, delta, orders=ORDERS):
    """Return the epsilon of `steps` Poisson-subsampled Gaussian steps (RDP)."""
    rdp = [steps * rdp_subsampled_gaussian(q, sigma, a) for a in orders]
    return epsilon_from_rdp(rdp, delta, orders)
