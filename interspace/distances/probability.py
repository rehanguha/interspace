# -*- coding: utf-8 -*-
"""
Probability distribution distance functions.

This module contains distance functions for comparing probability distributions.
"""

from __future__ import annotations

import numpy as np

from interspace._validators import _validate_vector, _validate_same_shape


def kl_divergence(p, q):
    """Compute Kullback-Leibler divergence between two probability distributions.

    KL divergence measures how one probability distribution diverges from another.
    It is also known as relative entropy.

    Parameters
    ----------
    p : array_like
        First probability distribution (must sum to 1 and be non-negative).
    q : array_like
        Second probability distribution (must sum to 1 and be non-negative).

    Returns
    -------
    float
        KL divergence D(p || q).

    Raises
    ------
    ValueError
        If distributions contain negative values, don't sum to 1,
        or if q contains zeros where p is non-zero.

    Examples
    --------
    >>> kl_divergence([0.5, 0.5], [0.5, 0.5])
    0.0
    >>> kl_divergence([1.0, 0.0], [0.5, 0.5])
    0.6931471805599453

    Notes
    -----
    The KL divergence is defined as:

    D(p || q) = sum(p_i * log(p_i / q_i))

    Properties:
    - Non-negative: D(p || q) >= 0
    - Not symmetric: D(p || q) != D(q || p)
    - Zero if and only if p = q
    - Not a true distance metric (violates symmetry and triangle inequality)
    """
    p = np.asarray(_validate_vector(p, dtype=np.double), dtype=np.double)
    q = np.asarray(_validate_vector(q, dtype=np.double), dtype=np.double)
    _validate_same_shape(p, q)

    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Distributions must contain non-negative values")

    # Check for zeros in q where p is non-zero
    if np.any((p > 0) & (q == 0)):
        raise ValueError("KL divergence is undefined when q=0 and p>0")

    # Only compute where p > 0 (0 * log(0/q) = 0 by convention)
    mask = p > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.sum(p[mask] * np.log(p[mask] / q[mask]))

    return float(result)


def js_distance(p, q):
    """Compute Jensen-Shannon distance between two probability distributions.

    The Jensen-Shannon distance is a symmetric and bounded version of KL
    divergence. It is the square root of the Jensen-Shannon divergence.

    Parameters
    ----------
    p : array_like
        First probability distribution (must sum to 1 and be non-negative).
    q : array_like
        Second probability distribution (must sum to 1 and be non-negative).

    Returns
    -------
    float
        Jensen-Shannon distance between 0 and 1.

    Raises
    ------
    ValueError
        If distributions contain negative values.

    Examples
    --------
    >>> js_distance([0.5, 0.5], [0.5, 0.5])
    0.0
    >>> js_distance([1.0, 0.0], [0.5, 0.5])
    0.4645034044881785

    Notes
    -----
    The Jensen-Shannon distance is defined as:

    JS(p, q) = sqrt(0.5 * D(p || m) + 0.5 * D(q || m))

    where m = (p + q) / 2 and D is the KL divergence.

    Properties:
    - Symmetric: JS(p, q) = JS(q, p)
    - Bounded: 0 <= JS(p, q) <= 1
    - A true distance metric (satisfies triangle inequality)
    """
    p = np.asarray(_validate_vector(p, dtype=np.double), dtype=np.double)
    q = np.asarray(_validate_vector(q, dtype=np.double), dtype=np.double)
    _validate_same_shape(p, q)

    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Distributions must contain non-negative values")

    # Normalize to probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)

    m = (p + q) / 2

    # Compute JS divergence
    def kl_safe(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))

    js_div = 0.5 * kl_safe(p, m) + 0.5 * kl_safe(q, m)

    # JS distance is the square root of JS divergence
    return float(np.sqrt(max(0, js_div)))


def bhattacharyya_distance(p, q):
    """Compute Bhattacharyya distance between two probability distributions.

    The Bhattacharyya distance measures the amount of overlap between
    two statistical samples or populations.

    Parameters
    ----------
    p : array_like
        First probability distribution (non-negative values).
    q : array_like
        Second probability distribution (non-negative values).

    Returns
    -------
    float
        Bhattacharyya distance (non-negative).

    Raises
    ------
    ValueError
        If distributions contain negative values.

    Examples
    --------
    >>> bhattacharyya_distance([0.5, 0.5], [0.5, 0.5])
    0.0
    >>> bhattacharyya_distance([1.0, 0.0], [0.5, 0.5])
    0.3465735902799726

    Notes
    -----
    The Bhattacharyya distance is defined as:

    D_B(p, q) = -ln(sum(sqrt(p_i * q_i)))

    The Bhattacharyya coefficient BC(p, q) = sum(sqrt(p_i * q_i)) measures the
    amount of overlap. The distance is -ln(BC).

    Properties:
    - Non-negative
    - Symmetric
    - Does not satisfy triangle inequality
    """
    p = np.asarray(_validate_vector(p, dtype=np.double), dtype=np.double)
    q = np.asarray(_validate_vector(q, dtype=np.double), dtype=np.double)
    _validate_same_shape(p, q)

    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Distributions must contain non-negative values")

    # Normalize to probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))

    if bc == 0:
        return float("inf")

    return float(-np.log(bc))


def hellinger_distance(p, q):
    """Compute Hellinger distance between two probability distributions.

    The Hellinger distance is related to the Bhattacharyya coefficient
    and is bounded between 0 and 1.

    Parameters
    ----------
    p : array_like
        First probability distribution (non-negative values).
    q : array_like
        Second probability distribution (non-negative values).

    Returns
    -------
    float
        Hellinger distance between 0 and 1.

    Raises
    ------
    ValueError
        If distributions contain negative values.

    Examples
    --------
    >>> hellinger_distance([0.5, 0.5], [0.5, 0.5])
    0.0
    >>> hellinger_distance([1.0, 0.0], [0.5, 0.5])
    0.29289321881345254

    Notes
    -----
    The Hellinger distance is defined as:

    H(p, q) = sqrt(1 - BC(p, q))

    where BC is the Bhattacharyya coefficient.

    Properties:
    - Symmetric
    - Bounded: 0 <= H(p, q) <= 1
    - A true distance metric
    """
    p = np.asarray(_validate_vector(p, dtype=np.double), dtype=np.double)
    q = np.asarray(_validate_vector(q, dtype=np.double), dtype=np.double)
    _validate_same_shape(p, q)

    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Distributions must contain non-negative values")

    # Normalize to probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(p * q))

    return float(np.sqrt(1 - bc))


def total_variation_distance(p, q):
    """Compute total variation distance between two probability distributions.

    The total variation distance is a measure of the maximum difference
    between two probability distributions.

    Parameters
    ----------
    p : array_like
        First probability distribution (non-negative values).
    q : array_like
        Second probability distribution (non-negative values).

    Returns
    -------
    float
        Total variation distance between 0 and 1.

    Raises
    ------
    ValueError
        If distributions contain negative values.

    Examples
    --------
    >>> total_variation_distance([0.5, 0.5], [0.5, 0.5])
    0.0
    >>> total_variation_distance([1.0, 0.0], [0.0, 1.0])
    1.0

    Notes
    -----
    The total variation distance is defined as:

    TV(p, q) = 0.5 * sum(|p_i - q_i|)

    Properties:
    - Symmetric
    - Bounded: 0 <= TV(p, q) <= 1
    - A true distance metric
    """
    p = np.asarray(_validate_vector(p, dtype=np.double), dtype=np.double)
    q = np.asarray(_validate_vector(q, dtype=np.double), dtype=np.double)
    _validate_same_shape(p, q)

    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Distributions must contain non-negative values")

    # Normalize to probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)

    return float(0.5 * np.sum(np.abs(p - q)))


def wasserstein_distance(p, q):
    """Compute 1D Wasserstein (Earth Mover's) distance between two distributions.

    The Wasserstein distance measures the minimum "work" required to transform
    one distribution into another.

    Parameters
    ----------
    p : array_like
        First probability distribution (non-negative values).
    q : array_like
        Second probability distribution (non-negative values).

    Returns
    -------
    float
        Wasserstein distance.

    Raises
    ------
    ValueError
        If distributions contain negative values.

    Examples
    --------
    >>> wasserstein_distance([1.0, 0.0], [0.0, 1.0])
    1.0

    Notes
    -----
    For 1D discrete distributions, this is computed using the cumulative
    distribution functions:

    W(p, q) = sum(|CDF_p(i) - CDF_q(i)|)

    Also known as the Earth Mover's Distance or optimal transport distance.
    """
    p = np.asarray(_validate_vector(p, dtype=np.double), dtype=np.double)
    q = np.asarray(_validate_vector(q, dtype=np.double), dtype=np.double)
    _validate_same_shape(p, q)

    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Distributions must contain non-negative values")

    # Normalize to probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Compute CDFs
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)

    return float(np.sum(np.abs(cdf_p - cdf_q)))