# -*- coding: utf-8 -*-
"""
Information theory functions.

This module contains functions from information theory for measuring
information content and relationships between variables.
"""

from __future__ import annotations

import numpy as np

from interspace._validators import _validate_vector, _validate_same_shape


def entropy(p, base=2):
    """Compute Shannon entropy of a probability distribution.

    Entropy measures the uncertainty or information content of a
    random variable.

    Parameters
    ----------
    p : array_like
        Probability distribution (must sum to 1 and contain non-negative values).
    base : float, default 2
        Base of the logarithm. Use 2 for bits, e for nats, 10 for bans.

    Returns
    -------
    float
        Shannon entropy.

    Raises
    ------
    ValueError
        If distribution contains negative values or does not sum to 1.

    Examples
    --------
    >>> entropy([0.5, 0.5])
    1.0
    >>> entropy([1.0, 0.0])
    0.0

    Notes
    -----
    Shannon entropy is defined as:

    H(X) = -sum(p_i * log(p_i))

    Properties:
    - Non-negative: H(X) >= 0
    - Zero if and only if the distribution is deterministic
    - Maximum for uniform distribution
    """
    p = np.asarray(_validate_vector(p, dtype=np.double), dtype=np.double)

    if np.any(p < 0):
        raise ValueError("Probability distribution must contain non-negative values")

    # Normalize to ensure it sums to 1
    p = p / np.sum(p)

    # Only compute where p > 0 (0 * log(0) = 0 by convention)
    mask = p > 0
    if base == 2:
        log_p = np.log2(p[mask])
    elif base == np.e:
        log_p = np.log(p[mask])
    elif base == 10:
        log_p = np.log10(p[mask])
    else:
        log_p = np.log(p[mask]) / np.log(base)

    return float(-np.sum(p[mask] * log_p))


def cross_entropy(p, q, base=2):
    """Compute cross-entropy between two probability distributions.

    Cross-entropy measures the average number of bits needed to encode
    data from distribution p using a code optimized for distribution q.

    Parameters
    ----------
    p : array_like
        True probability distribution.
    q : array_like
        Assumed probability distribution.
    base : float, default 2
        Base of the logarithm.

    Returns
    -------
    float
        Cross-entropy H(p, q).

    Raises
    ------
    ValueError
        If distributions contain negative values or q contains zeros where p is non-zero.

    Examples
    --------
    >>> cross_entropy([0.5, 0.5], [0.5, 0.5])
    1.0
    >>> cross_entropy([1.0, 0.0], [0.5, 0.5])
    1.0

    Notes
    -----
    Cross-entropy is defined as:

    H(p, q) = -sum(p_i * log(q_i))

    It is related to KL divergence by:
    H(p, q) = H(p) + D_KL(p || q)
    """
    p = np.asarray(_validate_vector(p, dtype=np.double), dtype=np.double)
    q = np.asarray(_validate_vector(q, dtype=np.double), dtype=np.double)
    _validate_same_shape(p, q)

    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Distributions must contain non-negative values")

    # Check for zeros in q where p is non-zero
    if np.any((p > 0) & (q == 0)):
        raise ValueError("Cross-entropy is undefined when q=0 and p>0")

    # Normalize
    p = p / np.sum(p)
    q = q / np.sum(q)

    # Only compute where p > 0
    mask = p > 0
    if base == 2:
        log_q = np.log2(q[mask])
    elif base == np.e:
        log_q = np.log(q[mask])
    elif base == 10:
        log_q = np.log10(q[mask])
    else:
        log_q = np.log(q[mask]) / np.log(base)

    return float(-np.sum(p[mask] * log_q))


def mutual_information(x, y, base=2):
    """Compute mutual information between two random variables.

    Mutual information measures the amount of information obtained about
    one random variable by observing another random variable.

    Parameters
    ----------
    x : array_like
        First random variable (discrete values).
    y : array_like
        Second random variable (discrete values).
    base : float, default 2
        Base of the logarithm.

    Returns
    -------
    float
        Mutual information I(X; Y).

    Raises
    ------
    ValueError
        If inputs have different lengths.

    Examples
    --------
    >>> mutual_information([0, 0, 1, 1], [0, 1, 0, 1])
    0.0
    >>> mutual_information([0, 0, 1, 1], [0, 0, 1, 1])
    1.0

    Notes
    -----
    Mutual information is defined as:

    I(X; Y) = sum_{x,y} p(x,y) * log(p(x,y) / (p(x) * p(y)))

    Properties:
    - Non-negative: I(X; Y) >= 0
    - Symmetric: I(X; Y) = I(Y; X)
    - I(X; Y) = 0 if and only if X and Y are independent
    - I(X; X) = H(X)
    """
    x = np.asarray(_validate_vector(x))
    y = np.asarray(_validate_vector(y))
    _validate_same_shape(x, y)

    n = len(x)
    if n == 0:
        return 0.0

    # Get unique values
    x_vals = np.unique(x)
    y_vals = np.unique(y)

    # Compute joint and marginal distributions
    joint = np.zeros((len(x_vals), len(y_vals)))
    x_margin = np.zeros(len(x_vals))
    y_margin = np.zeros(len(y_vals))

    for i, xv in enumerate(x_vals):
        for j, yv in enumerate(y_vals):
            joint[i, j] = np.sum((x == xv) & (y == yv)) / n
            x_margin[i] += joint[i, j]
            y_margin[j] += joint[i, j]

    # Compute mutual information
    mi = 0.0
    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            if joint[i, j] > 0:
                if base == 2:
                    log_ratio = np.log2(joint[i, j] / (x_margin[i] * y_margin[j]))
                elif base == np.e:
                    log_ratio = np.log(joint[i, j] / (x_margin[i] * y_margin[j]))
                elif base == 10:
                    log_ratio = np.log10(joint[i, j] / (x_margin[i] * y_margin[j]))
                else:
                    log_ratio = np.log(joint[i, j] / (x_margin[i] * y_margin[j])) / np.log(base)
                mi += joint[i, j] * log_ratio

    return float(mi)