# -*- coding: utf-8 -*-
"""
Normalized distance functions.

This module contains normalized and standardized distance functions.
"""

from __future__ import annotations

import numpy as np

from interspace._validators import _validate_vector, _validate_same_shape


def normalized_euclidean(x, y):
    """Compute normalized Euclidean distance between two vectors.

    The normalized Euclidean distance divides the Euclidean distance
    by the square root of the number of dimensions.

    Parameters
    ----------
    x : array_like
        First input vector.
    y : array_like
        Second input vector (must have same length as x).

    Returns
    -------
    float
        Normalized Euclidean distance.

    Raises
    ------
    ValueError
        If vectors have different shapes.

    Examples
    --------
    >>> normalized_euclidean([1, 2, 3], [4, 5, 6])
    2.9999999999999996

    Notes
    -----
    The normalized Euclidean distance is defined as:

    d(x, y) = ||x - y||_2 / sqrt(n)

    where n is the number of dimensions.
    """
    x = np.asarray(_validate_vector(x, dtype=np.double), dtype=np.double)
    y = np.asarray(_validate_vector(y, dtype=np.double), dtype=np.double)
    _validate_same_shape(x, y)

    n = len(x)
    if n == 0:
        return 0.0

    return float(np.linalg.norm(x - y) / np.sqrt(n))


def standardized_euclidean(x, y, variances):
    """Compute standardized Euclidean distance between two vectors.

    The standardized Euclidean distance normalizes each dimension
    by its variance, making it scale-invariant.

    Parameters
    ----------
    x : array_like
        First input vector.
    y : array_like
        Second input vector (must have same length as x).
    variances : array_like
        Variance for each dimension. Must have same length as x.

    Returns
    -------
    float
        Standardized Euclidean distance.

    Raises
    ------
    ValueError
        If vectors have different shapes or variance contains non-positive values.

    Examples
    --------
    >>> standardized_euclidean([1, 2], [3, 4], [1, 4])
    1.0

    Notes
    -----
    The standardized Euclidean distance is defined as:

    d(x, y) = sqrt(sum((x_i - y_i)^2 / var_i))
    """
    x = np.asarray(_validate_vector(x, dtype=np.double), dtype=np.double)
    y = np.asarray(_validate_vector(y, dtype=np.double), dtype=np.double)
    variances = np.asarray(_validate_vector(variances, dtype=np.double), dtype=np.double)

    _validate_same_shape(x, y)
    _validate_same_shape(x, variances)

    if np.any(variances <= 0):
        raise ValueError("Variances must be positive")

    return float(np.sqrt(np.sum((x - y) ** 2 / variances)))


def seuclidean(x, y, V):
    """Compute standardized Euclidean distance (alias for standardized_euclidean).

    Parameters
    ----------
    x : array_like
        First input vector.
    y : array_like
        Second input vector (must have same length as x).
    V : array_like
        Variance for each dimension.

    Returns
    -------
    float
        Standardized Euclidean distance.

    See Also
    --------
    standardized_euclidean : Same function.

    Examples
    --------
    >>> seuclidean([1, 2], [3, 4], [1, 4])
    1.0
    """
    return standardized_euclidean(x, y, V)


def chi2_distance(x, y):
    """Compute Chi-squared distance between two vectors.

    The Chi-squared distance is commonly used for comparing histograms
    and frequency distributions.

    Parameters
    ----------
    x : array_like
        First input vector (must be non-negative).
    y : array_like
        Second input vector (must be non-negative).

    Returns
    -------
    float
        Chi-squared distance.

    Raises
    ------
    ValueError
        If vectors have different shapes or contain negative values.

    Examples
    --------
    >>> chi2_distance([1, 2, 3], [2, 3, 4])
    0.2777777777777778

    Notes
    -----
    The Chi-squared distance is defined as:

    d(x, y) = sum((x_i - y_i)^2 / (x_i + y_i))
    """
    x = np.asarray(_validate_vector(x, dtype=np.double), dtype=np.double)
    y = np.asarray(_validate_vector(y, dtype=np.double), dtype=np.double)
    _validate_same_shape(x, y)

    if np.any(x < 0) or np.any(y < 0):
        raise ValueError("Chi-squared distance requires non-negative inputs")

    denom = x + y
    with np.errstate(divide="ignore", invalid="ignore"):
        # Handle cases where both are zero
        diff_sq = (x - y) ** 2
        result = np.where(denom == 0, 0.0, diff_sq / denom)

    return float(np.sum(result) / 2)


def gower_distance(x, y, types=None, ranges=None):
    """Compute Gower distance between two vectors with mixed variable types.

    The Gower distance can handle continuous, binary, and categorical
    variables in the same comparison.

    Parameters
    ----------
    x : array_like
        First input vector.
    y : array_like
        Second input vector (must have same length as x).
    types : list of str, optional
        Type of each variable: 'continuous', 'binary', or 'categorical'.
        If None, all variables are treated as continuous.
    ranges : list of float, optional
        Range (max - min) for each continuous variable.
        Required for continuous variables if types is provided.

    Returns
    -------
    float
        Gower distance between 0 and 1.

    Raises
    ------
    ValueError
        If vectors have different lengths.

    Examples
    --------
    >>> gower_distance([1, 0, 1], [2, 0, 0])
    0.3333333333333333

    Notes
    -----
    For continuous variables: d_i = |x_i - y_i| / range_i
    For binary variables: d_i = 0 if x_i == y_i, else 1
    For categorical variables: d_i = 0 if x_i == y_i, else 1

    The overall distance is the mean of individual distances.
    """
    x = np.asarray(_validate_vector(x, dtype=np.double), dtype=np.double)
    y = np.asarray(_validate_vector(y, dtype=np.double), dtype=np.double)
    _validate_same_shape(x, y)

    n = len(x)
    if n == 0:
        return 0.0

    if types is None:
        # Treat all as continuous
        if ranges is None:
            ranges = np.abs(x) + np.abs(y)
            # Avoid division by zero
            ranges = np.where(ranges == 0, 1.0, ranges)
        distances = np.abs(x - y) / np.asarray(ranges)
        return float(np.mean(distances))

    types = list(types)
    distances = np.zeros(n)

    for i, var_type in enumerate(types):
        if var_type == "continuous":
            if ranges is None or ranges[i] == 0:
                distances[i] = 0.0
            else:
                distances[i] = abs(x[i] - y[i]) / ranges[i]
        elif var_type in ("binary", "categorical"):
            distances[i] = 0.0 if x[i] == y[i] else 1.0
        else:
            raise ValueError(f"Unknown variable type: {var_type}")

    return float(np.mean(distances))