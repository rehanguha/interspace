# -*- coding: utf-8 -*-
"""
Distribution distance functions.

This module contains distance functions commonly used for comparing
probability distributions and ecological data.
"""

from __future__ import annotations

import numpy as np

from interspace._validators import _validate_vector, _validate_same_shape


def canberra_distance(x, y):
    """Compute Canberra distance between two vectors.

    The Canberra distance is a weighted version of Manhattan distance,
    sensitive to small changes near zero.

    Parameters
    ----------
    x : array_like
        First input vector.
    y : array_like
        Second input vector (must have same length as x).

    Returns
    -------
    float
        Canberra distance.

    Raises
    ------
    ValueError
        If vectors have different shapes.

    Examples
    --------
    >>> canberra_distance([1, 2, 3], [2, 2, 4])
    0.47619047619047616
    >>> canberra_distance([0, 0], [0, 0])
    0.0

    Notes
    -----
    The Canberra distance is defined as:

    d(x, y) = sum(|x_i - y_i| / (|x_i| + |y_i|))

    It is particularly sensitive to small changes when both values are small.
    Often used in clustering and classification tasks.
    """
    x = np.asarray(_validate_vector(x, dtype=np.double), dtype=np.double)
    y = np.asarray(_validate_vector(y, dtype=np.double), dtype=np.double)
    _validate_same_shape(x, y)
    denom = np.abs(x) + np.abs(y)
    numer = np.abs(x - y)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(denom == 0, 0.0, numer / denom)
    return float(np.sum(frac))


def braycurtis_distance(x, y):
    """Compute Bray-Curtis distance between two vectors.

    The Bray-Curtis distance is commonly used in ecology to measure
    dissimilarity between two sites based on species abundances.

    Parameters
    ----------
    x : array_like
        First input vector (non-negative values recommended).
    y : array_like
        Second input vector (must have same length as x).

    Returns
    -------
    float
        Bray-Curtis distance between 0 and 1.

    Raises
    ------
    ValueError
        If vectors have different shapes.

    Examples
    --------
    >>> braycurtis_distance([1, 2, 3], [2, 2, 4])
    0.14285714285714285
    >>> braycurtis_distance([0, 0], [0, 0])
    0.0

    Notes
    -----
    The Bray-Curtis distance is defined as:

    d(x, y) = sum(|x_i - y_i|) / sum(|x_i + y_i|)

    Also known as the Bray-Curtis dissimilarity. It is bounded between 0 and 1
    for non-negative inputs.
    """
    x = np.asarray(_validate_vector(x, dtype=np.double), dtype=np.double)
    y = np.asarray(_validate_vector(y, dtype=np.double), dtype=np.double)
    _validate_same_shape(x, y)
    denom = np.sum(np.abs(x + y))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(x - y)) / denom)


def correlation_distance(x, y):
    """Compute correlation distance between two vectors.

    The correlation distance is based on Pearson correlation coefficient.
    Distance = 1 - correlation(x, y).

    Parameters
    ----------
    x : array_like
        First input vector.
    y : array_like
        Second input vector (must have same length as x).

    Returns
    -------
    float
        Correlation distance between 0 and 2.

    Raises
    ------
    ValueError
        If vectors have fewer than 2 elements.

    Examples
    --------
    >>> correlation_distance([1, 2, 3], [1, 2, 3])
    0.0
    >>> correlation_distance([1, 2, 3], [3, 2, 1])
    2.0

    Notes
    -----
    The correlation distance is defined as:

    d(x, y) = 1 - rho_xy

    where rho_xy is the Pearson correlation coefficient.
    """
    x = np.asarray(_validate_vector(x, dtype=np.double), dtype=np.double)
    y = np.asarray(_validate_vector(y, dtype=np.double), dtype=np.double)
    _validate_same_shape(x, y)
    if x.size < 2:
        raise ValueError("At least two elements are required to compute correlation")
    return 1.0 - float(np.corrcoef(x, y)[0, 1])


def pearson_distance(x, y):
    """Compute Pearson distance between two vectors.

    Alias for correlation_distance. Computes 1 - Pearson correlation.

    Parameters
    ----------
    x : array_like
        First input vector.
    y : array_like
        Second input vector (must have same length as x).

    Returns
    -------
    float
        Pearson distance between 0 and 2.

    See Also
    --------
    correlation_distance : Same function.

    Examples
    --------
    >>> pearson_distance([1, 2, 3], [1, 2, 3])
    0.0
    """
    return correlation_distance(x, y)


def squared_chord_distance(x, y):
    """Compute squared chord distance between two vectors.

    The squared chord distance is useful in ecology for comparing
    species composition. Requires non-negative inputs.

    Parameters
    ----------
    x : array_like
        First input vector (must be non-negative).
    y : array_like
        Second input vector (must be non-negative).

    Returns
    -------
    float
        Squared chord distance.

    Raises
    ------
    ValueError
        If vectors have different shapes or contain negative values.

    Examples
    --------
    >>> squared_chord_distance([1, 2, 3], [2, 3, 4])
    0.1679497...

    Notes
    -----
    The squared chord distance is defined as:

    d(x, y) = sum((sqrt(x_i) - sqrt(y_i))^2) / sum((sqrt(x_i) + sqrt(y_i))^2)
    """
    x = np.asarray(_validate_vector(x, dtype=np.double), dtype=np.double)
    y = np.asarray(_validate_vector(y, dtype=np.double), dtype=np.double)
    _validate_same_shape(x, y)
    if np.any(x < 0) or np.any(y < 0):
        raise ValueError("Squared-chord requires non-negative inputs")
    num = np.sum((np.sqrt(x) - np.sqrt(y)) ** 2)
    den = np.sum((np.sqrt(x) + np.sqrt(y)) ** 2)
    if den == 0:
        return 0.0
    return float(num / den)