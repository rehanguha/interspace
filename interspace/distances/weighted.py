# -*- coding: utf-8 -*-
"""
Weighted distance functions.

This module contains weighted versions of common distance functions.
"""

from __future__ import annotations

import numpy as np

from interspace._validators import _validate_vector, _validate_weights, _validate_same_shape


def weighted_euclidean(x, y, w):
    """Compute weighted Euclidean distance between two vectors.

    The weighted Euclidean distance allows assigning different importance
    to different dimensions.

    Parameters
    ----------
    x : array_like
        First input vector.
    y : array_like
        Second input vector (must have same length as x).
    w : array_like
        Weight vector (must have same length as x and y).
        All weights must be non-negative.

    Returns
    -------
    float
        Weighted Euclidean distance.

    Raises
    ------
    ValueError
        If vectors have different shapes or weights contain negative values.

    Examples
    --------
    >>> weighted_euclidean([1, 2], [4, 6], [1, 1])
    5.0
    >>> weighted_euclidean([1, 2], [4, 6], [1, 0.5])
    4.301162633521313

    Notes
    -----
    The weighted Euclidean distance is defined as:

    d(x, y) = sqrt(sum(w_i * (x_i - y_i)^2))

    Weights allow emphasizing or de-emphasizing certain dimensions.
    """
    x = np.asarray(_validate_vector(x, dtype=np.double), dtype=np.double)
    y = np.asarray(_validate_vector(y, dtype=np.double), dtype=np.double)
    w = np.asarray(_validate_weights(w, dtype=np.double), dtype=np.double)
    _validate_same_shape(x, y)
    _validate_same_shape(x, w)
    return float(np.sqrt(np.sum(w * (x - y) ** 2)))


def weighted_manhattan(x, y, w):
    """Compute weighted Manhattan distance between two vectors.

    The weighted Manhattan distance allows assigning different importance
    to different dimensions.

    Parameters
    ----------
    x : array_like
        First input vector.
    y : array_like
        Second input vector (must have same length as x).
    w : array_like
        Weight vector (must have same length as x and y).
        All weights must be non-negative.

    Returns
    -------
    float
        Weighted Manhattan distance.

    Raises
    ------
    ValueError
        If vectors have different shapes or weights contain negative values.

    Examples
    --------
    >>> weighted_manhattan([1, 2], [4, 6], [1, 1])
    7.0
    >>> weighted_manhattan([1, 2], [4, 6], [2, 1])
    10.0

    Notes
    -----
    The weighted Manhattan distance is defined as:

    d(x, y) = sum(w_i * |x_i - y_i|)
    """
    x = np.asarray(_validate_vector(x, dtype=np.double), dtype=np.double)
    y = np.asarray(_validate_vector(y, dtype=np.double), dtype=np.double)
    w = np.asarray(_validate_weights(w, dtype=np.double), dtype=np.double)
    _validate_same_shape(x, y)
    _validate_same_shape(x, w)
    return float(np.sum(w * np.abs(x - y)))


def weighted_minkowski(x, y, w, p=2):
    """Compute weighted Minkowski distance between two vectors.

    The weighted Minkowski distance is a generalization of weighted
    Euclidean and weighted Manhattan distances.

    Parameters
    ----------
    x : array_like
        First input vector.
    y : array_like
        Second input vector (must have same length as x).
    w : array_like
        Weight vector (must have same length as x and y).
        All weights must be non-negative.
    p : float, default 2
        The order of the norm.

    Returns
    -------
    float
        Weighted Minkowski distance.

    Raises
    ------
    ValueError
        If vectors have different shapes or weights contain negative values.

    Examples
    --------
    >>> weighted_minkowski([1, 2], [4, 6], [1, 1], p=1)  # Weighted Manhattan
    7.0
    >>> weighted_minkowski([1, 2], [4, 6], [1, 1], p=2)  # Weighted Euclidean
    5.0

    Notes
    -----
    The weighted Minkowski distance is defined as:

    d(x, y) = (sum(w_i * |x_i - y_i|^p))^(1/p)
    """
    x = np.asarray(_validate_vector(x, dtype=np.double), dtype=np.double)
    y = np.asarray(_validate_vector(y, dtype=np.double), dtype=np.double)
    w = np.asarray(_validate_weights(w, dtype=np.double), dtype=np.double)
    _validate_same_shape(x, y)
    _validate_same_shape(x, w)
    return float(np.sum(w * np.abs(x - y) ** p) ** (1.0 / p))