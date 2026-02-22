# -*- coding: utf-8 -*-
"""
Vector distance functions.

This module contains basic vector distance and similarity functions.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from interspace._validators import _validate_vector, _validate_same_shape


def euclidean(vector_1, vector_2):
    """Compute Euclidean (L2) distance between two vectors.

    The Euclidean distance is the straight-line distance between two points
    in Euclidean space. It is the most common distance metric.

    Parameters
    ----------
    vector_1 : array_like
        First input vector.
    vector_2 : array_like
        Second input vector (must have same length as vector_1).

    Returns
    -------
    float
        Euclidean distance between the vectors.

    Raises
    ------
    ValueError
        If vectors have different shapes.

    Examples
    --------
    >>> euclidean([1, 2, 3], [4, 5, 6])
    5.196152422706632
    >>> euclidean([0, 0], [3, 4])
    5.0

    Notes
    -----
    The Euclidean distance is defined as:

    d(x, y) = sqrt(sum((x_i - y_i)^2))
    """
    v1 = np.asarray(_validate_vector(vector_1, dtype=np.double), dtype=np.double)
    v2 = np.asarray(_validate_vector(vector_2, dtype=np.double), dtype=np.double)
    _validate_same_shape(v1, v2)
    return float(np.linalg.norm(v1 - v2))


def manhattan(vector_1, vector_2):
    """Compute Manhattan (L1/cityblock) distance between two vectors.

    The Manhattan distance is the sum of absolute differences between
    corresponding elements of two vectors. Also known as cityblock distance
    or L1 norm.

    Parameters
    ----------
    vector_1 : array_like
        First input vector.
    vector_2 : array_like
        Second input vector (must have same length as vector_1).

    Returns
    -------
    float
        Manhattan distance between the vectors.

    Raises
    ------
    ValueError
        If vectors have different shapes.

    Examples
    --------
    >>> manhattan([1, 2, 3], [4, 5, 6])
    9.0
    >>> manhattan([0, 0], [3, 4])
    7.0

    Notes
    -----
    The Manhattan distance is defined as:

    d(x, y) = sum(|x_i - y_i|)

    This distance represents the shortest path between two points when
    movement is restricted to grid-like paths (like Manhattan streets).
    """
    v1 = np.asarray(_validate_vector(vector_1, dtype=np.double), dtype=np.double)
    v2 = np.asarray(_validate_vector(vector_2, dtype=np.double), dtype=np.double)
    _validate_same_shape(v1, v2)
    return float(np.sum(np.abs(v1 - v2)))


def minkowski(vector_1, vector_2, p=1):
    """Compute Minkowski distance between two vectors.

    The Minkowski distance is a generalization of Euclidean and Manhattan distances.
    It is parameterized by p, which determines the type of distance.

    Parameters
    ----------
    vector_1 : array_like
        First input vector.
    vector_2 : array_like
        Second input vector (must have same length as vector_1).
    p : float, default 1
        The order of the norm:
        - p=1: Manhattan distance
        - p=2: Euclidean distance
        - p->infinity: Chebyshev distance

    Returns
    -------
    float
        Minkowski distance between the vectors.

    Raises
    ------
    ValueError
        If vectors have different shapes.

    Examples
    --------
    >>> minkowski([1, 2], [4, 6], p=1)  # Manhattan
    7.0
    >>> minkowski([1, 2], [4, 6], p=2)  # Euclidean
    5.0
    >>> minkowski([1, 2], [4, 6], p=3)
    5.039684199579493

    Notes
    -----
    The Minkowski distance is defined as:

    d(x, y) = (sum(|x_i - y_i|^p))^(1/p)

    Special cases:
    - p = 1: Manhattan (L1) distance
    - p = 2: Euclidean (L2) distance
    - p -> infinity: Chebyshev (Linf) distance
    """
    v1 = np.asarray(_validate_vector(vector_1, dtype=np.double), dtype=np.double)
    v2 = np.asarray(_validate_vector(vector_2, dtype=np.double), dtype=np.double)
    _validate_same_shape(v1, v2)
    return float(np.sum(np.abs(v1 - v2) ** p) ** (1.0 / p))


def chebyshev_distance(x, y):
    """Compute Chebyshev (L-infinity/maximum) distance between two vectors.

    The Chebyshev distance is the maximum absolute difference between
    corresponding elements of two vectors.

    Parameters
    ----------
    x : array_like
        First input vector.
    y : array_like
        Second input vector (must have same length as x).

    Returns
    -------
    float
        Chebyshev distance.

    Raises
    ------
    ValueError
        If vectors have different shapes.

    Examples
    --------
    >>> chebyshev_distance([1, 2, 3], [4, 5, 6])
    3.0
    >>> chebyshev_distance([0, 0], [3, 4])
    4.0

    Notes
    -----
    The Chebyshev distance is defined as:

    d(x, y) = max(|x_i - y_i|)

    This is equivalent to the limit of Minkowski distance as p -> infinity.
    It is also known as the chessboard distance, as it represents the
    minimum number of moves a king would take to travel between two squares.
    """
    x = np.asarray(_validate_vector(x, dtype=np.double), dtype=np.double)
    y = np.asarray(_validate_vector(y, dtype=np.double), dtype=np.double)
    _validate_same_shape(x, y)
    return float(np.max(np.abs(x - y)))


def cosine_similarity(vector_1, vector_2):
    """Compute cosine similarity between two vectors.

    Cosine similarity measures the cosine of the angle between two vectors.
    Values range from -1 (opposite) to 1 (identical direction).

    Parameters
    ----------
    vector_1 : array_like
        First input vector.
    vector_2 : array_like
        Second input vector (must have same length as vector_1).

    Returns
    -------
    float
        Cosine similarity between -1 and 1.

    Raises
    ------
    ValueError
        If either vector is a zero-vector (undefined similarity).

    Examples
    --------
    >>> cosine_similarity([1, 0], [0, 1])
    0.0
    >>> cosine_similarity([1, 1], [1, 1])
    1.0
    >>> cosine_similarity([1, 0], [1, 0])
    1.0

    Notes
    -----
    The cosine similarity is defined as:

    similarity(x, y) = (x . y) / (||x|| * ||y||)

    Interpretation:
    - 1.0: Vectors point in the same direction
    - 0.0: Vectors are orthogonal (perpendicular)
    - -1.0: Vectors point in opposite directions
    """
    v1 = np.asarray(_validate_vector(vector_1, dtype=np.double), dtype=np.double)
    v2 = np.asarray(_validate_vector(vector_2, dtype=np.double), dtype=np.double)
    _validate_same_shape(v1, v2)
    num = np.dot(v1, v2)
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    if den == 0:
        raise ValueError("Zero-vector has undefined cosine similarity")
    return float(num / den)


def cosine_distance(x, y):
    """Compute cosine distance between two vectors.

    Cosine distance is defined as 1 - cosine_similarity.

    Parameters
    ----------
    x : array_like
        First input vector.
    y : array_like
        Second input vector (must have same length as x).

    Returns
    -------
    float
        Cosine distance between 0 and 2.

    Raises
    ------
    ValueError
        If either vector is a zero-vector.

    Examples
    --------
    >>> cosine_distance([1, 0], [0, 1])
    1.0
    >>> cosine_distance([1, 1], [1, 1])
    0.0

    Notes
    -----
    The cosine distance is defined as:

    d(x, y) = 1 - (x . y) / (||x|| * ||y||)
    """
    return 1.0 - cosine_similarity(x, y)


def mahalanobis(u, v, VI):
    """Compute Mahalanobis distance between two vectors.

    The Mahalanobis distance is a measure of distance between a point and a distribution,
    accounting for correlations between variables.

    Parameters
    ----------
    u : array_like
        First input vector.
    v : array_like
        Second input vector (must have same length as u).
    VI : array_like
        Inverse of the covariance matrix. Must be a square matrix with
        dimension matching the vectors.

    Returns
    -------
    float
        Mahalanobis distance.

    Examples
    --------
    >>> import numpy as np
    >>> VI = np.array([[1, 0.5], [0.5, 1]])  # Inverse covariance
    >>> mahalanobis([0, 0], [1, 1], VI)
    1.0

    Notes
    -----
    The Mahalanobis distance is defined as:

    d(u, v) = sqrt((u - v)^T * V^(-1) * (u - v))

    where V is the covariance matrix. This distance is scale-invariant
    and accounts for correlations between variables.
    """
    u = np.asarray(_validate_vector(u, dtype=np.double), dtype=np.double)
    v = np.asarray(_validate_vector(v, dtype=np.double), dtype=np.double)
    _validate_same_shape(u, v)
    VI = np.atleast_2d(VI)
    delta = u - v
    m = float(np.dot(np.dot(delta, VI), delta))
    return float(np.sqrt(m))