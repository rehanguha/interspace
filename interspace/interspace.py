# -*- coding: utf-8 -*-
"""
Interspace - Distance and Similarity Functions for Vectors, Sequences, and Distributions.

This module provides a comprehensive collection of distance and similarity metrics
commonly used in machine learning, data science, and scientific computing.

Example
-------
>>> import interspace
>>> interspace.euclidean([1, 2, 3], [4, 5, 6])
5.196152422706632
>>> interspace.cosine_similarity([1, 0, 0], [1, 0, 0])
1.0
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

__version__ = "0.0.11"

__all__ = [
    "haversine",
    "manhattan",
    "euclidean",
    "minkowski",
    "cosine_similarity",
    "cosine_distance",
    "hamming",
    "mahalanobis",
    "chebyshev_distance",
    "jaccard_distance",
    "canberra_distance",
    "braycurtis_distance",
    "correlation_distance",
    "dice_distance",
    "matching_distance",
    "overlap_distance",
    "pearson_distance",
    "squared_chord_distance",
]

# Type alias for input vectors
VectorLike = Union[list, tuple, np.ndarray, ArrayLike]


def _validate_vector(
    vector: VectorLike, dtype: type | None = None
) -> float | NDArray[np.floating]:
    """Validate that input is a 1-D vector-like object and cast dtype when provided.

    Parameters
    ----------
    vector : array_like
        Input vector to validate.
    dtype : type, optional
        Data type to cast the vector to.

    Returns
    -------
    float or ndarray
        A Python scalar when the input has length 1, otherwise a 1-D numpy array.

    Raises
    ------
    ValueError
        If input is not 1-D or cannot be converted to the specified dtype.
    """
    arr = np.asarray(vector, dtype=dtype)
    if arr.ndim > 1:
        raise ValueError("Input vector should be 1-D.")
    if arr.size == 1:
        return arr.item()
    return arr


def _validate_var_type(value: object, dtype: type) -> object:
    """Validate that a value is of the specified type.

    Parameters
    ----------
    value : object
        Value to validate.
    dtype : type
        Expected type.

    Returns
    -------
    object
        The input value if it matches the expected type.

    Raises
    ------
    ValueError
        If value is not of the expected type.
    """
    if isinstance(value, dtype):
        return value
    raise ValueError(f"Input value not of type: {dtype}")


def _validate_weights(w: VectorLike, dtype: type = np.double) -> float | NDArray[np.floating]:
    """Validate that weights are non-negative.

    Parameters
    ----------
    w : array_like
        Weight values to validate.
    dtype : type, default np.double
        Data type for the weights.

    Returns
    -------
    float or ndarray
        Validated weights.

    Raises
    ------
    ValueError
        If any weight is negative.
    """
    w = _validate_vector(w, dtype=dtype)
    if np.any(np.asarray(w) < 0):
        raise ValueError("Input weights should be all non-negative")
    return w


def _validate_same_shape(v1: NDArray, v2: NDArray) -> None:
    """Validate that two vectors have the same shape.

    Parameters
    ----------
    v1 : ndarray
        First vector.
    v2 : ndarray
        Second vector.

    Raises
    ------
    ValueError
        If vectors have different shapes.
    """
    if v1.shape != v2.shape:
        raise ValueError(
            f"Input vectors must have the same shape. Got {v1.shape} and {v2.shape}."
        )


def hamming(a: int | str | VectorLike, b: int | str | VectorLike) -> int | float:
    """Hamming distance for ints (bitwise), strings/iterables (per-position mismatch).

    Parameters
    ----------
    a : int, str, or array_like
        First input.
    b : int, str, or array_like
        Second input.

    Returns
    -------
    int or float
        For integers: count of differing bits.
        For sequences: number/proportion of positions with differing elements.

    Raises
    ------
    ValueError
        If sequences have unequal length.

    Examples
    --------
    >>> hamming(0b1010, 0b0011)
    2
    >>> hamming("abcd", "abcf")
    1
    >>> hamming([1, 2, 3], [1, 0, 3])
    1
    """
    # integers: bitwise xor popcount
    if isinstance(a, int) and isinstance(b, int):
        return bin(a ^ b).count("1")

    # strings or sequence-like: compare lengths then elementwise
    if isinstance(a, str) and isinstance(b, str):
        if len(a) != len(b):
            raise ValueError("Undefined for sequences of unequal length.")
        return sum(c1 != c2 for c1, c2 in zip(a, b))

    # try to treat as array-like
    xa = np.asarray(a)
    xb = np.asarray(b)
    if xa.shape != xb.shape:
        raise ValueError("Undefined for sequences of unequal shape.")
    return int(np.count_nonzero(xa != xb))


def haversine(
    coord1: tuple[float, float] | list[float],
    coord2: tuple[float, float] | list[float],
    R: float = 6372800.0,
) -> float:
    """Great-circle distance between two (lat, lon) pairs in degrees.

    Parameters
    ----------
    coord1 : tuple or list of float
        First coordinate as (latitude, longitude) in degrees.
    coord2 : tuple or list of float
        Second coordinate as (latitude, longitude) in degrees.
    R : float, default 6372800.0
        Radius of the sphere in meters. Default is Earth's mean radius.

    Returns
    -------
    float
        Great-circle distance in the same units as R (meters by default).

    Examples
    --------
    >>> haversine((42.5170365, 15.2778599), (51.5073219, -0.1276474))
    1231910.737...
    >>> haversine((0, 0), (0, 0))
    0.0
    """
    lat1, lon1 = _validate_vector(coord1, dtype=np.double)
    lat2, lon2 = _validate_vector(coord2, dtype=np.double)

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def manhattan(vector_1: VectorLike, vector_2: VectorLike) -> float:
    """Compute Manhattan (L1/cityblock) distance between two vectors.

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
    """
    v1 = np.asarray(_validate_vector(vector_1, dtype=np.double), dtype=np.double)
    v2 = np.asarray(_validate_vector(vector_2, dtype=np.double), dtype=np.double)
    _validate_same_shape(v1, v2)
    return float(np.sum(np.abs(v1 - v2)))


def euclidean(vector_1: VectorLike, vector_2: VectorLike) -> float:
    """Compute Euclidean (L2) distance between two vectors.

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
    """
    v1 = np.asarray(_validate_vector(vector_1, dtype=np.double), dtype=np.double)
    v2 = np.asarray(_validate_vector(vector_2, dtype=np.double), dtype=np.double)
    _validate_same_shape(v1, v2)
    return float(np.linalg.norm(v1 - v2))


def minkowski(vector_1: VectorLike, vector_2: VectorLike, p: float = 1) -> float:
    """Compute Minkowski distance between two vectors.

    The Minkowski distance is a generalization of Euclidean and Manhattan distances.

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
        - p→∞: Chebyshev distance

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
    """
    v1 = np.asarray(_validate_vector(vector_1, dtype=np.double), dtype=np.double)
    v2 = np.asarray(_validate_vector(vector_2, dtype=np.double), dtype=np.double)
    _validate_same_shape(v1, v2)
    return float(np.sum(np.abs(v1 - v2) ** p) ** (1.0 / p))


def cosine_similarity(vector_1: VectorLike, vector_2: VectorLike) -> float:
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
    """
    v1 = np.asarray(_validate_vector(vector_1, dtype=np.double), dtype=np.double)
    v2 = np.asarray(_validate_vector(vector_2, dtype=np.double), dtype=np.double)
    _validate_same_shape(v1, v2)
    num = np.dot(v1, v2)
    den = np.linalg.norm(v1) * np.linalg.norm(v2)
    if den == 0:
        raise ValueError("Zero-vector has undefined cosine similarity")
    return float(num / den)


def cosine_distance(x: VectorLike, y: VectorLike) -> float:
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
    """
    return 1.0 - cosine_similarity(x, y)


def mahalanobis(
    u: VectorLike, v: VectorLike, VI: ArrayLike
) -> float:
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
    """
    u = np.asarray(_validate_vector(u, dtype=np.double), dtype=np.double)
    v = np.asarray(_validate_vector(v, dtype=np.double), dtype=np.double)
    _validate_same_shape(u, v)
    VI = np.atleast_2d(VI)
    delta = u - v
    m = float(np.dot(np.dot(delta, VI), delta))
    return float(np.sqrt(m))


def chebyshev_distance(x: VectorLike, y: VectorLike) -> float:
    """Compute Chebyshev (L∞/maximum) distance between two vectors.

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
    """
    x = np.asarray(_validate_vector(x, dtype=np.double), dtype=np.double)
    y = np.asarray(_validate_vector(y, dtype=np.double), dtype=np.double)
    _validate_same_shape(x, y)
    return float(np.max(np.abs(x - y)))


def jaccard_distance(x: VectorLike, y: VectorLike) -> float:
    """Compute Jaccard distance between two sets.

    The Jaccard distance measures dissimilarity between sample sets.
    It is defined as 1 - Jaccard similarity (intersection over union).

    Parameters
    ----------
    x : array_like
        First input (treated as a set).
    y : array_like
        Second input (treated as a set).

    Returns
    -------
    float
        Jaccard distance between 0 and 1.

    Examples
    --------
    >>> jaccard_distance([1, 2, 3], [2, 3, 4])
    0.5
    >>> jaccard_distance([1, 2], [3, 4])
    1.0
    >>> jaccard_distance([1, 2], [1, 2])
    0.0
    """
    sx = set(x)
    sy = set(y)
    inter = len(sx & sy)
    union = len(sx | sy)
    if union == 0:
        return 0.0
    return 1.0 - inter / float(union)


def canberra_distance(x: VectorLike, y: VectorLike) -> float:
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
    """
    x = np.asarray(_validate_vector(x, dtype=np.double), dtype=np.double)
    y = np.asarray(_validate_vector(y, dtype=np.double), dtype=np.double)
    _validate_same_shape(x, y)
    denom = np.abs(x) + np.abs(y)
    numer = np.abs(x - y)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(denom == 0, 0.0, numer / denom)
    return float(np.sum(frac))


def braycurtis_distance(x: VectorLike, y: VectorLike) -> float:
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
    """
    x = np.asarray(_validate_vector(x, dtype=np.double), dtype=np.double)
    y = np.asarray(_validate_vector(y, dtype=np.double), dtype=np.double)
    _validate_same_shape(x, y)
    denom = np.sum(np.abs(x + y))
    if denom == 0:
        return 0.0
    return float(np.sum(np.abs(x - y)) / denom)


def correlation_distance(x: VectorLike, y: VectorLike) -> float:
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
    """
    x = np.asarray(_validate_vector(x, dtype=np.double), dtype=np.double)
    y = np.asarray(_validate_vector(y, dtype=np.double), dtype=np.double)
    _validate_same_shape(x, y)
    if x.size < 2:
        raise ValueError("At least two elements are required to compute correlation")
    return 1.0 - float(np.corrcoef(x, y)[0, 1])


def dice_distance(x: VectorLike, y: VectorLike) -> float:
    """Compute Dice distance between two sets.

    The Dice distance is based on the Dice coefficient (Sørensen index).
    It gives more weight to common elements than Jaccard.

    Parameters
    ----------
    x : array_like
        First input (treated as a set).
    y : array_like
        Second input (treated as a set).

    Returns
    -------
    float
        Dice distance between 0 and 1.

    Examples
    --------
    >>> dice_distance([1, 2, 3], [2, 3, 4])
    0.4
    >>> dice_distance([1, 2], [3, 4])
    1.0
    >>> dice_distance([1, 2], [1, 2])
    0.0
    """
    sx = set(x)
    sy = set(y)
    inter = len(sx & sy)
    denom = len(sx) + len(sy)
    if denom == 0:
        return 0.0
    return 1.0 - (2.0 * inter) / denom


def matching_distance(x: VectorLike, y: VectorLike) -> float:
    """Compute matching distance between two binary vectors.

    The matching distance is the proportion of positions where
    the elements differ.

    Parameters
    ----------
    x : array_like
        First input vector.
    y : array_like
        Second input vector (must have same length as x).

    Returns
    -------
    float
        Matching distance between 0 and 1.

    Raises
    ------
    ValueError
        If vectors have different lengths.

    Examples
    --------
    >>> matching_distance([1, 0, 1], [1, 1, 1])
    0.3333333333333333
    >>> matching_distance([1, 0, 1], [1, 0, 1])
    0.0
    """
    xa = np.asarray(x)
    ya = np.asarray(y)
    if xa.size != ya.size:
        raise ValueError("Inputs must have the same length")
    return float(np.count_nonzero(xa != ya) / float(xa.size))


def overlap_distance(x: VectorLike, y: VectorLike) -> float:
    """Compute overlap distance between two sets.

    The overlap distance measures dissimilarity based on the size
    of intersection relative to the smaller set.

    Parameters
    ----------
    x : array_like
        First input (treated as a set).
    y : array_like
        Second input (treated as a set).

    Returns
    -------
    float
        Overlap distance between 0 and 1.

    Examples
    --------
    >>> overlap_distance([1, 2, 3], [2, 3, 4])
    0.5
    >>> overlap_distance([1, 2], [3, 4])
    1.0
    >>> overlap_distance([1, 2], [1, 2])
    0.0
    """
    sx = set(x)
    sy = set(y)
    denom = float(min(len(sx), len(sy)))
    if denom == 0:
        return 0.0
    return 1.0 - float(len(sx & sy)) / denom


def pearson_distance(x: VectorLike, y: VectorLike) -> float:
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


def squared_chord_distance(x: VectorLike, y: VectorLike) -> float:
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