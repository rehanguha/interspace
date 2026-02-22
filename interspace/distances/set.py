# -*- coding: utf-8 -*-
"""
Set-based distance functions.

This module contains distance functions based on set operations.
"""

from __future__ import annotations

import numpy as np


def jaccard_distance(x, y):
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

    Notes
    -----
    The Jaccard distance is defined as:

    d(x, y) = 1 - |X intersection Y| / |X union Y|

    where X and Y are the sets of unique elements in x and y.
    """
    sx = set(x)
    sy = set(y)
    inter = len(sx & sy)
    union = len(sx | sy)
    if union == 0:
        return 0.0
    return 1.0 - inter / float(union)


def dice_distance(x, y):
    """Compute Dice distance between two sets.

    The Dice distance is based on the Dice coefficient (Sorensen index).
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

    Notes
    -----
    The Dice distance is defined as:

    d(x, y) = 1 - 2|X intersection Y| / (|X| + |Y|)

    Also known as the Sorensen-Dice coefficient. Commonly used in
    natural language processing and bioinformatics.
    """
    sx = set(x)
    sy = set(y)
    inter = len(sx & sy)
    denom = len(sx) + len(sy)
    if denom == 0:
        return 0.0
    return 1.0 - (2.0 * inter) / denom


def matching_distance(x, y):
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

    Notes
    -----
    The matching distance is defined as:

    d(x, y) = (number of mismatched positions) / n

    where n is the length of the vectors.
    """
    xa = np.asarray(x)
    ya = np.asarray(y)
    if xa.size != ya.size:
        raise ValueError("Inputs must have the same length")
    return float(np.count_nonzero(xa != ya) / float(xa.size))


def overlap_distance(x, y):
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

    Notes
    -----
    The overlap distance is defined as:

    d(x, y) = 1 - |X intersection Y| / min(|X|, |Y|)

    Also known as the overlap coefficient. A value of 1 means that the
    smaller set is a subset of the larger set.
    """
    sx = set(x)
    sy = set(y)
    denom = float(min(len(sx), len(sy)))
    if denom == 0:
        return 0.0
    return 1.0 - float(len(sx & sy)) / denom


def tanimoto_distance(x, y):
    """Compute Tanimoto distance between two vectors.

    The Tanimoto distance is an extended Jaccard coefficient for
    continuous or binary vectors. Also known as the generalized Jaccard.

    Parameters
    ----------
    x : array_like
        First input vector.
    y : array_like
        Second input vector (must have same length as x).

    Returns
    -------
    float
        Tanimoto distance between 0 and 1.

    Examples
    --------
    >>> tanimoto_distance([1, 2, 3], [2, 3, 4])
    0.5

    Notes
    -----
    The Tanimoto distance is defined as:

    d(x, y) = 1 - (x . y) / (|x|^2 + |y|^2 - x . y)

    For binary vectors, this is equivalent to Jaccard distance.
    """
    x = np.asarray(x, dtype=np.double)
    y = np.asarray(y, dtype=np.double)
    if x.shape != y.shape:
        raise ValueError("Inputs must have the same shape")
    
    dot = np.dot(x, y)
    norm_x_sq = np.dot(x, x)
    norm_y_sq = np.dot(y, y)
    
    denom = norm_x_sq + norm_y_sq - dot
    if denom == 0:
        return 0.0
    
    return 1.0 - dot / denom