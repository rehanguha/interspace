# -*- coding: utf-8 -*-
"""
Binary distance functions.

This module contains distance functions for binary (presence/absence) data.
"""

from __future__ import annotations

import numpy as np


def russell_rao_distance(x, y):
    """Compute Russell-Rao distance between two binary vectors.

    The Russell-Rao distance is based on the proportion of matching
    (both present) elements out of all elements.

    Parameters
    ----------
    x : array_like
        First binary vector.
    y : array_like
        Second binary vector (must have same length as x).

    Returns
    -------
    float
        Russell-Rao distance between 0 and 1.

    Raises
    ------
    ValueError
        If vectors have different lengths.

    Examples
    --------
    >>> russell_rao_distance([1, 0, 1, 0], [1, 1, 0, 0])
    0.75

    Notes
    -----
    The Russell-Rao distance is defined as:

    d(x, y) = 1 - a / n

    where a is the count of positions where both vectors have 1,
    and n is the total number of positions.
    """
    x = np.asarray(x, dtype=np.int32)
    y = np.asarray(y, dtype=np.int32)

    if x.shape != y.shape:
        raise ValueError("Inputs must have the same length")

    n = len(x)
    if n == 0:
        return 0.0

    # a = count where both are 1
    a = np.sum((x == 1) & (y == 1))

    return 1.0 - a / n


def sokal_sneath_distance(x, y):
    """Compute Sokal-Sneath distance between two binary vectors.

    The Sokal-Sneath distance emphasizes mismatches and gives more
    weight to double absences.

    Parameters
    ----------
    x : array_like
        First binary vector.
    y : array_like
        Second binary vector (must have same length as x).

    Returns
    -------
    float
        Sokal-Sneath distance between 0 and 1.

    Raises
    ------
    ValueError
        If vectors have different lengths.

    Examples
    --------
    >>> sokal_sneath_distance([1, 0, 1, 0], [1, 1, 0, 0])
    0.75

    Notes
    -----
    The Sokal-Sneath distance is defined as:

    d(x, y) = (b + c) / (a + b + c)

    where:
    - a = count where both are 1
    - b = count where x=1 and y=0
    - c = count where x=0 and y=1
    """
    x = np.asarray(x, dtype=np.int32)
    y = np.asarray(y, dtype=np.int32)

    if x.shape != y.shape:
        raise ValueError("Inputs must have the same length")

    a = np.sum((x == 1) & (y == 1))
    b = np.sum((x == 1) & (y == 0))
    c = np.sum((x == 0) & (y == 1))

    denom = a + b + c
    if denom == 0:
        return 0.0

    return (b + c) / denom


def kulczynski_distance(x, y):
    """Compute Kulczynski distance between two binary vectors.

    The Kulczynski distance is commonly used in ecology for comparing
    species presence/absence data.

    Parameters
    ----------
    x : array_like
        First binary vector.
    y : array_like
        Second binary vector (must have same length as x).

    Returns
    -------
    float
        Kulczynski distance.

    Raises
    ------
    ValueError
        If vectors have different lengths.

    Examples
    --------
    >>> kulczynski_distance([1, 0, 1, 1], [1, 1, 0, 1])
    0.5833333333333333

    Notes
    -----
    The Kulczynski distance is defined as:

    d(x, y) = 1 - (a/(a+b) + a/(a+c)) / 2

    where:
    - a = count where both are 1
    - b = count where x=1 and y=0
    - c = count where x=0 and y=1
    """
    x = np.asarray(x, dtype=np.int32)
    y = np.asarray(y, dtype=np.int32)

    if x.shape != y.shape:
        raise ValueError("Inputs must have the same length")

    a = np.sum((x == 1) & (y == 1))
    b = np.sum((x == 1) & (y == 0))
    c = np.sum((x == 0) & (y == 1))

    if a + b == 0 or a + c == 0:
        return 1.0

    similarity = (a / (a + b) + a / (a + c)) / 2
    return 1.0 - similarity