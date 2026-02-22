# -*- coding: utf-8 -*-
"""
Time series distance functions.

This module contains distance functions for comparing time series and sequences.
"""

from __future__ import annotations

import numpy as np

from interspace._validators import _validate_vector, _validate_same_shape


def dtw_distance(x, y):
    """Compute Dynamic Time Warping (DTW) distance between two sequences.

    DTW finds the optimal alignment between two sequences by warping the
    time axis non-linearly. It is commonly used for speech recognition,
    gesture recognition, and time series analysis.

    Parameters
    ----------
    x : array_like
        First time series.
    y : array_like
        Second time series.

    Returns
    -------
    float
        DTW distance.

    Examples
    --------
    >>> dtw_distance([1, 2, 3], [1, 2, 3])
    0.0
    >>> dtw_distance([1, 2, 3], [1, 2, 2, 3])
    0.0

    Notes
    -----
    DTW uses dynamic programming to find the optimal warping path.
    Time complexity: O(n*m) where n and m are the lengths of the sequences.
    """
    x = np.asarray(_validate_vector(x, dtype=np.double), dtype=np.double)
    y = np.asarray(_validate_vector(y, dtype=np.double), dtype=np.double)

    n, m = len(x), len(y)

    # Create cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # insertion
                dtw_matrix[i, j - 1],  # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )

    return float(dtw_matrix[n, m])


def euclidean_distance_1d(x, y):
    """Compute Euclidean distance between two 1D sequences.

    Optimized version for 1D time series. Sequences must have the same length.

    Parameters
    ----------
    x : array_like
        First time series.
    y : array_like
        Second time series (must have same length as x).

    Returns
    -------
    float
        Euclidean distance.

    Raises
    ------
    ValueError
        If sequences have different lengths.

    Examples
    --------
    >>> euclidean_distance_1d([1, 2, 3], [1, 2, 3])
    0.0
    >>> euclidean_distance_1d([1, 2, 3], [4, 5, 6])
    5.196152422706632

    Notes
    -----
    This is equivalent to the L2 norm of the difference between the two sequences.
    """
    x = np.asarray(_validate_vector(x, dtype=np.double), dtype=np.double)
    y = np.asarray(_validate_vector(y, dtype=np.double), dtype=np.double)
    _validate_same_shape(x, y)
    return float(np.linalg.norm(x - y))


def longest_common_subsequence(x, y):
    """Compute the length of the longest common subsequence (LCS) between two sequences.

    A subsequence is a sequence that can be derived from another sequence
    by deleting some or no elements without changing the order of the
    remaining elements.

    Parameters
    ----------
    x : array_like
        First sequence.
    y : array_like
        Second sequence.

    Returns
    -------
    int
        Length of the longest common subsequence.

    Examples
    --------
    >>> longest_common_subsequence([1, 2, 3, 4], [2, 3, 5])
    2
    >>> longest_common_subsequence("ABCBDAB", "BDCABA")
    4

    Notes
    -----
    Uses dynamic programming with O(n*m) time and O(n*m) space.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    n, m = len(x), len(y)

    # Create LCS matrix
    lcs_matrix = np.zeros((n + 1, m + 1), dtype=np.int32)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if x[i - 1] == y[j - 1]:
                lcs_matrix[i, j] = lcs_matrix[i - 1, j - 1] + 1
            else:
                lcs_matrix[i, j] = max(lcs_matrix[i - 1, j], lcs_matrix[i, j - 1])

    return int(lcs_matrix[n, m])