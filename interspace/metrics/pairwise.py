# -*- coding: utf-8 -*-
"""
Pairwise distance functions.

This module contains functions for computing distance matrices.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def pairwise_distance(X, Y=None, metric="euclidean", **kwargs):
    """Compute pairwise distances between vectors.

    Calculate distance between all pairs of vectors in X, or between
    vectors in X and Y.

    Parameters
    ----------
    X : array_like
        A 2D array where each row is a vector. Shape (n_samples_X, n_features).
    Y : array_like, optional
        A 2D array where each row is a vector. Shape (n_samples_Y, n_features).
        If None, computes distances between all pairs in X.
    metric : str or callable, default "euclidean"
        The distance metric to use. Can be a string name of any distance
        function in interspace, or a callable that takes two vectors and
        returns a distance.
    **kwargs : dict
        Additional keyword arguments passed to the distance function.

    Returns
    -------
    ndarray
        Distance matrix of shape (n_samples_X, n_samples_Y).

    Examples
    --------
    >>> X = [[1, 2], [3, 4], [5, 6]]
    >>> pairwise_distance(X, metric="euclidean")
    array([[0.        , 2.82842712, 5.65685425],
           [2.82842712, 0.        , 2.82842712],
           [5.65685425, 2.82842712, 0.        ]])

    Notes
    -----
    The distance matrix D is symmetric when Y is None, with D[i,j] = D[j,i].
    The diagonal elements are always zero (distance from a point to itself).
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.double))

    if Y is None:
        Y = X
    else:
        Y = np.atleast_2d(np.asarray(Y, dtype=np.double))

    n_X = X.shape[0]
    n_Y = Y.shape[0]

    # Get the distance function
    if isinstance(metric, str):
        # Import from main interspace module to get the function
        import interspace
        metric_func = getattr(interspace, metric, None)
        if metric_func is None:
            raise ValueError(f"Unknown metric: {metric}")
    else:
        metric_func = metric

    # Compute pairwise distances
    result = np.zeros((n_X, n_Y), dtype=np.double)
    for i in range(n_X):
        for j in range(n_Y):
            result[i, j] = metric_func(X[i], Y[j], **kwargs)

    return result