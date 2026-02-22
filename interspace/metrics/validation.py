# -*- coding: utf-8 -*-
"""
Distance metric validation functions.

This module contains functions for validating distance metric properties.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def is_distance_metric(distance_func, test_vectors=None, tolerance=1e-10):
    """Check if a function satisfies the properties of a distance metric.

    A distance metric must satisfy:
    1. Non-negativity: d(x, y) >= 0
    2. Identity: d(x, y) = 0 iff x = y
    3. Symmetry: d(x, y) = d(y, x)
    4. Triangle inequality: d(x, z) <= d(x, y) + d(y, z)

    Parameters
    ----------
    distance_func : callable
        A function that takes two vectors and returns a distance.
    test_vectors : list, optional
        List of test vectors to use for validation. If None, uses
        default test vectors.
    tolerance : float, default 1e-10
        Tolerance for floating-point comparisons.

    Returns
    -------
    bool
        True if the function appears to satisfy metric properties.

    Examples
    --------
    >>> import interspace
    >>> is_distance_metric(interspace.euclidean)
    True
    >>> is_distance_metric(interspace.cosine_distance)
    False  # Violates triangle inequality

    Notes
    -----
    This function performs empirical tests on the given vectors.
    A True result does not guarantee the function is a metric for all inputs,
    only that it satisfies the properties for the test cases.
    """
    if test_vectors is None:
        test_vectors = [
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 1],
            [2, 3],
            [-1, -1],
            [0.5, 0.5],
        ]

    for i, x in enumerate(test_vectors):
        for j, y in enumerate(test_vectors):
            d_xy = distance_func(x, y)
            d_yx = distance_func(y, x)

            # Non-negativity
            if d_xy < -tolerance:
                return False

            # Identity: d(x, x) = 0
            d_xx = distance_func(x, x)
            if abs(d_xx) > tolerance:
                return False

            # Symmetry
            if abs(d_xy - d_yx) > tolerance:
                return False

            # Triangle inequality
            for k, z in enumerate(test_vectors):
                d_xz = distance_func(x, z)
                d_yz = distance_func(y, z)
                if d_xy + d_yz < d_xz - tolerance:
                    return False

    return True