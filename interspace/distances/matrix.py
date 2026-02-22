# -*- coding: utf-8 -*-
"""
Matrix distance functions.

This module contains distance functions for comparing matrices.
"""

from __future__ import annotations

import numpy as np


def frobenius_distance(A, B):
    """Compute Frobenius distance between two matrices.

    The Frobenius distance is the Frobenius norm of the difference
    between two matrices. It is the matrix analog of the Euclidean
    distance for vectors.

    Parameters
    ----------
    A : array_like
        First matrix (2D array).
    B : array_like
        Second matrix (must have same shape as A).

    Returns
    -------
    float
        Frobenius distance.

    Raises
    ------
    ValueError
        If matrices have different shapes.

    Examples
    --------
    >>> A = [[1, 2], [3, 4]]
    >>> B = [[1, 2], [3, 4]]
    >>> frobenius_distance(A, B)
    0.0

    Notes
    -----
    The Frobenius distance is defined as:

    d(A, B) = ||A - B||_F = sqrt(sum(|A_ij - B_ij|^2))
    """
    A = np.atleast_2d(np.asarray(A, dtype=np.double))
    B = np.atleast_2d(np.asarray(B, dtype=np.double))

    if A.shape != B.shape:
        raise ValueError(f"Matrices must have the same shape. Got {A.shape} and {B.shape}")

    return float(np.linalg.norm(A - B, "fro"))


def spectral_distance(A, B):
    """Compute spectral distance between two matrices.

    The spectral distance is based on the largest singular value (spectral
    norm) of the difference between two matrices.

    Parameters
    ----------
    A : array_like
        First matrix (2D array).
    B : array_like
        Second matrix (must have same shape as A).

    Returns
    -------
    float
        Spectral distance.

    Raises
    ------
    ValueError
        If matrices have different shapes.

    Examples
    --------
    >>> A = [[1, 0], [0, 1]]
    >>> B = [[1, 0], [0, 2]]
    >>> spectral_distance(A, B)
    1.0

    Notes
    -----
    The spectral distance is defined as:

    d(A, B) = ||A - B||_2 = sigma_max(A - B)

    where sigma_max is the largest singular value.
    """
    A = np.atleast_2d(np.asarray(A, dtype=np.double))
    B = np.atleast_2d(np.asarray(B, dtype=np.double))

    if A.shape != B.shape:
        raise ValueError(f"Matrices must have the same shape. Got {A.shape} and {B.shape}")

    # Spectral norm is the largest singular value
    return float(np.linalg.norm(A - B, 2))


def trace_distance(A, B):
    """Compute trace distance between two matrices.

    The trace distance is half the sum of singular values of the
    difference matrix. Also known as the trace norm distance or
    nuclear norm distance.

    Parameters
    ----------
    A : array_like
        First matrix (2D array).
    B : array_like
        Second matrix (must have same shape as A).

    Returns
    -------
    float
        Trace distance.

    Raises
    ------
    ValueError
        If matrices have different shapes.

    Examples
    --------
    >>> A = [[1, 0], [0, 1]]
    >>> B = [[1, 0], [0, 0]]
    >>> trace_distance(A, B)
    0.5

    Notes
    -----
    The trace distance is defined as:

    d(A, B) = (1/2) * ||A - B||_1 = (1/2) * sum(sigma_i)

    where sigma_i are the singular values of (A - B).

    In quantum information, this is also known as the trace distance
    between density matrices.
    """
    A = np.atleast_2d(np.asarray(A, dtype=np.double))
    B = np.atleast_2d(np.asarray(B, dtype=np.double))

    if A.shape != B.shape:
        raise ValueError(f"Matrices must have the same shape. Got {A.shape} and {B.shape}")

    # Nuclear norm (sum of singular values)
    nuclear_norm = np.linalg.norm(A - B, "nuc")

    return float(nuclear_norm / 2)