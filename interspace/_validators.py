# -*- coding: utf-8 -*-
"""
Internal validation helper functions for interspace.

This module contains validation functions used internally by the distance
and similarity functions. These are not intended to be part of the public API.
"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

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


def _validate_non_negative(x: NDArray, name: str = "Input") -> None:
    """Validate that an array contains only non-negative values.

    Parameters
    ----------
    x : ndarray
        Input array to validate.
    name : str, default "Input"
        Name to use in error message.

    Raises
    ------
    ValueError
        If array contains negative values.
    """
    if np.any(x < 0):
        raise ValueError(f"{name} must contain non-negative values")