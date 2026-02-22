# -*- coding: utf-8 -*-
"""
Physics-related distance functions.

This module contains distance functions related to physics and geometry.
"""

from __future__ import annotations

import math
import numpy as np

from interspace._validators import _validate_vector, _validate_same_shape


def angular_distance(angle1, angle2, degrees=True):
    """Compute the shortest angular distance between two angles.

    Handles the wrap-around at 0/360 degrees (or 0/2*pi radians).

    Parameters
    ----------
    angle1 : float
        First angle.
    angle2 : float
        Second angle.
    degrees : bool, default True
        If True, angles are in degrees. If False, angles are in radians.

    Returns
    -------
    float
        Shortest angular distance (always non-negative).

    Examples
    --------
    >>> angular_distance(10, 350)
    20.0
    >>> angular_distance(0, 180)
    180.0
    >>> angular_distance(0, math.pi, degrees=False)
    3.141592653589793

    Notes
    -----
    The angular distance is computed as:

    d(theta1, theta2) = min(|theta1 - theta2|, 360 - |theta1 - theta2|)

    for degrees, or similarly for radians.
    """
    if degrees:
        diff = abs(angle1 - angle2) % 360
        return min(diff, 360 - diff)
    else:
        diff = abs(angle1 - angle2) % (2 * math.pi)
        return min(diff, 2 * math.pi - diff)


def spherical_law_of_cosines(coord1, coord2, R=6372800.0):
    """Compute great-circle distance using the spherical law of cosines.

    An alternative formula to haversine for computing great-circle distance.

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
        Great-circle distance in the same units as R.

    Examples
    --------
    >>> spherical_law_of_cosines((0, 0), (0, 1))
    111319.49...

    Notes
    -----
    The spherical law of cosines formula:

    d = arccos(sin(phi1) * sin(phi2) + cos(phi1) * cos(phi2) * cos(delta_lambda)) * R

    For very small distances, this formula may have numerical precision issues.
    In such cases, the haversine formula is more numerically stable.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    # Spherical law of cosines
    cos_d = math.sin(phi1) * math.sin(phi2) + math.cos(phi1) * math.cos(phi2) * math.cos(delta_lambda)

    # Clamp to [-1, 1] to handle numerical errors
    cos_d = max(-1.0, min(1.0, cos_d))

    return math.acos(cos_d) * R


def euclidean_3d(point1, point2):
    """Compute 3D Euclidean distance between two points.

    Parameters
    ----------
    point1 : array_like
        First 3D point as [x, y, z].
    point2 : array_like
        Second 3D point as [x, y, z].

    Returns
    -------
    float
        3D Euclidean distance.

    Raises
    ------
    ValueError
        If points do not have exactly 3 dimensions.

    Examples
    --------
    >>> euclidean_3d([0, 0, 0], [1, 2, 2])
    3.0
    >>> euclidean_3d([1, 1, 1], [1, 1, 1])
    0.0

    Notes
    -----
    The 3D Euclidean distance is defined as:

    d(p1, p2) = sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
    """
    p1 = np.asarray(_validate_vector(point1, dtype=np.double), dtype=np.double)
    p2 = np.asarray(_validate_vector(point2, dtype=np.double), dtype=np.double)

    if len(p1) != 3 or len(p2) != 3:
        raise ValueError("Points must have exactly 3 dimensions")

    return float(np.linalg.norm(p1 - p2))