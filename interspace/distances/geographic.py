# -*- coding: utf-8 -*-
"""
Geographic distance functions.

This module contains distance functions for geographic coordinates.
"""

from __future__ import annotations

import math
import numpy as np

from interspace._validators import _validate_vector


def haversine(coord1, coord2, R=6372800.0):
    """Great-circle distance between two (lat, lon) pairs in degrees.

    The haversine formula determines the great-circle distance between two points
    on a sphere given their longitudes and latitudes.

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

    Notes
    -----
    The haversine formula is given by:

    a = sin^2(delta_phi/2) + cos(phi1) * cos(phi2) * sin^2(delta_lambda/2)
    c = 2 * arctan2(sqrt(a), sqrt(1-a))
    d = R * c

    where phi is latitude, lambda is longitude, and R is Earth's radius.
    """
    lat1, lon1 = _validate_vector(coord1, dtype=np.double)
    lat2, lon2 = _validate_vector(coord2, dtype=np.double)

    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def vincenty_distance(coord1, coord2, max_iterations=200, tolerance=1e-12):
    """Compute Vincenty's formula for geodesic distance on an ellipsoid.

    More accurate than haversine for the Earth's ellipsoidal shape.

    Parameters
    ----------
    coord1 : tuple or list of float
        First coordinate as (latitude, longitude) in degrees.
    coord2 : tuple or list of float
        Second coordinate as (latitude, longitude) in degrees.
    max_iterations : int, default 200
        Maximum number of iterations for convergence.
    tolerance : float, default 1e-12
        Convergence tolerance in meters.

    Returns
    -------
    float
        Geodesic distance in meters.

    Raises
    ------
    ValueError
        If the algorithm fails to converge.

    Examples
    --------
    >>> vincenty_distance((0, 0), (0, 1))
    111319.49...

    Notes
    -----
    Uses WGS-84 ellipsoid parameters:
    - a (semi-major axis) = 6378137.0 meters
    - b (semi-minor axis) = 6356752.314245 meters
    - f (flattening) = 1/298.257223563
    """
    # WGS-84 ellipsoid parameters
    a = 6378137.0  # semi-major axis in meters
    f = 1 / 298.257223563  # flattening
    b = a * (1 - f)  # semi-minor axis

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    # Convert to radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    lambda1 = math.radians(lon1)
    lambda2 = math.radians(lon2)

    L = lambda2 - lambda1
    U1 = math.atan((1 - f) * math.tan(phi1))
    U2 = math.atan((1 - f) * math.tan(phi2))

    sin_U1 = math.sin(U1)
    cos_U1 = math.cos(U1)
    sin_U2 = math.sin(U2)
    cos_U2 = math.cos(U2)

    lambda_val = L
    for _ in range(max_iterations):
        sin_lambda = math.sin(lambda_val)
        cos_lambda = math.cos(lambda_val)
        sin_sigma = math.sqrt(
            (cos_U2 * sin_lambda) ** 2
            + (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lambda) ** 2
        )
        if sin_sigma == 0:
            return 0.0

        cos_sigma = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_lambda
        sigma = math.atan2(sin_sigma, cos_sigma)
        sin_alpha = cos_U1 * cos_U2 * sin_lambda / sin_sigma
        cos_sq_alpha = 1 - sin_alpha**2

        if cos_sq_alpha == 0:
            cos_2sigma_m = 0
        else:
            cos_2sigma_m = cos_sigma - 2 * sin_U1 * sin_U2 / cos_sq_alpha

        C = f / 16 * cos_sq_alpha * (4 + f * (4 - 3 * cos_sq_alpha))
        lambda_prev = lambda_val
        lambda_val = L + (1 - C) * f * sin_alpha * (
            sigma
            + C
            * sin_sigma
            * (cos_2sigma_m + C * cos_sigma * (-1 + 2 * cos_2sigma_m**2))
        )

        if abs(lambda_val - lambda_prev) < tolerance:
            break
    else:
        raise ValueError("Vincenty formula failed to converge")

    u_sq = cos_sq_alpha * (a**2 - b**2) / (b**2)
    A = 1 + u_sq / 16384 * (4096 + u_sq * (-768 + u_sq * (320 - 175 * u_sq)))
    B = u_sq / 1024 * (256 + u_sq * (-128 + u_sq * (74 - 47 * u_sq)))
    delta_sigma = (
        B
        * sin_sigma
        * (
            cos_2sigma_m
            + B
            / 4
            * (
                cos_sigma * (-1 + 2 * cos_2sigma_m**2)
                - B
                / 6
                * cos_2sigma_m
                * (-3 + 4 * sin_sigma**2)
                * (-3 + 4 * cos_2sigma_m**2)
            )
        )
    )

    distance = b * A * (sigma - delta_sigma)
    return float(distance)


def bearing(coord1, coord2):
    """Compute the bearing (direction) from one point to another.

    The bearing is the clockwise angle from north to the direction
    of travel between two points.

    Parameters
    ----------
    coord1 : tuple or list of float
        First coordinate as (latitude, longitude) in degrees.
    coord2 : tuple or list of float
        Second coordinate as (latitude, longitude) in degrees.

    Returns
    -------
    float
        Bearing in degrees from north (0-360).

    Examples
    --------
    >>> bearing((0, 0), (1, 0))
    0.0
    >>> bearing((0, 0), (0, 1))
    90.0

    Notes
    -----
    The initial bearing is calculated using:

    theta = arctan2(sin(delta_lambda) * cos(phi2),
                   cos(phi1) * sin(phi2) - sin(phi1) * cos(phi2) * cos(delta_lambda))
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(
        phi2
    ) * math.cos(delta_lambda)

    theta = math.atan2(y, x)
    bearing_deg = math.degrees(theta)

    # Normalize to 0-360
    return (bearing_deg + 360) % 360


def midpoint(coord1, coord2):
    """Compute the geographic midpoint between two coordinates.

    Parameters
    ----------
    coord1 : tuple or list of float
        First coordinate as (latitude, longitude) in degrees.
    coord2 : tuple or list of float
        Second coordinate as (latitude, longitude) in degrees.

    Returns
    -------
    tuple
        Midpoint as (latitude, longitude) in degrees.

    Examples
    --------
    >>> midpoint((0, 0), (0, 2))
    (0.0, 1.0)

    Notes
    -----
    The midpoint is computed by converting to Cartesian coordinates,
    averaging, and converting back to spherical coordinates.
    """
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1 = math.radians(lat1)
    lambda1 = math.radians(lon1)
    phi2 = math.radians(lat2)
    lambda2 = math.radians(lon2)

    # Convert to Cartesian coordinates
    x1 = math.cos(phi1) * math.cos(lambda1)
    y1 = math.cos(phi1) * math.sin(lambda1)
    z1 = math.sin(phi1)

    x2 = math.cos(phi2) * math.cos(lambda2)
    y2 = math.cos(phi2) * math.sin(lambda2)
    z2 = math.sin(phi2)

    # Average the Cartesian coordinates
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    z = (z1 + z2) / 2

    # Convert back to spherical coordinates
    lon = math.atan2(y, x)
    hyp = math.sqrt(x**2 + y**2)
    lat = math.atan2(z, hyp)

    return (math.degrees(lat), math.degrees(lon))


def destination_point(coord, bearing_deg, distance, R=6372800.0):
    """Compute the destination point given a start, bearing, and distance.

    Parameters
    ----------
    coord : tuple or list of float
        Starting coordinate as (latitude, longitude) in degrees.
    bearing_deg : float
        Bearing (direction) in degrees from north (0-360).
    distance : float
        Distance to travel in the same units as R (meters by default).
    R : float, default 6372800.0
        Radius of the sphere. Default is Earth's mean radius in meters.

    Returns
    -------
    tuple
        Destination coordinate as (latitude, longitude) in degrees.

    Examples
    --------
    >>> destination_point((0, 0), 0, 111319.49)
    (1.0, 0.0)

    Notes
    -----
    Uses the spherical law of cosines to compute the destination point.
    """
    lat1, lon1 = coord
    brng = math.radians(bearing_deg)

    phi1 = math.radians(lat1)
    lambda1 = math.radians(lon1)

    angular_dist = distance / R

    phi2 = math.asin(
        math.sin(phi1) * math.cos(angular_dist)
        + math.cos(phi1) * math.sin(angular_dist) * math.cos(brng)
    )

    lambda2 = lambda1 + math.atan2(
        math.sin(brng) * math.sin(angular_dist) * math.cos(phi1),
        math.cos(angular_dist) - math.sin(phi1) * math.sin(phi2),
    )

    # Normalize longitude to -180 to 180
    lon2 = (math.degrees(lambda2) + 540) % 360 - 180

    return (math.degrees(phi2), lon2)