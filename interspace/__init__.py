# -*- coding: utf-8 -*-
"""
Interspace - Distance and Similarity Functions for Vectors, Sequences, and Distributions.

This module provides a comprehensive collection of distance and similarity metrics
commonly used in machine learning, data science, and scientific computing.
"""

from .interspace import (
    __version__,
    haversine,
    manhattan,
    euclidean,
    minkowski,
    cosine_similarity,
    cosine_distance,
    hamming,
    mahalanobis,
    chebyshev_distance,
    jaccard_distance,
    canberra_distance,
    braycurtis_distance,
    correlation_distance,
    dice_distance,
    matching_distance,
    overlap_distance,
    pearson_distance,
    squared_chord_distance,
    # Internal helpers (for testing)
    _validate_vector,
    _validate_var_type,
    _validate_weights,
    _validate_same_shape,
)

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