# -*- coding: utf-8 -*-
"""
Distance functions module.

This module contains all distance and similarity functions organized by category.

Submodules
---------
vector : Basic vector distances (Euclidean, Manhattan, Minkowski, etc.)
weighted : Weighted distance functions
set : Set-based distances (Jaccard, Dice, etc.)
distribution : Distribution distances (Canberra, Bray-Curtis, etc.)
probability : Probability distribution distances (KL, JS, Bhattacharyya, etc.)
string : String/text distances (Levenshtein, Jaro, etc.)
geographic : Geographic distances (Haversine, Vincenty, etc.)
time_series : Time series distances (DTW, LCS, etc.)
matrix : Matrix distances (Frobenius, Spectral, etc.)
binary : Binary/categorical distances (Russell-Rao, etc.)
normalized : Normalized distances (Chi-squared, Gower, etc.)
physics : Physics-related distances (Angular, 3D Euclidean, etc.)
"""

# Import all functions for easy access
from interspace.distances.vector import (
    euclidean,
    manhattan,
    minkowski,
    chebyshev_distance,
    cosine_similarity,
    cosine_distance,
    mahalanobis,
)

from interspace.distances.weighted import (
    weighted_euclidean,
    weighted_manhattan,
    weighted_minkowski,
)

from interspace.distances.set import (
    jaccard_distance,
    dice_distance,
    matching_distance,
    overlap_distance,
    tanimoto_distance,
)

from interspace.distances.distribution import (
    canberra_distance,
    braycurtis_distance,
    correlation_distance,
    pearson_distance,
    squared_chord_distance,
)

from interspace.distances.probability import (
    kl_divergence,
    js_distance,
    bhattacharyya_distance,
    hellinger_distance,
    total_variation_distance,
    wasserstein_distance,
)

from interspace.distances.string import (
    hamming,
    hamming_distance_normalized,
    levenshtein_distance,
    damerau_levenshtein_distance,
    jaro_distance,
    jaro_winkler_distance,
)

from interspace.distances.geographic import (
    haversine,
    vincenty_distance,
    bearing,
    midpoint,
    destination_point,
)

from interspace.distances.time_series import (
    dtw_distance,
    euclidean_distance_1d,
    longest_common_subsequence,
)

from interspace.distances.matrix import (
    frobenius_distance,
    spectral_distance,
    trace_distance,
)

from interspace.distances.binary import (
    russell_rao_distance,
    sokal_sneath_distance,
    kulczynski_distance,
)

from interspace.distances.normalized import (
    normalized_euclidean,
    standardized_euclidean,
    seuclidean,
    chi2_distance,
    gower_distance,
)

from interspace.distances.physics import (
    angular_distance,
    spherical_law_of_cosines,
    euclidean_3d,
)

__all__ = [
    # Vector distances
    "euclidean",
    "manhattan",
    "minkowski",
    "chebyshev_distance",
    "cosine_similarity",
    "cosine_distance",
    "mahalanobis",
    # Weighted distances
    "weighted_euclidean",
    "weighted_manhattan",
    "weighted_minkowski",
    # Set distances
    "jaccard_distance",
    "dice_distance",
    "matching_distance",
    "overlap_distance",
    "tanimoto_distance",
    # Distribution distances
    "canberra_distance",
    "braycurtis_distance",
    "correlation_distance",
    "pearson_distance",
    "squared_chord_distance",
    # Probability distances
    "kl_divergence",
    "js_distance",
    "bhattacharyya_distance",
    "hellinger_distance",
    "total_variation_distance",
    "wasserstein_distance",
    # String distances
    "hamming",
    "hamming_distance_normalized",
    "levenshtein_distance",
    "damerau_levenshtein_distance",
    "jaro_distance",
    "jaro_winkler_distance",
    # Geographic distances
    "haversine",
    "vincenty_distance",
    "bearing",
    "midpoint",
    "destination_point",
    # Time series distances
    "dtw_distance",
    "euclidean_distance_1d",
    "longest_common_subsequence",
    # Matrix distances
    "frobenius_distance",
    "spectral_distance",
    "trace_distance",
    # Binary distances
    "russell_rao_distance",
    "sokal_sneath_distance",
    "kulczynski_distance",
    # Normalized distances
    "normalized_euclidean",
    "standardized_euclidean",
    "seuclidean",
    "chi2_distance",
    "gower_distance",
    # Physics distances
    "angular_distance",
    "spherical_law_of_cosines",
    "euclidean_3d",
]