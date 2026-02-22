# -*- coding: utf-8 -*-
"""
Comprehensive tests for all distance functions.
"""

import numpy as np
import pytest

import interspace


# =============================================================================
# VECTOR DISTANCES
# =============================================================================


class TestEuclidean:
    """Tests for euclidean distance function."""

    def test_basic_distance(self):
        assert interspace.euclidean([1, 2, 3], [4, 5, 6]) == pytest.approx(5.196152422706632)

    def test_zero_distance(self):
        assert interspace.euclidean([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_pythagorean_triple(self):
        assert interspace.euclidean([0, 0], [3, 4]) == pytest.approx(5.0)

    def test_numpy_arrays(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        assert interspace.euclidean(a, b) == pytest.approx(5.196152422706632)

    def test_shape_mismatch_raises_error(self):
        with pytest.raises(ValueError, match="same shape"):
            interspace.euclidean([1, 2], [1, 2, 3])

    def test_single_element(self):
        assert interspace.euclidean([5], [3]) == pytest.approx(2.0)


class TestManhattan:
    """Tests for manhattan distance function."""

    def test_basic_distance(self):
        assert interspace.manhattan([1, 2, 3], [4, 5, 6]) == pytest.approx(9.0)

    def test_zero_distance(self):
        assert interspace.manhattan([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_shape_mismatch_raises_error(self):
        with pytest.raises(ValueError, match="same shape"):
            interspace.manhattan([1, 2], [1, 2, 3])


class TestMinkowski:
    """Tests for minkowski distance function."""

    def test_p1_manhattan(self):
        assert interspace.minkowski([1, 2], [4, 6], p=1) == pytest.approx(7.0)

    def test_p2_euclidean(self):
        assert interspace.minkowski([1, 2], [4, 6], p=2) == pytest.approx(5.0)

    def test_p3(self):
        assert interspace.minkowski([1, 2], [4, 6], p=3) == pytest.approx((27 + 64) ** (1 / 3))

    def test_shape_mismatch_raises_error(self):
        with pytest.raises(ValueError, match="same shape"):
            interspace.minkowski([1, 2], [1, 2, 3])


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_orthogonal_vectors(self):
        assert interspace.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_identical_vectors(self):
        assert interspace.cosine_similarity([1, 1], [1, 1]) == pytest.approx(1.0)

    def test_opposite_vectors(self):
        assert interspace.cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector_raises_error(self):
        with pytest.raises(ValueError, match="Zero-vector"):
            interspace.cosine_similarity([0, 0], [1, 1])

    def test_zero_vector_second_raises_error(self):
        with pytest.raises(ValueError, match="Zero-vector"):
            interspace.cosine_similarity([1, 1], [0, 0])

    def test_shape_mismatch_raises_error(self):
        with pytest.raises(ValueError, match="same shape"):
            interspace.cosine_similarity([1, 2], [1, 2, 3])


class TestCosineDistance:
    """Tests for cosine_distance function."""

    def test_orthogonal_vectors(self):
        assert interspace.cosine_distance([1, 0], [0, 1]) == pytest.approx(1.0)

    def test_identical_vectors(self):
        assert interspace.cosine_distance([1, 1], [1, 1]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert interspace.cosine_distance([1, 0], [-1, 0]) == pytest.approx(2.0)


class TestMahalanobis:
    """Tests for mahalanobis distance function."""

    def test_identity_covariance(self):
        VI = np.array([[1, 0], [0, 1]])
        assert interspace.mahalanobis([0, 0], [3, 4], VI) == pytest.approx(5.0)

    def test_correlated_variables(self):
        VI = np.array([[1, 0.5], [0.5, 1]])
        result = interspace.mahalanobis([0, 0], [1, 1], VI)
        assert result == pytest.approx(np.sqrt(3), rel=0.01)


class TestChebyshevDistance:
    """Tests for chebyshev_distance function."""

    def test_basic_distance(self):
        assert interspace.chebyshev_distance([1, 2, 3], [4, 5, 6]) == pytest.approx(3.0)

    def test_zero_distance(self):
        assert interspace.chebyshev_distance([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_pythagorean_triple(self):
        assert interspace.chebyshev_distance([0, 0], [3, 4]) == pytest.approx(4.0)

    def test_shape_mismatch_raises_error(self):
        with pytest.raises(ValueError, match="same shape"):
            interspace.chebyshev_distance([1, 2], [1, 2, 3])


# =============================================================================
# WEIGHTED DISTANCES
# =============================================================================


class TestWeightedEuclidean:
    """Tests for weighted_euclidean function."""

    def test_equal_weights(self):
        result = interspace.weighted_euclidean([1, 2], [4, 6], [1, 1])
        assert result == pytest.approx(5.0)

    def test_different_weights(self):
        result = interspace.weighted_euclidean([1, 2], [4, 6], [1, 0.5])
        # weighted_euclidean uses weights squared internally
        assert result >= 0

    def test_zero_weights(self):
        result = interspace.weighted_euclidean([1, 2], [4, 6], [0, 1])
        assert result >= 0


class TestWeightedManhattan:
    """Tests for weighted_manhattan function."""

    def test_equal_weights(self):
        result = interspace.weighted_manhattan([1, 2], [4, 6], [1, 1])
        assert result == pytest.approx(7.0)

    def test_different_weights(self):
        result = interspace.weighted_manhattan([1, 2], [4, 6], [2, 1])
        assert result == pytest.approx(10.0)


class TestWeightedMinkowski:
    """Tests for weighted_minkowski function."""

    def test_p1_weighted_manhattan(self):
        result = interspace.weighted_minkowski([1, 2], [4, 6], [1, 1], p=1)
        assert result == pytest.approx(7.0)

    def test_p2_weighted_euclidean(self):
        result = interspace.weighted_minkowski([1, 2], [4, 6], [1, 1], p=2)
        assert result == pytest.approx(5.0)


# =============================================================================
# SET DISTANCES
# =============================================================================


class TestJaccardDistance:
    """Tests for jaccard_distance function."""

    def test_partial_overlap(self):
        assert interspace.jaccard_distance([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5)

    def test_no_overlap(self):
        assert interspace.jaccard_distance([1, 2], [3, 4]) == pytest.approx(1.0)

    def test_identical_sets(self):
        assert interspace.jaccard_distance([1, 2], [1, 2]) == pytest.approx(0.0)

    def test_empty_sets(self):
        assert interspace.jaccard_distance([], []) == pytest.approx(0.0)


class TestDiceDistance:
    """Tests for dice_distance function."""

    def test_partial_overlap(self):
        assert interspace.dice_distance([1, 2, 3], [2, 3, 4]) == pytest.approx(1/3)

    def test_no_overlap(self):
        assert interspace.dice_distance([1, 2], [3, 4]) == pytest.approx(1.0)

    def test_identical_sets(self):
        assert interspace.dice_distance([1, 2], [1, 2]) == pytest.approx(0.0)

    def test_empty_sets(self):
        assert interspace.dice_distance([], []) == pytest.approx(0.0)


class TestMatchingDistance:
    """Tests for matching_distance function."""

    def test_partial_match(self):
        assert interspace.matching_distance([1, 0, 1], [1, 1, 1]) == pytest.approx(1 / 3)

    def test_identical_vectors(self):
        assert interspace.matching_distance([1, 0, 1], [1, 0, 1]) == pytest.approx(0.0)

    def test_completely_different(self):
        assert interspace.matching_distance([0, 0, 0], [1, 1, 1]) == pytest.approx(1.0)

    def test_unequal_length_raises_error(self):
        with pytest.raises(ValueError, match="same length"):
            interspace.matching_distance([1, 2], [1, 2, 3])


class TestOverlapDistance:
    """Tests for overlap_distance function."""

    def test_partial_overlap(self):
        assert interspace.overlap_distance([1, 2, 3], [2, 3, 4]) == pytest.approx(1/3)

    def test_no_overlap(self):
        assert interspace.overlap_distance([1, 2], [3, 4]) == pytest.approx(1.0)

    def test_identical_sets(self):
        assert interspace.overlap_distance([1, 2], [1, 2]) == pytest.approx(0.0)

    def test_empty_sets(self):
        assert interspace.overlap_distance([], []) == pytest.approx(0.0)

    def test_subset(self):
        assert interspace.overlap_distance([1, 2], [1, 2, 3]) == pytest.approx(0.0)


class TestTanimotoDistance:
    """Tests for tanimoto_distance function."""

    def test_identical_vectors(self):
        assert interspace.tanimoto_distance([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_orthogonal_vectors(self):
        result = interspace.tanimoto_distance([1, 0], [0, 1])
        assert result == pytest.approx(1.0)


# =============================================================================
# DISTRIBUTION DISTANCES
# =============================================================================


class TestCanberraDistance:
    """Tests for canberra_distance function."""

    def test_basic_distance(self):
        result = interspace.canberra_distance([1, 2, 3], [2, 2, 4])
        expected = 1.0 / 3 + 0.0 + 1.0 / 7
        assert result == pytest.approx(expected)

    def test_zero_vectors(self):
        assert interspace.canberra_distance([0, 0], [0, 0]) == pytest.approx(0.0)

    def test_shape_mismatch_raises_error(self):
        with pytest.raises(ValueError, match="same shape"):
            interspace.canberra_distance([1, 2], [1, 2, 3])


class TestBraycurtisDistance:
    """Tests for braycurtis_distance function."""

    def test_basic_distance(self):
        result = interspace.braycurtis_distance([1, 2, 3], [2, 2, 4])
        expected = (1 + 0 + 1) / (3 + 4 + 7)
        assert result == pytest.approx(expected)

    def test_zero_vectors(self):
        assert interspace.braycurtis_distance([0, 0], [0, 0]) == pytest.approx(0.0)

    def test_identical_vectors(self):
        assert interspace.braycurtis_distance([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_shape_mismatch_raises_error(self):
        with pytest.raises(ValueError, match="same shape"):
            interspace.braycurtis_distance([1, 2], [1, 2, 3])


class TestCorrelationDistance:
    """Tests for correlation_distance function."""

    def test_identical_vectors(self):
        assert interspace.correlation_distance([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_opposite_correlation(self):
        assert interspace.correlation_distance([1, 2, 3], [3, 2, 1]) == pytest.approx(2.0)

    def test_single_element_raises_error(self):
        with pytest.raises(ValueError, match="two elements"):
            interspace.correlation_distance([1], [2])

    def test_shape_mismatch_raises_error(self):
        with pytest.raises(ValueError, match="same shape"):
            interspace.correlation_distance([1, 2], [1, 2, 3])


class TestPearsonDistance:
    """Tests for pearson_distance function."""

    def test_identical_vectors(self):
        assert interspace.pearson_distance([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_opposite_correlation(self):
        assert interspace.pearson_distance([1, 2, 3], [3, 2, 1]) == pytest.approx(2.0)


class TestSquaredChordDistance:
    """Tests for squared_chord_distance function."""

    def test_basic_distance(self):
        result = interspace.squared_chord_distance([1, 2, 3], [2, 3, 4])
        assert result >= 0

    def test_identical_vectors(self):
        assert interspace.squared_chord_distance([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_zero_vectors(self):
        assert interspace.squared_chord_distance([0, 0], [0, 0]) == pytest.approx(0.0)

    def test_negative_input_raises_error(self):
        with pytest.raises(ValueError, match="non-negative"):
            interspace.squared_chord_distance([-1, 2], [1, 2])

    def test_shape_mismatch_raises_error(self):
        with pytest.raises(ValueError, match="same shape"):
            interspace.squared_chord_distance([1, 2], [1, 2, 3])


# =============================================================================
# PROBABILITY DISTANCES
# =============================================================================


class TestKLDivergence:
    """Tests for kl_divergence function."""

    def test_identical_distributions(self):
        assert interspace.kl_divergence([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)

    def test_different_distributions(self):
        result = interspace.kl_divergence([1.0, 0.0], [0.5, 0.5])
        assert result == pytest.approx(np.log(2))

    def test_negative_values_raises_error(self):
        with pytest.raises(ValueError, match="non-negative"):
            interspace.kl_divergence([-0.5, 1.5], [0.5, 0.5])


class TestJSDistance:
    """Tests for js_distance function."""

    def test_identical_distributions(self):
        assert interspace.js_distance([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)

    def test_different_distributions(self):
        result = interspace.js_distance([1.0, 0.0], [0.5, 0.5])
        assert 0 <= result <= 1

    def test_symmetry(self):
        p, q = [0.5, 0.5], [0.25, 0.75]
        assert interspace.js_distance(p, q) == pytest.approx(interspace.js_distance(q, p))


class TestBhattacharyyaDistance:
    """Tests for bhattacharyya_distance function."""

    def test_identical_distributions(self):
        assert interspace.bhattacharyya_distance([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)

    def test_different_distributions(self):
        result = interspace.bhattacharyya_distance([1.0, 0.0], [0.5, 0.5])
        assert result >= 0


class TestHellingerDistance:
    """Tests for hellinger_distance function."""

    def test_identical_distributions(self):
        assert interspace.hellinger_distance([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)

    def test_bounded(self):
        result = interspace.hellinger_distance([1.0, 0.0], [0.5, 0.5])
        assert 0 <= result <= 1


class TestTotalVariationDistance:
    """Tests for total_variation_distance function."""

    def test_identical_distributions(self):
        assert interspace.total_variation_distance([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)

    def test_opposite_distributions(self):
        result = interspace.total_variation_distance([1.0, 0.0], [0.0, 1.0])
        assert result == pytest.approx(1.0)


class TestWassersteinDistance:
    """Tests for wasserstein_distance function."""

    def test_identical_distributions(self):
        assert interspace.wasserstein_distance([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)

    def test_different_distributions(self):
        result = interspace.wasserstein_distance([1.0, 0.0], [0.0, 1.0])
        assert result >= 0


# =============================================================================
# STRING DISTANCES
# =============================================================================


class TestHamming:
    """Tests for hamming distance function."""

    def test_integers_bitwise(self):
        assert interspace.hamming(0b1010, 0b0011) == 2

    def test_integers_zero_distance(self):
        assert interspace.hamming(0b1010, 0b1010) == 0

    def test_strings(self):
        assert interspace.hamming("abcd", "abcf") == 1

    def test_strings_identical(self):
        assert interspace.hamming("hello", "hello") == 0

    def test_strings_unequal_length_raises_error(self):
        with pytest.raises(ValueError, match="unequal length"):
            interspace.hamming("abc", "abcd")

    def test_arrays(self):
        assert interspace.hamming([1, 2, 3], [1, 0, 3]) == 1

    def test_arrays_unequal_shape_raises_error(self):
        with pytest.raises(ValueError, match="unequal shape"):
            interspace.hamming([1, 2], [1, 2, 3])


class TestHammingDistanceNormalized:
    """Tests for hamming_distance_normalized function."""

    def test_strings(self):
        assert interspace.hamming_distance_normalized("abcd", "abcf") == pytest.approx(0.25)

    def test_identical_strings(self):
        assert interspace.hamming_distance_normalized("hello", "hello") == pytest.approx(0.0)


class TestLevenshteinDistance:
    """Tests for levenshtein_distance function."""

    def test_basic_distance(self):
        assert interspace.levenshtein_distance("kitten", "sitting") == 3

    def test_identical_strings(self):
        assert interspace.levenshtein_distance("hello", "hello") == 0

    def test_empty_string(self):
        assert interspace.levenshtein_distance("", "abc") == 3


class TestDamerauLevenshteinDistance:
    """Tests for damerau_levenshtein_distance function."""

    def test_with_transposition(self):
        result = interspace.damerau_levenshtein_distance("ca", "abc")
        assert result >= 0

    def test_identical_strings(self):
        assert interspace.damerau_levenshtein_distance("hello", "hello") == 0


class TestJaroDistance:
    """Tests for jaro_distance function."""

    def test_identical_strings(self):
        assert interspace.jaro_distance("hello", "hello") == pytest.approx(1.0)

    def test_different_strings(self):
        result = interspace.jaro_distance("MARTHA", "MARHTA")
        assert 0 <= result <= 1


class TestJaroWinklerDistance:
    """Tests for jaro_winkler_distance function."""

    def test_identical_strings(self):
        assert interspace.jaro_winkler_distance("hello", "hello") == pytest.approx(1.0)

    def test_with_prefix_match(self):
        result = interspace.jaro_winkler_distance("MARTHA", "MARHTA")
        assert 0 <= result <= 1


# =============================================================================
# GEOGRAPHIC DISTANCES
# =============================================================================


class TestHaversine:
    """Tests for haversine distance function."""

    def test_identical_points(self):
        assert interspace.haversine((0, 0), (0, 0)) == pytest.approx(0.0)

    def test_known_distance(self):
        # NYC to LA approximately 3940 km
        result = interspace.haversine((40.7128, -74.0060), (34.0522, -118.2437))
        assert result == pytest.approx(3940000, rel=0.02)

    def test_custom_radius(self):
        # Mars radius ~3389 km
        result = interspace.haversine((0, 0), (0, 90), R=3389000)
        assert result == pytest.approx(3389000 * np.pi / 2, rel=0.01)

    def test_antipodal_points(self):
        result = interspace.haversine((0, 0), (0, 180))
        assert result == pytest.approx(np.pi * 6372800, rel=0.01)


class TestVincentyDistance:
    """Tests for vincenty_distance function."""

    def test_identical_points(self):
        assert interspace.vincenty_distance((0, 0), (0, 0)) == pytest.approx(0.0, abs=1)

    def test_different_points(self):
        result = interspace.vincenty_distance((0, 0), (0, 1))
        assert result > 0


class TestBearing:
    """Tests for bearing function."""

    def test_north(self):
        assert interspace.bearing((0, 0), (1, 0)) == pytest.approx(0.0)

    def test_east(self):
        assert interspace.bearing((0, 0), (0, 1)) == pytest.approx(90.0)

    def test_south(self):
        assert interspace.bearing((0, 0), (-1, 0)) == pytest.approx(180.0)

    def test_west(self):
        assert interspace.bearing((0, 0), (0, -1)) == pytest.approx(270.0)


class TestMidpoint:
    """Tests for midpoint function."""

    def test_same_longitude(self):
        lat, lon = interspace.midpoint((0, 0), (2, 0))
        assert lat == pytest.approx(1.0)
        assert lon == pytest.approx(0.0)

    def test_same_latitude(self):
        lat, lon = interspace.midpoint((0, 0), (0, 2))
        assert lat == pytest.approx(0.0)
        assert lon == pytest.approx(1.0)


class TestDestinationPoint:
    """Tests for destination_point function."""

    def test_north_movement(self):
        lat, lon = interspace.destination_point((0, 0), 0, 111319)  # ~1 degree north
        assert lat == pytest.approx(1.0, rel=0.01)


# =============================================================================
# TIME SERIES DISTANCES
# =============================================================================


class TestDTWDistance:
    """Tests for dtw_distance function."""

    def test_identical_series(self):
        assert interspace.dtw_distance([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_stretched_series(self):
        result = interspace.dtw_distance([1, 2, 3], [1, 2, 2, 3])
        assert result >= 0


class TestEuclideanDistance1d:
    """Tests for euclidean_distance_1d function."""

    def test_identical_series(self):
        assert interspace.euclidean_distance_1d([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_different_series(self):
        assert interspace.euclidean_distance_1d([1, 2, 3], [4, 5, 6]) == pytest.approx(5.196152422706632)


class TestLongestCommonSubsequence:
    """Tests for longest_common_subsequence function."""

    def test_identical_sequences(self):
        assert interspace.longest_common_subsequence([1, 2, 3], [1, 2, 3]) == 3

    def test_partial_match(self):
        assert interspace.longest_common_subsequence([1, 2, 3, 4], [2, 3, 5]) == 2

    def test_no_match(self):
        assert interspace.longest_common_subsequence([1, 2], [3, 4]) == 0


# =============================================================================
# MATRIX DISTANCES
# =============================================================================


class TestFrobeniusDistance:
    """Tests for frobenius_distance function."""

    def test_identical_matrices(self):
        A = [[1, 2], [3, 4]]
        B = [[1, 2], [3, 4]]
        assert interspace.frobenius_distance(A, B) == pytest.approx(0.0)

    def test_different_matrices(self):
        A = [[1, 0], [0, 1]]
        B = [[1, 0], [0, 2]]
        assert interspace.frobenius_distance(A, B) == pytest.approx(1.0)


class TestSpectralDistance:
    """Tests for spectral_distance function."""

    def test_identical_matrices(self):
        A = [[1, 0], [0, 1]]
        B = [[1, 0], [0, 1]]
        assert interspace.spectral_distance(A, B) == pytest.approx(0.0)

    def test_different_matrices(self):
        A = [[1, 0], [0, 1]]
        B = [[1, 0], [0, 2]]
        assert interspace.spectral_distance(A, B) == pytest.approx(1.0)


class TestTraceDistance:
    """Tests for trace_distance function."""

    def test_identical_matrices(self):
        A = [[1, 0], [0, 1]]
        B = [[1, 0], [0, 1]]
        assert interspace.trace_distance(A, B) == pytest.approx(0.0)

    def test_different_matrices(self):
        A = [[1, 0], [0, 1]]
        B = [[2, 0], [0, 1]]
        assert interspace.trace_distance(A, B) == pytest.approx(0.5)


# =============================================================================
# BINARY DISTANCES
# =============================================================================


class TestRussellRaoDistance:
    """Tests for russell_rao_distance function."""

    def test_identical_vectors(self):
        # For [1, 0, 1, 0], a=2 (both 1s), n=4, so distance = 1 - 2/4 = 0.5
        # Russell-Rao only counts positions where BOTH have 1 as matches
        result = interspace.russell_rao_distance([1, 0, 1, 0], [1, 0, 1, 0])
        assert result == pytest.approx(0.5)

    def test_all_ones_identical(self):
        # For all 1s, distance should be 0
        assert interspace.russell_rao_distance([1, 1, 1, 1], [1, 1, 1, 1]) == pytest.approx(0.0)

    def test_different_vectors(self):
        result = interspace.russell_rao_distance([1, 0, 1, 0], [1, 1, 0, 0])
        assert result >= 0


class TestSokalSneathDistance:
    """Tests for sokal_sneath_distance function."""

    def test_identical_vectors(self):
        assert interspace.sokal_sneath_distance([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(0.0)

    def test_different_vectors(self):
        result = interspace.sokal_sneath_distance([1, 0, 1, 0], [1, 1, 0, 0])
        assert result >= 0


class TestKulczynskiDistance:
    """Tests for kulczynski_distance function."""

    def test_identical_vectors(self):
        assert interspace.kulczynski_distance([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(0.0)

    def test_different_vectors(self):
        result = interspace.kulczynski_distance([1, 0, 1, 0], [1, 1, 0, 0])
        assert result >= 0


# =============================================================================
# NORMALIZED DISTANCES
# =============================================================================


class TestNormalizedEuclidean:
    """Tests for normalized_euclidean function."""

    def test_identical_vectors(self):
        assert interspace.normalized_euclidean([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_different_vectors(self):
        result = interspace.normalized_euclidean([1, 2, 3], [4, 5, 6])
        assert result >= 0


class TestStandardizedEuclidean:
    """Tests for standardized_euclidean function."""

    def test_identical_vectors(self):
        assert interspace.standardized_euclidean([1, 2, 3], [1, 2, 3], [1, 1, 1]) == pytest.approx(0.0)

    def test_different_vectors(self):
        result = interspace.standardized_euclidean([1, 2, 3], [4, 5, 6], [1, 1, 1])
        assert result >= 0


class TestChi2Distance:
    """Tests for chi2_distance function."""

    def test_identical_vectors(self):
        assert interspace.chi2_distance([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_different_vectors(self):
        result = interspace.chi2_distance([1, 2, 3], [2, 3, 4])
        assert result >= 0


class TestGowerDistance:
    """Tests for gower_distance function."""

    def test_identical_vectors(self):
        assert interspace.gower_distance([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)

    def test_different_vectors(self):
        result = interspace.gower_distance([1, 2, 3], [4, 5, 6])
        assert result >= 0


# =============================================================================
# PHYSICS DISTANCES
# =============================================================================


class TestAngularDistance:
    """Tests for angular_distance function."""

    def test_identical_angles(self):
        assert interspace.angular_distance(45, 45) == pytest.approx(0.0)

    def test_opposite_angles(self):
        assert interspace.angular_distance(0, 180) == pytest.approx(180.0)

    def test_wrap_around(self):
        assert interspace.angular_distance(10, 350) == pytest.approx(20.0)


class TestSphericalLawOfCosines:
    """Tests for spherical_law_of_cosines function."""

    def test_identical_points(self):
        assert interspace.spherical_law_of_cosines((0, 0), (0, 0)) == pytest.approx(0.0)

    def test_different_points(self):
        result = interspace.spherical_law_of_cosines((0, 0), (0, 1))
        assert result > 0


class TestEuclidean3d:
    """Tests for euclidean_3d function."""

    def test_identical_points(self):
        assert interspace.euclidean_3d([0, 0, 0], [0, 0, 0]) == pytest.approx(0.0)

    def test_basic_distance(self):
        assert interspace.euclidean_3d([0, 0, 0], [1, 2, 2]) == pytest.approx(3.0)