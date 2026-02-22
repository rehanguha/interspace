# -*- coding: utf-8 -*-
"""
Comprehensive tests for the interspace distance and similarity functions.
"""

import numpy as np
import pytest

import interspace


class TestValidateVarType:
    """Tests for _validate_var_type helper function."""

    def test_int_type_valid(self):
        assert interspace._validate_var_type(21, int) == 21

    def test_int_type_invalid_float(self):
        with pytest.raises(ValueError, match=r".*int.*"):
            interspace._validate_var_type(21.1, int)

    def test_int_type_invalid_string(self):
        with pytest.raises(ValueError, match=r".*int.*"):
            interspace._validate_var_type("string", int)

    def test_float_type_valid(self):
        assert interspace._validate_var_type(21.1, float) == 21.1

    def test_float_type_invalid_int(self):
        with pytest.raises(ValueError, match=r".*float.*"):
            interspace._validate_var_type(21, float)

    def test_float_type_invalid_string(self):
        with pytest.raises(ValueError, match=r".*float.*"):
            interspace._validate_var_type("string", float)

    def test_str_type_valid(self):
        assert interspace._validate_var_type("string", str) == "string"

    def test_str_type_invalid_float(self):
        with pytest.raises(ValueError, match=r".*str.*"):
            interspace._validate_var_type(21.1, str)

    def test_str_type_invalid_int(self):
        with pytest.raises(ValueError, match=r".*str.*"):
            interspace._validate_var_type(21, str)


class TestValidateVector:
    """Tests for _validate_vector helper function."""

    def test_single_element_returns_scalar(self):
        assert interspace._validate_vector([21], dtype=int) == 21

    def test_invalid_dtype_element(self):
        with pytest.raises(ValueError):
            interspace._validate_vector([21, "a"], dtype=int)

    def test_nested_list_raises_error(self):
        with pytest.raises(ValueError):
            interspace._validate_vector([1, [2]], dtype=int)

    def test_multidimensional_raises_error(self):
        with pytest.raises(ValueError, match="1-D"):
            interspace._validate_vector([[1, 2], [3, 4]])

    def test_returns_array_for_multiple_elements(self):
        result = interspace._validate_vector([1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert len(result) == 3


class TestValidateWeights:
    """Tests for _validate_weights helper function."""

    def test_valid_weight(self):
        assert interspace._validate_weights(21) == 21

    def test_negative_weight_raises_error(self):
        with pytest.raises(ValueError, match="non-negative"):
            interspace._validate_weights(-1)

    def test_invalid_weight_type(self):
        with pytest.raises(ValueError):
            interspace._validate_weights("a")


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


class TestHaversine:
    """Tests for haversine distance function."""

    def test_identical_points(self):
        assert interspace.haversine((0, 0), (0, 0)) == pytest.approx(0.0)

    def test_known_distance(self):
        # Test a known distance - New York to Los Angeles
        # NYC (40.7128, -74.0060) to LA (34.0522, -118.2437)
        result = interspace.haversine((40.7128, -74.0060), (34.0522, -118.2437))
        # Approx 3940 km
        assert result == pytest.approx(3940000, rel=0.02)

    def test_custom_radius(self):
        # Mars radius ~3389 km
        result = interspace.haversine((0, 0), (0, 90), R=3389000)
        assert result == pytest.approx(3389000 * np.pi / 2, rel=0.01)

    def test_antipodal_points(self):
        # Antipodal points should give distance = pi * R
        result = interspace.haversine((0, 0), (0, 180))
        assert result == pytest.approx(np.pi * 6372800, rel=0.01)


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


class TestDiceDistance:
    """Tests for dice_distance function."""

    def test_partial_overlap(self):
        # intersection=2, len(sx)=3, len(sy)=3
        # dice = 1 - (2*2)/(3+3) = 1 - 4/6 = 1/3
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
        # intersection=2, min(3,3)=3, overlap = 1 - 2/3 = 1/3
        assert interspace.overlap_distance([1, 2, 3], [2, 3, 4]) == pytest.approx(1/3)

    def test_no_overlap(self):
        assert interspace.overlap_distance([1, 2], [3, 4]) == pytest.approx(1.0)

    def test_identical_sets(self):
        assert interspace.overlap_distance([1, 2], [1, 2]) == pytest.approx(0.0)

    def test_empty_sets(self):
        assert interspace.overlap_distance([], []) == pytest.approx(0.0)

    def test_subset(self):
        # intersection=2, min(2,3)=2
        # overlap = 1 - 2/2 = 0
        assert interspace.overlap_distance([1, 2], [1, 2, 3]) == pytest.approx(0.0)


class TestPearsonDistance:
    """Tests for pearson_distance function (alias for correlation_distance)."""

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


class TestVersion:
    """Tests for package version."""

    def test_version_exists(self):
        assert hasattr(interspace, "__version__")
        assert isinstance(interspace.__version__, str)

    def test_version_format(self):
        # Check semantic versioning format
        parts = interspace.__version__.split(".")
        assert len(parts) >= 2
        assert all(part.isdigit() for part in parts[:2])


class TestAllExports:
    """Tests that all functions are properly exported."""

    def test_all_list_complete(self):
        expected_functions = [
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
        assert set(interspace.__all__) == set(expected_functions)

    def test_all_functions_callable(self):
        for name in interspace.__all__:
            func = getattr(interspace, name)
            assert callable(func)