# -*- coding: utf-8 -*-
"""Tests for metrics utility functions."""

import numpy as np
import pytest

import interspace


class TestPairwiseDistance:
    """Tests for pairwise_distance function."""

    def test_single_vector(self):
        X = [[1, 2, 3]]
        result = interspace.pairwise_distance(X)
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(0.0)

    def test_two_vectors(self):
        X = [[1, 2], [4, 6]]
        result = interspace.pairwise_distance(X)
        assert result.shape == (2, 2)
        assert result[0, 0] == pytest.approx(0.0)
        assert result[1, 1] == pytest.approx(0.0)
        assert result[0, 1] == result[1, 0]  # Symmetric

    def test_euclidean_metric(self):
        X = [[1, 2], [4, 6]]
        result = interspace.pairwise_distance(X, metric="euclidean")
        assert result[0, 1] == pytest.approx(5.0)

    def test_manhattan_metric(self):
        X = [[1, 2], [4, 6]]
        result = interspace.pairwise_distance(X, metric="manhattan")
        assert result[0, 1] == pytest.approx(7.0)

    def test_cosine_distance_metric(self):
        X = [[1, 0], [0, 1]]
        result = interspace.pairwise_distance(X, metric=interspace.cosine_distance)
        assert result[0, 1] == pytest.approx(1.0)

    def test_two_matrices(self):
        X = [[1, 2], [3, 4]]
        Y = [[1, 2], [5, 6]]
        result = interspace.pairwise_distance(X, Y)
        assert result.shape == (2, 2)

    def test_invalid_metric_raises_error(self):
        X = [[1, 2], [3, 4]]
        with pytest.raises(ValueError, match="Unknown metric"):
            interspace.pairwise_distance(X, metric="invalid_metric")


class TestIsDistanceMetric:
    """Tests for is_distance_metric function."""

    def test_euclidean_is_metric(self):
        assert interspace.is_distance_metric(interspace.euclidean) == True

    def test_manhattan_is_metric(self):
        assert interspace.is_distance_metric(interspace.manhattan) == True

    def test_euclidean_is_valid_metric(self):
        # Euclidean distance satisfies all metric properties
        assert interspace.is_distance_metric(interspace.euclidean) == True

    def test_manhattan_is_valid_metric(self):
        # Manhattan distance satisfies all metric properties
        assert interspace.is_distance_metric(interspace.manhattan) == True

    def test_custom_metric(self):
        # A function that doesn't satisfy metric properties
        def bad_distance(x, y):
            return -1  # Negative distance

        assert interspace.is_distance_metric(bad_distance) == False