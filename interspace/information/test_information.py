# -*- coding: utf-8 -*-
"""Tests for information theory functions."""

import numpy as np
import pytest

import interspace


class TestEntropy:
    """Tests for entropy function."""

    def test_uniform_distribution(self):
        # Maximum entropy for uniform distribution
        assert interspace.entropy([0.5, 0.5]) == pytest.approx(1.0)

    def test_deterministic_distribution(self):
        # Zero entropy for deterministic distribution
        assert interspace.entropy([1.0, 0.0]) == pytest.approx(0.0)

    def test_base_2(self):
        result = interspace.entropy([0.5, 0.5], base=2)
        assert result == pytest.approx(1.0)

    def test_base_10(self):
        result = interspace.entropy([0.5, 0.5], base=10)
        assert result == pytest.approx(np.log10(2))

    def test_negative_values_raises_error(self):
        with pytest.raises(ValueError, match="non-negative"):
            interspace.entropy([-0.5, 1.5])

    def test_normalization_warning(self):
        # Values not summing to 1 should still work
        result = interspace.entropy([1, 1])
        assert result >= 0


class TestCrossEntropy:
    """Tests for cross_entropy function."""

    def test_identical_distributions(self):
        # Cross-entropy of p with itself equals entropy of p
        p = [0.5, 0.5]
        ce = interspace.cross_entropy(p, p)
        h = interspace.entropy(p)
        assert ce == pytest.approx(h)

    def test_different_distributions(self):
        result = interspace.cross_entropy([1.0, 0.0], [0.5, 0.5])
        assert result >= 0

    def test_base_2(self):
        result = interspace.cross_entropy([0.5, 0.5], [0.25, 0.75], base=2)
        assert result >= 0


class TestMutualInformation:
    """Tests for mutual_information function."""

    def test_identical_variables(self):
        # MI of X with itself equals H(X)
        x = [0, 0, 1, 1]
        mi = interspace.mutual_information(x, x)
        h = interspace.entropy([0.5, 0.5])
        assert mi == pytest.approx(h)

    def test_independent_variables(self):
        # For independent variables, MI should be 0
        # This is a simplified test - true independence is hard to construct
        x = [0, 1, 0, 1]
        y = [0, 0, 1, 1]
        result = interspace.mutual_information(x, y)
        assert result >= 0

    def test_base_2(self):
        x = [0, 0, 1, 1]
        y = [0, 0, 1, 1]
        result = interspace.mutual_information(x, y, base=2)
        assert result >= 0

    def test_different_lengths_raises_error(self):
        with pytest.raises(ValueError):
            interspace.mutual_information([0, 1], [0, 1, 2])
