# -*- coding: utf-8 -*-
"""
Main tests for interspace package - version and exports validation.
"""

import pytest

import interspace


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

    def test_version_is_0_1_0(self):
        assert interspace.__version__ == "0.1.0"


class TestAllExports:
    """Tests that all functions are properly exported."""

    def test_all_list_exists(self):
        assert hasattr(interspace, "__all__")
        assert isinstance(interspace.__all__, list)

    def test_version_not_in_all(self):
        # __version__ should not be in __all__ (it's a string, not a function)
        assert "__version__" not in interspace.__all__

    def test_all_functions_callable(self):
        for name in interspace.__all__:
            func = getattr(interspace, name)
            assert callable(func), f"{name} is not callable"

    def test_key_functions_exported(self):
        """Test that key distance functions are exported."""
        key_functions = [
            # Vector distances
            "euclidean",
            "manhattan",
            "minkowski",
            "cosine_similarity",
            "cosine_distance",
            # Set distances
            "jaccard_distance",
            "dice_distance",
            # String distances
            "levenshtein_distance",
            "hamming",
            # Geographic distances
            "haversine",
            # Information theory
            "entropy",
            # Metrics
            "pairwise_distance",
        ]
        for func_name in key_functions:
            assert func_name in interspace.__all__, f"{func_name} not in __all__"
            assert hasattr(interspace, func_name), f"{func_name} not accessible"

    def test_submodules_accessible(self):
        """Test that submodules are accessible (but not in __all__)."""
        submodules = ["distances", "information", "metrics", "misc"]
        for module_name in submodules:
            assert hasattr(interspace, module_name), f"{module_name} module not accessible"


class TestSubmodules:
    """Tests for submodule access."""

    def test_distances_submodule(self):
        assert hasattr(interspace.distances, "vector")
        assert hasattr(interspace.distances, "string")
        assert hasattr(interspace.distances, "geographic")

    def test_information_submodule(self):
        assert hasattr(interspace.information, "theory")

    def test_metrics_submodule(self):
        assert hasattr(interspace.metrics, "pairwise")
        assert hasattr(interspace.metrics, "validation")


class TestFunctionCount:
    """Tests to verify the expected number of functions."""

    def test_minimum_function_count(self):
        # We expect at least 50 functions
        assert len(interspace.__all__) >= 50

    def test_function_count(self):
        # We expect 59 functions
        assert len(interspace.__all__) == 59
