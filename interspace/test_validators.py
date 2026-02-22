# -*- coding: utf-8 -*-
"""Tests for internal validator helper functions."""

import numpy as np
import pytest

from interspace._validators import (
    _validate_var_type,
    _validate_vector,
    _validate_weights,
    _validate_same_shape,
)


class TestValidateVarType:
    """Tests for _validate_var_type helper function."""

    def test_int_type_valid(self):
        assert _validate_var_type(21, int) == 21

    def test_int_type_invalid_float(self):
        with pytest.raises(ValueError, match=r".*int.*"):
            _validate_var_type(21.1, int)

    def test_int_type_invalid_string(self):
        with pytest.raises(ValueError, match=r".*int.*"):
            _validate_var_type("string", int)

    def test_float_type_valid(self):
        assert _validate_var_type(21.1, float) == 21.1

    def test_float_type_invalid_int(self):
        with pytest.raises(ValueError, match=r".*float.*"):
            _validate_var_type(21, float)

    def test_float_type_invalid_string(self):
        with pytest.raises(ValueError, match=r".*float.*"):
            _validate_var_type("string", float)

    def test_str_type_valid(self):
        assert _validate_var_type("string", str) == "string"

    def test_str_type_invalid_float(self):
        with pytest.raises(ValueError, match=r".*str.*"):
            _validate_var_type(21.1, str)

    def test_str_type_invalid_int(self):
        with pytest.raises(ValueError, match=r".*str.*"):
            _validate_var_type(21, str)


class TestValidateVector:
    """Tests for _validate_vector helper function."""

    def test_single_element_returns_scalar(self):
        assert _validate_vector([21], dtype=int) == 21

    def test_invalid_dtype_element(self):
        with pytest.raises(ValueError):
            _validate_vector([21, "a"], dtype=int)

    def test_nested_list_raises_error(self):
        with pytest.raises(ValueError):
            _validate_vector([1, [2]], dtype=int)

    def test_multidimensional_raises_error(self):
        with pytest.raises(ValueError, match="1-D"):
            _validate_vector([[1, 2], [3, 4]])

    def test_returns_array_for_multiple_elements(self):
        result = _validate_vector([1, 2, 3])
        assert isinstance(result, np.ndarray)
        assert len(result) == 3


class TestValidateWeights:
    """Tests for _validate_weights helper function."""

    def test_valid_weight(self):
        assert _validate_weights(21) == 21

    def test_negative_weight_raises_error(self):
        with pytest.raises(ValueError, match="non-negative"):
            _validate_weights(-1)

    def test_invalid_weight_type(self):
        with pytest.raises(ValueError):
            _validate_weights("a")


class TestValidateSameShape:
    """Tests for _validate_same_shape helper function."""

    def test_same_shape_arrays(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        # Should not raise
        _validate_same_shape(a, b)

    def test_different_shape_raises_error(self):
        a = np.array([1, 2])
        b = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="same shape"):
            _validate_same_shape(a, b)