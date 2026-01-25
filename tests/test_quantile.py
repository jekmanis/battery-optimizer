"""
Tests for the _quantile utility function.
"""

import math

import pytest

from battery_optimizer_lib import _quantile


class TestQuantile:
    """Test cases for the _quantile function."""

    def test_empty_list(self):
        """Empty list should return 0."""
        assert _quantile([], 0.5) == 0.0

    def test_single_value(self):
        """Single value list should return that value for any quantile."""
        assert _quantile([5.0], 0.0) == 5.0
        assert _quantile([5.0], 0.5) == 5.0
        assert _quantile([5.0], 1.0) == 5.0

    def test_q_zero_returns_minimum(self):
        """Quantile 0 should return the minimum value."""
        values = [5, 2, 8, 1, 9]
        assert _quantile(values, 0.0) == 1

    def test_q_one_returns_maximum(self):
        """Quantile 1 should return the maximum value."""
        values = [5, 2, 8, 1, 9]
        assert _quantile(values, 1.0) == 9

    def test_q_negative_returns_minimum(self):
        """Negative quantile should clamp to minimum."""
        values = [1, 5, 10]
        assert _quantile(values, -0.5) == 1

    def test_q_greater_than_one_returns_maximum(self):
        """Quantile > 1 should clamp to maximum."""
        values = [1, 5, 10]
        assert _quantile(values, 1.5) == 10

    def test_median_odd_list(self):
        """Median of odd-length list."""
        values = [1, 2, 3, 4, 5]
        result = _quantile(values, 0.5)
        assert result == 3.0

    def test_median_even_list(self):
        """Median of even-length list uses linear interpolation."""
        values = [1, 2, 3, 4]
        result = _quantile(values, 0.5)
        assert result == 2.5

    def test_quartiles(self):
        """Test Q1, Q2, Q3 on a known dataset."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        q1 = _quantile(values, 0.25)
        q2 = _quantile(values, 0.50)
        q3 = _quantile(values, 0.75)

        # Q1 should be around 3.25
        assert 3.0 <= q1 <= 3.5
        # Q2 (median) should be around 5.5
        assert 5.0 <= q2 <= 6.0
        # Q3 should be around 7.75
        assert 7.5 <= q3 <= 8.0

    def test_linear_interpolation(self):
        """Test that interpolation works correctly."""
        values = [0.0, 10.0]

        # At q=0.5, should be exactly 5.0
        assert _quantile(values, 0.5) == 5.0

        # At q=0.25, should be 2.5
        assert _quantile(values, 0.25) == 2.5

        # At q=0.75, should be 7.5
        assert _quantile(values, 0.75) == 7.5

    def test_unsorted_input(self):
        """Function should work with unsorted input."""
        unsorted = [9, 1, 5, 3, 7]
        sorted_version = [1, 3, 5, 7, 9]

        assert _quantile(unsorted, 0.5) == _quantile(sorted_version, 0.5)

    def test_duplicate_values(self):
        """Handle lists with duplicate values."""
        values = [5, 5, 5, 5, 5]
        assert _quantile(values, 0.0) == 5
        assert _quantile(values, 0.5) == 5
        assert _quantile(values, 1.0) == 5

    def test_floating_point_values(self):
        """Handle floating point values correctly."""
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        result = _quantile(values, 0.5)
        assert math.isclose(result, 0.3, rel_tol=1e-9)

    def test_negative_values(self):
        """Handle negative values."""
        values = [-10, -5, 0, 5, 10]
        assert _quantile(values, 0.0) == -10
        assert _quantile(values, 0.5) == 0
        assert _quantile(values, 1.0) == 10

    def test_large_dataset(self):
        """Test performance with larger dataset."""
        values = list(range(1000))

        assert _quantile(values, 0.0) == 0
        assert _quantile(values, 1.0) == 999

        # 90th percentile should be around 899
        q90 = _quantile(values, 0.90)
        assert 895 <= q90 <= 905
