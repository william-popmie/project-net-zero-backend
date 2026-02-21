import pytest
from src.app.math_utils import add


class TestAdd:
    """Test suite for the add function."""

    def test_add_positive_numbers(self):
        """Test addition of two positive integers."""
        assert add(2, 3) == 5

    def test_add_negative_numbers(self):
        """Test addition of two negative integers."""
        assert add(-2, -3) == -5

    def test_add_mixed_sign_numbers(self):
        """Test addition of positive and negative integers."""
        assert add(-2, 3) == 1
        assert add(2, -3) == -1

    def test_add_with_zero(self):
        """Test addition with zero."""
        assert add(0, 5) == 5
        assert add(5, 0) == 5
        assert add(0, 0) == 0

    def test_add_large_numbers(self):
        """Test addition with large integers."""
        assert add(1000000, 2000000) == 3000000
        assert add(10**10, 10**10) == 2 * 10**10

    def test_add_commutative_property(self):
        """Test that addition is commutative (a + b == b + a)."""
        assert add(3, 5) == add(5, 3)
        assert add(-7, 12) == add(12, -7)

    def test_add_identity_property(self):
        """Test that zero is the identity element for addition."""
        assert add(42, 0) == 42
        assert add(0, 42) == 42

    def test_add_negative_result(self):
        """Test addition resulting in negative number."""
        assert add(-10, 5) == -5
        assert add(5, -10) == -5


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1, 1, 2),
        (0, 0, 0),
        (-1, 1, 0),
        (100, -100, 0),
        (999, 1, 1000),
        (-50, -50, -100),
    ],
)
def test_add_parametrized(a: int, b: int, expected: int):
    """Parametrized test for various input combinations."""
    assert add(a, b) == expected

import pytest
from src.app.math_utils import multiply


class TestMultiply:
    """Tests for the multiply function."""

    def test_multiply_positive_numbers(self):
        """Test multiplication of two positive integers."""
        assert multiply(3, 4) == 12

    def test_multiply_negative_numbers(self):
        """Test multiplication of two negative integers."""
        assert multiply(-3, -4) == 12

    def test_multiply_positive_and_negative(self):
        """Test multiplication of a positive and negative integer."""
        assert multiply(3, -4) == -12
        assert multiply(-3, 4) == -12

    def test_multiply_by_zero(self):
        """Test multiplication when one operand is zero."""
        assert multiply(0, 5) == 0
        assert multiply(5, 0) == 0

    def test_multiply_both_zero(self):
        """Test multiplication when both operands are zero."""
        assert multiply(0, 0) == 0

    def test_multiply_by_one(self):
        """Test multiplication by one (identity)."""
        assert multiply(1, 7) == 7
        assert multiply(7, 1) == 7

    def test_multiply_by_negative_one(self):
        """Test multiplication by negative one."""
        assert multiply(-1, 7) == -7
        assert multiply(7, -1) == -7

    def test_multiply_large_numbers(self):
        """Test multiplication of large integers."""
        assert multiply(1000000, 1000000) == 1000000000000

    def test_multiply_commutativity(self):
        """Test that multiplication is commutative (a * b == b * a)."""
        assert multiply(5, 3) == multiply(3, 5)
        assert multiply(-7, 4) == multiply(4, -7)

    def test_multiply_associativity_property(self):
        """Test associativity-like property with sequential multiplications."""
        a, b, c = 2, 3, 4
        assert multiply(multiply(a, b), c) == multiply(a, multiply(b, c))


class TestMultiplyEdgeCases:
    """Edge case tests for the multiply function."""

    def test_multiply_max_int_values(self):
        """Test multiplication with very large integers."""
        large_num = 10**18
        assert multiply(large_num, 1) == large_num

    def test_multiply_negative_zero_interaction(self):
        """Test that zero result is consistent regardless of negative operands."""
        assert multiply(-5, 0) == 0
        assert multiply(0, -5) == 0

import pytest
from src.app.math_utils import subtract


class TestSubtract:
    """Tests for the subtract function."""

    def test_subtract_positive_numbers(self):
        """Test subtracting two positive numbers."""
        assert subtract(10, 3) == 7

    def test_subtract_negative_numbers(self):
        """Test subtracting two negative numbers."""
        assert subtract(-5, -3) == -2

    def test_subtract_positive_from_negative(self):
        """Test subtracting a positive number from a negative number."""
        assert subtract(-5, 3) == -8

    def test_subtract_negative_from_positive(self):
        """Test subtracting a negative number from a positive number."""
        assert subtract(5, -3) == 8

    def test_subtract_zero_from_number(self):
        """Test subtracting zero from a number."""
        assert subtract(5, 0) == 5

    def test_subtract_number_from_zero(self):
        """Test subtracting a number from zero."""
        assert subtract(0, 5) == -5

    def test_subtract_zeros(self):
        """Test subtracting zero from zero."""
        assert subtract(0, 0) == 0

    def test_subtract_same_numbers(self):
        """Test subtracting a number from itself."""
        assert subtract(42, 42) == 0

    def test_subtract_large_numbers(self):
        """Test subtracting large numbers."""
        assert subtract(1000000, 999999) == 1

    def test_subtract_results_in_negative(self):
        """Test subtraction that results in a negative number."""
        assert subtract(3, 10) == -7

    def test_subtract_with_negative_result_from_negatives(self):
        """Test subtraction of negative numbers resulting in negative."""
        assert subtract(-3, -1) == -2

    def test_subtract_large_negative_numbers(self):
        """Test subtracting large negative numbers."""
        assert subtract(-1000000, -999999) == -1


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (10, 5, 5),
        (5, 10, -5),
        (0, 0, 0),
        (-1, -1, 0),
        (100, 50, 50),
        (-100, 50, -150),
        (100, -50, 150),
    ],
)
def test_subtract_parametrized(a: int, b: int, expected: int):
    """Parametrized test for various subtraction scenarios."""
    assert subtract(a, b) == expected

import pytest

from src.app.math_utils import divide


class TestDivide:
    """Test suite for the divide function."""

    def test_divide_positive_numbers(self):
        """Test division of two positive numbers."""
        result = divide(10.0, 2.0)
        assert result == 5.0

    def test_divide_negative_numbers(self):
        """Test division of two negative numbers."""
        result = divide(-10.0, -2.0)
        assert result == 5.0

    def test_divide_positive_by_negative(self):
        """Test division of positive number by negative number."""
        result = divide(10.0, -2.0)
        assert result == -5.0

    def test_divide_negative_by_positive(self):
        """Test division of negative number by positive number."""
        result = divide(-10.0, 2.0)
        assert result == -5.0

    def test_divide_zero_by_number(self):
        """Test division of zero by a non-zero number."""
        result = divide(0.0, 5.0)
        assert result == 0.0

    def test_divide_by_zero_raises_value_error(self):
        """Test that dividing by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(10.0, 0.0)

    def test_divide_zero_by_zero_raises_value_error(self):
        """Test that dividing zero by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(0.0, 0.0)

    def test_divide_floats_with_decimal_places(self):
        """Test division with floating point numbers."""
        result = divide(7.5, 2.5)
        assert result == 3.0

    def test_divide_returns_float(self):
        """Test that divide returns a float type."""
        result = divide(10.0, 3.0)
        assert isinstance(result, float)

    def test_divide_integer_inputs(self):
        """Test division with integer inputs (should still work due to type coercion)."""
        result = divide(10, 4)
        assert result == 2.5

    def test_divide_small_numbers(self):
        """Test division with very small numbers."""
        result = divide(0.001, 0.001)
        assert result == pytest.approx(1.0)

    def test_divide_large_numbers(self):
        """Test division with large numbers."""
        result = divide(1e10, 1e5)
        assert result == pytest.approx(1e5)

    def test_divide_result_precision(self):
        """Test division result with floating point precision."""
        result = divide(1.0, 3.0)
        assert result == pytest.approx(0.3333333333333333)

    def test_divide_one_by_number(self):
        """Test division of 1 by a number."""
        result = divide(1.0, 4.0)
        assert result == 0.25

    def test_divide_number_by_one(self):
        """Test division of a number by 1."""
        result = divide(42.0, 1.0)
        assert result == 42.0

    def test_divide_number_by_itself(self):
        """Test division of a number by itself."""
        result = divide(7.0, 7.0)
        assert result == 1.0

    def test_divide_negative_zero_raises_value_error(self):
        """Test that dividing by negative zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(10.0, -0.0)
