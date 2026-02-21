"""Auto-generated tests by AI Spec Generator."""

import pytest
from src.app.math_utils import subtract


class TestSubtract:
    """Test suite for the subtract function."""
    
    def test_subtract_positive_numbers(self):
        """Test subtraction of two positive numbers."""
        assert subtract(10, 5) == 5
        assert subtract(100, 25) == 75
        assert subtract(1, 1) == 0
    
    def test_subtract_negative_numbers(self):
        """Test subtraction of two negative numbers."""
        assert subtract(-10, -5) == -5
        assert subtract(-5, -10) == 5
        assert subtract(-100, -100) == 0
    
    def test_subtract_mixed_sign_numbers(self):
        """Test subtraction with mixed positive and negative numbers."""
        assert subtract(10, -5) == 15
        assert subtract(-10, 5) == -15
        assert subtract(5, -10) == 15
        assert subtract(-5, 10) == -15
    
    def test_subtract_with_zero(self):
        """Test subtraction involving zero."""
        assert subtract(0, 0) == 0
        assert subtract(10, 0) == 10
        assert subtract(0, 10) == -10
        assert subtract(-10, 0) == -10
        assert subtract(0, -10) == 10
    
    def test_subtract_large_numbers(self):
        """Test subtraction with large integer values."""
        assert subtract(1000000, 500000) == 500000
        assert subtract(9999999, 1) == 9999998
        assert subtract(-1000000, -500000) == -500000
    
    def test_subtract_resulting_in_negative(self):
        """Test subtraction where result is negative."""
        assert subtract(5, 10) == -5
        assert subtract(0, 100) == -100
        assert subtract(-50, 50) == -100
    
    def test_subtract_commutative_property(self):
        """Verify that subtraction is not commutative."""
        assert subtract(10, 5) != subtract(5, 10)
        assert subtract(10, 5) == -(subtract(5, 10))
    
    def test_subtract_identity_property(self):
        """Test that subtracting zero returns the original number."""
        for num in [0, 1, -1, 100, -100]:
            assert subtract(num, 0) == num
    
    def test_subtract_inverse_property(self):
        """Test that subtracting a number from itself returns zero."""
        for num in [0, 1, -1, 100, -100, 999999]:
            assert subtract(num, num) == 0

import pytest
from src.app.math_utils import divide


class TestDivide:
    """Test suite for the divide function."""
    
    def test_divide_positive_numbers(self):
        """Test division with positive numbers."""
        assert divide(10, 2) == 5.0
        assert divide(15, 3) == 5.0
        assert divide(100, 4) == 25.0
    
    def test_divide_negative_numbers(self):
        """Test division with negative numbers."""
        assert divide(-10, 2) == -5.0
        assert divide(10, -2) == -5.0
        assert divide(-10, -2) == 5.0
    
    def test_divide_with_floats(self):
        """Test division with floating point numbers."""
        assert divide(7.5, 2.5) == 3.0
        assert divide(1.5, 0.5) == 3.0
        assert pytest.approx(divide(10, 3)) == 3.333333333333333
    
    def test_divide_by_one(self):
        """Test that dividing by 1 returns the original number."""
        assert divide(5, 1) == 5.0
        assert divide(-5, 1) == -5.0
        assert divide(0, 1) == 0.0
    
    def test_divide_zero_as_numerator(self):
        """Test that dividing zero by any non-zero number returns zero."""
        assert divide(0, 5) == 0.0
        assert divide(0, -5) == 0.0
        assert divide(0, 0.5) == 0.0
    
    def test_divide_by_negative_one(self):
        """Test that dividing by -1 negates the number."""
        assert divide(5, -1) == -5.0
        assert divide(-5, -1) == 5.0
    
    def test_divide_small_numbers(self):
        """Test division with very small numbers."""
        assert pytest.approx(divide(0.001, 0.01)) == 0.1
        assert pytest.approx(divide(1e-10, 1e-5)) == 1e-5
    
    def test_divide_large_numbers(self):
        """Test division with very large numbers."""
        assert divide(1e10, 1e5) == 1e5
        assert divide(1000000, 1000) == 1000.0
    
    def test_divide_by_zero_raises_value_error(self):
        """Test that dividing by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(5, 0)
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(-5, 0)
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(0, 0)
    
    def test_divide_by_zero_float_raises_value_error(self):
        """Test that dividing by 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(10, 0.0)
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(10, -0.0)
    
    def test_divide_result_is_float(self):
        """Test that the result is always a float type."""
        assert isinstance(divide(4, 2), float)
        assert isinstance(divide(5, 2), float)
        assert isinstance(divide(0, 1), float)

import pytest
from src.app.math_utils import subtract


def test_subtract_positive_numbers():
    """Test subtraction of two positive numbers."""
    assert subtract(10, 3) == 7


def test_subtract_negative_numbers():
    """Test subtraction of two negative numbers."""
    assert subtract(-5, -3) == -2


def test_subtract_mixed_signs():
    """Test subtraction with mixed positive and negative numbers."""
    assert subtract(5, -3) == 8
    assert subtract(-5, 3) == -8


def test_subtract_zero():
    """Test subtraction involving zero."""
    assert subtract(5, 0) == 5
    assert subtract(0, 5) == -5
    assert subtract(0, 0) == 0


def test_subtract_same_numbers():
    """Test subtraction of equal numbers results in zero."""
    assert subtract(7, 7) == 0
    assert subtract(-7, -7) == 0


def test_subtract_large_numbers():
    """Test subtraction with large numbers."""
    assert subtract(1000000, 999999) == 1
    assert subtract(10**9, 10**8) == 900000000


def test_subtract_result_negative():
    """Test subtraction where result is negative."""
    assert subtract(3, 10) == -7


@pytest.mark.parametrize("a,b,expected", [
    (100, 50, 50),
    (1, 1, 0),
    (-1, -1, 0),
    (0, -5, 5),
    (-10, 5, -15),
])
def test_subtract_parametrized(a, b, expected):
    """Parametrized test for various subtraction cases."""
    assert subtract(a, b) == expected

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

    def test_subtract_mixed_signs(self):
        """Test subtracting numbers with different signs."""
        assert subtract(5, -3) == 8
        assert subtract(-5, 3) == -8

    def test_subtract_zero(self):
        """Test subtracting zero from a number."""
        assert subtract(5, 0) == 5

    def test_subtract_from_zero(self):
        """Test subtracting a number from zero."""
        assert subtract(0, 5) == -5

    def test_subtract_both_zero(self):
        """Test subtracting zero from zero."""
        assert subtract(0, 0) == 0

    def test_subtract_same_numbers(self):
        """Test subtracting a number from itself."""
        assert subtract(42, 42) == 0

    def test_subtract_large_numbers(self):
        """Test subtracting large numbers."""
        assert subtract(1000000000, 999999999) == 1

    def test_subtract_result_negative(self):
        """Test subtraction that results in a negative number."""
        assert subtract(3, 10) == -7

    def test_subtract_large_negative_result(self):
        """Test subtraction with large negative result."""
        assert subtract(-1000000, 1000000) == -2000000


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (10, 5, 5),
        (5, 10, -5),
        (0, 0, 0),
        (-1, -1, 0),
        (100, 50, 50),
        (-10, 5, -15),
        (10, -5, 15),
    ],
)
def test_subtract_parametrized(a, b, expected):
    """Parametrized test for various subtract inputs."""
    assert subtract(a, b) == expected

import pytest
from src.app.math_utils import subtract


class TestSubtract:
    """Test cases for the subtract function."""

    def test_subtract_positive_numbers(self):
        """Test subtraction of two positive numbers."""
        assert subtract(10, 3) == 7

    def test_subtract_negative_numbers(self):
        """Test subtraction of two negative numbers."""
        assert subtract(-5, -3) == -2

    def test_subtract_mixed_sign_numbers(self):
        """Test subtraction with mixed positive and negative numbers."""
        assert subtract(5, -3) == 8
        assert subtract(-5, 3) == -8

    def test_subtract_zero_from_number(self):
        """Test subtracting zero from a number."""
        assert subtract(5, 0) == 5
        assert subtract(-5, 0) == -5

    def test_subtract_number_from_zero(self):
        """Test subtracting a number from zero."""
        assert subtract(0, 5) == -5
        assert subtract(0, -5) == 5

    def test_subtract_zero_from_zero(self):
        """Test subtracting zero from zero."""
        assert subtract(0, 0) == 0

    def test_subtract_same_numbers(self):
        """Test subtracting a number from itself."""
        assert subtract(7, 7) == 0
        assert subtract(-7, -7) == 0

    def test_subtract_large_numbers(self):
        """Test subtraction with large numbers."""
        assert subtract(1000000, 999999) == 1
        assert subtract(10**10, 10**9) == 9 * 10**9

    def test_subtract_result_negative(self):
        """Test subtraction where result is negative."""
        assert subtract(3, 10) == -7

    def test_subtract_result_positive(self):
        """Test subtraction where result is positive."""
        assert subtract(10, 3) == 7


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (10, 5, 5),
        (5, 10, -5),
        (0, 0, 0),
        (-10, -5, -5),
        (-5, -10, 5),
        (100, 1, 99),
        (1, 100, -99),
    ],
)
def test_subtract_parametrized(a: int, b: int, expected: int):
    """Parametrized test for various subtraction cases."""
    assert subtract(a, b) == expected

import pytest

from src.app.math_utils import divide


class TestDivide:
    """Tests for the divide function."""

    def test_divide_positive_numbers(self):
        """Test dividing two positive numbers."""
        assert divide(10, 2) == 5.0

    def test_divide_negative_numbers(self):
        """Test dividing two negative numbers."""
        assert divide(-10, -2) == 5.0

    def test_divide_positive_by_negative(self):
        """Test dividing a positive number by a negative number."""
        assert divide(10, -2) == -5.0

    def test_divide_negative_by_positive(self):
        """Test dividing a negative number by a positive number."""
        assert divide(-10, 2) == -5.0

    def test_divide_zero_by_number(self):
        """Test dividing zero by a non-zero number."""
        assert divide(0, 5) == 0.0

    def test_divide_by_zero_raises_value_error(self):
        """Test that dividing by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(10, 0)

    def test_divide_zero_by_zero_raises_value_error(self):
        """Test that dividing zero by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(0, 0)

    def test_divide_floats(self):
        """Test dividing float numbers."""
        assert divide(7.5, 2.5) == 3.0

    def test_divide_returns_float(self):
        """Test that divide returns a float type."""
        result = divide(10, 4)
        assert isinstance(result, float)
        assert result == 2.5

    def test_divide_large_numbers(self):
        """Test dividing large numbers."""
        assert divide(1e10, 1e5) == 1e5

    def test_divide_small_numbers(self):
        """Test dividing small numbers."""
        assert divide(1e-10, 1e-5) == pytest.approx(1e-5)

    def test_divide_result_precision(self):
        """Test division result with floating point precision."""
        result = divide(1, 3)
        assert result == pytest.approx(0.3333333333333333)

    def test_divide_by_one(self):
        """Test dividing by one returns the same number."""
        assert divide(42.5, 1) == 42.5

    def test_divide_by_negative_one(self):
        """Test dividing by negative one negates the number."""
        assert divide(42.5, -1) == -42.5

import pytest

from src.app.math_utils import divide


class TestDivide:
    """Tests for the divide function."""

    def test_divide_positive_numbers(self):
        """Test division of two positive numbers."""
        assert divide(10, 2) == 5.0

    def test_divide_negative_numbers(self):
        """Test division of two negative numbers."""
        assert divide(-10, -2) == 5.0

    def test_divide_positive_by_negative(self):
        """Test division of positive by negative number."""
        assert divide(10, -2) == -5.0

    def test_divide_negative_by_positive(self):
        """Test division of negative by positive number."""
        assert divide(-10, 2) == -5.0

    def test_divide_zero_by_number(self):
        """Test division of zero by a non-zero number."""
        assert divide(0, 5) == 0.0

    def test_divide_returns_float(self):
        """Test that divide returns a float type."""
        result = divide(10, 4)
        assert isinstance(result, float)

    def test_divide_with_floats(self):
        """Test division with float inputs."""
        assert divide(7.5, 2.5) == 3.0

    def test_divide_with_mixed_int_float(self):
        """Test division with mixed integer and float inputs."""
        assert divide(10, 2.5) == 4.0
        assert divide(7.5, 3) == 2.5

    def test_divide_small_numbers(self):
        """Test division with small numbers."""
        result = divide(0.001, 0.001)
        assert result == pytest.approx(1.0)

    def test_divide_large_numbers(self):
        """Test division with large numbers."""
        result = divide(1e10, 1e5)
        assert result == pytest.approx(1e5)

    def test_divide_by_zero_raises_value_error(self):
        """Test that dividing by zero raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            divide(10, 0)
        assert str(exc_info.value) == "Cannot divide by zero"

    def test_divide_zero_by_zero_raises_value_error(self):
        """Test that dividing zero by zero raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            divide(0, 0)
        assert str(exc_info.value) == "Cannot divide by zero"

    def test_divide_negative_by_zero_raises_value_error(self):
        """Test that dividing negative number by zero raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            divide(-10, 0)
        assert str(exc_info.value) == "Cannot divide by zero"

    def test_divide_float_by_zero_raises_value_error(self):
        """Test that dividing float by zero raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            divide(3.14, 0)
        assert str(exc_info.value) == "Cannot divide by zero"

    def test_divide_by_zero_float_raises_value_error(self):
        """Test that dividing by 0.0 (float zero) raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            divide(10, 0.0)
        assert str(exc_info.value) == "Cannot divide by zero"

    def test_divide_result_precision(self):
        """Test division result precision for non-terminating decimals."""
        result = divide(1, 3)
        assert result == pytest.approx(0.3333333333333333)

    def test_divide_very_small_divisor(self):
        """Test division by a very small number."""
        result = divide(1, 1e-10)
        assert result == pytest.approx(1e10)

    def test_divide_one_by_one(self):
        """Test division of one by one."""
        assert divide(1, 1) == 1.0

    def test_divide_same_numbers(self):
        """Test division of same numbers returns one."""
        assert divide(42, 42) == 1.0
        assert divide(-7, -7) == 1.0

import pytest

from src.app.math_utils import divide


class TestDivide:
    """Tests for the divide function."""

    def test_divide_positive_numbers(self):
        """Test dividing two positive numbers."""
        result = divide(10.0, 2.0)
        assert result == 5.0

    def test_divide_negative_numbers(self):
        """Test dividing two negative numbers."""
        result = divide(-10.0, -2.0)
        assert result == 5.0

    def test_divide_positive_by_negative(self):
        """Test dividing a positive number by a negative number."""
        result = divide(10.0, -2.0)
        assert result == -5.0

    def test_divide_negative_by_positive(self):
        """Test dividing a negative number by a positive number."""
        result = divide(-10.0, 2.0)
        assert result == -5.0

    def test_divide_zero_by_number(self):
        """Test dividing zero by a non-zero number."""
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

    def test_divide_with_floats(self):
        """Test dividing with floating point numbers."""
        result = divide(7.5, 2.5)
        assert result == 3.0

    def test_divide_with_integers_as_floats(self):
        """Test dividing integers cast as floats."""
        result = divide(9, 3)
        assert result == 3.0

    def test_divide_result_is_float(self):
        """Test that the result is always a float."""
        result = divide(10, 2)
        assert isinstance(result, float)

    def test_divide_small_numbers(self):
        """Test dividing very small numbers."""
        result = divide(0.0001, 0.01)
        assert pytest.approx(result, rel=1e-9) == 0.01

    def test_divide_large_numbers(self):
        """Test dividing large numbers."""
        result = divide(1e10, 1e5)
        assert result == 1e5

    def test_divide_by_one(self):
        """Test dividing by one returns the dividend."""
        result = divide(42.0, 1.0)
        assert result == 42.0

    def test_divide_by_negative_one(self):
        """Test dividing by negative one negates the dividend."""
        result = divide(42.0, -1.0)
        assert result == -42.0

    def test_divide_same_numbers(self):
        """Test dividing a number by itself returns one."""
        result = divide(7.0, 7.0)
        assert result == 1.0

    def test_divide_fractional_result(self):
        """Test division that results in a non-whole number."""
        result = divide(1.0, 3.0)
        assert pytest.approx(result, rel=1e-9) == 0.3333333333333333
