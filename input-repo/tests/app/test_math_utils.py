import pytest

from src.app.math_utils import add


class TestAdd:
    """Test suite for the add function."""

    def test_add_positive_numbers(self):
        """Test adding two positive integers."""
        assert add(2, 3) == 5

    def test_add_negative_numbers(self):
        """Test adding two negative integers."""
        assert add(-2, -3) == -5

    def test_add_positive_and_negative(self):
        """Test adding a positive and negative integer."""
        assert add(5, -3) == 2
        assert add(-5, 3) == -2

    def test_add_with_zero(self):
        """Test adding with zero."""
        assert add(0, 5) == 5
        assert add(5, 0) == 5
        assert add(0, 0) == 0

    def test_add_large_numbers(self):
        """Test adding large integers."""
        assert add(1000000, 2000000) == 3000000
        assert add(10**18, 10**18) == 2 * 10**18

    def test_add_large_negative_numbers(self):
        """Test adding large negative integers."""
        assert add(-1000000, -2000000) == -3000000

    def test_add_commutative_property(self):
        """Test that addition is commutative (a + b == b + a)."""
        assert add(3, 7) == add(7, 3)
        assert add(-5, 10) == add(10, -5)

    def test_add_identity_property(self):
        """Test that zero is the identity element for addition."""
        assert add(42, 0) == 42
        assert add(0, 42) == 42

    def test_add_returns_integer(self):
        """Test that add returns an integer type."""
        result = add(1, 2)
        assert isinstance(result, int)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1, 1, 2),
        (0, 0, 0),
        (-1, 1, 0),
        (100, 200, 300),
        (-100, -200, -300),
        (1, -1, 0),
    ],
)
def test_add_parametrized(a, b, expected):
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
        """Test multiplication by zero."""
        assert multiply(5, 0) == 0
        assert multiply(0, 5) == 0
        assert multiply(0, 0) == 0

    def test_multiply_by_one(self):
        """Test multiplication by one (identity)."""
        assert multiply(7, 1) == 7
        assert multiply(1, 7) == 7
        assert multiply(1, 1) == 1

    def test_multiply_by_negative_one(self):
        """Test multiplication by negative one."""
        assert multiply(5, -1) == -5
        assert multiply(-1, 5) == -5
        assert multiply(-1, -1) == 1

    def test_multiply_large_numbers(self):
        """Test multiplication of large integers."""
        assert multiply(1000000, 1000000) == 1000000000000

    def test_multiply_commutative_property(self):
        """Test that multiplication is commutative (a * b == b * a)."""
        assert multiply(7, 3) == multiply(3, 7)
        assert multiply(-5, 8) == multiply(8, -5)


@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 6),
    (0, 100, 0),
    (-2, -3, 6),
    (-2, 3, -6),
    (1, 999, 999),
    (10, 10, 100),
])
def test_multiply_parametrized(a, b, expected):
    """Parametrized test for various multiplication cases."""
    assert multiply(a, b) == expected

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

    def test_subtract_zero_from_zero(self):
        """Test subtracting zero from zero."""
        assert subtract(0, 0) == 0

    def test_subtract_same_numbers(self):
        """Test subtracting a number from itself."""
        assert subtract(42, 42) == 0

    def test_subtract_large_numbers(self):
        """Test subtracting large numbers."""
        assert subtract(1000000, 999999) == 1

    def test_subtract_result_is_negative(self):
        """Test subtraction that results in a negative number."""
        assert subtract(3, 10) == -7

    def test_subtract_large_positive_and_negative(self):
        """Test subtracting a large negative from a large positive."""
        assert subtract(1000000, -1000000) == 2000000


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (10, 5, 5),
        (5, 10, -5),
        (0, 0, 0),
        (-1, -1, 0),
        (100, 50, 50),
    ],
)
def test_subtract_parametrized(a, b, expected):
    """Test subtract with various input combinations."""
    assert subtract(a, b) == expected

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
        """Test division of positive number by negative number."""
        assert divide(10, -2) == -5.0

    def test_divide_negative_by_positive(self):
        """Test division of negative number by positive number."""
        assert divide(-10, 2) == -5.0

    def test_divide_zero_by_number(self):
        """Test division of zero by a non-zero number."""
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
        """Test division of floating point numbers."""
        assert divide(7.5, 2.5) == 3.0

    def test_divide_returns_float(self):
        """Test that divide returns a float type."""
        result = divide(10, 4)
        assert isinstance(result, float)
        assert result == 2.5

    def test_divide_large_numbers(self):
        """Test division of large numbers."""
        assert divide(1e10, 1e5) == 1e5

    def test_divide_small_numbers(self):
        """Test division of small numbers."""
        assert divide(1e-10, 1e-5) == pytest.approx(1e-5)

    def test_divide_result_is_fraction(self):
        """Test division that results in a non-integer fraction."""
        assert divide(1, 3) == pytest.approx(0.3333333333333333)

    def test_divide_by_one(self):
        """Test division by one returns the dividend."""
        assert divide(42, 1) == 42.0

    def test_divide_by_negative_one(self):
        """Test division by negative one negates the dividend."""
        assert divide(42, -1) == -42.0

    def test_divide_same_numbers(self):
        """Test division of a number by itself returns one."""
        assert divide(7, 7) == 1.0
