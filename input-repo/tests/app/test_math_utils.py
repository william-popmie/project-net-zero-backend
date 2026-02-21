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

    def test_add_mixed_sign_numbers(self):
        """Test adding a positive and negative integer."""
        assert add(5, -3) == 2
        assert add(-5, 3) == -2

    def test_add_zero(self):
        """Test adding zero to a number."""
        assert add(0, 5) == 5
        assert add(5, 0) == 5
        assert add(0, 0) == 0

    def test_add_large_numbers(self):
        """Test adding large integers."""
        assert add(1000000, 2000000) == 3000000
        assert add(10**18, 10**18) == 2 * 10**18

    def test_add_commutativity(self):
        """Test that addition is commutative (a + b == b + a)."""
        assert add(3, 7) == add(7, 3)
        assert add(-5, 10) == add(10, -5)

    def test_add_identity_property(self):
        """Test the identity property of addition (a + 0 == a)."""
        assert add(42, 0) == 42
        assert add(0, 42) == 42

    def test_add_negative_result(self):
        """Test addition that results in a negative number."""
        assert add(-10, 3) == -7
        assert add(3, -10) == -7


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (1, 1, 2),
        (0, 0, 0),
        (-1, 1, 0),
        (100, 200, 300),
        (-50, -50, -100),
        (999, 1, 1000),
    ],
)
def test_add_parametrized(a, b, expected):
    """Parametrized test for various add inputs."""
    assert add(a, b) == expected

import pytest

from src.app.math_utils import multiply


class TestMultiply:
    """Test suite for the multiply function."""

    def test_multiply_positive_integers(self):
        """Test multiplication of two positive integers."""
        assert multiply(3, 4) == 12

    def test_multiply_negative_integers(self):
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
        assert multiply(7, -1) == -7
        assert multiply(-1, 7) == -7
        assert multiply(-1, -1) == 1

    def test_multiply_large_numbers(self):
        """Test multiplication of large integers."""
        assert multiply(1000000, 1000000) == 1000000000000

    def test_multiply_very_large_numbers(self):
        """Test multiplication of very large integers (Python handles arbitrary precision)."""
        large_num = 10**100
        assert multiply(large_num, 2) == 2 * 10**100

    def test_multiply_commutative_property(self):
        """Test that multiplication is commutative (a * b == b * a)."""
        assert multiply(5, 7) == multiply(7, 5)
        assert multiply(-3, 8) == multiply(8, -3)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (2, 3, 6),
        (0, 100, 0),
        (-5, 5, -25),
        (-10, -10, 100),
        (1, 999, 999),
        (100, 100, 10000),
    ],
)
def test_multiply_parametrized(a, b, expected):
    """Parametrized test for various multiplication cases."""
    assert multiply(a, b) == expected

import pytest
from src.app.math_utils import subtract


def test_subtract_positive_numbers():
    """Test subtracting two positive numbers."""
    assert subtract(10, 3) == 7


def test_subtract_negative_numbers():
    """Test subtracting two negative numbers."""
    assert subtract(-5, -3) == -2


def test_subtract_positive_from_negative():
    """Test subtracting a positive number from a negative number."""
    assert subtract(-5, 3) == -8


def test_subtract_negative_from_positive():
    """Test subtracting a negative number from a positive number."""
    assert subtract(5, -3) == 8


def test_subtract_zero_from_number():
    """Test subtracting zero from a number."""
    assert subtract(5, 0) == 5


def test_subtract_number_from_zero():
    """Test subtracting a number from zero."""
    assert subtract(0, 5) == -5


def test_subtract_zero_from_zero():
    """Test subtracting zero from zero."""
    assert subtract(0, 0) == 0


def test_subtract_same_numbers():
    """Test subtracting a number from itself."""
    assert subtract(42, 42) == 0


def test_subtract_large_numbers():
    """Test subtracting large numbers."""
    assert subtract(1000000, 999999) == 1


def test_subtract_result_is_negative():
    """Test subtraction resulting in a negative number."""
    assert subtract(3, 10) == -7


@pytest.mark.parametrize("a, b, expected", [
    (100, 50, 50),
    (50, 100, -50),
    (-100, -50, -50),
    (-50, -100, 50),
    (0, 0, 0),
])
def test_subtract_parametrized(a, b, expected):
    """Test subtract with various input combinations."""
    assert subtract(a, b) == expected

import pytest

from src.app.math_utils import divide


class TestDivide:
    """Tests for the divide function."""

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

    def test_divide_zero_by_positive(self):
        """Test division of zero by a positive number."""
        result = divide(0.0, 5.0)
        assert result == 0.0

    def test_divide_zero_by_negative(self):
        """Test division of zero by a negative number."""
        result = divide(0.0, -5.0)
        assert result == 0.0

    def test_divide_by_zero_raises_value_error(self):
        """Test that division by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(10.0, 0.0)

    def test_divide_zero_by_zero_raises_value_error(self):
        """Test that dividing zero by zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(0.0, 0.0)

    def test_divide_with_floats(self):
        """Test division with floating point numbers."""
        result = divide(7.5, 2.5)
        assert result == 3.0

    def test_divide_result_is_float(self):
        """Test that division result is always a float."""
        result = divide(10.0, 4.0)
        assert result == 2.5
        assert isinstance(result, float)

    def test_divide_small_numbers(self):
        """Test division with small floating point numbers."""
        result = divide(0.001, 0.001)
        assert result == pytest.approx(1.0)

    def test_divide_large_numbers(self):
        """Test division with large numbers."""
        result = divide(1e10, 1e5)
        assert result == pytest.approx(1e5)

    def test_divide_with_integers_as_floats(self):
        """Test division when integers are passed (type coercion)."""
        result = divide(10, 3)
        assert result == pytest.approx(3.333333333333333)

    def test_divide_one_by_number(self):
        """Test division of 1 by a number (reciprocal)."""
        result = divide(1.0, 4.0)
        assert result == 0.25

    def test_divide_number_by_one(self):
        """Test division by 1 returns the original number."""
        result = divide(42.5, 1.0)
        assert result == 42.5

    def test_divide_same_numbers(self):
        """Test division of a number by itself returns 1."""
        result = divide(7.0, 7.0)
        assert result == 1.0

    def test_divide_by_negative_zero_raises_value_error(self):
        """Test that division by negative zero raises ValueError."""
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            divide(10.0, -0.0)
