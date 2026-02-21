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
        assert add(-3, 5) == 2

    def test_add_with_zero(self):
        """Test adding with zero."""
        assert add(0, 5) == 5
        assert add(5, 0) == 5
        assert add(0, 0) == 0

    def test_add_large_numbers(self):
        """Test adding large integers."""
        assert add(1000000000, 2000000000) == 3000000000

    def test_add_commutative_property(self):
        """Test that addition is commutative (a + b == b + a)."""
        assert add(7, 3) == add(3, 7)

    def test_add_negative_result(self):
        """Test addition resulting in a negative number."""
        assert add(-10, 3) == -7


@pytest.mark.parametrize("a,b,expected", [
    (1, 1, 2),
    (0, 0, 0),
    (-1, 1, 0),
    (100, 200, 300),
    (-50, -50, -100),
])
def test_add_parametrized(a, b, expected):
    """Parametrized test for various input combinations."""
    assert add(a, b) == expected

import pytest

from src.app.math_utils import multiply


class TestMultiply:
    """Test suite for the multiply function."""

    def test_multiply_positive_numbers(self):
        """Test multiplication of two positive integers."""
        assert multiply(3, 4) == 12

    def test_multiply_negative_numbers(self):
        """Test multiplication of two negative integers."""
        assert multiply(-3, -4) == 12

    def test_multiply_mixed_sign_numbers(self):
        """Test multiplication of positive and negative integers."""
        assert multiply(-3, 4) == -12
        assert multiply(3, -4) == -12

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

    def test_multiply_commutative_property(self):
        """Test that multiplication is commutative (a * b == b * a)."""
        assert multiply(5, 3) == multiply(3, 5)
        assert multiply(-2, 7) == multiply(7, -2)

    def test_multiply_associative_with_sequential_calls(self):
        """Test associative-like behavior with sequential multiplications."""
        # (2 * 3) * 4 should equal 2 * (3 * 4)
        result1 = multiply(multiply(2, 3), 4)
        result2 = multiply(2, multiply(3, 4))
        assert result1 == result2 == 24


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (2, 3, 6),
        (0, 100, 0),
        (-5, -5, 25),
        (-10, 10, -100),
        (1, 999, 999),
        (100, 100, 10000),
    ],
)
def test_multiply_parametrized(a, b, expected):
    """Parametrized test for various multiplication scenarios."""
    assert multiply(a, b) == expected

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
    """Test subtracting a number from itself."""
    assert subtract(42, 42) == 0
    assert subtract(-10, -10) == 0


def test_subtract_large_numbers():
    """Test subtraction with large numbers."""
    assert subtract(1000000, 999999) == 1
    assert subtract(10**10, 10**9) == 9000000000


def test_subtract_result_negative():
    """Test subtraction resulting in a negative number."""
    assert subtract(3, 10) == -7


@pytest.mark.parametrize("a, b, expected", [
    (100, 50, 50),
    (1, 1, 0),
    (-1, -1, 0),
    (0, -5, 5),
    (-100, 50, -150),
])
def test_subtract_parametrized(a: int, b: int, expected: int):
    """Test subtraction with various input combinations."""
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
        """Test division of positive by negative number."""
        assert divide(10, -2) == -5.0

    def test_divide_negative_by_positive(self):
        """Test division of negative by positive number."""
        assert divide(-10, 2) == -5.0

    def test_divide_zero_by_number(self):
        """Test division of zero by a non-zero number."""
        assert divide(0, 5) == 0.0

    def test_divide_returns_float(self):
        """Test that division returns a float."""
        result = divide(10, 4)
        assert isinstance(result, float)
        assert result == 2.5

    def test_divide_with_floats(self):
        """Test division with float inputs."""
        assert divide(7.5, 2.5) == 3.0

    def test_divide_with_mixed_int_and_float(self):
        """Test division with mixed integer and float inputs."""
        assert divide(10, 2.5) == 4.0
        assert divide(7.5, 3) == 2.5

    def test_divide_small_numbers(self):
        """Test division with small numbers."""
        result = divide(0.001, 0.001)
        assert result == pytest.approx(1.0)

    def test_divide_large_numbers(self):
        """Test division with large numbers."""
        assert divide(1000000, 1000) == 1000.0

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

    def test_divide_by_zero_float_raises_value_error(self):
        """Test that dividing by zero as float raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            divide(10, 0.0)
        assert str(exc_info.value) == "Cannot divide by zero"

    def test_divide_result_precision(self):
        """Test division result precision for repeating decimals."""
        result = divide(1, 3)
        assert result == pytest.approx(0.3333333333333333)

    def test_divide_by_one(self):
        """Test division by one returns the dividend."""
        assert divide(42, 1) == 42.0
        assert divide(-42, 1) == -42.0

    def test_divide_by_negative_one(self):
        """Test division by negative one negates the dividend."""
        assert divide(42, -1) == -42.0
        assert divide(-42, -1) == 42.0
