"""Auto-generated tests by AI Spec Generator."""

import pytest
from src.app.math_utils import subtract


class TestSubtract:
    """Test suite for the subtract function."""
    
    def test_subtract_positive_numbers(self):
        """Test subtraction with two positive numbers."""
        assert subtract(10, 3) == 7
        assert subtract(100, 25) == 75
        assert subtract(1, 1) == 0
    
    def test_subtract_negative_numbers(self):
        """Test subtraction with two negative numbers."""
        assert subtract(-5, -3) == -2
        assert subtract(-10, -20) == 10
        assert subtract(-1, -1) == 0
    
    def test_subtract_mixed_sign_numbers(self):
        """Test subtraction with positive and negative numbers."""
        assert subtract(5, -3) == 8
        assert subtract(-5, 3) == -8
        assert subtract(10, -10) == 20
        assert subtract(-10, 10) == -20
    
    def test_subtract_with_zero(self):
        """Test subtraction involving zero."""
        assert subtract(0, 0) == 0
        assert subtract(5, 0) == 5
        assert subtract(0, 5) == -5
        assert subtract(-5, 0) == -5
        assert subtract(0, -5) == 5
    
    def test_subtract_large_numbers(self):
        """Test subtraction with large integer values."""
        assert subtract(1000000, 500000) == 500000
        assert subtract(999999999, 1) == 999999998
        assert subtract(-1000000, -500000) == -500000
    
    def test_subtract_resulting_in_negative(self):
        """Test subtraction where result is negative."""
        assert subtract(3, 10) == -7
        assert subtract(0, 100) == -100
        assert subtract(-50, 50) == -100
    
    def test_subtract_same_numbers(self):
        """Test subtraction of identical numbers results in zero."""
        assert subtract(42, 42) == 0
        assert subtract(-17, -17) == 0
        assert subtract(0, 0) == 0
    
    def test_subtract_order_matters(self):
        """Test that order of operands affects the result."""
        assert subtract(10, 3) == 7
        assert subtract(3, 10) == -7
        assert subtract(10, 3) != subtract(3, 10)

import pytest
from src.app.math_utils import divide


class TestDivide:
    """Test suite for the divide function."""

    def test_divide_positive_numbers(self):
        """Test division of two positive numbers."""
        result = divide(10.0, 2.0)
        assert result == 5.0

    def test_divide_negative_dividend(self):
        """Test division with negative dividend."""
        result = divide(-10.0, 2.0)
        assert result == -5.0

    def test_divide_negative_divisor(self):
        """Test division with negative divisor."""
        result = divide(10.0, -2.0)
        assert result == -5.0

    def test_divide_both_negative(self):
        """Test division with both numbers negative."""
        result = divide(-10.0, -2.0)
        assert result == 5.0

    def test_divide_by_one(self):
        """Test that dividing by one returns the original number."""
        result = divide(42.0, 1.0)
        assert result == 42.0

    def test_divide_zero_dividend(self):
        """Test that zero divided by any non-zero number is zero."""
        result = divide(0.0, 5.0)
        assert result == 0.0

    def test_divide_decimal_numbers(self):
        """Test division with decimal numbers."""
        result = divide(7.5, 2.5)
        assert result == 3.0

    def test_divide_small_numbers(self):
        """Test division with small decimal numbers."""
        result = divide(0.1, 0.01)
        assert result == pytest.approx(10.0)

    def test_divide_large_numbers(self):
        """Test division with large numbers."""
        result = divide(1000000.0, 1000.0)
        assert result == 1000.0

    def test_divide_result_is_decimal(self):
        """Test division that results in a decimal number."""
        result = divide(10.0, 3.0)
        assert result == pytest.approx(3.33333333333333, rel=1e-9)

    def test_divide_by_zero_raises_value_error(self):
        """Test that dividing by zero raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            divide(10.0, 0.0)
        assert str(excinfo.value) == 'Cannot divide by zero'

    def test_divide_by_negative_zero_raises_value_error(self):
        """Test that dividing by negative zero raises ValueError."""
        with pytest.raises(ValueError) as excinfo:
            divide(10.0, -0.0)
        assert str(excinfo.value) == 'Cannot divide by zero'

    def test_divide_one_by_number(self):
        """Test reciprocal calculation (1 divided by a number)."""
        result = divide(1.0, 4.0)
        assert result == 0.25

    def test_divide_integer_inputs_as_floats(self):
        """Test division with integer values passed as floats."""
        result = divide(15.0, 3.0)
        assert result == 5.0
        assert isinstance(result, float)
