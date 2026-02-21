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
