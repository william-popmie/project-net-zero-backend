"""Optimized functions from src/app/math_utils.py."""

def add(a: int, b: int) -> int:
    return a + b

def subtract(a: int, b: int) -> int:
    """Subtract b from a."""
    return a - b

def divide(a: float, b: float) -> float:
    if b == 0:
        raise ValueError('Cannot divide by zero')
    return a / b
