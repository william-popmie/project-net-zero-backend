from src.app.math_utils import add, multiply


def test_add_returns_sum() -> None:
    assert add(2, 3) == 5


def test_multiply_returns_product() -> None:
    assert multiply(4, 3) == 12
