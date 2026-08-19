import numpy as np
import pytest
import newton


def test_successful_optimization():
    def f(x):
        return (x - 3) ** 2

    result, iteration = newton.optimize(0, f)

    assert abs(result - 3) < 1e-4


def test_unsuccessful_optimization():
    def f(x):
        return x**4 / 4 - x**3 - x

    with pytest.raises(ValueError):
        newton.optimize(2, f)


def test_invalid_function_input():
    with pytest.raises(TypeError):
        newton.optimize(0, 5)