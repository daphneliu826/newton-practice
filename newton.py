import numpy as np


# Estimate the first and second derivatives.
def derivative(f, x, eps=1e-5):
    """Estimate the first derivative of a function using finite differences."""
    return (f(x + eps) - f(x)) / eps


def second_derivative(f, x, eps=1e-5):
    """Estimate the second derivative of a function using finite differences."""
    return derivative(lambda y: derivative(f, y, eps), x, eps)


def optimize(x0, f, tol=1e-6, max_iter=100):
    """Minimize a univariate function using Newton's method.

    Parameters
    ----------
    x0 : float
        Starting value for the optimization.
    f : function
        Function to minimize.
    tol : float
        Tolerance for the stopping criterion.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    x0 : float
        Estimated location of the minimum.
    iteration : int
        Number of iterations performed.
    """
    if not callable(f):
        raise TypeError(f"Argument is not a function, it is of type {type(f)}")
    
    if x0 > 1e7:
        raise RuntimeError(f"At iteration {iter}, optimization appears to be diverging")

    for iteration in range(max_iter):
        fp = derivative(f, x0)
        fpp = second_derivative(f, x0)

        if np.abs(fp) < tol:
            break

        if np.abs(fpp) < 1e-10:
            raise ValueError(
                "Second derivative is too close to zero."
            )

        x0 = x0 - fp / fpp
    return x0, iteration
