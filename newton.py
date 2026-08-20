import numpy as np


# Estimate the first and second derivatives.
def derivative(f, x, eps=1e-5):
    """Estimate the first derivative of a function using central differences."""
    return (f(x + eps) - f(x - eps)) / (2 * eps)


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

def gradient(f, x, eps=1e-5):
    """Estimate the gradient of a multivariate function."""
    grad = np.zeros_like(x, dtype=float)

    for i in range(len(x)):
        x_step = x.copy()
        x_step[i] += eps
        grad[i] = (f(x_step) - f(x)) / eps

    return grad


def hessian(f, x, eps=1e-5):
    """Estimate the Hessian matrix of a multivariate function."""
    n = len(x)
    hess = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            x_ij = x.copy()
            x_i = x.copy()
            x_j = x.copy()

            x_ij[i] += eps
            x_ij[j] += eps
            x_i[i] += eps
            x_j[j] += eps

            hess[i, j] = (
                f(x_ij) - f(x_i) - f(x_j) + f(x)
            ) / eps**2

    return hess


def optimize_multivariate(x0, f, tol=1e-6, max_iter=100):
    """Minimize a multivariate function using Newton's method."""
    x = np.array(x0, dtype=float)

    for iteration in range(max_iter):
        grad = gradient(f, x)
        hess = hessian(f, x)

        if np.linalg.norm(grad) < tol:
            return x, iteration

        step = np.linalg.solve(hess, grad)
        x_new = x - step

        if np.linalg.norm(x_new - x) < tol:
            return x_new, iteration + 1

        x = x_new

    return x, max_iter