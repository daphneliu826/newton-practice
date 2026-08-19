import numpy as np

def derivative(f, x, eps=1e-5):
    return (f(x + eps) - f(x)) / eps

def second_derivative(f, x, eps=1e-5):
    return derivative(lambda y: derivative(f, y, eps), x, eps)
    
def optimize(x0, f, tol=1e-6, max_iter=100):
    for iteration in range(max_iter):
        
        fp = derivative(f, x0)
        fpp = second_derivative(f, x0)

        if np.abs(fp)< tol:
            break
            
        x0 = x0 - fp / fpp
    return x0, iteration

