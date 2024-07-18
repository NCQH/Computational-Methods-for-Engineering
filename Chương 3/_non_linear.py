import numpy as np

def _bisection(f, a, b, tol=1e-6, max_iter=10000):
    '''
    Input:
        f: function
        a: left endpoint
        b: right endpoint
        tol: tolerance
        max_iter: maximum number of iterations
    Output:
        x: root of f
    '''
    if f(a) * f(b) > 0:
        print("Error: f(a) and f(b) must have different signs")
        return
    if f(a) == 0:
        return a
    if f(b) == 0:
        return b
    for i in range(max_iter):
        c = (a + b) / 2
        if f(c) == 0:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        if abs(f(c)) < tol:
            return c
    print("Error: Method failed after {} iterations".format(max_iter))
    return 

def _regula_falsi(f, a, b, tol=1e-6, max_iter=10000):
    '''
    Input:
        f: function
        a: left endpoint
        b: right endpoint
        tol: tolerance
        max_iter: maximum number of iterations
    Output:
        x: root of f
    '''
    if f(a) * f(b) > 0:
        print("Error: f(a) and f(b) must have different signs")
        return
    if f(a) == 0:
        return a
    if f(b) == 0:
        return b
    for i in range(max_iter):
        c = b - f(b) * (a - b) / (f(a) - f(b))
        if f(c) == 0:
            return c
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
        if abs(f(c)) < tol:
            return c
    print("Error: Method failed after {} iterations".format(max_iter))
    return


def _fixed_point(g, x0, tol=1e-6, max_iter=10000):
    '''
    Input:
        g: function
        x0: initial guess
        tol: tolerance
        max_iter: maximum number of iterations
    Output:
        x: root of f
    '''
    for i in range(max_iter):
        x = g(x0)
        if abs(x - x0) < tol:
            return x
        x0 = x
    print("Error: Method failed after {} iterations".format(max_iter))
    return 


def _newton_rahpson(f, x0, tol=1e-6, max_iter=10000):
    '''
    Input:
        f: function
        df: derivative of f
        x0: initial guess
        tol: tolerance
        max_iter: maximum number of iterations
    Output:
        x: root of f
    '''

    for i in range(max_iter):
        x = x0 - f(x0) / _df(f, x0, tol)
        if abs(x - x0) < tol:
            return x
        x0 = x
    print("Error: Method failed after {} iterations".format(max_iter))
    return

def _secant(f, x0, x1, tol=1e-6, max_iter=10000):
    '''
    Input:
        f: function
        x0: initial guess
        x1: initial guess
        tol: tolerance
        max_iter: maximum number of iterations
    Output:
        x: root of f
    '''
    for i in range(max_iter):
        x = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(x - x1) < tol:
            return x
        x0 = x1
        x1 = x
    print("Error: Method failed after {} iterations".format(max_iter))
    return

def _df(f, x, h = 1e-6):
    """
    Calculate the derivative of a function using the central difference method.

    f: callable function
    x: point at which to evaluate the derivative
    h: step size for the finite difference (optional, default is 1e-6)

    Returns the approximate derivative of the function at the given point.
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def _df(x1, x2, y1, y2):
    return (y2 - y1) / (x2 - x1)

def fx(x):
    return x*x +3*x -5

print(_bisection(fx, 1, 1.5, 0.01))