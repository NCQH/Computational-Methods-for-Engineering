import numpy as np

def _trapzoid_rule(f, a, b, n):
    '''
    Input:
        f: function
        a: left endpoint
        b: right endpoint
        n: number of subintervals
    Output:
        I: integral of f
    '''
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    y = f(x)
    I = h * (y[0] + y[-1]) / 2 + h * np.sum(y[1:-1])
    return I

def _simpson_rule13(f, a, b, n):
    '''
    Input:
        f: function
        a: left endpoint
        b: right endpoint
        n: number of subintervals
    Output:
        I: integral of f
    '''
    h = (b - a) / n
    x = np.linspace(a, b, n+1)
    for i in range(x):
        y[i] = f(x[i])
    I = h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]))
    return I

def _romberg_integration(f, a, b, tol=1e-6, max_iter=10000):
    '''
    Input:
        f: function
        a: left endpoint
        b: right endpoint
        tol: tolerance
        max_iter: maximum number of iterations
    Output:
        I: integral of f
    '''
    R = np.zeros(shape=(max_iter, max_iter))
    R[0, 0] = _trapzoid_rule(f, a, b, 1)
    for i in range(1, max_iter):
        R[i, 0] = _trapzoid_rule(f, a, b, 2 ** i)
        for j in range(1, i+1):
            R[i, j] = R[i, j-1] + (R[i, j-1] - R[i-1, j-1]) / (4 ** j - 1)
        if abs(R[i, i] - R[i-1, i-1]) < tol:
            return R[i, i]
    print("Error: Method failed after {} iterations".format(max_iter))
    return



def f(x):
    return np.exp(np.math.cos(x)) + x * x

print(_simpson_rule13(f, 0, 6, 20))