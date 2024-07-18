import numpy as np
import _non_linear as non_linear


def _explicit_euler_method(f, y0, a, b, h):
    '''
    Input:
        f: function
        y0: initial value
        a: left endpoint
        b: right endpoint
        h: step size
    Output:
        y: array of y values
    '''
    n = int((b - a) / h)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        y[i + 1] = y[i] + h * f(a + i * h, y[i])
    return y

def _implicit_euler_method(f, y0, a, b, h, tol=1e-6, max_iter=10000):
    '''
    Input:
        f: function
        y0: initial value
        a: left endpoint
        b: right endpoint
        h: step size
        tol: tolerance
        max_iter: maximum number of iterations
    Output:
        y: array of y values
    '''
    n = int((b - a) / h)
    y = np.zeros(n + 1)
    y[0] = y0
    for i in range(n):
        y[i + 1] = non_linear._newton_rahpson(lambda y: y - y0 - h * f(a + (i + 1) * h, y), y[i], y[i] + 2 * h * f(a + (i + 1) * h, y[i]), tol, max_iter)
    return y

def runge_kutta_2(f, x0, y0, h, n = 100):
    """
    Solve the initial value problem using the second-order Runge-Kutta method.

    f: callable function f(x, y) that defines the ODE dy/dx
    x0: initial x value
    y0: initial y value
    h: step size
    n: number of steps to take

    Returns two arrays containing the x and y values.
    """
    x_values = [x0]
    y_values = [y0]
    
    for _ in range(n):
        x = x_values[-1]
        y = y_values[-1]
        k1 = h * f(x, y)
        k2 = h * f(x + h, y + k1)
        x_values.append(x + h)
        y_values.append(y + 1/2*(k1 + k2))
    
    return x_values, y_values

def runge_kutta_4(f, x0, y0, h, n):
    """
    Solve the initial value problem using the fourth-order Runge-Kutta method.

    f: callable function f(x, y) that defines the ODE dy/dx
    x0: initial x value
    y0: initial y value
    h: step size
    n: number of steps to take

    Returns two arrays containing the x and y values.
    """
    x_values = [x0]
    y_values = [y0]
    
    for _ in range(n):
        x = x_values[-1]
        y = y_values[-1]
        k1 = h * f(x, y)
        k2 = h * f(x + h/2, y + k1/2)
        k3 = h * f(x + h/2, y + k2/2)
        k4 = h * f(x + h, y + k3)
        x_values.append(x + h)
        y_values.append(y + (k1 + 2*k2 + 2*k3 + k4)/6)
    
    return x_values, y_values


def _adams_bashforth_():
    pass

def _adams_moulton_():
    pass

def _predictor_corrector_():
    pass

