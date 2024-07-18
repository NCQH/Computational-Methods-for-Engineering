import _polynomial_approx_interpolation as poly
import _non_linear as nlinear
import numpy as np


def _diff_direct_fit_poly(x, y, x0, n = 1):
    '''
    Returns the nth derivative of the direct fit polynomial at x0
    '''
    f = poly._direct_fit_polynomial(x, y)
    d = nlinear._df(f, x0)
        
    return d

def _diff_lagrange_poly(x, y, x0, n = 1):
    '''
    Returns the derivative of the lagrange polynomial at x0
    '''
    f = poly._lagrange_polynomials(x, y)
    d = nlinear._df(f, x0)

    return d

def _diff_divided_poly():
    pass

def _diff_newton_forward_poly():
    '''
    Returns the derivative of the newton forward polynimial at x0
    '''
    pass

