import numpy as np
import _linear
import math

'''Power method'''


def _direct_power_method(A, x0, tol=1e-6, max_iter=10000,  no_scale=1, print_mt=False):
    """
    Computes the dominant eigenvalue and eigenvector of a square matrix A using the direct power method.

    Inputs:
    - A: a square matrix
    - x0: an initial guess for the eigenvector
    - tol: convergence tolerance (default 1e-6)
    - max_iter: maximum number of iterations (default 10000)
    - no_scale: index of the component of x that should be scaled (default 1)
    - print_mt: whether to print intermediate results (default False)

    Outputs:
    - lambda_new: the dominant eigenvalue of A
    - x: the corresponding eigenvector
    """
    if print_mt:
        print("---Direct power method---")
    A = np.array(A, dtype=float)
    x = np.array(x0, dtype=float)

    if print_mt:
        print("A = \n", A)
        print("x0 = \n", x0)

    lambda_old = 0

    if print_mt:
        header = ["k", "lambda", "x1", "x2", "x3"]
        print("{:<10} {:<20}".format(*header), end='')
        for i in range(x.shape[0]):
            print(" {:<20}".format("x" + str(i+1)), end='')
        print()

    for i in range(max_iter):
        if print_mt:
            print("{:<10} {:<20}".format(
                i, str(lambda_old)), end='')
            for i in range (x.shape[0]):
                print(" {:<20f}".format(float(x[i])), end='')
            print()
        y = A @ x
        lambda_new = y[no_scale-1]
        x = y / lambda_new
        if abs(lambda_new - lambda_old) < tol:
            break
        lambda_old = lambda_new
    if i == max_iter - 1:
        print('Error: Maximum iteration reached')
        return
    return lambda_new, x


'''Inverse power method'''


def _inverse_power_method(A, x0, tol=1e-6, max_iter=10000, no_scale=1, print_mt=False):
    """
    Computes the eigenvalue closest to a given value lambda using the inverse power method.

    Inputs:
    - A: a square matrix
    - x0: an initial guess for the eigenvector
    - tol: convergence tolerance (default 1e-6)
    - max_iter: maximum number of iterations (default 10000)
    - no_scale: index of the component of x that should be scaled (default 1)
    - print_mt: whether to print intermediate results (default False)

    Outputs:
    - lambda_new: the eigenvalue closest to lambda
    - x: the eigen vector corresponding to lambda_new
    """
    if print_mt:
        print("---The Inverse Power Method---")
    A = np.array(A, dtype=float)
    x = np.array(x, dtype=float)

    L, U = _linear._lu_factorization(A)

    if print_mt:
        print("A = \n", A)
        print("L = \n", L)
        print("U = \n", U)
        print("x0 = \n", x0)

    lambda_old = 0

    if print_mt:
        header = ["k", "lambda_inverse", "x1", "x2", "x3"]
        print("{:<10} {:<20}".format(*header), end='')
        for i in range(x.shape[0]):
            print(" {:<20f}".format("x" + str(i+1)), end='')
        print()

    for i in range(max_iter):
        if print_mt:
            print("{:<10} {:<20}".format(
                i, str(lambda_old)), end='')
            for i in range (x.shape[0]):
                print(" {:<20f}".format(float(x[i])), end='')
            print()

        y = _linear._solve_lu_factorization(L, U, x)

        lambda_new = y[no_scale-1]
        x = y / lambda_new
        if abs(lambda_new - lambda_old) < tol:
            break
        lambda_old = lambda_new

    if i == max_iter - 1:
        print('Error: Maximum iteration reached')
        return
    
    return 1/lambda_new, x


'''Shifted'''


def _shifted_power_method(A, lambda_, x0, tol=1e-6, max_iter=10000, no_scale=1, print_mt=False):
    """
    Computes the eigenvalue closest to a given value lambda using the shifted power method.

    Inputs:
    - A: a square matrix
    - x0: an initial guess for the eigenvector
    - tol: convergence tolerance (default 1e-6)
    - max_iter: maximum number of iterations (default 10000)
    - no_scale: index of the component of x that should be scaled (default 1)
    - print_mt: whether to print intermediate results (default False)

    Outputs:
    - lambda_new: the eigenvalue closest to lambda
    - x: the eigen vector corresponding to lambda_new
    """
    if print_mt:
        print("---The Shifted Power Moethod---")
    A = np.array(A, dtype=float)
    x = np.array(x0, dtype=float)
    As = A - np.eye(A.shape[0]) * lambda_

    if print_mt:
        print("A = \n", A)
        print("Ashifted = \n", As)
        print("x = \n", x)

    lambda_old = 0


    if print_mt:
        header = ["k", "lambda shifted"]
        print("{:<10} {:<20}".format(*header), end='')
        for i in range(x.shape[0]):
            print(" {:<20f}".format("x" + str(i+1)), end='')
        print()

    for i in range(max_iter):
        if print_mt:
            print("{:<10} {:<20}".format(
                i, str(lambda_old)), end='')
            for i in range (x.shape[0]):
                print(" {:<20f}".format(float(x[i])), end='')
            print()
        y = As @ x
        lambda_new = y[no_scale-1]
        x = y / lambda_new
        if abs(lambda_new - lambda_old) < tol:
            break
        lambda_old = lambda_new

    if i == max_iter - 1:
        print('Error: Maximum iteration reached')
        return
    
    lambda_new = lambda_ + lambda_new
    if print_mt:
        print("lambda = ", lambda_new)
        print("x = \n", x)
    return lambda_new , x

def _qr_method(A, tol=1e-6, max_iter=80, print_mt=False):
    """
    Computes the eigenvalues and eigenvectors of a matrix using the QR method.

    Inputs:
    - A: a square matrix
    - tol: convergence tolerance (default 1e-6)
    - max_iter: maximum number of iterations (default 10000)
    - print_mt: whether to print intermediate results (default False)

    Outputs:
    - lambda_new: the eigenvalue closest to lambda
    - x: the eigen vector corresponding to lambda_new
    """
    if print_mt:
        print("---The QR Method---")
    A = np.array(A, dtype=float)
    
    _a = np.zeros((1, A.shape[1]), dtype=float)
    Q = np.zeros((A.shape[0], A.shape[1]), dtype=float)
    R = np.zeros((A.shape[0], A.shape[1]), dtype=float)

    if print_mt:
        print("A = \n", A)

    for i in range(max_iter):
        if print_mt:
            print("Iteration ", i)
            
        _A = np.copy(A)
        _a[:, 0] = _square_root_sum(A[:, 0])
        Q[:, 0] = A[:, 0] / _a[:, 0]
        for j in range(1, A.shape[1]):
            for k in range(j):
                _A[:, j] = _A[:, j] - Q[:, k].T @ A[:, j] * Q[:, k]
            _a[:, j] = _square_root_sum(_A[:, j])
            Q[:, j] = _A[:, j] / _a[:, j]
        for j in range(A.shape[0]):
            for k in range(j, A.shape[1]):
                if j == k:
                    R[j, k] = _a[:, j]
                elif j < k:
                    R[j, k] = Q[:, j].T @ A[:, k]
        A = R @ Q
        if print_mt:
            print("Q = \n", Q)
            print("R = \n", R)
            print("A = \n", A)
            
            
        near_triangle = True

        for j in range(A.shape[0] - 1):
            if abs(A[j , j+1]) > tol:
                near_triangle = False

        if near_triangle:
            break

   
    if i == max_iter - 1:
        print('Error: Maximum iteration reached')
        return
    return 

def _square_root_sum(A):
    return np.sqrt(np.sum(A**2))