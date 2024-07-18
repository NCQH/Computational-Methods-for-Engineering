import numpy as np
import decimal

''' Gauss Elimination'''
# without pivoting


def _gauss_elimination(A, print_mt=False):
    '''
    Input:
        A: 2D array
        print_mt: print matrix
    Output:
        A: 2D array after elimination by Gauss method
    '''
    if print_mt:
        print("---Gauss elimination---")
    A = np.array(A, dtype=decimal.Decimal)
    col = A.shape[1]
    row = A.shape[0]

    # Check for zero diagonal elements
    for i in range(row):
        if A[i,i] == 0:
            for j in range(i+1, row):
                if A[j,i] != 0:
                    A[[i,j]] = A[[j,i]]  # swap rows i and j
                    break
            else:
                print('Error: Diagonal element is zero')
                return
            
    for i in range(row):
        for j in range(i+1, row):
            if A[i, i] == 0:
                print('Error: Elimination failed')
                return 
            else:
                # elimination multiplier
                em = A[j, i]/A[i, i]
                for k in range(i, col):
                    A[j, k] = A[j, k] - em * A[i, k]
    if print_mt:
        print("A = \n", A)
    return A


'''Calculate Derteminant'''


def _combine(A, B):
    '''
    Input:
        A: 2D array
        B: 2D array same number of rows as A
    Output:
        C: 2D array with A and B combined
    '''
    if A.shape[1] != B.shape[0]:
        print("A and B must have compatible dimensions")
        return
    return np.concatenate((A, B), axis=1)


def _det_cofactor_method(A):
    '''
    Input:
        A: 2D array
    Output:
        det: determinant of A
    '''
    n = len(A)
    if n == 1:
        return A[0, 0]
    elif n == 2:
        return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    elif n > 2:
        if A.shape[0] != A.shape[1]:
            return ("Matrix must be square")
        det = 0
        for i in range(n):
            cofactor = (-1)**(0+i)*_det_cofactor_method(_get_minor(A, 0, i))
            det += A[0][i] * cofactor
        return det
    else:
        return ("Matrix must be at least 2x2")


def _get_minor(A, i, j):
    '''
    Input:
        A: 2D array
        i: row index
        j: column index
    Output:
        minor: minor of A at (i, j)
    '''
    if A.shape[0] < 2 or A.shape[1] < 2:
        return ("Matrix must be at least 2x2")
    minor = np.delete(np.delete(A, i, 0), j, 1)
    return minor


'''Solve Equations'''


def _solve_gauss_elimination(A, B, print_mt=False):
    '''
    Input:
        A: 2D array
        B: 2D array same number of rows as A
        print_mt: print matrix
    Output:
        X: 2D array solution of AX = B
    '''
    A = np.array(A, dtype=decimal.Decimal)
    B = np.array(B, dtype=decimal.Decimal)

    # Check if A and B have compatible dimensions
    
    A = _combine(A, B)
    if print_mt:
        print("---Solve equation using Gauss elimination---")
        print("A = \n", A)
        print("B = \n", B)
        print("A|B = \n", A)

    row = A.shape[0]
    col = A.shape[1]

    A = _gauss_elimination(A, print_mt=print_mt)
    X = np.zeros(shape=(A.shape[0], B.shape[1]), dtype=decimal.Decimal)

    # Back substitution
    for k in range(X.shape[1]):
        for i in range(row-1, -1, -1):
            X[i, k] = A[i, col-1]
            for j in range(i+1, row):
                X[i, k] = X[i, k] - A[i, j] * X[j, k]
            X[i, k] = X[i, k]/A[i, i]

    if print_mt:
        print("Solution (Back subtituition): \n", X)
    return X


def _solve_cramer_rule(A, B, print_mt=False):
    '''
    Input:
        A: 2D array
        B: 2D array same number of rows as A
        print_mt: print matrix
    Output:
        X: 2D array solution of AX = B
    '''
    if print_mt:
        print("---Solve equation using Cramer---")
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    # Check if A and B have compatible dimensions
    if A.shape[0] != A.shape[1]:
        return ("A must be square")
    if A.shape[1] != B.shape[0]:
        return ("A and B must have compatible dimensions")
    
    det_A = _det_cofactor_method(A)
    if det_A == 0:
        return ("A must be invertible")
    else:
        X = np.zeros(shape=(A.shape[1], B.shape[1]), dtype=float)
        for i in range(A.shape[1]):
            A_temp = A.copy()
            A_temp[:, i] = B[:, 0]
            X[i] = _det_cofactor_method(A_temp)/det_A
        if print_mt:
            print("Solution (Cramer): \n", X)
    return X


def _solve_inverse_method(A_I, B, print_mt=False):
    '''
    Input:
        A_I: 2D array inverse of A
        B: 2D array same number of rows as A
        print_mt: print matrix
    Output:
        X: 2D array solution of AX = B
    '''
    if print_mt:
        print("---Solve equation using inverse matrix---")
    A_I = np.array(A_I, dtype=float)
    B = np.array(B, dtype=float)

    if A_I.shape[0] != A_I.shape[1]:
        return "Error: A is not a square matrix"
    
    print("A^-1 = \n", A_I)
    print("B = \n", B)
    X = np.dot(A_I, B)

    if print_mt:
        print("Solution (A^-1 * B): \n", X)
    return X


def _solve_gauss_jordan(A, B, print_mt=False):
    '''
    Input:
        A: 2D array
        B: 2D array same number of rows as A
        print_mt: print matrix
    Output:
        X: 2D array solution of AX = B
    '''
    if print_mt:
        print("---Solve equation using Gauss-Jordan---")
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)

    if print_mt:
        print("A = \n", A)
        print("B = \n", B)
    A = _combine(A, B)

    if print_mt:
        print("A|B = \n", A)
    col = A.shape[1]
    row = A.shape[0]

    for i in range(row):
        if A[i, i] == 0:
            print('Error: Elimination failed')
            return
        A[i] = A[i]/A[i, i]
        for j in range(i+1, row):
            # elimination multiplier - Nhân tử khử
            em = A[j, i]/A[i, i]
            for k in range(i, col):
                A[j, k] = A[j, k] - em * A[i, k]
        for j in range(row):
            if j != i:
                A[j, :] = A[j, :] - A[j, i] * A[i, :]

    sol = A[:, row:]
    if print_mt:
        print("A|B after elimination = \n", A)
        print("Solution (Columns): \n", sol)
    return sol


def _solve_lu_factorization(L, U, B, print_mt=False):
    '''
    Input:
        L: 2D array lower triangular matrix
        U: 2D array upper triangular matrix
        B: 2D array same number of rows as A
        print_mt: print matrix
    Output:
        X: 2D array solution of AX = B
    '''
    if print_mt:
        print("---Solve equation using LU factorization---")

    L = np.array(L, dtype=float)
    U = np.array(U, dtype=float)
    B = np.array(B, dtype=float)

    if print_mt:
        print("L = \n", L)
        print("U = \n", U)
        print("B = \n", B)

    # forward substitution
    B = _solve_gauss_elimination(L, B)
    if print_mt:
        print("B' = \n", B)

    # backward substitution
    X = _solve_gauss_elimination(U, B)
    if print_mt:
        print("Solution: \n", X)
    return X


def _lu_factorization(A, print_mt=False):
    '''
    Input:
        A: 2D array
        print_mt: print matrix
    Output:
        L: 2D array lower triangular matrix
        U: 2D array upper triangular matrix
    '''
    A = np.array(A, dtype=float)
    if print_mt:
        print("A = \n", A)

    row = A.shape[0]
    col = A.shape[1]
    L = np.zeros(shape=(row, col), dtype=float)
    U = np.zeros(shape=(row, col), dtype=float)

    for i in range(row):
        for j in range(i+1, row):
            if A[i, i] == 0:
                return 'Error: Elimination failed'
            else:
                # elimination multiplier
                em = A[j, i]/A[i, i]
                for k in range(i, col):
                    A[j, k] = A[j, k] - em * A[i, k]
                # store elimination multiplier - L matrix
                L[j, i] = em
        
        L[i, i] = 1

        for j in range(i, col):
            U[i, j] = A[i, j]
    if print_mt:
        print("L = \n", L)
        print("U = \n", U)
        
    return L, U

_solve_gauss_elimination([[2.6, -4.5, -2], [3, 3, 4.3], [-6, 3.5, 3]], [[19.07], [3.21], [-18.25]], print_mt=True)