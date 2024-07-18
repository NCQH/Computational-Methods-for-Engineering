import numpy as np


'''The Jacobi Iteration Method'''
def _jacobi_iteration(A, B, x0, tol=1e-6, max_iter=10000):
    '''
    Input:
        A: matrix A
        B: matrix B
        x0: initial value
        tol: tolerance
        max_iter: maximum iteration
    Output:
        x: solution
    '''
    print("---Lặp Jacobi---")
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    x0 = np.array(x0, dtype=float)
    x = np.copy(x0)

    if x0.shape[1] != B.shape[1]:
        return 'Error: x0 size error'
    
    for k in range(max_iter):
        print("Lần lặp thứ", k)
        print(x0)
        for i in range(A.shape[0]):
            R = B[i, 0]
            for j in range(x0.shape[0]):
                R = R - A[i, j] * x0[j]
            x[i] = x0[i] + R / A[i, i]
        if max(abs(x - x0)) <= tol:
            break
        x0 = np.copy(x)

    return x


'''The Gauss-Seidel Iteration Method'''
def _gauss_seidel_iteration(A, B, x0, tol=1e-6, max_iter=10000):
    '''
    Input:
        A: matrix A
        B: matrix B
        x0: initial value
        tol: tolerance
        max_iter: maximum iteration
    Output:
        x: solution
    '''
    print("--Gauss Seidel iteration, e=", tol, "---")
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    x0 = np.array(x0, dtype=float)
    x = np.copy(x0)

    if x0.shape[1] != B.shape[1]:
        return 'Error: x0 size error'
    
    for k in range(max_iter):
        print("Lần lặp thứ", k)
        print(x0)
        for i in range(A.shape[0]):
            R = B[i, 0]
            for j in range(x0.shape[0]):
                if j < i:
                    R = R - A[i, j] * x[j]
                else:
                    R = R - A[i, j] * x0[j]
            x[i] = x0[i] + R / A[i, i]
        if max(abs(x - x0)) <= tol:
            break
        x0 = np.copy(x)

    return x

'''The SOR Iteration Method'''
def _sor_iteration(A, B, x0, w, tol=1e-6, max_iter=10000):
    '''
    Input:
        A: matrix A
        B: matrix B
        x0: initial value
        w: relaxation factor
        tol: tolerance
        max_iter: maximum iteration
    Output:
        x: solution
    '''
    print("---SOR iteration, w=", w, "e=", tol, "---")
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    x0 = np.array(x0, dtype=float)
    x = np.copy(x0)

    if x0.shape[1] != B.shape[1]:
        return 'Error: x0 size error'
    
    for k in range(max_iter):
        print("Lần lặp thứ", k)
        print(x0)
        for i in range(A.shape[0]):
            R = B[i, 0]
            for j in range(x0.shape[0]):
                if j < i:
                    R = R - A[i, j] * x[j]
                else:
                    R = R - A[i, j] * x0[j]
            x[i] = x0[i] + w * R / A[i, i]
        if max(abs(x - x0)) <= tol:
            break
        x0 = np.copy(x)

    return x