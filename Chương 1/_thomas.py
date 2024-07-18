'''Thomas algorithm and its helper function'''
import numpy as np

def _matrix_to_thomas_form(A):
    '''
    Input:
        A: 2D array
    Output:
        A_2: 2D array in Thomas form
    '''
    A = np.array(A, dtype=float)
    row = A.shape[0]
    col = A.shape[1]
    A_2 = np.zeros((row, 3))

    for i in range(row):
        if i == 0:
            A_2[i][1] = A[i][i]
            A_2[i][2] = A[i][i+1]
        elif i == row-1:
            A_2[i][0] = A[i][i-1]
            A_2[i][1] = A[i][i]
        else:
            A_2[i][0] = A[i][i-1]
            A_2[i][1] = A[i][i]
            A_2[i][2] = A[i][i+1]

    return A_2


def _thomas_matrix_transform(A):
    '''
    Input:
        A: 2D array in Thomas form
    Output:
        A: 2D array in Thomas form after transformation
    '''
    for i in range(1, A.shape[0]):
        # store em
        A[i][0] = A[i][0] / A[i - 1][1]
        # eliminate
        A[i][1] = A[i][1] - A[i][0] * A[i - 1][2]
    return A


def _lu_factorization_thomas(A, transfered=False, print_mt=False):
    '''
    Input:
        A: 2D array
        transfered: if A is in Thomas form
        print_mt: print matrix
    Output:
        L: 2D array lower triangular matrix
        U: 2D array upper triangular matrix
    '''
    if print_mt:
        print("---LU factorization---")

    if not transfered:
        A = _matrix_to_thomas_form(A)
    if A.shape[1] != 3:
        return 'Error: Input or transpose matrix error'
    A = _thomas_matrix_transform(A)
    L = np.zeros(shape=(A.shape[0], A.shape[0]))
    U = np.zeros(shape=(A.shape[0], A.shape[0]))
    row = A.shape[0]

    for i in range(row):
        L[i][i] = 1
        U[i][i] = A[i][1]
    for i in range(row - 1):
        L[i + 1][i] = A[i + 1][0]
        U[i][i + 1] = A[i][2]

    if print_mt:
        print("---Ma trận L---")
        print(L)
        print("---Ma trận U---")
        print(U)

    return L, U


def _solve_thomas_algorithm(A, B_t, transfered=False, print_mt=False):
    '''
    Input:
        A: 2D array
        B_t: 2D array
        transfered: if A is in Thomas form
        print_mt: print matrix
    Output:
        X: 2D array solution
    '''
    if print_mt:
        print("---Solve equation using Thomas algorithm---")

    if not transfered:
        A = _matrix_to_thomas_form(A)
    if A.shape[1] != 3:
        return 'Error: Input or transpose matrix error'
    
    A = _thomas_matrix_transform(A)  # 4 * 3 matrix
    
    B = np.array(B_t, dtype=float) # 4 * 1

    row = A.shape[0]
    col = B.shape[1]
    X = np.zeros(shape=(row, col)) #4*1
    
    for i in range(0, col): #number of column
        for j in range(1, row): #number of row
            B[j][i] = B[j][i] - A[j][0]*B[j-1][i]
        X[X.shape[0]-1][i] = B[X.shape[0]-1][i]/A[X.shape[0]-1][1]
        for j in range(X.shape[0]-2, -1, -1):
            X[j][i] = (B[j][i] - A[j][2]*X[j+1][i])/A[j][1]
    if print_mt:
        print("---Solution---")
        print(X)
    return X

def _inverse_thomas_algorithm(A, print_mt=False):
    '''
    Input:
        A: 2D array
        print_mt: print matrix
    Output:
        A_I: 2D array inverse matrix
    '''
    if print_mt:
        print("---Tìm ma trận nghịch đảo---")
    A = np.array(A, dtype=float)
    I = np.eye(A.shape[0])

    A_I = _solve_thomas_algorithm(A, I, transfered=False)
    if print_mt:
        print("---Ma trận nghịch đảo---")
        print(A_I)
    return A_I

def _determinant_thomas_algorithm(A, print_mt=False):
    '''
    Input:
        A: 2D array
        print_mt: print matrix
    Output:
        det: float determinant
    '''
    if print_mt:
        print("---Tính định thức---")
    A = np.array(A, dtype=float)
    A = _matrix_to_thomas_form(A)
    A = _thomas_matrix_transform(A)
    det = 1
    for i in range(A.shape[0]):
        det *= A[i][1]
        
    if print_mt:
        print("Định thức của ma trận:", det)
    return det