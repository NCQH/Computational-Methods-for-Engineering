import numpy as np
import _linear as linear
import decimal

'''Uneuqal Spacing'''


def _direct_fit_polynomial(x, y):
    '''
    Đa thức phù hợp trực tiếp
    Input:
        x: list of x values
        y: list of y values
    Output:
        f: function
        f_str: string of function
    '''
    if len(x) != len(y):
        print("Error: Number of x values must equal number of y values")
        return

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)

    n = y.shape[0]

    X = np.zeros(shape=(n, n), dtype=float)
    Y = y.reshape(n, 1)

    X[:, 0] = 1
    for i in range(0, n):
        for j in range(1, n):
            X[i, j] = x[i] ** j
    
    A = linear._solve_gauss_elimination(X, Y)

    f_str = ''
    for i in range(0, n-1):
        f_str += str(A[i, 0]) + ' * x**' + str(i) + ' + '
    f_str += str(A[n-1, 0]) + '* x **' + str(n-1)
    f = 'lambda x: ' + f_str
    f = eval(f)

    return f, f_str

def _lagrange_polynomials(x, y):
    if len(x) != len(y):
        print("Error: Number of x values must equal number of y values")
        return
    n = len(x)

    f = 0
    f_str = ''
    for i in range(n):
        l = 1
        l_str = ''
        for j in range(n):
            if i != j:
                l *= (x[i] - x[j])
                l_str += '(x - ' + str(x[j]) + ')*'
        l = y[i] / l
        l_str = str(l) + '*' + l_str
        f += l
        f_str += l_str
        if i != n-1:
            f_str = f_str[:-1] + ' + '
    f_str = f_str[:-1]
    f = 'lambda x: ' + f_str
    f = eval(f)
    
    return f, f_str

def _neville_algorithm(x, y, x0):
    if len(x) != len(y):
        print("Length X = ", len(x), "Length Y = ", len(y))
        print("Error: Number of x values must equal number of y values")
        return
    n = len(x)
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, 0] = y[i]
        for j in range(1, i+1):
            Q[i, j] = ((x0 - x[i-j]) * Q[i, j-1] - (x0 - x[i]) * Q[i-1, j-1]) / (x[i] - x[i-j])
    return Q[n-1, n-1]


# Not done
def _divided_difference(x, y):
    if len(x) != len(y):
        print("Error: Number of x values must equal number of y values")
        return
    n = len(x)

    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, 0] = y[i]
        for j in range(1, i+1):
            Q[i, j] = (Q[i, j-1] - Q[i-1, j-1]) / (x[i] - x[i-j])
    return Q


'''Equal Spacing'''


def _newton_forward_polynomial(x, y, x0):
    if len(x) != len(y):
        print("Error: Number of x values must equal number of y values")
        return
    if not _is_equal_space(x):
        print("Error: x values must be equally spaced")
        return
    h = x[1] - x[0]
    s = (x0 - x[0]) / h

    n = len(x)
    Q = np.zeros((n, n))
    Q[:, 0] = y
    for i in range(1, n):
        for j in range(n-i):
            Q[j, i] = Q[j+1, i-1] - Q[j, i-1]
    f = y[0]
    for i in range(1, n):
        p = 1
        for j in range(i):
            p *= (s - j)
        f += p * Q[0, i] / np.math.factorial(i)
    return f
    

def _least_square_polynomial(x, y, degree):
    '''
    Xấp xỉ đa thức bằng phương pháp bình phương tối thiểu
    Input:
        x: list of x values
        y: list of y values
        n: degree of polynomial
    Output:
        f: function
    '''
    decimal.getcontext().prec = 30
    
    if len(x) != len(y):
        print("Error: Number of x values must equal number of y values")
        return
    
    n = len(x)
    X = np.zeros((degree+1, degree+1), dtype=decimal.Decimal)
    Y = np.zeros((degree+1, 1), dtype=decimal.Decimal)

    X[0, 0] = n
    for i in range(degree):
        X[0, i+1] = _sum_(x, i+1)
    Y[0, 0] = _sum_(y, 1)

    for i in range(1, degree+1):
        for j in range(degree+1):
            X[i, j] = _sum_(x, j + i)
        Y[i, 0] = _sum_(pow(x, i) * y, 1)

    A = linear._solve_gauss_elimination(X, Y, print_mt = False)

    return A[::-1].transpose()

def _sum_(x, degree):
    sum = 0
    n = len(x)
    for i in range(n):
        sum += x[i]**degree
    return sum

def pow(x, n):
    x = np.array(x, dtype=float)
    X = 1
    for i in range(n):
        X *= x
    return X

def _is_equal_space(x):
    if len(x) < 3:
        return True
    x = np.array(x, dtype=float)
    dx = np.diff(x)
    ddx = np.diff(dx)
    eps = np.finfo(float).eps
    if sum(ddx) < eps:
        return True
    return False


x = [0, -1, -2, -3, -4, -5, -6, -7, -8]
y = [2.16, 1.88, 1.51, 1.19, 0.94, 0.67, 0.36, 0.04, -0.27]

A = _least_square_polynomial(x, y, 1)
print(A)