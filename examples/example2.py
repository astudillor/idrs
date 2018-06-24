from idrs import idrs
from Convergence import Convergence


import numpy as np
import time


def example_4_6(n):
    A = np.zeros((n, n), dtype = np.float)
    for i in range(0, n):
        A[i, i] = float(i + 1)
        if i < n - 1:
            A[i, i + 1] = 1e-5

    A[0][0] = 1.0 * 1.0e-8
    A[1][1] = 2.0 * 1.0e-8
    A[2][2] = 3.0 * 1.0e-8
    A[3][3] = 4.0 * 1.0e-8
    A[4][4] = 5.0 * 1.0e-8
    b = np.ones((n,))
    return A, b


def get_smallest_eigenvectors(A):
    evalues, evectors = np.linalg.eig(A)
    n = A.shape[0]
    index = []
    for i, ev in enumerate(evalues):
        if np.abs(ev) < 1.0:
            index.append(i)
    return evectors[:, index]

def main():
    n = 100
    A, b = example_4_6(n)
    msg = "Method {:8} Time = {:6.3f} Matvec = {:d} Residual = {:g}"

    def residual(x):
        return np.linalg.norm(b - A.dot(x)) / np.linalg.norm(b)

    conv = Convergence(only_counter_iters = True)
    t = time.time()
    x, info = idrs(A, b, tol=1e-8, s=5, callback=conv)
    elapsed_time = time.time() - t
    matvec = len(conv)
    print(msg.format('IDR(5)', elapsed_time, matvec, residual(x)))

    conv = Convergence(only_counter_iters = True)
    options = dict()
    t = time.time()
    options['U0'] = get_smallest_eigenvectors(A)
    x, info = idrs(A, b, tol=1e-8, s=5, callback=conv, options = options)
    elapsed_time = time.time() - t
    matvec = len(conv)
    print(msg.format('IDR(5)', elapsed_time, matvec, residual(x)))



if __name__ == '__main__':
    main()
