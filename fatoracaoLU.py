import numpy as np

def solveLU(A, b):
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()

    # Decomposição LU
    for k in range(n-1):
        for i in range(k+1, n):
            m = U[i, k] / U[k, k]
            L[i, k] = m
            for j in range(k, n):
                U[i, j] = U[i, j] - m * U[k, j]

    # Substituição progressiva: Ly = b
    y = np.zeros(n)
    for i in range(n):
        soma = 0
        for j in range(i):
            soma += L[i, j] * y[j]
        y[i] = (b[i] - soma) / L[i, i]

    # Substituição regressiva: Ux = y
    x = np.zeros(n)
    for i in reversed(range(n)):
        soma = 0
        for j in range(i+1, n):
            soma += U[i, j] * x[j]
        x[i] = (y[i] - soma) / U[i, i]

    return x, L, U