import numpy as np

def gauss_eliminacao(A, b):
    """
    Eliminação Gaussiana sem pivoteamento
    A: matriz de coeficientes (numpy array)
    b: vetor de termos independentes (numpy array)
    Retorna: vetor x solução do sistema Ax = b
    """
    A = A.astype(float)  # Garantir operações em ponto flutuante
    b = b.astype(float)
    n = len(b)

    # Fase de eliminação
    for k in range(n-1):
        for i in range(k+1, n):
            m = A[i, k] / A[k, k]
            for j in range(k, n):
                A[i, j] = A[i, j] - m * A[k, j]
            b[i] = b[i] - m * b[k]

    # Substituição regressiva
    x = np.zeros(n)
    x[n-1] = b[n-1] / A[n-1, n-1]
    for i in range(n-2, -1, -1):
        soma = 0
        for j in range(i+1, n):
            soma += A[i, j] * x[j]
        x[i] = (b[i] - soma) / A[i, i]

    return x,A
