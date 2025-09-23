import numpy as np

def matrix_to_couplings(Q):
    """
    Converts a symmetric matrix Q into coupling (J) and bias (h) dictionaries for use in Ising or QUBO models.

    Parameters:
        Q (numpy.ndarray): A square matrix representing the quadratic coefficients of a QUBO or Ising problem.

    Returns:
        tuple:
            J (dict): Dictionary of couplings, where keys are tuples (i, j) with i < j, and values are the sum Q[i, j] + Q[j, i].
            h (dict): Dictionary of biases, where keys are indices i and values are the diagonal elements Q[i, i].

    Notes:
        - The input matrix Q is symmetrized before extracting couplings and biases.
        - Suitable function name: `qubo_matrix_to_ising_couplings`
    """
    Q = 0.5 * (Q + Q.T)
    # Return J, h
    N = Q.shape[0]
    J = {}
    h = {}
    for i in range(N):
        for j in range(i + 1, N):
            if Q[i, j] != 0:
                J[(i, j)] = Q[i, j] + Q[j, i]
        if Q[i, i] != 0:
            h[i] = Q[i, i]
    return J, h

def qubo_to_ising(Q):
    """
    Convert a QUBO matrix Q to Ising model parameters J and h.
    Q: numpy array (n x n), not necessarily symmetric.
    Returns:
        J: dict of (i, j): value for i < j
        h: dict of i: value
    """
    Q = np.array(Q)
    Q = (Q + Q.T) / 2

    n = Q.shape[0]
    J = {}
    h = {}

    # Couplings
    for i in range(n):
        for j in range(i + 1, n):
            if Q[i, j] != 0:
                J[(i, j)] = Q[i, j] / 2

    # Local fields
    for i in range(n):
        h[i] = sum(-Q[i, j] / 2 for j in range(n))

    # Energy shift (constant term)
    c = np.sum(Q) / 4 + np.sum(np.diag(Q)) / 4
    return J, h, c

