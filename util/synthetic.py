import numpy as np


def generate_sparse_least_squares(m, n, rho):
    """
    Generates a Sparse Least Squares problem according to the specifications in Nesterov's paper.

    Parameters:
    m (int): Number of rows in matrix A.
    n (int): Number of columns in matrix A, n > m.
    rho (float): Sparsity and magnitude control parameter.

    Returns:
    A (numpy.ndarray): Generated dense matrix A of shape (m, n).
    b (numpy.ndarray): Generated vector b of shape (m,).
    x_star (numpy.ndarray): Sparse solution x* of shape (n,).

    Example:
    m, n, rho = 10, 20, 0.1  # dimensions and sparsity level
    A, b, x_star = generate_sparse_least_squares(m, n, rho)

    # print("Matrix A:\n", A)
    # print("Vector b:\n", b)
    print("Sparse solution x*:\n", x_star)
    """
    # Generate a dense matrix A with elements uniformly distributed in [-1, 1]
    A = np.random.uniform(-1, 1, (m, n))

    # Generate a sparse solution x_star
    x_star = np.zeros(n)
    # Ensure sparsity in the solution
    non_zero_indices = np.random.choice(n, int(n * rho), replace=False)
    x_star[non_zero_indices] = np.random.normal(0, 1, int(n * rho))

    # Calculate the vector b = A*x_star + noise
    noise = np.random.normal(0, 0.1, m)  # adding small Gaussian noise
    b = np.dot(A, x_star) + noise

    return A, b, x_star


def generate_sparse_least_squares_2(m, n, rho):
    """
    Generates a Sparse Least Squares problem as described by Nesterov, with rho influencing
    the magnitude of components in x*.

    Parameters:
    m (int): Number of rows in matrices B and A.
    n (int): Number of columns in matrices B and A, typically n > m.
    rho (float): Parameter influencing the threshold for non-zero components in x*.

    Returns:
    A (numpy.ndarray): Generated matrix A of shape (m, n).
    b (numpy.ndarray): Generated vector b of shape (m).
    x_star (numpy.ndarray): Sparse solution vector x* of shape (n).

    Example:
    m, n = 10, 20  # dimensions of the matrix
    rho = 0.1        # control parameter for the magnitude of x*
    A, b, x_star = generate_sparse_least_squares(m, n, rho)
    print("Matrix A:\n", A[:5])  # print first 5 rows of A to keep output manageable
    print("Vector b:\n", b[:5])  # print first 5 elements of b
    print("Sparse solution x* (non-zero values):\n", x_star)
    """
    # Generate matrix B with elements uniformly distributed in [-1, 1]
    B = np.random.uniform(-1, 1, (m, n))

    # Generate random vector v* with elements uniformly distributed in [0, 1]
    v_star = np.random.uniform(0, 1, m)

    # Normalize v* to get y*
    y_star = v_star / np.linalg.norm(v_star)

    # Compute B^T * y* and sort indices by decreasing order of their absolute values
    B_transpose_y_star = np.dot(B.T, y_star)
    sorted_indices = np.argsort(-np.abs(B_transpose_y_star))

    # Initialize x* with zeros
    x_star = np.zeros(n)

    # Threshold for significant non-zero values
    threshold = np.abs(B_transpose_y_star[sorted_indices[int(n * rho)]])

    # Assign non-zero values to x* based on rho and threshold
    for i in range(n):
        if np.abs(B_transpose_y_star[i]) >= threshold:
            sign = np.sign(B_transpose_y_star[i])
            x_star[i] = sign * np.random.uniform(
                0.1 * rho, rho
            )  # random value influenced by rho

    # Compute matrix A as B with scaled columns
    A = B * x_star

    # Compute the vector b = Ax*
    b = np.dot(A, x_star)

    return A, b, x_star
