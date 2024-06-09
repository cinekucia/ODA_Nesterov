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
    # noise = np.random.normal(0, 0.1, m)  # adding small Gaussian noise
    b = np.dot(A, x_star) # + noise

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


def generate_test_data(m, m_star, n, rho):
    """_summary_

    Args:
        m (_type_): _description_
        m_star (_type_): m_star < m, the number of components of the optimal 
        solution x_star of problem
        n (_type_): _description_
        rho (_type_): rho > 0, parameter responsible for the size of x_star
    """
    # Generate randomly matrix B in R^{m x n} with elements uniformly distributed in [-1, 1]
    B = np.random.uniform(-1, 1, (m, n))
    # Generate randomly vector v_star in R^m with elements uniformly distributed in [0, 1]
    v_star = np.random.uniform(0, 1, m)
    # Define y_star = v_star / ||v_star||_2
    y_star = v_star / np.linalg.norm(v_star, ord=2) # normalize v_star
    # Compute B^T * y_star
    # sort the entries of B^T * y_star in decreasing order of their absolute values
    B_transpose_y_star = np.dot(B.T, y_star)
    sorted_indices = np.argsort(-np.abs(B_transpose_y_star))
    B_sorted = B[:, sorted_indices]
    # for i = 1, 2, ..., n, define a_i = alpha_i * B_i, where
    # alpha_i = 1 / |<b_i, y_star>| for i = 1, 2, ..., m_star
    # alpha_i = epsilon_i / |<b_i, y_star>| otherwise
    # where epsilon_i are uniformly distributed in [0, 1]
    A = np.zeros((m, n))
    for i in range(n):
        denominator = np.abs(np.dot(B_sorted[:, i], y_star))
        if i < m_star:
            numerator = 1
        else:
            numerator = np.random.uniform(0, 1)  # epsilon_i
        alpha_i = numerator / denominator
        A[:, i] = alpha_i * B_sorted[:, i]
    # for i = 1, 2, ..., n, generate components of the primal solution x_star:
    # x_star_i = epsilon_i * sign(<a_i, y_star>) for i <= m_star and 0 otherwise
    # where epsilon _i are uniformly distributed in [0, rho / sqrt(m_star)]
    x_star = np.zeros(n)
    for i in range(m_star):
        epsilon_i = np.random.uniform(0, rho / np.sqrt(m_star))
        x_star[i] = epsilon_i * np.sign(np.dot(A[:, i], y_star))
    # Generate the vector b = A * x_star
    b = y_star + np.dot(A, x_star)
    phi_star = 1/2 * np.linalg.norm(y_star)**2 + np.linalg.norm(x_star, ord=1)
    return A, b, x_star, phi_star
